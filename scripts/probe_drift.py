"""Offline fused-feature drift probe on held-out nuisances.

Loads a trained checkpoint (or the public HF checkpoint) and measures how far
the fused conditioning representation (``encode_prefix_pooled``) drifts under
nuisance views the training objective never saw: held-out corruptions
(glass_blur, zoom_blur) and held-out paraphrase types (sentence_structure,
verbosity).  This is the load-bearing invariance probe from
``RESEARCH_DIRECTION.md`` as a standalone eval — the training-time telemetry
only covers the TRAIN nuisance types.

The nuisance view plan (frame selection, corruption type/severity, paraphrase
choice) is derived from a single seeded generator, so runs with the same seed
apply IDENTICAL views to every checkpoint: arm-to-arm differences are model
differences, not view noise.

Examples:
    uv run python scripts/probe_drift.py --checkpoint-dir checkpoints/sft/spatial_both_seed42_v3/last
    uv run python scripts/probe_drift.py --nuisances train   # sanity vs training telemetry
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import typer

from vla.models.smolvla import SmolVLAPolicy
from vla.results_registry import get_git_info, now_iso, write_json
from vla.training.invariance import (
    DEFAULT_HELDOUT_NOISE_TYPES,
    DEFAULT_HELDOUT_VARIANT_TYPES,
    DEFAULT_TRAIN_NOISE_TYPES,
    DEFAULT_TRAIN_VARIANT_TYPES,
    _import_noise,
    _to_float01,
    check_corruption_deps,
    feature_drift_per_sample,
    load_instruction_variants,
)
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VARIANTS_PATH = "smolvla_language_pilot/instruction_variants.json"


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_resamples: int = 10_000
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summarize(
    values: list[float], labels: list[str], rng: np.random.Generator
) -> dict:
    """Mean + CI overall and per nuisance-type label."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    lo, hi = _bootstrap_ci(arr, rng)
    out = {"n": int(arr.size), "mean": float(arr.mean()), "ci95": [lo, hi], "per_type": {}}
    for t in sorted(set(labels)):
        sub = arr[np.asarray([lb == t for lb in labels])]
        t_lo, t_hi = _bootstrap_ci(sub, rng)
        out["per_type"][t] = {"n": int(sub.size), "mean": float(sub.mean()), "ci95": [t_lo, t_hi]}
    return out


def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    checkpoint_dir: Path = typer.Option(
        None,
        "--checkpoint-dir",
        help="Trained checkpoint dir (e.g. checkpoints/sft/spatial_both_seed42_v3/last). "
        "Omit to probe the base HF checkpoint.",
    ),
    libero_suite: str = typer.Option("spatial", "--libero-suite", "-l"),
    num_demos: int = typer.Option(None, "--num-demos", help="Demos to load (default: all)"),
    num_frames: int = typer.Option(512, "--num-frames", help="Frames sampled for the probe."),
    micro_batch: int = typer.Option(16, "--micro-batch"),
    seed: int = typer.Option(42, "--seed"),
    nuisances: str = typer.Option(
        "heldout", "--nuisances", help="heldout (eval nuisances) | train (sanity vs telemetry)."
    ),
    severities: str = typer.Option(
        None,
        "--severities",
        help="Comma-separated corruption severities (default: 3,4,5 heldout / 1,2,3 train).",
    ),
    variants_path: str = typer.Option(DEFAULT_VARIANTS_PATH, "--variants-path"),
    out_dir: Path = typer.Option(Path("results/probes"), "--out-dir"),
) -> None:
    """Measure per-modality fused-feature drift of a checkpoint under nuisances."""
    if nuisances not in ("heldout", "train"):
        raise typer.BadParameter("--nuisances must be 'heldout' or 'train'")
    noise_types = DEFAULT_HELDOUT_NOISE_TYPES if nuisances == "heldout" else DEFAULT_TRAIN_NOISE_TYPES
    variant_types = (
        DEFAULT_HELDOUT_VARIANT_TYPES if nuisances == "heldout" else DEFAULT_TRAIN_VARIANT_TYPES
    )
    sev = tuple(int(s) for s in severities.split(",")) if severities else (
        (3, 4, 5) if nuisances == "heldout" else (1, 2, 3)
    )
    NoiseConfig, apply_noise = _import_noise()
    check_corruption_deps(noise_types)

    seed_everything(seed)
    device = get_device()

    from vla.data.libero import LiberoSFTDataset

    dataset = LiberoSFTDataset(libero_suite, num_demos=num_demos, seed=seed)
    action_dim, state_dim = dataset.action_dim, dataset.state_dim
    if checkpoint_dir is not None:
        ckpt_data = torch.load(checkpoint_dir / "policy.pt", map_location="cpu", weights_only=False)
        action_dim = ckpt_data.get("action_dim", action_dim)
        state_dim = ckpt_data.get("state_dim", state_dim)
    policy = SmolVLAPolicy(checkpoint=checkpoint, action_dim=action_dim, state_dim=state_dim, device=str(device))
    if checkpoint_dir is not None:
        policy.load_checkpoint(checkpoint_dir)
        logger.info("Loaded checkpoint from %s", checkpoint_dir)
    policy.model.eval()

    # Per-type variant tables so language drift can be reported per held-out
    # type.  No template fallback here: an instruction without listed variants
    # of the requested type is EXCLUDED from the language/both axes (falling
    # back to train-type templates would contaminate the held-out measurement;
    # passing it clean would dilute drift toward zero).
    variants_by_type = {vt: load_instruction_variants(variants_path, (vt,)) for vt in variant_types}

    # --- deterministic view plan -------------------------------------------
    gen = torch.Generator().manual_seed(seed)
    frame_idx = torch.randperm(len(dataset), generator=gen)[:num_frames].tolist()

    plan: list[dict] = []
    for i, idx in enumerate(frame_idx):
        sample = dataset[idx]
        instr = str(sample["instruction"])
        noise_type = noise_types[i % len(noise_types)]  # stratified
        severity = int(sev[int(torch.randint(0, len(sev), (1,), generator=gen).item())])
        vt = variant_types[i % len(variant_types)]  # stratified
        choices = variants_by_type[vt].get(instr, [])
        para = None
        if choices:
            para = choices[int(torch.randint(0, len(choices), (1,), generator=gen).item())]
        plan.append(
            {
                "idx": idx,
                "instruction": instr,
                "noise_type": noise_type,
                "severity": severity,
                "variant_type": vt,
                "paraphrase": para,
            }
        )
    n_lang = sum(1 for p in plan if p["paraphrase"] is not None)
    logger.info(
        "Probe plan: %d frames (%s), language coverage %d/%d (uncovered frames excluded from language/both axes)",
        len(plan),
        nuisances,
        n_lang,
        len(plan),
    )

    # --- encode and measure -------------------------------------------------
    drifts: dict[str, list[float]] = {"vision": [], "language": [], "both": []}
    labels: dict[str, list[str]] = {"vision": [], "language": [], "both": []}
    autocast = torch.autocast(device_type=policy.device.type, dtype=policy.dtype)

    for start in range(0, len(plan), micro_batch):
        chunk = plan[start : start + micro_batch]
        imgs = torch.stack([_to_float01(dataset[p["idx"]]["image"]) for p in chunk])
        states = torch.stack([dataset[p["idx"]]["state"] for p in chunk])
        instr = [p["instruction"] for p in chunk]

        def _corrupt(img: torch.Tensor, p: dict) -> torch.Tensor:
            # Images may be multi-view (V, C, H, W); one (type, severity) per
            # sample shared across its views, matching corrupt_images().
            nconf = NoiseConfig(noise_type=p["noise_type"], severity=p["severity"])
            if img.ndim == 4:
                return torch.stack([apply_noise(view, nconf) for view in img])
            return apply_noise(img, nconf)

        corrupted = torch.stack([_corrupt(img, p) for img, p in zip(imgs, chunk, strict=True)])
        para_instr = [p["paraphrase"] if p["paraphrase"] is not None else p["instruction"] for p in chunk]
        covered = [p["paraphrase"] is not None for p in chunk]

        with torch.no_grad(), autocast:
            z_clean = policy.encode_prefix_pooled(imgs, instr, states)
            z_vis = policy.encode_prefix_pooled(corrupted, instr, states)
            z_lang = policy.encode_prefix_pooled(imgs, para_instr, states)
            z_both = policy.encode_prefix_pooled(corrupted, para_instr, states)

        d_vis = feature_drift_per_sample(z_clean, z_vis).cpu().tolist()
        d_lang = feature_drift_per_sample(z_clean, z_lang).cpu().tolist()
        d_both = feature_drift_per_sample(z_clean, z_both).cpu().tolist()
        for j, p in enumerate(chunk):
            drifts["vision"].append(d_vis[j])
            labels["vision"].append(p["noise_type"])
            if covered[j]:
                drifts["language"].append(d_lang[j])
                labels["language"].append(p["variant_type"])
                drifts["both"].append(d_both[j])
                labels["both"].append(f"{p['noise_type']}+{p['variant_type']}")

    rng = np.random.default_rng(seed)
    results = {axis: _summarize(drifts[axis], labels[axis], rng) for axis in drifts}
    for axis, r in results.items():
        if r.get("n"):
            logger.info(
                "%s drift: %.5f  ci95=[%.5f, %.5f]  n=%d", axis, r["mean"], r["ci95"][0], r["ci95"][1], r["n"]
            )

    ckpt_tag = (
        f"{checkpoint_dir.parent.name}_{checkpoint_dir.name}"
        if checkpoint_dir is not None
        else checkpoint.replace("/", "_")
    )
    record = {
        "record_type": "drift_probe",
        "recorded_at": now_iso(),
        "checkpoint": checkpoint,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
        "suite": libero_suite,
        "seed": seed,
        "nuisances": nuisances,
        "noise_types": list(noise_types),
        "severities": list(sev),
        "variant_types": list(variant_types),
        "variants_path": variants_path,
        "num_frames": len(plan),
        "language_covered_frames": n_lang,
        "results": results,
        **get_git_info(),
    }
    out_path = out_dir / f"drift_{ckpt_tag}_{nuisances}_seed{seed}.json"
    write_json(out_path, record)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    typer.run(main)
