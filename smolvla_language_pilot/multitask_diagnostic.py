"""
Sweep all instruction variants across every rollout in a LIBERO h5 file.

For each demo the script:
  1. Runs the policy once with the base instruction to get reference actions.
  2. Runs the policy once per variant (loaded from the variants JSON).
  3. Computes the absolute L2 distance between variant actions and reference
     actions at every timestep.

After processing all rollouts the averaged L2 curve (over all tasks and
rollouts) is plotted per variant type and saved to --output-dir.

Usage:
    uv run python -m smolvla_language_pilot.variant_sweep \
        --rollout data/h5/libero/libero_object_tasks5_rollouts50.h5 \
        --variants-json smolvla_language_pilot/instruction_variants.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from vla.models.smolvla import smolvla


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants-json",
        default="smolvla_language_pilot/instruction_variants.json",
        help="Path to the instruction_variants.json produced by instruction_variants.py.",
    )
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="smolvla_language_pilot/outputs",
        help="Directory where the output plot is saved.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _load_policy(checkpoint: str, device: str) -> dict:
    policy_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    policy, model_id, _ = smolvla(checkpoint, str(policy_device))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(policy_device)}},
    )
    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": policy_device,
        "dtype": next(policy.parameters()).dtype,
    }


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run(instruction: str, images: torch.Tensor, states: torch.Tensor, bundle: dict, seed: int) -> torch.Tensor:
    """Run the policy on a trajectory, re-querying at every timestep and taking only the
    first predicted action from each chunk (receding-horizon, window=1).

    The model sees a fresh observation at every step — eliminating the chunking sawtooth
    and giving a pure measure of how the language instruction affects the immediate next action.
    """
    _set_seed(seed)
    policy = bundle["policy"]
    pre = bundle["preprocessor"]
    post = bundle["postprocessor"]
    device = bundle["device"]
    dtype = bundle["dtype"]
    use_amp = device.type == "cuda"

    actions = []
    with torch.inference_mode():
        for img, state in zip(images, states):
            batch = {"observation.state": state.unsqueeze(0).to(device, dtype=dtype), "task": [instruction]}
            for key in policy.config.input_features:
                if key.startswith("observation.images."):
                    batch[key] = img.unsqueeze(0).to(device, dtype=dtype)
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Fresh forward pass at every step; only the first action of the chunk is used.
                chunk = policy.predict_action_chunk(batch)   # (1, chunk_size, action_dim)
            actions.append(post(chunk[:, 0, :]).squeeze(0).cpu())
    return torch.stack(actions)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_variants(json_path: str) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """Return (variants_map, h5_paths).

    variants_map: {base_instruction: {variant_type: [variant, ...]}}
    h5_paths: deduplicated list of rollout h5 files referenced in the JSON.
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, list[str]]] = {}
    h5_paths: list[str] = []
    seen_paths: set[str] = set()
    for entry in payload.get("rollouts", []):
        instruction = entry.get("base_instruction", "").strip()
        variants_by_type = entry.get("variants", {})
        rollout_path = entry.get("rollout_path", "")
        if isinstance(variants_by_type, dict) and instruction:
            mapping[instruction] = variants_by_type
        if rollout_path and rollout_path not in seen_paths:
            h5_paths.append(rollout_path)
            seen_paths.add(rollout_path)
    return mapping, h5_paths


def _iter_demos(h5_path: str, keep_instructions: set[str] | None = None):
    """Yield (task_index, task_instruction, images, states) for every demo.

    If keep_instructions is provided, only demos whose instruction is in that
    set are yielded.
    """
    with h5py.File(h5_path, "r") as f:
        demos = f["demonstrations"]
        for demo_key in sorted(demos.keys()):
            demo = demos[demo_key]
            task_index = int(demo.attrs.get("task_index", -1))
            task_instruction = str(demo.attrs.get("task", "")).strip()
            if keep_instructions is not None and task_instruction not in keep_instructions:
                continue
            raw = np.array(demo["observations/cam0"])                                    # (T, H, W, C)
            images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0          # (T, C, H, W)
            states = torch.tensor(np.array(demo["states"]), dtype=torch.float32)         # (T, S)
            yield task_index, task_instruction, images, states


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading variants JSON …")
    variants_map, h5_paths = _load_variants(args.variants_json)
    if not variants_map:
        raise ValueError(f"No variants found in {args.variants_json}. Run instruction_variants.py first.")
    if not h5_paths:
        raise ValueError(f"No rollout_path entries found in {args.variants_json}.")
    known = "\n    ".join(variants_map.keys())
    print(f"  Found variants for {len(variants_map)} instructions:\n    {known}")
    print(f"  H5 files to sweep: {h5_paths}")

    print("Loading policy …")
    bundle = _load_policy(args.checkpoint, args.device)

    type_l2_curves: dict[str, list[torch.Tensor]] = defaultdict(list)
    n_processed = 0

    for h5_path in h5_paths:
        print(f"\nSweeping {h5_path} …")
        for task_index, task_instruction, images, states in _iter_demos(h5_path, keep_instructions=set(variants_map)):
            variants_by_type = variants_map[task_instruction]
            n_variants = sum(len(v) for v in variants_by_type.values())
            print(f"  [task {task_index}] '{task_instruction}' — {images.shape[0]} steps, {n_variants} variants", flush=True)

            base_actions = _run(task_instruction, images, states, bundle, args.seed)

            for variant_type, variant_list in variants_by_type.items():
                for i, variant_text in enumerate(variant_list):
                    print(f"    [{variant_type} {i+1}/{len(variant_list)}] '{variant_text}'", flush=True)
                    variant_actions = _run(variant_text, images, states, bundle, args.seed)
                    T = min(base_actions.shape[0], variant_actions.shape[0])
                    l2 = torch.norm(variant_actions[:T] - base_actions[:T], dim=-1)
   zzz                 type_l2_curves[variant_type].append(l2)

            n_processed += 1

    print(f"\nProcessed {n_processed} demos.")

    if n_processed == 0:
        raise RuntimeError(
            f"No matching demos found in {h5_paths} for the instructions in {args.variants_json}.\n"
            f"Re-run instruction_variants.py with the correct --rollout flag."
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def _stack(curves: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        max_T = max(c.shape[0] for c in curves)
        padded = np.full((len(curves), max_T), np.nan, dtype=np.float32)
        for i, c in enumerate(curves):
            padded[i, : c.shape[0]] = c.numpy()
        return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

    type_mean: dict[str, np.ndarray] = {}
    type_std: dict[str, np.ndarray] = {}
    for vtype, curves in type_l2_curves.items():
        type_mean[vtype], type_std[vtype] = _stack(curves)

    all_curves = [c for curves in type_l2_curves.values() for c in curves]
    overall_mean, overall_std = _stack(all_curves)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for vtype, mean in type_mean.items():
        x = np.arange(len(mean))
        ax.plot(x, mean, label=vtype)
        ax.fill_between(x, mean - type_std[vtype], mean + type_std[vtype], alpha=0.15)

    x_all = np.arange(len(overall_mean))
    ax.plot(x_all, overall_mean, color="black", linewidth=2, linestyle="--", label="overall mean")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean L2 Action Distance")
    ax.set_title(f"Average L2 Action Distance per Variant Type\n({n_processed} demos × all variants)")
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "variant_sweep_avg_l2.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot → {out_path}")

    summary = {
        "n_demos_processed": n_processed,
        "variant_types": {
            vtype: {
                "mean_l2_per_timestep": type_mean[vtype].tolist(),
                "std_l2_per_timestep": type_std[vtype].tolist(),
                "n_rollouts": len(type_l2_curves[vtype]),
            }
            for vtype in type_mean
        },
        "overall": {
            "mean_l2_per_timestep": overall_mean.tolist(),
            "std_l2_per_timestep": overall_std.tolist(),
        },
    }
    summary_path = output_dir / "variant_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary → {summary_path}")


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
