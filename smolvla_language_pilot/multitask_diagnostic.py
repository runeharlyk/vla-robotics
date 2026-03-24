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

from vla.models.smolvla import SmolVLAPolicy


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
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="smolvla_language_pilot/outputs",
        help="Directory where the output plot is saved.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Stop after processing this many demos in total. Useful for quick smoke-tests (e.g. --max-demos 1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _load_policy(checkpoint: str, device: str) -> dict:
    policy_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    policy = SmolVLAPolicy(checkpoint, action_dim=7, device=str(policy_device))
    policy.eval()
    return {
        "policy": policy,
        "device": policy_device,
        "dtype": policy.dtype,
    }


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run(instruction: str, images: torch.Tensor, states: torch.Tensor, bundle: dict, seed: int) -> torch.Tensor:
    """Run the policy on a trajectory, re-querying at every timestep and taking only the
    first predicted action from each chunk (receding-horizon, window=1).

    The model sees a fresh observation at every step — eliminating the chunking sawtooth
    and giving a pure measure of how the language instruction affects the immediate next action.
    """
    _set_seed(seed)
    policy = bundle["policy"]
    device = bundle["device"]

    actions = []
    with torch.inference_mode():
        for img, state in zip(images, states):
            action = policy.predict_action(img, instruction, state)
            actions.append(action.cpu())
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


def _plot_l2_curves(
    type_mean: dict[str, np.ndarray],
    type_std: dict[str, np.ndarray],
    overall_mean: np.ndarray,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for vtype, mean in type_mean.items():
        x = np.arange(len(mean))
        ax.plot(x, mean, label=vtype)
        ax.fill_between(x, mean - type_std[vtype], mean + type_std[vtype], alpha=0.15)

    x_all = np.arange(len(overall_mean))
    ax.plot(x_all, overall_mean, color="black", linewidth=2, linestyle="--", label="overall mean")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot → {out_path}")


def _safe_task_dir_name(task_instruction: str) -> str:
    task_name = task_instruction.strip().lower()
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in task_name)
    return safe[:80] if safe else "unknown_task"


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
    type_rel_l2_curves: dict[str, list[torch.Tensor]] = defaultdict(list)
    task_type_l2_curves: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    task_type_rel_l2_curves: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
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
                    base_norm = torch.norm(base_actions[:T], dim=-1).clamp(min=1e-8)
                    rel_l2 = l2 / base_norm
                    type_l2_curves[variant_type].append(l2)
                    type_rel_l2_curves[variant_type].append(rel_l2)
                    task_type_l2_curves[task_instruction][variant_type].append(l2)
                    task_type_rel_l2_curves[task_instruction][variant_type].append(rel_l2)

            n_processed += 1
            if args.max_demos is not None and n_processed >= args.max_demos:
                break
        if args.max_demos is not None and n_processed >= args.max_demos:
            break

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

    type_rel_mean: dict[str, np.ndarray] = {}
    type_rel_std: dict[str, np.ndarray] = {}
    for vtype, curves in type_rel_l2_curves.items():
        type_rel_mean[vtype], type_rel_std[vtype] = _stack(curves)

    all_rel_curves = [c for curves in type_rel_l2_curves.values() for c in curves]
    overall_rel_mean, overall_rel_std = _stack(all_rel_curves)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    _plot_l2_curves(
        type_mean=type_mean,
        type_std=type_std,
        overall_mean=overall_mean,
        out_path=output_dir / "multitask_avg_l2.png",
        title=f"Average L2 Action Distance per Variant Type\n({n_processed} demos × all variants)",
        y_label="Mean L2 Action Distance",
    )

    _plot_l2_curves(
        type_mean=type_rel_mean,
        type_std=type_rel_std,
        overall_mean=overall_rel_mean,
        out_path=output_dir / "multitask_avg_rel_l2.png",
        title=f"Average Relative L2 Action Distance per Variant Type\n({n_processed} demos × all variants)",
        y_label="Mean Relative L2 Action Distance",
    )

    per_task_dir = output_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)
    for task_instruction, by_type_abs in task_type_l2_curves.items():
        by_type_rel = task_type_rel_l2_curves[task_instruction]

        task_mean_abs: dict[str, np.ndarray] = {}
        task_std_abs: dict[str, np.ndarray] = {}
        for vtype, curves in by_type_abs.items():
            task_mean_abs[vtype], task_std_abs[vtype] = _stack(curves)

        task_mean_rel: dict[str, np.ndarray] = {}
        task_std_rel: dict[str, np.ndarray] = {}
        for vtype, curves in by_type_rel.items():
            task_mean_rel[vtype], task_std_rel[vtype] = _stack(curves)

        task_all_abs = [c for curves in by_type_abs.values() for c in curves]
        task_all_rel = [c for curves in by_type_rel.values() for c in curves]
        task_overall_abs, _ = _stack(task_all_abs)
        task_overall_rel, _ = _stack(task_all_rel)

        task_dir = per_task_dir / _safe_task_dir_name(task_instruction)
        task_dir.mkdir(parents=True, exist_ok=True)
        _plot_l2_curves(
            type_mean=task_mean_abs,
            type_std=task_std_abs,
            overall_mean=task_overall_abs,
            out_path=task_dir / "task_avg_l2.png",
            title=f"Task: {task_instruction}\nAverage L2 Action Distance per Variant Type",
            y_label="Mean L2 Action Distance",
        )
        _plot_l2_curves(
            type_mean=task_mean_rel,
            type_std=task_std_rel,
            overall_mean=task_overall_rel,
            out_path=task_dir / "task_avg_rel_l2.png",
            title=f"Task: {task_instruction}\nAverage Relative L2 Action Distance per Variant Type",
            y_label="Mean Relative L2 Action Distance",
        )

    summary = {
        "n_demos_processed": n_processed,
        "variant_types": {
            vtype: {
                "mean_l2_per_timestep": type_mean[vtype].tolist(),
                "std_l2_per_timestep": type_std[vtype].tolist(),
                "mean_rel_l2_per_timestep": type_rel_mean[vtype].tolist(),
                "std_rel_l2_per_timestep": type_rel_std[vtype].tolist(),
                "n_rollouts": len(type_l2_curves[vtype]),
            }
            for vtype in type_mean
        },
        "overall": {
            "mean_l2_per_timestep": overall_mean.tolist(),
            "std_l2_per_timestep": overall_std.tolist(),
            "mean_rel_l2_per_timestep": overall_rel_mean.tolist(),
            "std_rel_l2_per_timestep": overall_rel_std.tolist(),
        },
        "per_task": {
            task_instruction: {
                "variant_types": {
                    vtype: {
                        "mean_l2_per_timestep": _stack(task_type_l2_curves[task_instruction][vtype])[0].tolist(),
                        "std_l2_per_timestep": _stack(task_type_l2_curves[task_instruction][vtype])[1].tolist(),
                        "mean_rel_l2_per_timestep": _stack(task_type_rel_l2_curves[task_instruction][vtype])[0].tolist(),
                        "std_rel_l2_per_timestep": _stack(task_type_rel_l2_curves[task_instruction][vtype])[1].tolist(),
                        "n_rollouts": len(task_type_l2_curves[task_instruction][vtype]),
                    }
                    for vtype in task_type_l2_curves[task_instruction]
                }
            }
            for task_instruction in task_type_l2_curves
        },
    }
    summary_path = output_dir / "multitask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary → {summary_path}")


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
