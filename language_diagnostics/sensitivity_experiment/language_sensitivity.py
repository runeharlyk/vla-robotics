"""
Language sensitivity analysis: measure how action outputs vary with prompt modifications.

This script tests how sensitive the policy is to language variations by:
  1. Loading prompts from variant_prompt_plan_full.json
  2. For each task, running the policy with the "original" prompt to get reference actions
  3. Running the policy with all variant prompts
  4. Computing L2 distances between variant and reference actions at each timestep
  5. Aggregating and plotting results by variant type

Outputs:
  - Per-task L2 curves (absolute and relative)
  - Overall aggregated L2 curves across all tasks
  - Summary JSON with statistics
  
Usage:
    uv run python -m language_diagnostics.sensitivity_experiment.language_sensitivity \
        --rollout data/h5/libero/libero_object_tasks5_rollouts50.h5 \
        --variants-json language_diagnostics/variant_prompt_plan_full.json \
        --output-dir language_diagnostics/sensitivity_experiment/outputs
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
from vla.utils.device import get_device
from vla.utils.seed import seed_everything


device = get_device()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout",
        required=True,
        help="Path to the LIBERO h5 rollout file (e.g., data/h5/libero/libero_object_tasks5_rollouts50.h5).",
    )
    parser.add_argument(
        "--variants-json",
        default="language_diagnostics/variant_prompt_plan_full.json",
        help="Path to the variant_prompt_plan_full.json with prompt variants.",
    )
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="language_diagnostics/sensitivity_experiment/outputs",
        help="Directory where output plots and summary JSON are saved.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Stop after processing this many demos. Useful for smoke-tests (e.g. --max-demos 1).",
    )
    return parser.parse_args()


def _run(
    instruction: str,
    images: torch.Tensor,
    states: torch.Tensor,
    policy: SmolVLAPolicy,
    seed: int,
) -> torch.Tensor:
    """Run the policy on a trajectory, re-querying at every timestep.

    Takes only the first predicted action from each chunk (receding-horizon, window=1).
    The model sees a fresh observation at every step — eliminating the chunking sawtooth
    and giving a pure measure of how the language instruction affects the immediate next action.
    """
    seed_everything(seed)

    actions = []
    with torch.inference_mode():
        for img, state in zip(images, states):
            action = policy.predict_action(img, instruction, state)
            actions.append(action.cpu())
    return torch.stack(actions)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_variant_prompts(json_path: str) -> dict[str, dict[str, dict[str, str]]]:
    """Load prompts from variant_prompt_plan_full.json.

    Returns:
        {task_description: {variant_type: {variant_name: prompt_text}}}
    """
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    prompts_by_task: dict[str, dict[str, dict[str, str]]] = {}

    for task in payload.get("tasks", []):
        task_desc = task.get("task_description", "").strip()
        task_prompts = task.get("prompts", {})

        if task_desc and isinstance(task_prompts, dict):
            # Group prompts by variant type
            variants_by_type: dict[str, dict[str, str]] = defaultdict(dict)
            for variant_name, prompt_text in task_prompts.items():
                variants_by_type[variant_name][variant_name] = prompt_text

            prompts_by_task[task_desc] = dict(variants_by_type)

    return prompts_by_task


def _iter_demos(h5_path: str, keep_instructions: set[str] | None = None):
    """Yield (task_index, task_instruction, images, states) for every demo.

    If keep_instructions is provided, only demos whose instruction is in that set are yielded.
    """
    with h5py.File(h5_path, "r") as f:
        demos = f["demonstrations"]
        for demo_key in sorted(demos.keys()):
            demo = demos[demo_key]
            task_index = int(demo.attrs.get("task_index", -1))
            task_instruction = str(demo.attrs.get("task", "")).strip()
            if keep_instructions is not None and task_instruction not in keep_instructions:
                continue
            raw = np.array(demo["observations/cam0"])  # (T, H, W, C)
            images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            states = torch.tensor(np.array(demo["states"]), dtype=torch.float32)  # (T, S)
            yield task_index, task_instruction, images, states


def _plot_l2_curves(
    type_mean: dict[str, np.ndarray],
    type_std: dict[str, np.ndarray],
    overall_mean: np.ndarray,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    """Plot L2 distance curves with error bands."""
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
    """Convert task instruction to a safe directory name."""
    task_name = task_instruction.strip().lower()
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in task_name)
    return safe[:80] if safe else "unknown_task"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    """Run the full sensitivity sweep."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prompt variants …")
    prompts_by_task = _load_variant_prompts(args.variants_json)
    if not prompts_by_task:
        raise ValueError(f"No prompts found in {args.variants_json}.")
    known = "\n    ".join(prompts_by_task.keys())
    print(f"  Found prompts for {len(prompts_by_task)} tasks:\n    {known}")

    print("Loading policy …")
    policy = SmolVLAPolicy(args.checkpoint, action_dim=7, device=str(device))
    policy.eval()

    # Storage: {variant_type: [L2 curves across all tasks]}
    type_l2_curves: dict[str, list[torch.Tensor]] = defaultdict(list)
    type_rel_l2_curves: dict[str, list[torch.Tensor]] = defaultdict(list)

    # Per-task storage: {task_instruction: {variant_type: [L2 curves]}}
    task_type_l2_curves: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_type_rel_l2_curves: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )

    n_processed = 0
    print(f"\nSweeping {args.rollout} …")
    for task_index, task_instruction, images, states in _iter_demos(
        args.rollout, keep_instructions=set(prompts_by_task.keys())
    ):
        task_variants = prompts_by_task[task_instruction]
        n_variants = sum(len(v) for v in task_variants.values())
        print(
            f"  [task {task_index}] '{task_instruction}' — {images.shape[0]} steps, {n_variants} variants",
            flush=True,
        )

        # Get reference actions from the original/base prompt
        if "original" not in task_variants:
            print(f"    WARNING: No 'original' variant found for '{task_instruction}'. Skipping.")
            continue

        original_variant = task_variants["original"]
        # Get the actual prompt text from the nested dict
        original_text = next(iter(original_variant.values())) if original_variant else task_instruction
        base_actions = _run(original_text, images, states, policy, args.seed)

        # Compute L2 distances for each variant
        for variant_type, variants_dict in task_variants.items():
            if variant_type == "original":
                continue
            for variant_name, variant_text in variants_dict.items():
                print(f"    [{variant_type}] '{variant_text}'", flush=True)
                variant_actions = _run(variant_text, images, states, policy, args.seed)
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

    print(f"\nProcessed {n_processed} demos.")

    if n_processed == 0:
        raise RuntimeError(
            f"No matching demos found in {args.rollout} for the tasks in {args.variants_json}.\n"
            f"Ensure the task descriptions in the h5 file match those in the JSON."
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def _stack(curves: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """Stack curves of varying lengths, padding with NaN."""
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
        out_path=output_dir / "overall_avg_l2.png",
        title="Overall Average L2 Action Distance per Variant Type",
        y_label="Mean L2 Action Distance",
    )
    _plot_l2_curves(
        type_mean=type_rel_mean,
        type_std=type_rel_std,
        overall_mean=overall_rel_mean,
        out_path=output_dir / "overall_avg_rel_l2.png",
        title="Overall Average Relative L2 Action Distance per Variant Type",
        y_label="Mean Relative L2 Action Distance",
    )

    # Per-task plots
    per_task_dir = output_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    for task_instruction in sorted(task_type_l2_curves.keys()):
        by_type_abs = task_type_l2_curves[task_instruction]
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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
    summary_path = output_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary → {summary_path}")


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
