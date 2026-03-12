"""
Generate per-task relative L2 plots from a pre-computed multitask_summary.json.

Requires the summary to have been produced by multitask_diagnostic.py (which saves
a "per_task" section with mean/std rel_l2 per variant type per task).

Usage:
    uv run python -m smolvla_language_pilot.plot_per_task_from_summary \
        --summary smolvla_language_pilot/outputs/multitask_summary.json \
        --output-dir smolvla_language_pilot/outputs/per_task_rel_l2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="smolvla_language_pilot/outputs/multitask_summary.json",
        help="Path to multitask_summary.json produced by multitask_diagnostic.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="smolvla_language_pilot/outputs/per_task_rel_l2",
        help="Directory where per-task plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    per_task = summary.get("per_task")
    if not per_task:
        raise ValueError(
            "The summary has no 'per_task' section. "
            "Re-run multitask_diagnostic.py to regenerate the summary with per-task data."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_instruction, type_data in per_task.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        all_means: list[np.ndarray] = []

        for vtype, stats in type_data.items():
            mean = np.array(stats["mean_rel_l2_per_timestep"])
            std = np.array(stats["std_rel_l2_per_timestep"])
            x = np.arange(len(mean))
            ax.plot(x, mean, label=vtype)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            all_means.append(mean)

        # overall mean across variant types (equal-weight average of per-type means)
        if all_means:
            max_T = max(m.shape[0] for m in all_means)
            padded = np.full((len(all_means), max_T), np.nan, dtype=np.float32)
            for i, m in enumerate(all_means):
                padded[i, : m.shape[0]] = m
            overall = np.nanmean(padded, axis=0)
            ax.plot(np.arange(max_T), overall, color="black", linewidth=2, linestyle="--", label="overall mean")

        n_rollouts = max(v["n_rollouts"] for v in type_data.values()) if type_data else 0
        short_title = task_instruction[:60] + "…" if len(task_instruction) > 60 else task_instruction

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Relative L2 Action Distance")
        ax.set_title(f"Relative L2 per Variant Type\n{short_title}\n({n_rollouts} rollouts)")
        ax.legend()
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-]+", "_", task_instruction)[:60]
        out_path = output_dir / f"{safe_name}_rel_l2.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved → {out_path}")

    print(f"\nDone. {len(per_task)} task plot(s) saved to {output_dir}")

    # ------------------------------------------------------------------
    # Single-rollout plots (rollout 0 per task)
    # ------------------------------------------------------------------
    single_rollout_dir = output_dir / "single_rollout"
    single_rollout_dir.mkdir(exist_ok=True)

    for task_instruction, type_data in per_task.items():
        # Check that at least one variant type has rollout-0 data
        if not any(type_data[vt].get("rollout_0_rel_l2_per_timestep") for vt in type_data):
            print(f"  (skipping single-rollout plot for '{task_instruction[:50]}' — no rollout_0 data in summary)")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        all_r0: list[np.ndarray] = []

        for vtype, stats in type_data.items():
            r0 = stats.get("rollout_0_rel_l2_per_timestep")
            if not r0:
                continue
            r0_arr = np.array(r0)
            x = np.arange(len(r0_arr))
            ax.plot(x, r0_arr, label=vtype)
            all_r0.append(r0_arr)

        if all_r0:
            max_T = max(a.shape[0] for a in all_r0)
            padded = np.full((len(all_r0), max_T), np.nan, dtype=np.float32)
            for i, a in enumerate(all_r0):
                padded[i, : a.shape[0]] = a
            overall = np.nanmean(padded, axis=0)
            ax.plot(np.arange(max_T), overall, color="black", linewidth=2, linestyle="--", label="overall mean")

        short_title = task_instruction[:60] + "…" if len(task_instruction) > 60 else task_instruction
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Relative L2 Action Distance")
        ax.set_title(f"Single-Rollout Relative L2 per Variant Type\n{short_title}\n(rollout 0)")
        ax.legend()
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-]+", "_", task_instruction)[:60]
        out_path = single_rollout_dir / f"{safe_name}_rollout0_rel_l2.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved → {out_path}")

    print(f"\nDone. Single-rollout plots saved to {single_rollout_dir}")


if __name__ == "__main__":
    main()
