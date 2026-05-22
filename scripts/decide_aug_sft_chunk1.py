"""Post-eval decision matrix for the augmented-SFT plan.

Reads the eval JSONs produced by ``jobs/eval_aug_sft_chunk1_protocol.sh``
(via ``scripts/evaluate.py`` -> ``results/evals/<wandb_run_name>.json``),
builds a comparison table for baseline / Job A (control) / Job B (augmented),
and applies the 4-way decision rules from the augmented-sft-chunk1 plan.

Usage:

    uv run python scripts/decide_aug_sft_chunk1.py
    uv run python scripts/decide_aug_sft_chunk1.py --evals-dir custom/evals/dir

The script does *not* require all evals to be present; missing entries are
reported but do not crash. Run it after the eval LSF job completes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import typer

from vla.constants import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class EvalCell:
    label: str
    arm: str
    suite: str
    n_action_steps: int
    wandb_pattern: str


CELLS: list[EvalCell] = [
    EvalCell("baseline @ chunk-1, spatial", "baseline", "spatial", 1, "eval_baseline_spatial_chunk1_seed42"),
    EvalCell("baseline @ chunk-2, spatial", "baseline", "spatial", 2, "eval_baseline_spatial_chunk2_seed42"),
    EvalCell("Job A control @ chunk-1, spatial", "jobA", "spatial", 1, "eval_aug_sft_jobA_control_spatial_chunk1_seed42"),
    EvalCell("Job A control @ chunk-2, spatial", "jobA", "spatial", 2, "eval_aug_sft_jobA_control_spatial_chunk2_seed42"),
    EvalCell("Job A control @ chunk-1, object", "jobA", "object", 1, "eval_aug_sft_jobA_control_object_chunk1_seed42"),
    EvalCell("Job A control @ chunk-1, goal", "jobA", "goal", 1, "eval_aug_sft_jobA_control_goal_chunk1_seed42"),
    EvalCell("Job A control @ chunk-1, long", "jobA", "long", 1, "eval_aug_sft_jobA_control_long_chunk1_seed42"),
    EvalCell("Job B augmented @ chunk-1, spatial", "jobB", "spatial", 1, "eval_aug_sft_jobB_augmented_spatial_chunk1_seed42"),
    EvalCell("Job B augmented @ chunk-2, spatial", "jobB", "spatial", 2, "eval_aug_sft_jobB_augmented_spatial_chunk2_seed42"),
    EvalCell("Job B augmented @ chunk-1, object", "jobB", "object", 1, "eval_aug_sft_jobB_augmented_object_chunk1_seed42"),
    EvalCell("Job B augmented @ chunk-1, goal", "jobB", "goal", 1, "eval_aug_sft_jobB_augmented_goal_chunk1_seed42"),
    EvalCell("Job B augmented @ chunk-1, long", "jobB", "long", 1, "eval_aug_sft_jobB_augmented_long_chunk1_seed42"),
]

BASELINE_TARGET_PCT = 74.4
EQUIVALENCE_BAND_PP = 1.0


def _load_eval(evals_dir: Path, pattern: str) -> dict | None:
    for path in evals_dir.glob(f"{pattern}*.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
    return None


def _format_pct(value: float | None) -> str:
    if value is None:
        return "    -- "
    return f"{value * 100:5.1f}%"


def _verdict(rows: dict[str, float | None]) -> str:
    a_clean = rows.get("Job A control @ chunk-1, spatial")
    b_clean = rows.get("Job B augmented @ chunk-1, spatial")

    if a_clean is None or b_clean is None:
        missing = []
        if a_clean is None:
            missing.append("Job A chunk-1 spatial")
        if b_clean is None:
            missing.append("Job B chunk-1 spatial")
        return f"VERDICT: incomplete — missing {', '.join(missing)}. Re-run eval once both finish."

    a_pct = a_clean * 100
    b_pct = b_clean * 100
    delta_pp = b_pct - a_pct

    lines: list[str] = []
    lines.append("Decision matrix (clean spatial chunk-1 = primary metric):")
    lines.append(f"  baseline target       : {BASELINE_TARGET_PCT:5.1f}%")
    lines.append(f"  Job A (control)       : {a_pct:5.1f}%  (delta vs baseline: {a_pct - BASELINE_TARGET_PCT:+5.1f} pp)")
    lines.append(f"  Job B (augmented)     : {b_pct:5.1f}%  (delta vs baseline: {b_pct - BASELINE_TARGET_PCT:+5.1f} pp)")
    lines.append(f"  Job B - Job A         : {delta_pp:+5.1f} pp")

    if a_pct < BASELINE_TARGET_PCT and b_pct < BASELINE_TARGET_PCT:
        lines.append(
            "VERDICT: BOTH < baseline. Cross-suite dilution is likely; fall back to"
            " spatial-weighted sampling (2:1:1:1) and consider dropping the long suite first."
        )
    elif delta_pp >= EQUIVALENCE_BAND_PP and b_pct >= BASELINE_TARGET_PCT:
        lines.append(
            "VERDICT: PROMOTE Job B (augmented). Augmentation lifts clean SR by"
            f" {delta_pp:+.1f} pp on top of Job A and beats the baseline. Strongest story."
        )
    elif a_pct - b_pct >= EQUIVALENCE_BAND_PP:
        lines.append(
            "VERDICT: PROMOTE Job A (control). More training was the lever;"
            f" augmentation under-performed by {a_pct - b_pct:+.1f} pp on clean SR."
            " Use Job A as the spatial-only RL init. Consider weakening augmentation"
            " strength (variants=3, camera-pos-std=0.010, brightness=0.05) and"
            " re-running ONLY Job B if perturbed-eval is critical."
        )
    else:
        lines.append(
            f"VERDICT: A ≈ B on clean SR (|Δ| < {EQUIVALENCE_BAND_PP:.1f} pp)."
            " Promote Job B for the LIBERO-Pro/Plus robustness story IF the perturbed"
            " eval (not in this protocol — wire LIBERO-Pro first) shows B ≫ A. Otherwise"
            " promote Job A (cheaper, no augmentation maintenance burden)."
        )

    a_chunk2 = rows.get("Job A control @ chunk-2, spatial")
    b_chunk2 = rows.get("Job B augmented @ chunk-2, spatial")
    if a_chunk2 is not None and a_clean is not None and (a_clean - a_chunk2) > 0.10:
        lines.append(
            "  WARNING: Job A chunk-1 vs chunk-2 gap > 10 pp — confirm chunk-1 lift"
            " isn't a pathological action-chunk collapse before promotion."
        )
    if b_chunk2 is not None and b_clean is not None and (b_clean - b_chunk2) > 0.10:
        lines.append(
            "  WARNING: Job B chunk-1 vs chunk-2 gap > 10 pp — confirm chunk-1 lift"
            " isn't a pathological action-chunk collapse before promotion."
        )

    return "\n".join(lines)


@app.command()
def main(
    evals_dir: Path = typer.Option(
        RESULTS_DIR / "evals",
        "--evals-dir",
        help="Directory containing eval_*.json result files.",
    ),
) -> None:
    """Print the augmented-SFT plan's decision matrix from eval result JSONs."""
    if not evals_dir.exists():
        typer.echo(f"Evals directory does not exist: {evals_dir}")
        raise typer.Exit(1)

    rows: dict[str, float | None] = {}
    cross_suite: dict[str, dict[str, float | None]] = {"jobA": {}, "jobB": {}}

    print("=" * 96)
    print(f"{'cell':<48} {'arm':<10} {'suite':<8} {'n_act':<6} {'SR':>8}")
    print("-" * 96)
    for cell in CELLS:
        record = _load_eval(evals_dir, cell.wandb_pattern)
        sr = record.get("success_rate") if record else None
        rows[cell.label] = sr
        if cell.suite != "spatial" and cell.n_action_steps == 1:
            cross_suite[cell.arm][cell.suite] = sr
        print(f"{cell.label:<48} {cell.arm:<10} {cell.suite:<8} {cell.n_action_steps:<6} {_format_pct(sr)}")
    print("=" * 96)
    print()

    print(_verdict(rows))

    print()
    print("Cross-suite chunk-1 comparison (secondary metric — checks for regression):")
    print(f"{'suite':<10} {'Job A':>10} {'Job B':>10} {'B - A':>10}")
    for suite in ["object", "goal", "long"]:
        a = cross_suite["jobA"].get(suite)
        b = cross_suite["jobB"].get(suite)
        delta = (b - a) * 100 if (a is not None and b is not None) else None
        delta_str = f"{delta:+5.1f} pp" if delta is not None else "    --"
        print(f"{suite:<10} {_format_pct(a):>10} {_format_pct(b):>10} {delta_str:>10}")


if __name__ == "__main__":
    app()
