"""
Analyze visual sensitivity experiment results.

This script loads the CSV output from visual_sensitivity.py and reports:
  1. Overall success rate per noise condition
  2. Noise type ranking (which corruption is most disruptive)
  3. Per-task breakdown
  4. Severity comparison (when multiple severities were tested)
  5. Interpretation guide

Usage:
    uv run python -m visual_diagnostic.analyze_results \
        --results-csv visual_diagnostic/outputs/visual_noise_rollouts_raw.csv

    # Filter to a specific severity:
    uv run python -m visual_diagnostic.analyze_results \
        --results-csv visual_diagnostic/outputs/visual_noise_rollouts_raw.csv \
        --severity 3
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load_csv(path: str) -> list[dict[str, str]]:
    """Load the rollout CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def _success_rate(rows: list[dict[str, str]]) -> tuple[int, int, float]:
    """Return (successes, total, rate)."""
    total = len(rows)
    successes = sum(1 for r in rows if _parse_bool(r["success"]))
    rate = successes / total if total else 0.0
    return successes, total, rate


def _mean_episode_length(rows: list[dict[str, str]]) -> float:
    lengths = [int(r["episode_length"]) for r in rows]
    return sum(lengths) / len(lengths) if lengths else 0.0


# ---------------------------------------------------------------------------
# Print sections
# ---------------------------------------------------------------------------


def print_overall_stats(rows: list[dict[str, str]]) -> None:
    """Print overall success rate across all conditions."""
    print("=" * 80)
    print("OVERALL VISUAL SENSITIVITY STATISTICS")
    print("=" * 80)

    # Clean baseline
    clean_rows = [r for r in rows if r["noise_type"] == "clean"]
    noised_rows = [r for r in rows if r["noise_type"] != "clean"]

    if clean_rows:
        s, t, rate = _success_rate(clean_rows)
        ml = _mean_episode_length(clean_rows)
        print(f"\n  Clean baseline:  {s}/{t} success ({rate:.1%}), mean ep. length={ml:.1f}")

    if noised_rows:
        s, t, rate = _success_rate(noised_rows)
        ml = _mean_episode_length(noised_rows)
        print(f"  All noised:      {s}/{t} success ({rate:.1%}), mean ep. length={ml:.1f}")

    if clean_rows and noised_rows:
        _, _, clean_rate = _success_rate(clean_rows)
        _, _, noised_rate = _success_rate(noised_rows)
        drop = clean_rate - noised_rate
        print(f"\n  Success rate drop from noise: {drop:+.1%}")


def print_noise_ranking(rows: list[dict[str, str]]) -> None:
    """Rank noise types by success rate (lowest = most disruptive)."""
    print("\n" + "=" * 80)
    print("NOISE TYPE RANKING (by success rate, lowest = most disruptive)")
    print("=" * 80)

    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        nt = r["noise_type"]
        sev = r["severity"]
        key = f"{nt}" if nt == "clean" else f"{nt}_s{sev}"
        by_condition[key].append(r)

    rankings = []
    for key, cond_rows in by_condition.items():
        s, t, rate = _success_rate(cond_rows)
        ml = _mean_episode_length(cond_rows)
        rankings.append((key, s, t, rate, ml))

    # Sort by success rate ascending (most disruptive first)
    rankings.sort(key=lambda x: x[3])

    print(f"\n{'Rank':<6} {'Condition':<25} {'Success':<12} {'Rate':<10} {'Avg Length':<12}")
    print("-" * 65)
    for i, (key, s, t, rate, ml) in enumerate(rankings, 1):
        print(f"{i:<6} {key:<25} {s}/{t:<9} {rate:<10.1%} {ml:<12.1f}")


def print_severity_comparison(rows: list[dict[str, str]]) -> None:
    """Compare success rates across severity levels."""
    severities = sorted({r["severity"] for r in rows if r["noise_type"] != "clean"})
    if len(severities) <= 1:
        return

    print("\n" + "=" * 80)
    print("SEVERITY COMPARISON")
    print("=" * 80)

    noise_types = sorted({r["noise_type"] for r in rows if r["noise_type"] != "clean"})

    # Header
    sev_header = "".join(f"{'s' + str(s):<10}" for s in severities)
    print(f"\n{'Noise Type':<20} {sev_header}")
    print("-" * (20 + 10 * len(severities)))

    for nt in noise_types:
        line = f"{nt:<20} "
        for sev in severities:
            sev_rows = [r for r in rows if r["noise_type"] == nt and r["severity"] == str(sev)]
            if sev_rows:
                _, _, rate = _success_rate(sev_rows)
                line += f"{rate:<10.1%}"
            else:
                line += f"{'—':<10}"
        print(line)


def print_task_sensitivity(rows: list[dict[str, str]]) -> None:
    """Analyze sensitivity per task."""
    print("\n" + "=" * 80)
    print("PER-TASK SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Group by task
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_task[r["task_description"]].append(r)

    task_stats = []
    for task_desc, task_rows in by_task.items():
        clean = [r for r in task_rows if r["noise_type"] == "clean"]
        noised = [r for r in task_rows if r["noise_type"] != "clean"]

        _, _, clean_rate = _success_rate(clean) if clean else (0, 0, 0.0)
        _, _, noised_rate = _success_rate(noised) if noised else (0, 0, 0.0)
        drop = clean_rate - noised_rate
        task_stats.append((task_desc, clean_rate, noised_rate, drop))

    # Sort by drop (largest drop = most sensitive)
    task_stats.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'Task':<55} {'Clean':<10} {'Noised':<10} {'Drop':<10}")
    print("-" * 85)
    for task, clean_r, noised_r, drop in task_stats:
        task_short = task[:52] + "..." if len(task) > 55 else task
        print(f"{task_short:<55} {clean_r:<10.1%} {noised_r:<10.1%} {drop:<+10.1%}")


def print_interpretation() -> None:
    """Provide interpretation of visual sensitivity results."""
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)

    print("""
SUCCESS RATE:
  - Measures whether the policy completes the task under visual corruption
  - Clean baseline gives the upper bound
  - Drop = Clean rate − Noised rate

WHAT TO LOOK FOR:
  ✓ fog/zoom_blur with SMALL drop → Policy is robust to mild distortions (good!)
  ✓ Gradual degradation across severities → Predictable, well-behaved model
  ✗ glass_blur/motion_blur with LARGE drop → Policy relies on fine visual details
  ✗ Sudden cliff at a specific severity → Brittle perception threshold
  ✗ Inconsistent drops across tasks → Policy robustness is task-dependent

SEVERITY LEVELS (1–5):
  s1 = barely perceptible noise
  s3 = moderate corruption (standard test point)
  s5 = severe corruption (stress test)
""")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to visual_noise_rollouts_raw.csv",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        help="Filter results to a specific severity level (e.g. --severity 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results_csv)
    if not path.exists():
        print(f"ERROR: Results CSV not found at {path}")
        print("Run the visual sensitivity experiment first:")
        print("  uv run python -m visual_diagnostic.visual_sensitivity \\")
        print("    --suites object --severity 3")
        return

    rows = _load_csv(str(path))
    if not rows:
        print("ERROR: CSV is empty.")
        return

    # Optional severity filter
    if args.severity is not None:
        sev_str = str(args.severity)
        rows = [r for r in rows if r["noise_type"] == "clean" or r["severity"] == sev_str]
        print(f"Filtered to severity={args.severity} ({len(rows)} rows)")

    print(f"\nLoaded {len(rows)} rollout rows from: {path}\n")

    print_overall_stats(rows)
    print_noise_ranking(rows)
    print_severity_comparison(rows)
    print_task_sensitivity(rows)
    print_interpretation()


if __name__ == "__main__":
    main()
