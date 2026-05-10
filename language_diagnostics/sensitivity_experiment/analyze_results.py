"""
Example analysis script: How to interpret sensitivity experiment results.

This script demonstrates:
1. Loading the summary JSON
2. Finding which variant types are most/least sensitive
3. Comparing sensitivity across tasks
4. Generating simple statistics
"""

import json
import numpy as np
from pathlib import Path


def load_summary(summary_path: str) -> dict:
    """Load the sensitivity_summary.json file."""
    with open(summary_path) as f:
        return json.load(f)


def print_overall_stats(summary: dict) -> None:
    """Print overall sensitivity statistics."""
    print("=" * 80)
    print("OVERALL LANGUAGE SENSITIVITY STATISTICS")
    print("=" * 80)
    
    overall = summary["overall"]
    mean_l2 = np.array(overall["mean_l2_per_timestep"])
    mean_rel_l2 = np.array(overall["mean_rel_l2_per_timestep"])
    
    print(f"\nAbsolute L2 Distance (averaged across timesteps):")
    print(f"  Mean: {mean_l2.mean():.4f}")
    print(f"  Std:  {mean_l2.std():.4f}")
    print(f"  Min:  {mean_l2.min():.4f}")
    print(f"  Max:  {mean_l2.max():.4f}")
    
    print(f"\nRelative L2 Distance (averaged across timesteps):")
    print(f"  Mean: {mean_rel_l2.mean():.4f}")
    print(f"  Std:  {mean_rel_l2.std():.4f}")
    print(f"  Min:  {mean_rel_l2.min():.4f}")
    print(f"  Max:  {mean_rel_l2.max():.4f}")


def print_variant_ranking(summary: dict) -> None:
    """Rank variant types by mean L2 sensitivity."""
    print("\n" + "=" * 80)
    print("VARIANT TYPE RANKING (by mean absolute L2 distance)")
    print("=" * 80)
    
    variants = summary["variant_types"]
    
    # Compute overall mean L2 for each variant
    rankings = []
    for vtype, data in variants.items():
        mean_l2 = np.array(data["mean_l2_per_timestep"]).mean()
        mean_rel_l2 = np.array(data["mean_rel_l2_per_timestep"]).mean()
        n_rollouts = data["n_rollouts"]
        rankings.append((vtype, mean_l2, mean_rel_l2, n_rollouts))
    
    # Sort by L2 (highest = most sensitive)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Variant Type':<20} {'Abs L2':<10} {'Rel L2':<10} {'Rollouts':<10}")
    print("-" * 60)
    for i, (vtype, mean_l2, mean_rel_l2, n_rollouts) in enumerate(rankings, 1):
        print(f"{i:<6} {vtype:<20} {mean_l2:<10.4f} {mean_rel_l2:<10.4f} {n_rollouts:<10}")


def print_task_sensitivity(summary: dict) -> None:
    """Analyze sensitivity per task."""
    print("\n" + "=" * 80)
    print("PER-TASK SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    per_task = summary["per_task"]
    
    task_sensitivities = []
    for task_desc, task_data in per_task.items():
        # Average sensitivity across all variants for this task
        variant_l2s = []
        for vtype, vdata in task_data["variant_types"].items():
            mean_l2 = np.array(vdata["mean_l2_per_timestep"]).mean()
            variant_l2s.append(mean_l2)
        
        avg_l2 = np.mean(variant_l2s)
        task_sensitivities.append((task_desc, avg_l2, len(variant_l2s)))
    
    # Sort by sensitivity
    task_sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Task Description':<50} {'Avg L2':<10} {'Variants':<10}")
    print("-" * 70)
    for task, avg_l2, n_variants in task_sensitivities:
        task_short = task[:47] + "..." if len(task) > 50 else task
        print(f"{task_short:<50} {avg_l2:<10.4f} {n_variants:<10}")


def print_sensitivity_interpretation(summary: dict) -> None:
    """Provide interpretation of results."""
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    
    print("""
ABSOLUTE L2 DISTANCE:
  - Measures raw distance between variant and reference actions
  - Units: same as action space (typically [-1, 1] for normalized actions)
  - High values (>0.2): Strong language sensitivity
  - Medium values (0.05-0.2): Moderate sensitivity
  - Low values (<0.05): Weak sensitivity (robust to variation)

RELATIVE L2 DISTANCE:
  - Measures L2 normalized by action magnitude
  - Useful for comparing across tasks with different action scales
  - High values (>0.1): Actions scaled up/down significantly
  - Medium values (0.02-0.1): Moderate scaling
  - Low values (<0.02): Minimal scaling effect

WHAT TO LOOK FOR:
  ✓ Politeness/verb_paraphrase with LOW L2 → Policy is robust (good!)
  ✓ carefully/quickly with HIGH L2 → Policy responds to adverbs (good!)
  ✗ do_not_move with LOW L2 → Policy ignores negation (bad!)
  ✗ randomly varying L2 per task → Policy behavior is inconsistent
""")


def main(summary_path: str = "language_diagnostics/sensitivity_experiment/outputs/sensitivity_summary.json"):
    """Run full analysis."""
    path = Path(summary_path)
    if not path.exists():
        print(f"ERROR: Summary file not found at {path}")
        print("Run the sensitivity experiment first:")
        print("  python -m language_diagnostics.sensitivity_experiment.run_sensitivity_experiment \\")
        print("    --rollout-key object_5_50")
        return
    
    summary = load_summary(summary_path)
    
    print(f"\nLoaded summary from: {summary_path}")
    print(f"Processed {summary['n_demos_processed']} demos\n")
    
    print_overall_stats(summary)
    print_variant_ranking(summary)
    print_task_sensitivity(summary)
    print_sensitivity_interpretation(summary)


if __name__ == "__main__":
    main()
