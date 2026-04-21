"""Visual noise robustness evaluation — main entry point.

Usage::

    python -m smolvla_visual_pilot.run_evaluation \\
        --rollout data/h5/libero/libero_plus.h5 \\
        --output-dir smolvla_visual_pilot/outputs \\
        --max-demos 2   # smoke-test cap

The pipeline:
    1. Loads all demos from the h5 file.
    2. Loads the SmolVLA policy.
    3. For each demo:
       a. Obtains reference actions (h5 ground truth, or clean model predictions
          if the h5 lacks an ``actions`` dataset).
       b. For each noise variant (5 types × severity 3):
          - Replays the trajectory with noisy observations.
          - Computes per-timestep L2 distances vs reference.
    4. Writes results to JSON + CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .config import DEFAULT_EVAL_CAMERAS, EvalConfig, NOISE_TYPES, NOISE_SEVERITY
    from .data_loader import iter_demos
    from .inference import load_policy_bundle, run_trajectory
    from .logger import ResultLogger
    from .metrics import compute_l2_distances, compute_relative_l2_distances
    from .noise import get_noise_configs
except ImportError:
    from config import DEFAULT_EVAL_CAMERAS, EvalConfig, NOISE_TYPES, NOISE_SEVERITY
    from data_loader import iter_demos
    from inference import load_policy_bundle, run_trajectory
    from logger import ResultLogger
    from metrics import compute_l2_distances, compute_relative_l2_distances
    from noise import get_noise_configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SmolVLA robustness under Libero+ visual noise.",
    )
    p.add_argument(
        "--rollout",
        required=True,
        help="Path to the combined Libero+ h5 file.",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_EVAL_CAMERAS),
        help=(
            "Camera streams under /videos for new-format h5 files. "
            "Default uses front+wrist for dual-camera inference."
        ),
    )
    p.add_argument("--camera", default=None, help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--noise-types",
        nargs="*",
        default=None,
        help=f"Noise types to evaluate (default: all 5). Choices: {NOISE_TYPES}",
    )
    p.add_argument(
        "--noise-severity",
        type=int,
        default=NOISE_SEVERITY,
        help="Corruption severity level 1-5 (default: 3).",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Cap on distinct tasks to process (default: all).",
    )
    p.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Cap on demos per task to process (default: all).",
    )
    p.add_argument(
        "--output-dir",
        default="smolvla_visual_pilot/outputs",
        help="Directory for result JSON, CSV, and summary.",
    )
    return p.parse_args()


def parse_csv_tokens(raw_values: list[str]) -> list[str]:
    tokens = []
    for value in raw_values:
        for token in value.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> None:
    out_dir = cfg.resolve_output_dir()

    # -- load demos ---
    print(f"Loading demos from {cfg.rollout_path} …")
    demos = iter_demos(
        cfg.rollout_path,
        max_tasks=cfg.max_tasks,
        max_demos=cfg.max_demos,
        cameras=cfg.cameras,
    )
    if not demos:
        raise RuntimeError(f"No demos found in {cfg.rollout_path}.")

    n_tasks = len({d.task_index for d in demos})
    print(f"  Loaded {len(demos)} demos across {n_tasks} tasks.")
    print(f"  Cameras: {cfg.cameras}")

    # -- load policy --
    print("Loading SmolVLA policy …")
    policy_bundle = load_policy_bundle(cfg.checkpoint, cfg.device)

    # -- noise configs --
    noise_configs = get_noise_configs(cfg.noise_types, severity=cfg.noise_severity)
    print(f"  Noise variants: {[str(nc) for nc in noise_configs]}")

    # -- logger --
    logger = ResultLogger()
    t0 = time.time()

    # -- iterate --
    for demo_idx, demo in enumerate(demos):
        print(
            f"\n[Demo {demo_idx + 1}/{len(demos)}] "
            f"task={demo.task_index} "
            f"instruction='{demo.task_instruction}' "
            f"T={demo.images.shape[0]}",
            flush=True,
        )

        # --- obtain reference actions ---
        if demo.gt_actions is not None:
            ref_actions = demo.gt_actions
            ref_source = "h5_ground_truth"
        else:
            # Fall back: run model on clean observations
            print("  (no GT actions in h5 — running clean baseline)")
            ref_actions = run_trajectory(
                demo.images,
                demo.states,
                demo.task_instruction,
                policy_bundle,
                noise_config=None,
                seed=cfg.seed,
            )
            ref_source = "clean_model_prediction"

        print(f"  Reference: {ref_source} — shape {tuple(ref_actions.shape)}")

        # --- run each noise variant ---
        for nc in noise_configs:
            print(f"    [{nc}] running …", end="", flush=True)

            predicted = run_trajectory(
                demo.images,
                demo.states,
                demo.task_instruction,
                policy_bundle,
                noise_config=nc,
                seed=cfg.seed,
            )

            l2 = compute_l2_distances(predicted, ref_actions)
            rel_l2 = compute_relative_l2_distances(predicted, ref_actions)

            logger.log_trajectory(
                task_index=demo.task_index,
                task_name=demo.task_instruction,
                rollout_index=demo_idx,
                noise_type=nc.noise_type,
                noise_severity=nc.severity,
                l2_distances=l2.numpy(),
                rel_l2_distances=rel_l2.numpy(),
            )

            mean_l2 = float(l2.mean())
            print(f"  mean L2 = {mean_l2:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone — {logger.n_records} timestep records in {elapsed:.1f}s")

    # -- save --
    logger.save_csv(out_dir / "results.csv")
    logger.save_json(out_dir / "results.json")

    summary = logger.get_summary()
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"Saved summary → {summary_path}")

    # -- print summary --
    print("\n" + "=" * 60)
    print("SUMMARY — Mean L2 per noise type")
    print("=" * 60)
    for noise_type, stats in summary["by_noise_type"].items():
        print(f"  {noise_type:20s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    camera_raw_values = [args.camera] if args.camera else args.cameras
    cameras = parse_csv_tokens(camera_raw_values)
    if not cameras:
        raise SystemExit("No cameras selected. Pass --cameras <camera1> [camera2 ...].")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but CUDA is not available in this environment. "
            "Use --device cpu to run on CPU."
        )

    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        seed=args.seed,
        rollout_path=args.rollout,
        cameras=cameras,
        noise_types=args.noise_types or list(NOISE_TYPES),
        noise_severity=args.noise_severity,
        max_tasks=args.max_tasks,
        max_demos=args.max_demos,
        output_dir=args.output_dir,
    )

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
