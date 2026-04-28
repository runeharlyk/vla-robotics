"""Visual noise robustness evaluation — main entry point.

Usage::

    python -m smolvla_visual_pilot.run_evaluation \\
        --rollout libero_demo_samples/combined_data
        --output-dir smolvla_visual_pilot/outputs \\
        --max-demos 2   # smoke-test cap

The pipeline:
    1. Discovers all h5 files (single-file path or recursively from a folder).
    2. Loads the SmolVLA policy.
    3. For each demo in each file:
       a. Obtains reference actions from the selected source:
          - model (default): run a clean no-noise replay.
          - demo: use h5 ground-truth actions.
          - fallback to the other source if the selected source is unavailable.
       b. For each noise variant (5 types × severity 3):
          - At timestep t, predicts one noisy action from observation/state at t.
          - Compares it to the reference action at the same timestep t.
          - Repeats for all timesteps in the episode.
    4. Writes results to JSON + CSV and saves a summary deviation diagram.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .config import DEFAULT_EVAL_CAMERAS, NOISE_SEVERITY, NOISE_TYPES, EvalConfig
    from .data_loader import iter_demos
    from .inference import load_policy_bundle, run_trajectories_for_noises, run_trajectory
    from .logger import ResultLogger
    from .metrics import (
        compute_l2_distances,
        compute_per_dimension_absolute_errors,
        compute_per_dimension_relative_errors,
        compute_quality_degradation,
        compute_relative_l2_distances,
    )
    from .noise import get_noise_configs
except ImportError:
    from config import DEFAULT_EVAL_CAMERAS, NOISE_SEVERITY, NOISE_TYPES, EvalConfig
    from data_loader import iter_demos
    from inference import load_policy_bundle, run_trajectories_for_noises, run_trajectory
    from logger import ResultLogger
    from metrics import (
        compute_l2_distances,
        compute_per_dimension_absolute_errors,
        compute_per_dimension_relative_errors,
        compute_quality_degradation,
        compute_relative_l2_distances,
    )
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
        default="libero_demo_samples/combined_data",
        help=(
            "Path to one Libero+ h5 file OR to a directory that contains "
            "chunk-*/episode_*.h5 files (recursively discovered)."
        ),
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
        "--reference-source",
        choices=["model", "demo"],
        default="model",
        help=(
            "Reference action source. "
            "'model' (default): clean no-noise model actions. "
            "'demo': h5 demo actions when available."
        ),
    )
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
        "--noise-batch-size",
        type=int,
        default=0,
        help=("Number of noise variants to evaluate in one model forward pass. 0 means all configured noise types."),
    )
    p.add_argument(
        "--timestep-batch-size",
        type=int,
        default=1,
        help=(
            "Number of timesteps to evaluate per model forward pass. Increase for throughput, decrease if you hit OOM."
        ),
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


def discover_rollout_files(rollout_path: str) -> list[Path]:
    """Resolve input to a sorted list of h5 files.

    If ``rollout_path`` is a file, only that file is used.
    If it is a directory, all ``*.h5`` files under it are evaluated.
    """
    root = Path(rollout_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Rollout path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() != ".h5":
            raise ValueError(f"Expected an h5 file, got: {root}")
        return [root]

    files = sorted(p for p in root.rglob("*.h5") if p.is_file())
    if not files:
        raise RuntimeError(f"No .h5 files found under directory: {root}")
    return files


def to_source_id(h5_path: Path, rollout_root: Path) -> str:
    """Return a stable source identifier for logging and summaries."""
    if rollout_root.is_dir():
        try:
            return str(h5_path.relative_to(rollout_root)).replace("\\", "/")
        except ValueError:
            pass
    return str(h5_path).replace("\\", "/")


def parse_chunk_episode(h5_path: Path) -> tuple[str, str]:
    """Extract chunk and episode identifiers from an h5 path."""
    chunk = h5_path.parent.name if h5_path.parent.name.startswith("chunk-") else "unknown"
    episode = h5_path.stem
    return chunk, episode


def save_noise_deviation_diagram(summary: dict, output_path: Path) -> None:
    """Save a PNG diagram showing deviation per noise type."""
    by_noise_type = summary.get("by_noise_type", {})
    if not by_noise_type:
        print("No noise statistics available; skipping diagram generation.")
        return

    by_noise_type_relative = summary.get("by_noise_type_relative", {})

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping diagram generation.")
        return

    noise_labels = list(by_noise_type.keys())
    x_positions = list(range(len(noise_labels)))
    means = [by_noise_type[label]["mean"] for label in noise_labels]
    stds = [by_noise_type[label]["std"] for label in noise_labels]

    rel_means = [by_noise_type_relative.get(label, {}).get("mean", 0.0) for label in noise_labels]
    rel_stds = [by_noise_type_relative.get(label, {}).get("std", 0.0) for label in noise_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    nice_labels = [label.replace("_", " ") for label in noise_labels]

    axes[0].bar(x_positions, means, color="#2c7fb8", alpha=0.9)
    axes[0].errorbar(x_positions, means, yerr=stds, fmt="none", ecolor="black", capsize=4)
    axes[0].set_title("Absolute L2 Deviation")
    axes[0].set_ylabel("Mean L2")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(nice_labels, rotation=25, ha="right")

    axes[1].bar(x_positions, rel_means, color="#f03b20", alpha=0.9)
    axes[1].errorbar(x_positions, rel_means, yerr=rel_stds, fmt="none", ecolor="black", capsize=4)
    axes[1].set_title("Relative L2 Deviation")
    axes[1].set_ylabel("Mean Relative L2")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(nice_labels, rotation=25, ha="right")

    fig.suptitle("SmolVLA Deviation by Libero+ Noise Type")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved deviation diagram → {output_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> None:
    out_dir = cfg.resolve_output_dir()

    if cfg.reference_source not in {"model", "demo"}:
        raise ValueError(f"Invalid reference_source='{cfg.reference_source}'. Expected one of: model, demo.")

    rollout_root = Path(cfg.rollout_path).expanduser()
    rollout_files = discover_rollout_files(cfg.rollout_path)
    n_chunks = len({p.parent.name for p in rollout_files if p.parent.name.startswith("chunk-")})

    print(f"Discovered {len(rollout_files)} h5 files under {rollout_root}.")
    if rollout_root.is_dir():
        print(f"  Chunk folders detected: {n_chunks}")
    print(f"  Cameras: {cfg.cameras}")
    print(f"  Reference source mode: {cfg.reference_source}")

    # -- load policy --
    print("Loading SmolVLA policy …")
    policy_bundle = load_policy_bundle(cfg.checkpoint, cfg.device)

    # -- noise configs --
    noise_configs = get_noise_configs(cfg.noise_types, severity=cfg.noise_severity)
    resolved_noise_batch = len(noise_configs) if cfg.noise_batch_size <= 0 else cfg.noise_batch_size
    resolved_noise_batch = min(resolved_noise_batch, len(noise_configs))
    print(f"  Noise variants: {[str(nc) for nc in noise_configs]}")
    print(f"  Noise batch size: {resolved_noise_batch}")
    print(f"  Timestep batch size: {cfg.timestep_batch_size}")

    # -- logger --
    logger = ResultLogger()
    t0 = time.time()
    seen_tasks: dict[int, int] = {}  # task_index -> processed demos
    processed_demos = 0
    skipped_files: list[str] = []

    # -- iterate --
    for file_idx, h5_path in enumerate(rollout_files, start=1):
        source_h5 = to_source_id(h5_path, rollout_root)
        source_chunk, source_episode = parse_chunk_episode(h5_path)

        print(f"\n[File {file_idx}/{len(rollout_files)}] {source_h5}", flush=True)

        try:
            demos = iter_demos(
                str(h5_path),
                max_tasks=None,
                max_demos=None,
                cameras=cfg.cameras,
            )
        except Exception as exc:
            skipped_files.append(source_h5)
            print(f"  Skipping unreadable file ({type(exc).__name__}): {exc}")
            continue

        if not demos:
            print("  No demos found in this file; skipping.")
            continue

        for local_demo_idx, demo in enumerate(demos, start=1):
            task_count = seen_tasks.get(demo.task_index, 0)

            if demo.task_index not in seen_tasks:
                if cfg.max_tasks is not None and len(seen_tasks) >= cfg.max_tasks:
                    continue
                seen_tasks[demo.task_index] = 0

            if cfg.max_demos is not None and task_count >= cfg.max_demos:
                continue

            seen_tasks[demo.task_index] = seen_tasks.get(demo.task_index, 0) + 1
            rollout_index = processed_demos
            processed_demos += 1

            print(
                f"  [Demo {local_demo_idx}/{len(demos)} | global={processed_demos}] "
                f"task={demo.task_index} "
                f"instruction='{demo.task_instruction}' "
                f"T={demo.images.shape[0]}",
                flush=True,
            )

            # --- obtain reference actions (selected source, with fallback) ---
            if cfg.reference_source == "demo":
                if demo.gt_actions is not None:
                    ref_actions = demo.gt_actions
                    ref_source = "h5_ground_truth"
                else:
                    print("    (reference-source=demo but no GT actions; running clean baseline)")
                    try:
                        ref_actions = run_trajectory(
                            demo.images,
                            demo.states,
                            demo.task_instruction,
                            policy_bundle,
                            noise_config=None,
                            seed=cfg.seed,
                            timestep_batch_size=cfg.timestep_batch_size,
                        )
                        ref_source = "clean_model_prediction_fallback"
                    except Exception as exc:
                        print(
                            "    (demo reference unavailable and clean baseline failed; "
                            f"skipping demo: {type(exc).__name__}: {exc})"
                        )
                        continue
            else:
                try:
                    ref_actions = run_trajectory(
                        demo.images,
                        demo.states,
                        demo.task_instruction,
                        policy_bundle,
                        noise_config=None,
                        seed=cfg.seed,
                        timestep_batch_size=cfg.timestep_batch_size,
                    )
                    ref_source = "clean_model_prediction"
                except Exception as exc:
                    if demo.gt_actions is None:
                        print(
                            "    (clean baseline failed and no GT actions available; "
                            f"skipping demo: {type(exc).__name__}: {exc})"
                        )
                        continue

                    print(f"    (clean baseline failed; falling back to GT actions: {type(exc).__name__}: {exc})")
                    ref_actions = demo.gt_actions
                    ref_source = "h5_ground_truth_fallback"

            print(f"    Reference: {ref_source} — shape {tuple(ref_actions.shape)}")

            # Optional quality metric anchor: compare noisy-vs-GT relative to clean-vs-GT.
            # If clean actions are not already available, try to compute them once here.
            clean_actions_for_quality: torch.Tensor | None = None
            if ref_source.startswith("clean_model_prediction"):
                clean_actions_for_quality = ref_actions
            elif demo.gt_actions is not None:
                try:
                    clean_actions_for_quality = run_trajectory(
                        demo.images,
                        demo.states,
                        demo.task_instruction,
                        policy_bundle,
                        noise_config=None,
                        seed=cfg.seed,
                        timestep_batch_size=cfg.timestep_batch_size,
                    )
                except Exception as exc:
                    print(
                        f"    (clean baseline unavailable for quality-degradation metric: {type(exc).__name__}: {exc})"
                    )

            # --- run all noise variants with batched model inference ---
            print(
                "      running noisy variants "
                f"(noise_batch={resolved_noise_batch}, timestep_batch={cfg.timestep_batch_size}) ...",
                flush=True,
            )
            predicted_by_noise = run_trajectories_for_noises(
                demo.images,
                demo.states,
                demo.task_instruction,
                policy_bundle,
                noise_configs=noise_configs,
                seed=cfg.seed,
                noise_batch_size=resolved_noise_batch,
                timestep_batch_size=cfg.timestep_batch_size,
            )

            for nc, predicted in zip(noise_configs, predicted_by_noise, strict=True):
                print(f"      [{nc}] scoring ...", end="", flush=True)

                # For each timestep t, compare noisy prediction a_noisy[t] against
                # reference action a_ref[t] (clean-model or demo), then aggregate over T.
                l2 = compute_l2_distances(predicted, ref_actions)
                rel_l2 = compute_relative_l2_distances(predicted, ref_actions)
                per_dim_abs = compute_per_dimension_absolute_errors(predicted, ref_actions)
                per_dim_rel = compute_per_dimension_relative_errors(predicted, ref_actions)

                quality_delta_l2 = None
                quality_ratio_l2 = None
                if clean_actions_for_quality is not None and demo.gt_actions is not None:
                    quality = compute_quality_degradation(
                        noisy_predicted=predicted,
                        clean_predicted=clean_actions_for_quality,
                        ground_truth=demo.gt_actions,
                    )
                    quality_delta_l2 = quality["degradation_delta_l2"].numpy()
                    quality_ratio_l2 = quality["degradation_ratio_l2"].numpy()

                logger.log_trajectory(
                    task_index=demo.task_index,
                    task_name=demo.task_instruction,
                    rollout_index=rollout_index,
                    source_h5=source_h5,
                    source_chunk=source_chunk,
                    source_episode=source_episode,
                    noise_type=nc.noise_type,
                    noise_severity=nc.severity,
                    l2_distances=l2.numpy(),
                    rel_l2_distances=rel_l2.numpy(),
                    per_dim_abs_errors=per_dim_abs.numpy(),
                    per_dim_rel_errors=per_dim_rel.numpy(),
                    quality_delta_l2=quality_delta_l2,
                    quality_ratio_l2=quality_ratio_l2,
                )

                mean_l2 = float(l2.mean())
                print(f"  mean L2 = {mean_l2:.4f}")

    if logger.n_records == 0:
        raise RuntimeError(
            "No evaluation records were produced. Check cameras, task/demo caps, and input rollout files."
        )

    elapsed = time.time() - t0
    print(f"\nDone — {logger.n_records} timestep records in {elapsed:.1f}s")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} unreadable files.")

    # -- save --
    logger.save_csv(out_dir / "results.csv")
    logger.save_json(out_dir / "results.json")

    summary = logger.get_summary()
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"Saved summary → {summary_path}")

    save_noise_deviation_diagram(summary, out_dir / "noise_deviation.png")

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
    if args.noise_batch_size < 0:
        raise SystemExit("--noise-batch-size must be >= 0.")
    if args.timestep_batch_size < 1:
        raise SystemExit("--timestep-batch-size must be >= 1.")

    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        seed=args.seed,
        reference_source=args.reference_source,
        rollout_path=args.rollout,
        cameras=cameras,
        noise_types=args.noise_types or list(NOISE_TYPES),
        noise_severity=args.noise_severity,
        noise_batch_size=args.noise_batch_size,
        timestep_batch_size=args.timestep_batch_size,
        max_tasks=args.max_tasks,
        max_demos=args.max_demos,
        output_dir=args.output_dir,
    )

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
