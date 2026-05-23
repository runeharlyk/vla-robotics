"""Visual sensitivity analysis: run LIBERO rollouts with visual noise perturbations.

This script tests how sensitive the policy is to visual corruptions by:
  1. Running the policy with clean observations to get baseline success rates
  2. Running the policy with noised observations for each noise type × severity
  3. Recording success/failure and episode length for each rollout
  4. Saving results incrementally (checkpoint-safe) and producing a summary

Outputs:
  - Per-rollout CSV with success/failure and episode length
  - Checkpoint JSON for resume capability
  - Summary JSON with aggregated statistics

Usage:
    uv run python -m visual_diagnostic.visual_sensitivity \
        --suites object \
        --tasks-per-suite 5 \
        --episodes 50 \
        --severity 3 \
        --output-dir visual_diagnostic/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch

from language_diagnostics.libero_prompt_variant_generate import TaskSpec
from language_diagnostics.libero_prompt_variant_run import (
    _fmt_seconds,
    _save_rows_csv,
)

from visual_diagnostic.noise import NoiseConfig, apply_noise

from vla.envs import make_env_factory
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device, seed_everything
from vla.utils.tensor import action_to_numpy

DEFAULT_SFT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

NOISE_TYPES: tuple[str, ...] = (
    "motion_blur",
    "gaussian_blur",
    "zoom_blur",
    "fog",
    "glass_blur",
)

FALLBACK_TASK_DESCRIPTIONS: dict[str, list[str]] = {
    "spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    ],
    "object": [
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the ketchup and place it in the basket",
    ],
    "goal": [
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside",
        "put the bowl on top of the cabinet",
    ],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["spatial", "object", "goal"])
    parser.add_argument("--tasks-per-suite", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument(
        "--severity",
        type=int,
        nargs="+",
        default=[3],
        help="Severity level(s) to test (1–5). E.g. --severity 1 2 3 4 5",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=list(NOISE_TYPES),
        help="Noise types to test. Default: all five Libero+ types.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--output-dir", default="visual_diagnostic/outputs")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--step-progress-every", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task collection
# ---------------------------------------------------------------------------


def _task_description_for(suite: str, task_id: int, state_dim: int) -> str:
    env_factory = make_env_factory("libero", suite=suite, state_dim=state_dim, task_id=task_id)
    env = env_factory(0)
    try:
        return env.task_description
    finally:
        env.close()


def _collect_task_specs(
    suites: list[str],
    tasks_per_suite: int,
    state_dim: int,
) -> list[TaskSpec]:
    """Build task specs — tries environment, falls back to hardcoded descriptions."""
    try:
        specs: list[TaskSpec] = []
        for suite in suites:
            env_factory = make_env_factory("libero", suite=suite, state_dim=state_dim)
            n = env_factory.num_tasks
            task_ids = list(range(min(tasks_per_suite, n)))
            print(f"Suite={suite}: selected task_ids={task_ids}")
            for task_id in task_ids:
                specs.append(
                    TaskSpec(
                        suite=suite,
                        task_id=task_id,
                        task_description=_task_description_for(suite, task_id, state_dim),
                    )
                )
        return specs
    except ModuleNotFoundError:
        print("LIBERO unavailable; using fallback task descriptions.")
        specs = []
        for suite in suites:
            descs = FALLBACK_TASK_DESCRIPTIONS.get(suite.lower(), [])
            for tid, desc in enumerate(descs[:tasks_per_suite]):
                specs.append(TaskSpec(suite=suite.lower(), task_id=tid, task_description=desc))
        return specs


# ---------------------------------------------------------------------------
# Episode seeds (same deterministic scheme as language diagnostics)
# ---------------------------------------------------------------------------


def _episode_seeds_for_task(
    global_seed: int, suite: str, task_id: int, num_episodes: int
) -> list[int]:
    suite_offset = {
        "spatial": 0,
        "object": 100_000,
        "goal": 200_000,
        "long": 300_000,
    }.get(suite, 400_000)
    base_seed = global_seed + suite_offset + task_id * 1_000
    return [base_seed + i for i in range(num_episodes)]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


@dataclass
class _RunKey:
    """Identifies one (suite, task_id, noise_type, severity) combination."""

    suite: str
    task_id: int
    noise_type: str
    severity: int

    def to_str(self) -> str:
        return f"{self.suite}|{self.task_id}|{self.noise_type}|{self.severity}"

    @staticmethod
    def from_str(s: str) -> "_RunKey":
        parts = s.split("|")
        return _RunKey(suite=parts[0], task_id=int(parts[1]), noise_type=parts[2], severity=int(parts[3]))


def _load_checkpoint(path: Path) -> tuple[set[str], list[dict[str, object]]]:
    """Load checkpoint: returns (completed_keys, accumulated_rows)."""
    if not path.exists():
        return set(), []
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("completed", [])), data.get("rows", [])


def _save_checkpoint(
    path: Path,
    completed: set[str],
    rows: list[dict[str, object]],
) -> None:
    """Save checkpoint atomically."""
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": sorted(completed),
        "rows": rows,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Noise-aware rollout
# ---------------------------------------------------------------------------


def _apply_noise_to_batch(
    batch: dict[str, torch.Tensor],
    noise_config: NoiseConfig | None,
) -> dict[str, torch.Tensor]:
    """Apply visual noise to all image tensors in a batch dict (in-place safe)."""
    if noise_config is None:
        return batch

    noised = dict(batch)
    for key in list(noised.keys()):
        if not key.startswith("observation.images."):
            continue
        img = noised[key]
        original_device = img.device

        # Handle batched images: (B, C, H, W) or (C, H, W)
        if img.ndim == 4:
            imgs = [apply_noise(img[i].cpu(), noise_config).to(original_device) for i in range(img.shape[0])]
            noised[key] = torch.stack(imgs)
        elif img.ndim == 3:
            noised[key] = apply_noise(img.cpu(), noise_config).to(original_device)
        elif img.ndim == 5:
            # (T, B, C, H, W)
            frames = []
            for t in range(img.shape[0]):
                batch_frames = [apply_noise(img[t, b].cpu(), noise_config).to(original_device) for b in range(img.shape[1])]
                frames.append(torch.stack(batch_frames))
            noised[key] = torch.stack(frames)
        # else: leave unchanged (e.g. 2D grayscale)

    return noised


def _evaluate_with_noise(
    policy: SmolVLAPolicy,
    suite: str,
    task_id: int,
    prompt: str,
    episode_seeds: list[int],
    max_steps: int,
    noise_config: NoiseConfig | None,
    progress_every: int,
    step_progress_every: int,
    marker_label: str,
) -> list[dict[str, object]]:
    """Run rollouts in the simulator, optionally applying visual noise to observations."""
    env_factory = make_env_factory("libero", suite=suite, state_dim=policy.state_dim, task_id=task_id)

    def _policy_fn(batch: dict) -> torch.Tensor:
        image_keys = sorted(k for k in batch if k.startswith("observation.images."))
        if not image_keys:
            raise ValueError(f"No image keys in batch. Keys: {list(batch.keys())}")
        cam_views: list[torch.Tensor] = []
        for key in image_keys:
            img = batch[key]
            if img.ndim in (4, 5):
                img = img[0]
            if img.ndim == 2:
                img = img.unsqueeze(0)
            cam_views.append(img)
        image = torch.stack(cam_views, dim=0) if len(cam_views) > 1 else cam_views[0]
        state = batch.get("observation.state")
        if state is not None and state.ndim == 2:
            state = state[0]
        return policy.predict_action(image, prompt, state)

    env = env_factory(0)
    try:
        capped_max_steps = min(max_steps, env.max_episode_steps)
        t0 = time.perf_counter()
        rollout_rows: list[dict[str, object]] = []

        for rollout_index, episode_seed in enumerate(episode_seeds):
            raw_obs, _info = env.reset(seed=episode_seed)
            ep_len = 0
            success = False
            ep_t0 = time.perf_counter()

            for step_idx in range(capped_max_steps):
                batch = env.obs_to_batch(raw_obs, device=policy.device)
                # Apply visual noise before policy inference
                batch = _apply_noise_to_batch(batch, noise_config)
                action_np = action_to_numpy(_policy_fn(batch))
                raw_obs, reward, terminated, truncated, info = env.step(action_np)
                _ = reward
                ep_len += 1

                step_done = step_idx + 1
                if step_progress_every > 0 and (
                    step_done == 1
                    or step_done == capped_max_steps
                    or step_done % step_progress_every == 0
                ):
                    ep_elapsed = time.perf_counter() - ep_t0
                    print(
                        f"        step-progress [{marker_label}] rollout={rollout_index + 1}/{len(episode_seeds)}, "
                        f"step={step_done}/{capped_max_steps}, elapsed={_fmt_seconds(ep_elapsed)}"
                    )

                if env.is_success(info):
                    success = True
                    break
                if terminated or truncated:
                    break

            rollout_rows.append(
                {
                    "rollout_index": rollout_index,
                    "episode_seed": episode_seed,
                    "success": success,
                    "episode_length": ep_len,
                }
            )

            done = rollout_index + 1
            if done == 1 or done == len(episode_seeds) or (progress_every > 0 and done % progress_every == 0):
                elapsed = time.perf_counter() - t0
                print(
                    f"      progress [{marker_label}] {done}/{len(episode_seeds)} rollouts, "
                    f"elapsed={_fmt_seconds(elapsed)}"
                )

        return rollout_rows
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "visual_noise_rollouts_raw.csv"
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "visual_noise_rollouts_summary.json"

    # Load checkpoint
    completed_keys, all_rows = _load_checkpoint(checkpoint_path)
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} condition(s) already completed, {len(all_rows)} rows loaded.")

    # Collect tasks
    task_specs = _collect_task_specs(
        suites=args.suites,
        tasks_per_suite=args.tasks_per_suite,
        state_dim=args.state_dim,
    )

    # Load policy
    device = get_device(args.device)
    policy = SmolVLAPolicy(
        checkpoint=DEFAULT_SFT_CHECKPOINT,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        device=str(device),
    )
    policy.eval()

    # Build the condition matrix: clean + each (noise_type, severity)
    conditions: list[tuple[str, int]] = [("clean", 0)]
    for sev in sorted(args.severity):
        for nt in args.noise_types:
            conditions.append((nt, sev))

    total_conditions = len(task_specs) * len(conditions)
    conditions_done = len(completed_keys)

    for spec in task_specs:
        episode_seeds = _episode_seeds_for_task(args.seed, spec.suite, spec.task_id, args.episodes)

        for noise_type, severity in conditions:
            run_key = _RunKey(suite=spec.suite, task_id=spec.task_id, noise_type=noise_type, severity=severity)
            key_str = run_key.to_str()

            if key_str in completed_keys:
                continue

            # Build noise config (None for clean)
            noise_config: NoiseConfig | None = None
            if noise_type != "clean":
                noise_config = NoiseConfig(noise_type=noise_type, severity=severity)

            label = f"{spec.suite}/task{spec.task_id}/{noise_type}" + (f"_s{severity}" if severity > 0 else "")
            print(f"\n  [{label}] {spec.task_description}")

            prompt_start = time.perf_counter()
            rollout_rows = _evaluate_with_noise(
                policy=policy,
                suite=spec.suite,
                task_id=spec.task_id,
                prompt=spec.task_description,
                episode_seeds=episode_seeds,
                max_steps=args.max_steps,
                noise_config=noise_config,
                progress_every=args.progress_every,
                step_progress_every=args.step_progress_every,
                marker_label=label,
            )

            # Append rows
            for rollout in rollout_rows:
                all_rows.append(
                    {
                        "suite": spec.suite,
                        "task_id": spec.task_id,
                        "task_description": spec.task_description,
                        "noise_type": noise_type,
                        "severity": severity,
                        "rollout_index": rollout["rollout_index"],
                        "episode_seed": rollout["episode_seed"],
                        "success": rollout["success"],
                        "episode_length": rollout["episode_length"],
                    }
                )

            # Update checkpoint
            completed_keys.add(key_str)
            conditions_done += 1
            _save_checkpoint(checkpoint_path, completed_keys, all_rows)
            _save_rows_csv(csv_path, all_rows)

            successes = sum(1 for r in rollout_rows if bool(r["success"]))
            lengths = [cast(int, r["episode_length"]) for r in rollout_rows]
            mean_len = sum(lengths) / max(len(lengths), 1)
            elapsed = time.perf_counter() - prompt_start
            print(
                f"    done [{label}] condition={conditions_done}/{total_conditions}, "
                f"success={successes}/{len(rollout_rows)}, mean_len={mean_len:.1f}, "
                f"elapsed={_fmt_seconds(elapsed)}"
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "noise_types_tested": args.noise_types,
        "severities_tested": args.severity,
        "num_rollout_rows": len(all_rows),
        "conditions_completed": len(completed_keys),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_rows_csv(csv_path, all_rows)
    print(f"\nSaved raw results to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Total wall time: {_fmt_seconds(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
