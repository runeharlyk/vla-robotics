"""Evaluate a saved SmolVLA policy checkpoint in ManiSkill or Libero.

Usage:
    uv run python scripts/evaluate.py --checkpoint-dir checkpoints/sft/peginsertionside_v1_demos10_seed42/best
    uv run python scripts/evaluate.py --checkpoint-dir ... --simulator libero --suite spatial
    uv run python scripts/evaluate.py --checkpoint-dir ... --simulator maniskill --env PickCube-v1
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", "-d", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    simulator: str = typer.Option("maniskill", "--simulator", "-s", help="Simulator backend: maniskill or libero"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    suite: str = typer.Option("all", "--suite", help="Libero suite (spatial/object/goal/long/all)"),
    num_episodes: int = typer.Option(100, "--num-episodes", "-n"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max steps (default: from metadata)"),
    seed: int = typer.Option(0, "--seed"),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    control_mode: str = typer.Option(None, "--control-mode", help="Override control mode (default: from checkpoint)"),
) -> None:
    """Evaluate a saved policy and print metrics.

    Supports both ManiSkill and Libero simulators.  ``env_id``,
    ``instruction``, and ``control_mode`` are loaded from the checkpoint's
    saved metadata unless explicitly overridden via CLI flags.
    """
    seed_everything(seed)
    device = get_device()

    policy = SmolVLAPolicy(checkpoint=checkpoint, action_dim=8, state_dim=0, device=str(device))
    env_meta = policy.load_checkpoint(checkpoint_dir)
    logging.info(
        "Loaded checkpoint from %s (action_dim=%d, state_dim=%d, env_metadata=%s)",
        checkpoint_dir,
        policy.action_dim,
        policy.state_dim,
        env_meta,
    )

    resolved_env_id = env_id or env_meta.get("env_id", "PickCube-v1")
    resolved_instruction = instruction or env_meta.get("instruction", "complete the manipulation task")
    resolved_control_mode = control_mode or env_meta.get("control_mode", "pd_joint_delta_pos")
    resolved_max_steps = max_steps or 200

    logging.info(
        "Evaluating: simulator=%s  env_id=%s  instruction=%r  control_mode=%s  max_steps=%d  suite=%s",
        simulator,
        resolved_env_id,
        resolved_instruction,
        resolved_control_mode,
        resolved_max_steps,
        suite,
    )

    metrics = evaluate_smolvla(
        policy,
        instruction=resolved_instruction,
        simulator=simulator,
        env_id=resolved_env_id,
        num_episodes=num_episodes,
        max_steps=resolved_max_steps,
        seed=seed,
        control_mode=resolved_control_mode,
        suite=suite,
    )
    print_metrics(metrics, tag=str(checkpoint_dir))


if __name__ == "__main__":
    typer.run(main)
