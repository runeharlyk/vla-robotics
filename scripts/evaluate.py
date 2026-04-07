"""Evaluate a SmolVLA policy checkpoint in ManiSkill or Libero.

Usage:
    # Evaluate the base HuggingFace SFT checkpoint directly:
    uv run python scripts/evaluate.py --checkpoint HuggingFaceVLA/smolvla_libero --simulator libero --suite spatial

    # Evaluate a fine-tuned checkpoint (RL or SFT):
    uv run python scripts/evaluate.py --checkpoint-dir checkpoints/sparse_rl/best --simulator libero --suite spatial
    uv run python scripts/evaluate.py --checkpoint-dir checkpoints/sft/best --simulator maniskill --env PickCube-v1
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer

from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.env_metadata import EnvMetadata
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(
    checkpoint_dir: Path = typer.Option(None, "--checkpoint-dir", "-d", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    simulator: str = typer.Option("maniskill", "--simulator", "-s", help="Simulator backend: maniskill or libero"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    suite: str = typer.Option("all", "--suite", help="Libero suite (spatial/object/goal/long/all)"),
    num_episodes: int = typer.Option(100, "--num-episodes", "-n"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max steps (default: from metadata)"),
    seed: int = typer.Option(0, "--seed"),
    num_envs: int = typer.Option(4, "--num-envs"),
    task_id: int | None = typer.Option(None, "--task-id", help="Optional LIBERO task id override"),
    fixed_noise_seed: int | None = typer.Option(
        None,
        "--fixed-noise-seed",
        help="Use deterministic seeded evaluation noise instead of fresh sampling",
    ),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    control_mode: str = typer.Option(None, "--control-mode", help="Override control mode (default: from checkpoint)"),
    action_dim: int = typer.Option(7, "--action-dim", help="Action dimension (used when no checkpoint-dir)"),
    state_dim: int = typer.Option(8, "--state-dim", help="State dimension (used when no checkpoint-dir)"),
) -> None:
    """Evaluate a saved policy and print metrics.

    Supports both ManiSkill and Libero simulators.  When ``--checkpoint-dir``
    is provided, model weights and metadata are loaded from the saved
    ``policy.pt``.  When omitted, the base HuggingFace checkpoint is
    evaluated directly (useful for SFT baseline evaluation).

    ``env_id``, ``instruction``, and ``control_mode`` are loaded from the
    checkpoint's saved metadata unless explicitly overridden via CLI flags.
    """
    seed_everything(seed)
    device = get_device()

    if checkpoint_dir is not None:
        ckpt_data = torch.load(checkpoint_dir / "policy.pt", map_location="cpu", weights_only=False)
        action_dim = ckpt_data.get("action_dim", action_dim)
        state_dim = ckpt_data.get("state_dim", state_dim)

    policy = SmolVLAPolicy(checkpoint=checkpoint, action_dim=action_dim, state_dim=state_dim, device=str(device))

    if checkpoint_dir is not None:
        env_meta = policy.load_checkpoint(checkpoint_dir)
        logging.info(
            "Loaded checkpoint from %s (action_dim=%d, state_dim=%d, env_metadata=%s)",
            checkpoint_dir,
            policy.action_dim,
            policy.state_dim,
            env_meta,
        )
    else:
        env_meta = EnvMetadata()
        logging.info(
            "Using base HuggingFace checkpoint %s (action_dim=%d, state_dim=%d)",
            checkpoint,
            policy.action_dim,
            policy.state_dim,
        )

    tag = str(checkpoint_dir) if checkpoint_dir else checkpoint
    resolved_env_id = env_id or env_meta.env_id
    resolved_instruction = instruction or env_meta.instruction
    resolved_control_mode = control_mode or env_meta.control_mode
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
        num_envs=num_envs,
        task_id=task_id,
        max_steps=resolved_max_steps,
        seed=seed,
        control_mode=resolved_control_mode,
        suite=suite,
        fixed_noise_seed=fixed_noise_seed,
    )
    print_metrics(metrics, tag=tag)


if __name__ == "__main__":
    typer.run(main)
