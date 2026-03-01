"""Evaluate a saved SmolVLA policy checkpoint in ManiSkill.

Usage:
    uv run python scripts/evaluate.py --checkpoint-dir checkpoints/sft/demos10_seed42/best
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from vla.diagnostics.eval import evaluate, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device, seed_everything

PICK_CUBE_INSTRUCTION = "pick up the red cube and move it to the green goal"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", "-d", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    action_dim: int = typer.Option(8, "--action-dim"),
    env_id: str = typer.Option("PickCube-v1", "--env"),
    num_episodes: int = typer.Option(100, "--num-episodes", "-n"),
    max_steps: int = typer.Option(200, "--max-steps"),
    seed: int = typer.Option(0, "--seed"),
    instruction: str = typer.Option(PICK_CUBE_INSTRUCTION, "--instruction"),
) -> None:
    """Evaluate a saved policy and print metrics."""
    seed_everything(seed)
    device = get_device()

    policy = SmolVLAPolicy(checkpoint=checkpoint, action_dim=action_dim, state_dim=0, device=str(device))
    policy.load_checkpoint(checkpoint_dir)
    logging.info(f"Loaded checkpoint from {checkpoint_dir} (state_dim={policy.state_dim})")

    metrics = evaluate(
        policy_fn=policy.predict_action,
        instruction=instruction,
        env_id=env_id,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    print_metrics(metrics, tag=str(checkpoint_dir))


if __name__ == "__main__":
    typer.run(main)
