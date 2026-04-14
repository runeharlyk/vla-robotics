"""CLI entry point for checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path

import typer

from vla.evaluation.evaluate import run_evaluation
from vla.evaluation.runtime import EvalConfig


def main(
    checkpoint_dir: Path | None = typer.Option(None, "--checkpoint-dir", "-d", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    simulator: str = typer.Option("maniskill", "--simulator", "-s", help="Simulator backend: maniskill or libero"),
    env_id: str | None = typer.Option(None, "--env", "--env-id", help="Override env id (default: from checkpoint)"),
    suite: str = typer.Option("all", "--suite", help="Libero suite (spatial/object/goal/long/all)"),
    num_episodes: int = typer.Option(100, "--num-episodes", "-n"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override max steps (default: from metadata)"),
    seed: int = typer.Option(0, "--seed"),
    num_envs: int = typer.Option(4, "--num-envs"),
    task_id: int | None = typer.Option(None, "--task-id", help="Optional LIBERO task id override"),
    fixed_noise_seed: int | None = typer.Option(
        None,
        "--fixed-noise-seed",
        help="Use deterministic seeded evaluation noise instead of fresh sampling",
    ),
    use_wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log live eval metrics to Weights & Biases"),
    wandb_project: str = typer.Option("vla-eval", "--wandb-project", help="W&B project name for evaluation runs"),
    wandb_name: str | None = typer.Option(None, "--wandb-name", help="Optional W&B run name"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity", help="Optional W&B entity/team"),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        help="Override instruction (default: from checkpoint)",
    ),
    control_mode: str | None = typer.Option(
        None,
        "--control-mode",
        help="Override control mode (default: from checkpoint)",
    ),
    action_dim: int = typer.Option(7, "--action-dim", help="Action dimension (used when no checkpoint-dir)"),
    state_dim: int = typer.Option(8, "--state-dim", help="State dimension (used when no checkpoint-dir)"),
    device: str = typer.Option("cuda", "--device", help="Torch device to use for the policy"),
) -> None:
    run_evaluation(
        EvalConfig(
            checkpoint_dir=checkpoint_dir,
            checkpoint=checkpoint,
            simulator=simulator,
            env_id=env_id,
            suite=suite,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            num_envs=num_envs,
            task_id=task_id,
            fixed_noise_seed=fixed_noise_seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            wandb_entity=wandb_entity,
            instruction=instruction,
            control_mode=control_mode,
            action_dim=action_dim,
            state_dim=state_dim,
            device=device,
        )
    )


if __name__ == "__main__":
    typer.run(main)
