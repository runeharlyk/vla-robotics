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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import typer

from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.env_metadata import EnvMetadata
from vla.models.smolvla import SmolVLAPolicy
from vla.training.metrics_logger import MetricsLogger
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_eval_log_prefix(simulator: str, suite: str, task_payload: dict[str, Any]) -> str:
    if simulator.lower() == "libero":
        return f"eval/{suite}/task_{task_payload['task_id']}"
    return f"eval/{simulator}/task_{task_payload['task_id']}"


def _log_eval_metrics(
    metrics_logger: MetricsLogger | None,
    simulator: str,
    suite: str,
    task_payload: dict[str, Any],
) -> None:
    if metrics_logger is None:
        return

    prefix = _build_eval_log_prefix(simulator, suite, task_payload)
    log_data = {
        f"{prefix}/success_rate": task_payload["success_rate"],
        f"{prefix}/successes": task_payload["successes"],
        f"{prefix}/num_episodes": task_payload["num_episodes"],
        f"{prefix}/mean_reward": task_payload["mean_reward"],
        f"{prefix}/mean_episode_length": task_payload["mean_episode_length"],
    }
    if "task_index" in task_payload:
        log_data["eval/progress/tasks_completed"] = int(task_payload["task_index"]) + 1
    if "tasks_total" in task_payload:
        log_data["eval/progress/tasks_total"] = task_payload["tasks_total"]
    metrics_logger.log(log_data)


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
    use_wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log live eval metrics to Weights & Biases"),
    wandb_project: str = typer.Option("vla-eval", "--wandb-project", help="W&B project name for evaluation runs"),
    wandb_name: str | None = typer.Option(None, "--wandb-name", help="Optional W&B run name"),
    wandb_entity: str | None = typer.Option(None, "--wandb-entity", help="Optional W&B entity/team"),
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
    wandb_run = None
    metrics_logger: MetricsLogger | None = None
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

    # If evaluating a base checkpoint (no checkpoint_dir), provide better defaults for LIBERO
    if checkpoint_dir is None:
        if simulator == "libero":
            resolved_env_id = env_id or f"libero_{suite}"
            resolved_instruction = instruction or "follow the task instruction"
            resolved_control_mode = control_mode or "relative"
        elif simulator == "maniskill":
            resolved_env_id = env_id or "PickCube-v1"
            resolved_instruction = instruction or "pick up the cube"
            resolved_control_mode = control_mode or "pd_joint_delta_pos"

    resolved_max_steps = max_steps or 220

    # Build a concise log message with only relevant information
    log_bits = [f"simulator={simulator}"]
    if simulator == "libero":
        log_bits.append(f"suite={suite}")
        if task_id is not None:
            log_bits.append(f"task_id={task_id}")
    else:
        log_bits.append(f"env_id={resolved_env_id}")

    log_bits.append(f"control_mode={resolved_control_mode}")
    log_bits.append(f"max_steps={resolved_max_steps}")

    if instruction:  # Only log instruction if it was explicitly provided via CLI
        log_bits.append(f"instruction={instruction!r}")

    logging.info("Evaluating: %s", "  ".join(log_bits))

    if use_wandb:
        import wandb

        resolved_run_name = wandb_name or (
            f"eval_{simulator}_{suite}_{checkpoint_dir.name if checkpoint_dir else checkpoint.split('/')[-1]}"
        )
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=resolved_run_name,
            config={
                "checkpoint": checkpoint,
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "simulator": simulator,
                "suite": suite,
                "env_id": resolved_env_id,
                "num_episodes": num_episodes,
                "num_envs": num_envs,
                "max_steps": resolved_max_steps,
                "seed": seed,
                "fixed_noise_seed": fixed_noise_seed,
                "task_id": task_id,
            },
        )
        metrics_logger = MetricsLogger(wandb_run=wandb_run)

    task_callback: Callable[[int, dict[str, Any]], None] | None = None
    if metrics_logger is not None:
        task_callback = lambda _task_id, payload: _log_eval_metrics(metrics_logger, simulator, suite, payload)

    try:
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
            task_metrics_callback=task_callback,
        )
        print_metrics(metrics, tag=tag)
        if metrics_logger is not None:
            overall_prefix = f"eval/{suite}" if simulator.lower() == "libero" else f"eval/{simulator}"
            metrics_logger.log(
                {
                    f"{overall_prefix}/overall/success_rate": metrics.success_rate,
                    f"{overall_prefix}/overall/successes": metrics.successes,
                    f"{overall_prefix}/overall/num_episodes": metrics.num_episodes,
                    f"{overall_prefix}/overall/mean_reward": metrics.mean_reward,
                    f"{overall_prefix}/overall/mean_episode_length": metrics.mean_episode_length,
                    f"{overall_prefix}/overall/median_episode_length": metrics.median_episode_length,
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
