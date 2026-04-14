"""Checkpoint evaluation runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.evaluation.runtime import EvalConfig, resolve_eval_runtime
from vla.results_registry import (
    RESULTS_DIR,
    find_training_metadata,
    flatten_task_metrics,
    get_git_info,
    get_scheduler_info,
    load_json_if_exists,
    now_iso,
    sanitize_name,
    write_eval_registry,
    write_json,
)
from vla.training.metrics_logger import MetricsLogger
from vla.utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def _default_eval_name(
    simulator: str,
    suite: str,
    checkpoint_dir: Path | None,
    checkpoint: str,
    wandb_name: str | None,
) -> str:
    if wandb_name:
        return wandb_name
    checkpoint_tag = checkpoint_dir.name if checkpoint_dir else checkpoint.split("/")[-1]
    return f"eval_{simulator}_{suite}_{checkpoint_tag}"


def _append_task_payload(
    task_payloads: list[dict[str, Any]],
    metrics_logger: MetricsLogger | None,
    simulator: str,
    suite: str,
    _task_id: int,
    payload: dict[str, Any],
) -> None:
    task_payloads.append(dict(payload))
    _log_eval_metrics(metrics_logger, simulator, suite, payload)


def run_evaluation(config: EvalConfig) -> None:
    """Evaluate a saved policy checkpoint and persist the result."""
    wandb_run = None
    metrics_logger: MetricsLogger | None = None
    task_payloads: list[dict[str, Any]] = []

    seed_everything(config.seed)
    runtime = resolve_eval_runtime(config)

    if config.checkpoint_dir is not None:
        logger.info(
            "Loaded checkpoint from %s (action_dim=%d, state_dim=%d, env_metadata=%s)",
            config.checkpoint_dir,
            runtime.action_dim,
            runtime.state_dim,
            runtime.env_meta,
        )
    else:
        logger.info(
            "Using base checkpoint %s (action_dim=%d, state_dim=%d)",
            config.checkpoint,
            runtime.action_dim,
            runtime.state_dim,
        )

    tag = str(config.checkpoint_dir) if config.checkpoint_dir else config.checkpoint

    log_bits = [f"simulator={config.simulator}"]
    if config.simulator.lower() == "libero":
        log_bits.append(f"suite={config.suite}")
        if config.task_id is not None:
            log_bits.append(f"task_id={config.task_id}")
    else:
        log_bits.append(f"env_id={runtime.env_id}")
    log_bits.append(f"device={runtime.device}")
    log_bits.append(f"control_mode={runtime.control_mode}")
    log_bits.append(f"max_steps={runtime.max_steps}")
    if config.instruction:
        log_bits.append(f"instruction={config.instruction!r}")
    logger.info("Evaluating: %s", "  ".join(log_bits))

    resolved_run_name = _default_eval_name(
        config.simulator,
        config.suite,
        config.checkpoint_dir,
        config.checkpoint,
        config.wandb_name,
    )

    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=resolved_run_name,
            config={
                "checkpoint": config.checkpoint,
                "checkpoint_dir": str(config.checkpoint_dir) if config.checkpoint_dir else None,
                "simulator": config.simulator,
                "suite": config.suite,
                "env_id": runtime.env_id,
                "num_episodes": config.num_episodes,
                "num_envs": config.num_envs,
                "max_steps": runtime.max_steps,
                "seed": config.seed,
                "fixed_noise_seed": config.fixed_noise_seed,
                "task_id": config.task_id,
                "device": str(runtime.device),
            },
        )
        metrics_logger = MetricsLogger(wandb_run=wandb_run)

    try:
        metrics = evaluate_smolvla(
            runtime.policy,
            instruction=runtime.instruction,
            simulator=config.simulator,
            env_id=runtime.env_id,
            num_episodes=config.num_episodes,
            num_envs=config.num_envs,
            task_id=config.task_id,
            max_steps=runtime.max_steps,
            seed=config.seed,
            control_mode=runtime.control_mode,
            suite=config.suite,
            fixed_noise_seed=config.fixed_noise_seed,
            task_metrics_callback=lambda current_task_id, payload: _append_task_payload(
                task_payloads,
                metrics_logger,
                config.simulator,
                config.suite,
                current_task_id,
                payload,
            ),
        )
        print_metrics(metrics, tag=tag)

        if metrics_logger is not None:
            if config.simulator.lower() == "libero":
                overall_prefix = f"eval/{config.suite}"
            else:
                overall_prefix = f"eval/{config.simulator}"
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

        training_metadata_path = find_training_metadata(config.checkpoint_dir)
        training_metadata = load_json_if_exists(training_metadata_path)
        eval_record = {
            "record_type": "evaluation",
            "recorded_at": now_iso(),
            "eval_name": resolved_run_name,
            "checkpoint": config.checkpoint,
            "checkpoint_dir": str(config.checkpoint_dir) if config.checkpoint_dir else "",
            "tag": tag,
            "simulator": config.simulator,
            "suite": config.suite,
            "env_id": runtime.env_id,
            "instruction": runtime.instruction,
            "control_mode": runtime.control_mode,
            "num_episodes": config.num_episodes,
            "num_envs": config.num_envs,
            "max_steps": runtime.max_steps,
            "seed": config.seed,
            "fixed_noise_seed": config.fixed_noise_seed,
            "task_id": config.task_id,
            "device": str(runtime.device),
            "wandb_run_name": resolved_run_name if config.use_wandb else "",
            "success_rate": metrics.success_rate,
            "successes": metrics.successes,
            "mean_reward": metrics.mean_reward,
            "mean_episode_length": metrics.mean_episode_length,
            "median_episode_length": metrics.median_episode_length,
            "task_metrics": task_payloads,
            "training_metadata_path": str(training_metadata_path) if training_metadata_path else "",
            "training_method": (training_metadata or {}).get("method", ""),
            "training_save_dir": (training_metadata or {}).get("save_dir", ""),
            "training_git_commit": (training_metadata or {}).get("git_commit", ""),
            **get_git_info(),
            **get_scheduler_info(),
        }
        eval_slug = sanitize_name(resolved_run_name)
        eval_json_path = RESULTS_DIR / "evals" / f"{eval_slug}.json"
        write_json(eval_json_path, eval_record)

        eval_registry_row = {k: v for k, v in eval_record.items() if k not in {"task_metrics", "instruction"}}
        eval_registry_row["result_json"] = str(eval_json_path)
        eval_registry_row.update(flatten_task_metrics(task_payloads))
        write_eval_registry(eval_registry_row)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
