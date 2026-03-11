"""Supervised fine-tuning (behaviour cloning) for SmolVLA.

Mirrors the LeRobot training recipe for SmolVLA fine-tuning:
  - AdamW with betas=(0.9, 0.95), weight_decay=1e-10
  - Cosine decay LR scheduler with linear warmup
  - Gradient clipping at max_norm=10
  - MEAN_STD action/state normalisation
  - Periodic rollout evaluation (success rate in simulator)
  - Checkpoint resume (optimizer + scheduler + epoch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from vla.base_config import BaseTrainingConfig
from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.env_metadata import EnvMetadata
from vla.models.smolvla import SmolVLAPolicy
from vla.training.checkpoint import save_best_checkpoint
from vla.training.lr_scheduler import cosine_decay_with_warmup_lambda_lr
from vla.training.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

TRAINING_STATE_FILE = "training_state.pt"


# ---------------------------------------------------------------------------
# Training state persistence
# ---------------------------------------------------------------------------


def _save_training_state(
    save_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_success: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_success": best_success,
        },
        save_dir / TRAINING_STATE_FILE,
    )


def _load_training_state(
    resume_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore optimizer/scheduler state and return (start_epoch, global_step, best_success)."""
    state_path = resume_dir / TRAINING_STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"No training state at {state_path}. Cannot resume.")
    # weights_only=False: optimizer and scheduler state dicts contain Python objects
    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    logger.info(
        "Resumed from epoch %d, step %d (best_success=%.2f%%)",
        state["epoch"],
        state["global_step"],
        state["best_success"] * 100,
    )
    return state["epoch"], state["global_step"], state["best_success"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig(BaseTrainingConfig):
    """Hyperparameters for supervised fine-tuning (behaviour cloning).

    Defaults match the LeRobot SmolVLA fine-tuning recipe.
    Inherits common fields from :class:`BaseTrainingConfig`.
    """

    lr: float = 1e-4
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    batch_size: int = 64
    micro_batch_size: int = 4
    num_epochs: int = 50
    eval_every: int = 5
    max_steps: int = 200
    save_dir: str = "checkpoints/sft"
    eval_suite: str = "all"
    control_mode: str = "pd_joint_delta_pos"
    resume_from: str | None = None


# ---------------------------------------------------------------------------
# Gradient step helper
# ---------------------------------------------------------------------------


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    trainable: list[torch.nn.Parameter],
    grad_clip_norm: float,
) -> float:
    """Clip gradients, step optimizer & scheduler, zero grads.

    Returns:
        The (possibly clipped) gradient norm.
    """
    if grad_clip_norm > 0:
        gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip_norm)
    else:
        gn = torch.nn.utils.clip_grad_norm_(trainable, float("inf"), error_if_nonfinite=False)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return gn.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_sft(
    policy: SmolVLAPolicy,
    dataset: Dataset,
    config: SFTConfig,
    metrics_logger: MetricsLogger | None = None,
    instruction: str | None = None,
) -> SmolVLAPolicy:
    """Run supervised fine-tuning (behaviour cloning) on a few-demo dataset.

    Follows the LeRobot SmolVLA training recipe:
      - AdamW with betas=(0.9, 0.95), weight_decay=1e-10
      - Cosine decay LR with linear warmup (auto-scaled to total steps)
      - Gradient clipping at ``config.max_grad_norm``
      - grad_norm logged every epoch

    Supports checkpoint resume via ``config.resume_from``.

    Periodic evaluation uses the simulator specified by ``config.simulator``
    (``"maniskill"`` or ``"libero"``), so you can train on ManiSkill data
    and evaluate in Libero, or vice-versa.

    Args:
        policy: SmolVLA policy to fine-tune.
        dataset: Dataset returning dicts with ``image``, ``state``, ``action``,
            ``instruction`` keys.  Must expose ``norm_stats`` attribute.
        config: SFT hyperparameters.
        metrics_logger: Optional :class:`MetricsLogger` for W&B and JSONL
            persistence.  When ``None``, metrics are not logged.
        instruction: Override instruction for evaluation (default: from dataset).

    Returns:
        The fine-tuned policy.
    """
    policy.set_normalization(
        dataset.norm_stats.action_mean,
        dataset.norm_stats.action_std,
        dataset.norm_stats.state_mean,
        dataset.norm_stats.state_std,
    )
    logger.info(
        "Action stats: mean=%s  std=%s",
        dataset.norm_stats.action_mean.tolist(),
        dataset.norm_stats.action_std.tolist(),
    )

    optimizer, trainable = config.build_optimizer(policy)

    micro_bs = min(config.micro_batch_size, config.batch_size)
    grad_accum_steps = max(config.batch_size // micro_bs, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_bs,
        shuffle=True,
        drop_last=False,
        pin_memory=policy.device.type == "cuda",
        num_workers=2,
        prefetch_factor=2,
    )
    micro_batches_per_epoch = max(len(dataloader), 1)
    optimizer_steps_per_epoch = max(micro_batches_per_epoch // grad_accum_steps, 1)
    total_optimizer_steps = config.num_epochs * optimizer_steps_per_epoch
    logger.info(
        "Training for %d epochs × %d optimizer steps = %d total  "
        "(micro_bs=%d, grad_accum=%d, effective_bs=%d, lr=%.2e, warmup=%d, decay=%d)",
        config.num_epochs,
        optimizer_steps_per_epoch,
        total_optimizer_steps,
        micro_bs,
        grad_accum_steps,
        micro_bs * grad_accum_steps,
        config.lr,
        config.warmup_steps,
        config.decay_steps,
    )

    scheduler = cosine_decay_with_warmup_lambda_lr(
        optimizer,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        peak_lr=config.lr,
        decay_lr=config.decay_lr,
        total_steps=total_optimizer_steps,
    )

    eval_instruction = instruction or getattr(dataset, "instruction", "complete the manipulation task")
    control_mode = config.control_mode
    env_id = config.env_id
    save_path = Path(config.save_dir)
    best_success = -1.0
    global_step = 0
    start_epoch = 0

    if config.resume_from:
        resume_dir = Path(config.resume_from)
        start_epoch, global_step, best_success = _load_training_state(resume_dir, optimizer, scheduler, policy.device)

    _save_meta = EnvMetadata(
        env_id=env_id,
        instruction=eval_instruction,
        control_mode=control_mode,
    )

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        policy.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_micro_batches = 0
        num_optimizer_steps = 0
        optimizer.zero_grad()

        for batch in dataloader:
            images = batch["image"].to(policy.device)
            target_actions = batch["action"].to(policy.device)
            states = batch["state"].to(policy.device)
            batch_instructions = batch["instruction"]
            unique_instrs = set(batch_instructions)
            instr_input: str | list[str] = batch_instructions if len(unique_instrs) > 1 else list(unique_instrs)[0]

            with torch.autocast(device_type=policy.device.type, dtype=policy.dtype):
                out = policy(images, instr_input, target_actions, states=states)
            loss = out["loss"] / grad_accum_steps
            loss.backward()

            epoch_loss += out["loss"].item()
            num_micro_batches += 1

            if num_micro_batches % grad_accum_steps == 0:
                epoch_grad_norm += _optimizer_step(optimizer, scheduler, trainable, config.max_grad_norm)
                num_optimizer_steps += 1
                global_step += 1

        if num_micro_batches % grad_accum_steps != 0:
            epoch_grad_norm += _optimizer_step(optimizer, scheduler, trainable, config.max_grad_norm)
            num_optimizer_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_micro_batches, 1)
        avg_grad_norm = epoch_grad_norm / max(num_optimizer_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "SFT epoch %d/%d  loss=%.6f  grad_norm=%.4f  lr=%.2e  step=%d",
            epoch,
            config.num_epochs,
            avg_loss,
            avg_grad_norm,
            current_lr,
            global_step,
        )
        if metrics_logger is not None:
            metrics_logger.log(
                {
                    "sft/loss": avg_loss,
                    "sft/grad_norm": avg_grad_norm,
                    "sft/lr": current_lr,
                    "sft/epoch": epoch,
                    "sft/step": global_step,
                }
            )

        if epoch % config.eval_every == 0 or epoch == config.num_epochs:
            metrics = evaluate_smolvla(
                policy,
                instruction=eval_instruction,
                simulator=config.simulator,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 10000,
                control_mode=control_mode,
                suite=config.eval_suite,
            )
            print_metrics(metrics, tag=f"SFT epoch {epoch}")
            eval_log = {
                "sft/success_rate": metrics.success_rate,
                "sft/mean_reward": metrics.mean_reward,
                "sft/mean_ep_len": metrics.mean_episode_length,
                "sft/epoch": epoch,
            }
            if metrics_logger is not None:
                metrics_logger.log(eval_log)

            def _save(
                _epoch: int = epoch,
                _step: int = global_step,
                _best: float = best_success,
            ) -> None:
                policy.save_checkpoint(save_path / "best", env_metadata=_save_meta)
                _save_training_state(save_path / "best", optimizer, scheduler, _epoch, _step, _best)

            best_success = save_best_checkpoint(
                metrics.success_rate, best_success, _save, tag="SFT",
            )

        policy.save_checkpoint(save_path / "last", env_metadata=_save_meta)
        _save_training_state(save_path / "last", optimizer, scheduler, epoch, global_step, best_success)

    return policy
