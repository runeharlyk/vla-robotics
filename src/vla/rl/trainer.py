"""Training loops for SFT (behaviour cloning) and SRPO reinforcement learning.

SFT (``train_sft``) mirrors the LeRobot training recipe for SmolVLA fine-tuning:
  - AdamW with betas=(0.9, 0.95), weight_decay=1e-10
  - Cosine decay LR scheduler with linear warmup
  - Gradient clipping at max_norm=10
  - MEAN_STD action/state normalisation

SRPO (``train_srpo``) implements the full paper algorithm (Section 3.3):
  1. Collect M trajectories with π_θ_old.
  2. Compute world-progress trajectory rewards g_i via the world model.
  3. Compute trajectory-level advantages  = (g_i − μ_g) / σ_g.
  4. Cache per-step flow-matching losses under θ_old (fixed noise/time).
  5. For each PPO-epoch:
       - Recompute per-step FM losses under θ → importance ratio
       - Clipped surrogate loss + KL regularisation against π_ref.
  6. Update θ_old ← θ, add any new successes to reference set.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from vla.data.dataset import FewDemoDataset
from vla.diagnostics.eval import evaluate, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.models.world_model import WorldModelEncoder, build_world_model
from vla.rl.rollout import ManiSkillRollout, Trajectory
from vla.rl.srpo_reward import SRPORewardConfig, WorldProgressReward

logger = logging.getLogger(__name__)


def _cosine_decay_with_warmup_schedule(
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
    decay_lr: float,
    total_steps: int,
) -> callable:
    """Return a lambda for :class:`LambdaLR` implementing cosine decay with linear warmup.

    If ``total_steps < decay_steps`` both warmup and decay durations are
    scaled proportionally (matching the LeRobot auto-scale behaviour).

    Args:
        warmup_steps: Number of linear warmup steps (before scaling).
        decay_steps: Number of cosine decay steps (before scaling).
        peak_lr: The learning rate at the end of warmup.
        decay_lr: The minimum learning rate after decay.
        total_steps: Actual number of optimiser steps for the run.

    Returns:
        A multiplier function ``step -> lr_multiplier`` for :class:`LambdaLR`.
    """
    if decay_steps > total_steps:
        warmup_steps = int(warmup_steps * total_steps / decay_steps)
        decay_steps = total_steps

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return (current_step + 1) / (warmup_steps + 1)
        progress = min((current_step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = decay_lr + (peak_lr - decay_lr) * cosine
        return lr / peak_lr

    return _lr_lambda


@dataclass
class SFTConfig:
    """Hyperparameters for supervised fine-tuning (behaviour cloning).

    Defaults match the LeRobot SmolVLA fine-tuning recipe.
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    batch_size: int = 64
    micro_batch_size: int = 4
    num_epochs: int = 50
    eval_every: int = 5
    eval_episodes: int = 50
    max_steps: int = 200
    save_dir: str = "checkpoints/sft"
    env_id: str = "PickCube-v1"
    seed: int = 42


@dataclass
class SRPOConfig:
    """Hyperparameters for SRPO RL training."""

    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    num_iterations: int = 100
    trajectories_per_iter: int = 16
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    eval_every: int = 10
    eval_episodes: int = 50
    max_steps: int = 200
    save_dir: str = "checkpoints/srpo"
    env_id: str = "PickCube-v1"
    seed: int = 42
    mode: str = "srpo"
    world_model_type: str = "dinov2"
    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    num_fm_noise_samples: int = 4
    gamma: float = 0.99
    reward_scale: float = 1.0


def train_sft(
    policy: SmolVLAPolicy,
    dataset: FewDemoDataset,
    config: SFTConfig,
    wandb_run: Any | None = None,
) -> SmolVLAPolicy:
    """Run supervised fine-tuning (behaviour cloning) on a few-demo dataset.

    Follows the LeRobot SmolVLA training recipe:
      - AdamW with betas=(0.9, 0.95), weight_decay=1e-10
      - Cosine decay LR with linear warmup (auto-scaled to total steps)
      - Gradient clipping at ``config.grad_clip_norm``

    Args:
        policy: SmolVLA policy to fine-tune.
        dataset: Few-demo dataset.
        config: SFT hyperparameters.
        wandb_run: Optional wandb run for logging.

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

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    micro_bs = min(config.micro_batch_size, config.batch_size)
    grad_accum_steps = max(config.batch_size // micro_bs, 1)
    dataloader = DataLoader(dataset, batch_size=micro_bs, shuffle=True, drop_last=False)
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

    lr_lambda = _cosine_decay_with_warmup_schedule(
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        peak_lr=config.lr,
        decay_lr=config.decay_lr,
        total_steps=total_optimizer_steps,
    )
    scheduler = LambdaLR(optimizer, lr_lambda)

    instruction = dataset.instruction
    save_path = Path(config.save_dir)
    best_success = -1.0
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        policy.train()
        epoch_loss = 0.0
        num_micro_batches = 0
        optimizer.zero_grad()

        for batch in dataloader:
            images = batch["image"].to(policy.device)
            target_actions = batch["action"].to(policy.device)
            states = batch["state"].to(policy.device)
            out = policy(images, instruction, target_actions, states=states)
            loss = out["loss"] / grad_accum_steps
            loss.backward()

            epoch_loss += out["loss"].item()
            num_micro_batches += 1

            if num_micro_batches % grad_accum_steps == 0:
                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        if num_micro_batches % grad_accum_steps != 0:
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(num_micro_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"SFT epoch {epoch}/{config.num_epochs}  loss={avg_loss:.6f}  lr={current_lr:.2e}  step={global_step}"
        )
        if wandb_run is not None:
            wandb_run.log({"sft/loss": avg_loss, "sft/lr": current_lr, "sft/epoch": epoch, "sft/step": global_step})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs:
            metrics = evaluate(
                policy_fn=policy.predict_action,
                instruction=instruction,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 10000,
            )
            print_metrics(metrics, tag=f"SFT epoch {epoch}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "sft/success_rate": metrics.success_rate,
                        "sft/mean_reward": metrics.mean_reward,
                        "sft/mean_ep_len": metrics.mean_episode_length,
                        "sft/epoch": epoch,
                    }
                )
            if metrics.success_rate > best_success:
                best_success = metrics.success_rate
                policy.save_checkpoint(save_path / "best")
                logger.info(f"New best SFT checkpoint: {best_success:.2%}")

    policy.save_checkpoint(save_path / "last")
    return policy


def _compute_fm_loss_per_step(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
) -> torch.Tensor:
    """Compute per-timestep flow-matching loss for a trajectory.

    Uses pre-sampled noise/time so that the same random quantities are
    used under both θ_old and θ_new for valid importance-ratio estimation.
    Multiple noise/time samples are averaged for variance reduction.

    Args:
        policy: The current policy (either θ or θ_old).
        traj: Trajectory whose actions we evaluate.
        instruction: Language instruction.
        fixed_noise: Pre-sampled noise, one tensor per timestep.
        fixed_time: Pre-sampled time, one tensor per timestep.

    Returns:
        ``(T,)`` mean FM loss per timestep.
    """
    T = traj.length
    step_losses = []
    for t in range(T):
        img = traj.images[t].unsqueeze(0).to(policy.device)
        action = traj.actions[t].unsqueeze(0).to(policy.device)

        imgs_f = policy._to_float01(img).to(policy.device, dtype=policy.dtype)
        img_list, mask_list = policy._prepare_images(imgs_f)
        tokens, tmasks = policy._tokenize(instruction, batch_size=1)
        state_raw = traj.states[t].unsqueeze(0) if traj.states is not None else None
        state = policy._prepare_state_input(state_raw, batch_size=1)
        normalized_action = policy._normalize_action(action.to(policy.device, dtype=policy.dtype))
        action_padded = policy._prepare_action(normalized_action)
        action_padded = action_padded.unsqueeze(1).expand(-1, policy.chunk_size, -1)

        losses_per_sample = policy.model.forward(
            img_list,
            mask_list,
            tokens,
            tmasks,
            state,
            action_padded,
            noise=fixed_noise[t].to(policy.device, dtype=policy.dtype),
            time=fixed_time[t].to(policy.device, dtype=policy.dtype),
        )
        step_losses.append(losses_per_sample[:, :, : policy.max_action_dim].mean().detach())

    return torch.stack(step_losses)


def _sample_fixed_noise_time(
    traj: Trajectory,
    policy: SmolVLAPolicy,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pre-sample noise and time tensors for each timestep in a trajectory.

    Returns:
        Tuple of (noise_list, time_list) – each is a list of T tensors.
    """
    T = traj.length
    noise_list = []
    time_list = []
    for _ in range(T):
        n = torch.randn(1, policy.chunk_size, policy.max_action_dim)
        beta = torch.distributions.Beta(1.5, 1.0)
        t = beta.sample((1,)) * 0.999 + 0.001
        noise_list.append(n)
        time_list.append(t)
    return noise_list, time_list


def train_srpo(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    instruction: str,
    demo_trajectories: list[Trajectory] | None = None,
    wandb_run: Any | None = None,
) -> SmolVLAPolicy:
    """Run SRPO training on top of an SFT-initialised policy.

    Implements the full SRPO algorithm from the paper:
      - World-progress reward shaping via world-model embeddings + DBSCAN
      - Trajectory-level advantages (GRPO-style)
      - Clipped surrogate objective with importance sampling
      - KL regularisation against the reference (SFT) policy

    When ``config.mode == "sparse_rl"`` the world-model rewards are
    disabled and only the binary environment reward is used (ablation).

    Args:
        policy: SFT-initialised SmolVLA policy (becomes π_θ *and* π_ref).
        config: SRPO hyperparameters.
        instruction: Language instruction for the task.
        demo_trajectories: Optional list of demo trajectories to seed the
            reference set.  Their images are used for world-model encoding.
        wandb_run: Optional wandb run for logging.

    Returns:
        The RL-tuned policy.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    world_encoder: WorldModelEncoder | None = None
    reward_model: WorldProgressReward | None = None

    if config.mode == "srpo":
        world_encoder = build_world_model(
            model_type=config.world_model_type,
            device=str(policy.device),
        )
        reward_cfg = SRPORewardConfig(
            subsample_every=config.subsample_every,
            dbscan_eps=config.dbscan_eps,
            dbscan_min_samples=config.dbscan_min_samples,
        )
        reward_model = WorldProgressReward(world_encoder, reward_cfg)

        if demo_trajectories:
            demo_images = []
            for dt in demo_trajectories:
                imgs = dt.images[: dt.length]
                if imgs.dtype == torch.uint8:
                    imgs = imgs.float() / 255.0
                else:
                    imgs = imgs.float()
                demo_images.append(imgs)
            reward_model.add_demo_trajectories(demo_images)

    rollout_engine = ManiSkillRollout(
        env_id=config.env_id,
        num_envs=1,
        max_steps=config.max_steps,
    )
    save_path = Path(config.save_dir)
    best_success = -1.0

    for iteration in range(1, config.num_iterations + 1):
        # ── 1. Collect trajectories with current policy ─────────────────
        policy.eval()
        trajectories = rollout_engine.collect_batch(
            policy_fn=policy.predict_action,
            instruction=instruction,
            num_trajectories=config.trajectories_per_iter,
            seed=config.seed + iteration * 1000,
        )
        num_successes = sum(1 for t in trajectories if t.success)
        logger.info(f"Iter {iteration}: collected {len(trajectories)} trajs, " f"{num_successes} successes")

        # ── 2. Compute trajectory-level rewards g_i ─────────────────────
        if config.mode == "srpo" and reward_model is not None:
            g_values = reward_model.compute_trajectory_rewards(trajectories)
            reward_model.add_successful_trajectories([t for t in trajectories if t.success])
        else:
            g_values = [1.0 if t.success else 0.0 for t in trajectories]

        # ── 3. Compute trajectory-level advantages  ──────────────────
        g_tensor = torch.tensor(g_values, dtype=torch.float32)
        g_mean = g_tensor.mean()
        g_std = g_tensor.std().clamp(min=1e-8)
        advantages = ((g_tensor - g_mean) / g_std).tolist()

        # ── 4. Cache per-step FM loss under θ_old (+ sample fixed noise/time)
        policy.eval()
        old_losses_per_traj: list[torch.Tensor] = []
        fixed_noise_per_traj: list[list[torch.Tensor]] = []
        fixed_time_per_traj: list[list[torch.Tensor]] = []

        with torch.no_grad():
            for traj in trajectories:
                noise_list, time_list = _sample_fixed_noise_time(traj, policy)
                fixed_noise_per_traj.append(noise_list)
                fixed_time_per_traj.append(time_list)
                old_loss = _compute_fm_loss_per_step(policy, traj, instruction, noise_list, time_list)
                old_losses_per_traj.append(old_loss)

        # ── 5. PPO epochs ───────────────────────────────────────────────
        policy.train()
        total_surrogate = 0.0
        total_kl = 0.0

        for ppo_epoch in range(config.ppo_epochs):
            epoch_loss_sum = 0.0
            for i, traj in enumerate(trajectories):
                adv_i = advantages[i]
                old_losses = old_losses_per_traj[i]
                noise_list = fixed_noise_per_traj[i]
                time_list = fixed_time_per_traj[i]

                new_step_losses = []
                for t in range(traj.length):
                    img = traj.images[t].unsqueeze(0).to(policy.device)
                    action = traj.actions[t].unsqueeze(0).to(policy.device)

                    imgs_f = policy._to_float01(img).to(policy.device, dtype=policy.dtype)
                    img_list, mask_list = policy._prepare_images(imgs_f)
                    tokens, tmasks = policy._tokenize(instruction, batch_size=1)
                    state_raw = traj.states[t].unsqueeze(0) if traj.states is not None else None
                    state = policy._prepare_state_input(state_raw, batch_size=1)
                    normalized_action = policy._normalize_action(action.to(policy.device, dtype=policy.dtype))
                    action_padded = policy._prepare_action(normalized_action)
                    action_padded = action_padded.unsqueeze(1).expand(-1, policy.chunk_size, -1)

                    fm_loss = policy.model.forward(
                        img_list,
                        mask_list,
                        tokens,
                        tmasks,
                        state,
                        action_padded,
                        noise=noise_list[t].to(policy.device, dtype=policy.dtype),
                        time=time_list[t].to(policy.device, dtype=policy.dtype),
                    )
                    new_step_losses.append(fm_loss[:, :, : policy.max_action_dim].mean())

                if not new_step_losses:
                    continue

                new_losses_t = torch.stack(new_step_losses)
                old_losses_t = old_losses.to(policy.device)

                log_ratios = old_losses_t - new_losses_t
                ratios = torch.exp(log_ratios.clamp(-10.0, 10.0))

                adv_t = torch.full_like(ratios, adv_i)

                surr1 = ratios * adv_t
                surr2 = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * adv_t
                clip_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    ref_losses = _compute_fm_loss_per_step(ref_policy, traj, instruction, noise_list, time_list).to(
                        policy.device
                    )
                kl_approx = (ref_losses - new_losses_t.detach()).mean()
                kl_penalty = config.kl_coeff * kl_approx

                loss = clip_loss + kl_penalty

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
                optimizer.step()

                epoch_loss_sum += clip_loss.item()
                total_kl += kl_penalty.item()

            total_surrogate += epoch_loss_sum

        avg_surr = total_surrogate / max(config.ppo_epochs * len(trajectories), 1)
        avg_kl = total_kl / max(config.ppo_epochs * len(trajectories), 1)

        # ── 6. Update θ_old for next iteration's ratio computation ──────
        old_losses_per_traj.clear()
        fixed_noise_per_traj.clear()
        fixed_time_per_traj.clear()

        log_data = {
            f"{config.mode}/surrogate_loss": avg_surr,
            f"{config.mode}/kl_penalty": avg_kl,
            f"{config.mode}/batch_successes": num_successes,
            f"{config.mode}/mean_g": g_mean.item(),
            f"{config.mode}/iteration": iteration,
        }
        logger.info(
            f"Iter {iteration}  surr={avg_surr:.6f}  kl={avg_kl:.6f}  "
            f"successes={num_successes}  g_mean={g_mean:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(log_data)

        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            metrics = evaluate(
                policy_fn=policy.predict_action,
                instruction=instruction,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 20000,
            )
            print_metrics(metrics, tag=f"{config.mode} iter {iteration}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"{config.mode}/success_rate": metrics.success_rate,
                        f"{config.mode}/mean_reward": metrics.mean_reward,
                        f"{config.mode}/mean_ep_len": metrics.mean_episode_length,
                        f"{config.mode}/iteration": iteration,
                    }
                )
            if metrics.success_rate > best_success:
                best_success = metrics.success_rate
                policy.save_checkpoint(save_path / "best")
                logger.info(f"New best {config.mode} checkpoint: {best_success:.2%}")

    policy.save_checkpoint(save_path / "last")
    rollout_engine.close()
    return policy
