"""PPO policy update."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched


def ppo_update(
    policy: SmolVLAPolicy,
    ref_policy: SmolVLAPolicy,
    sft_policy: SmolVLAPolicy | None,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run PPO clipped-surrogate policy update with KL regularisation."""
    B = config.fm_batch_size
    M = len(trajectories)
    if M == 0:
        return UpdateMetrics()

    old_losses_per_traj: list[torch.Tensor] = []
    ref_losses_per_traj: list[torch.Tensor] = []
    sft_losses_per_traj: list[torch.Tensor] = []
    sft_kl_coeff = float(getattr(config, "sft_kl_coeff", 0.0))

    with torch.no_grad():
        for i in range(M):
            traj = trajectories[i]
            old_loss = _compute_fm_loss_batched(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
            )
            old_losses_per_traj.append(old_loss.detach())

            ref_loss = _compute_fm_loss_batched(
                ref_policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
            )
            ref_losses_per_traj.append(ref_loss.detach())

            if sft_kl_coeff > 0:
                if sft_policy is None:
                    raise ValueError("PPO SFT-anchor KL requested, but sft_policy was not provided.")
                sft_loss = _compute_fm_loss_batched(
                    sft_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                )
                sft_losses_per_traj.append(sft_loss.detach())

    policy.train()
    total_surrogate = 0.0
    total_kl = 0.0
    total_sft_kl = 0.0
    used_count_total = 0

    for _ppo_epoch in range(config.ppo_epochs):
        optimizer.zero_grad()
        epoch_clip_loss = 0.0
        epoch_kl = 0.0
        epoch_sft_kl = 0.0
        used_count = 0

        for i in range(M):
            traj = trajectories[i]
            adv_i = advantages[i]
            old_losses_t = old_losses_per_traj[i]
            ref_losses_t = ref_losses_per_traj[i]

            new_losses_t = _compute_fm_loss_batched(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
            )

            log_ratios = old_losses_t - new_losses_t
            ratios = torch.exp(log_ratios.clamp(-10.0, 10.0))

            adv_t = torch.full_like(ratios, adv_i)
            surr1 = ratios * adv_t
            surr2 = (
                torch.clamp(
                    ratios,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon_high,
                )
                * adv_t
            )
            clip_loss = -torch.min(surr1, surr2).mean()

            log_ratio_ref = ref_losses_t - new_losses_t
            kl_approx = 0.5 * (log_ratio_ref**2).mean()
            kl_penalty = config.kl_coeff * kl_approx

            traj_loss = (clip_loss + kl_penalty) / M
            if sft_losses_per_traj:
                sft_losses_t = sft_losses_per_traj[i]
                log_ratio_sft = sft_losses_t - new_losses_t
                sft_kl_approx = 0.5 * (log_ratio_sft**2).mean()
                sft_kl_penalty = sft_kl_coeff * sft_kl_approx
                traj_loss = traj_loss + sft_kl_penalty / M
                epoch_sft_kl += sft_kl_penalty.item()
            traj_loss.backward()

            epoch_clip_loss += clip_loss.item()
            epoch_kl += kl_penalty.item()
            used_count += 1

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_surrogate += epoch_clip_loss
        total_kl += epoch_kl
        total_sft_kl += epoch_sft_kl
        used_count_total += used_count

    denom = max(used_count_total, 1)
    return UpdateMetrics(
        avg_loss=total_surrogate / denom,
        avg_kl=total_kl / denom,
        avg_sft_kl=total_sft_kl / denom,
    )


__all__ = ["ppo_update"]
