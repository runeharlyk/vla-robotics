"""AWR policy update."""

from __future__ import annotations

import math

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched


def awr_update(
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
    """Run advantage-weighted regression (AWR) policy update."""
    B = config.fm_batch_size
    M = len(trajectories)
    if M == 0:
        return UpdateMetrics()

    ref_losses_per_traj: list[torch.Tensor] = []
    if config.kl_coeff > 0:
        if ref_policy is None:
            raise ValueError("AWR KL requested, but ref_policy was not provided.")
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                ref_loss = _compute_fm_loss_batched(
                    ref_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction="mean",
                )
                ref_losses_per_traj.append(ref_loss.detach())

    sft_kl_coeff = float(getattr(config, "sft_kl_coeff", 0.0))
    sft_losses_per_traj: list[torch.Tensor] = []
    if sft_kl_coeff > 0:
        if sft_policy is None:
            raise ValueError("AWR SFT-anchor KL requested, but sft_policy was not provided.")
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                sft_loss = _compute_fm_loss_batched(
                    sft_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction="mean",
                )
                sft_losses_per_traj.append(sft_loss.detach())

    policy.train()
    total_weighted_loss = 0.0
    total_kl = 0.0
    total_sft_kl = 0.0
    total_weight = 0.0
    used_count_total = 0

    for _ in range(config.awr_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_sft_kl = 0.0
        epoch_weight = 0.0
        used_count = 0

        for i in range(M):
            traj = trajectories[i]
            adv_i = advantages[i]
            weight = min(math.exp(adv_i / config.awr_temperature), config.awr_weight_clip)

            fm_loss_t = _compute_fm_loss_batched(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
                reduction="mean",
            )

            fm_loss = fm_loss_t.mean()
            traj_loss = weight * fm_loss / M

            if config.kl_coeff > 0 and ref_losses_per_traj:
                ref_fm_t = ref_losses_per_traj[i]
                kl_per_step = 0.5 * (ref_fm_t - fm_loss_t) ** 2
                kl_approx = kl_per_step.mean()
                traj_loss = traj_loss + config.kl_coeff * kl_approx / M
                epoch_kl += (config.kl_coeff * kl_approx).item()

            if sft_losses_per_traj:
                sft_fm_t = sft_losses_per_traj[i]
                sft_kl_per_step = 0.5 * (sft_fm_t - fm_loss_t) ** 2
                sft_kl_approx = sft_kl_per_step.mean()
                traj_loss = traj_loss + sft_kl_coeff * sft_kl_approx / M
                epoch_sft_kl += (sft_kl_coeff * sft_kl_approx).item()

            traj_loss.backward()
            epoch_loss += (weight * fm_loss).item()
            epoch_weight += weight
            used_count += 1

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_weighted_loss += epoch_loss
        total_kl += epoch_kl
        total_sft_kl += epoch_sft_kl
        total_weight += epoch_weight
        used_count_total += used_count

    denom = max(used_count_total, 1)
    return UpdateMetrics(
        avg_loss=total_weighted_loss / denom,
        avg_kl=total_kl / denom,
        avg_sft_kl=total_sft_kl / denom,
        avg_weight=total_weight / denom,
    )


__all__ = ["awr_update"]
