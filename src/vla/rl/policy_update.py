"""AWR and PPO policy update algorithms for SRPO.

Extracted from the training loop to allow independent testing and
keep :mod:`vla.rl.trainer` focused on orchestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

# ---------------------------------------------------------------------------
# Batched FM-loss helpers
# ---------------------------------------------------------------------------


def _sample_fixed_noise_time(
    traj: Trajectory,
    policy: SmolVLAPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-sample noise and time tensors for an entire trajectory.

    Returns:
        noise: ``(T, chunk_size, max_action_dim)``
        time:  ``(T,)``
    """
    T = traj.length
    noise = torch.randn(T, policy.chunk_size, policy.max_action_dim)
    beta = torch.distributions.Beta(1.5, 1.0)
    time = beta.sample((T,)) * 0.999 + 0.001
    return noise, time


def _compute_fm_loss_batched(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    fixed_noise: torch.Tensor,
    fixed_time: torch.Tensor,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute per-timestep FM loss in mini-batches.

    Delegates to :meth:`SmolVLAPolicy.compute_fm_loss_batched` so that
    all FM-loss computation lives inside the policy and can be tested
    independently of the training loop.
    """
    T = traj.length
    return policy.compute_fm_loss_batched(
        images=traj.images[:T],
        actions=traj.actions[:T],
        states=traj.states[:T] if traj.states is not None else None,
        instruction=instruction,
        fixed_noise=fixed_noise[:T],
        fixed_time=fixed_time[:T],
        batch_size=batch_size,
    )


def _merge_trajectory_data(
    trajectories: list[Trajectory],
    noise_list: list[torch.Tensor],
    time_list: list[torch.Tensor],
) -> tuple[Trajectory, torch.Tensor, torch.Tensor, list[int]]:
    """Concatenate trajectory data for mega-batched FM-loss computation.

    Returns a virtual merged ``Trajectory``, merged noise/time tensors,
    and the per-trajectory lengths needed to split results back.
    """
    lengths = [t.length for t in trajectories]
    has_states = trajectories[0].states is not None
    merged = Trajectory(
        images=torch.cat([t.images[: t.length] for t in trajectories]),
        actions=torch.cat([t.actions[: t.length] for t in trajectories]),
        states=(
            torch.cat([t.states[: t.length] for t in trajectories])
            if has_states
            else trajectories[0].states
        ),
        rewards=torch.zeros(1),
        dones=torch.zeros(1),
        success=False,
        length=sum(lengths),
    )
    return merged, torch.cat(noise_list), torch.cat(time_list), lengths


# ---------------------------------------------------------------------------
# Policy update algorithms
# ---------------------------------------------------------------------------


@dataclass
class UpdateMetrics:
    """Metrics returned by a single policy update step (AWR or PPO)."""

    avg_loss: float = 0.0
    avg_kl: float = 0.0
    avg_weight: float = 0.0


def awr_update(
    policy: SmolVLAPolicy,
    ref_policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    skipped_task_ids: set[str],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run advantage-weighted regression (AWR) policy update.

    Returns:
        :class:`UpdateMetrics` with average loss, KL, and weight.
    """
    B = config.fm_batch_size
    M = len(trajectories)

    ref_losses_per_traj: list[torch.Tensor] = []
    if config.kl_coeff > 0:
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                ref_loss = _compute_fm_loss_batched(
                    ref_policy, traj, instrs_per_traj[i],
                    fixed_noise[i], fixed_time[i],
                    batch_size=B,
                )
                ref_losses_per_traj.append(ref_loss.detach())

    policy.train()
    total_weighted_loss = 0.0
    total_kl = 0.0
    total_weight = 0.0

    for _epoch in range(config.awr_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_weight = 0.0

        for i, traj in enumerate(trajectories):
            if traj.task_id in skipped_task_ids:
                continue
            adv_i = advantages[i]
            weight = min(math.exp(adv_i / config.awr_temperature), config.awr_weight_clip)

            fm_loss = _compute_fm_loss_batched(
                policy, traj, instrs_per_traj[i],
                fixed_noise[i], fixed_time[i],
                batch_size=B,
            ).mean()

            traj_loss = weight * fm_loss / M

            if config.kl_coeff > 0 and ref_losses_per_traj:
                ref_fm = ref_losses_per_traj[i].mean()
                kl_approx = 0.5 * (ref_fm - fm_loss) ** 2
                traj_loss = traj_loss + config.kl_coeff * kl_approx / M
                epoch_kl += (config.kl_coeff * kl_approx).item()

            traj_loss.backward()
            epoch_loss += (weight * fm_loss).item()
            epoch_weight += weight

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_weighted_loss += epoch_loss
        total_kl += epoch_kl
        total_weight += epoch_weight

    denom = max(config.awr_epochs * M, 1)
    return UpdateMetrics(
        avg_loss=total_weighted_loss / denom,
        avg_kl=total_kl / denom,
        avg_weight=total_weight / denom,
    )


def ppo_update(
    policy: SmolVLAPolicy,
    ref_policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run PPO clipped-surrogate policy update with KL regularisation.

    Returns:
        :class:`UpdateMetrics` with average surrogate loss and KL penalty.
    """
    B = config.fm_batch_size
    M = len(trajectories)

    old_losses_per_traj: list[torch.Tensor] = []
    ref_losses_per_traj: list[torch.Tensor] = []

    with torch.no_grad():
        for i, traj in enumerate(trajectories):
            old_loss = _compute_fm_loss_batched(
                policy, traj, instrs_per_traj[i],
                fixed_noise[i], fixed_time[i],
                batch_size=B,
            )
            old_losses_per_traj.append(old_loss.detach())

            ref_loss = _compute_fm_loss_batched(
                ref_policy, traj, instrs_per_traj[i],
                fixed_noise[i], fixed_time[i],
                batch_size=B,
            )
            ref_losses_per_traj.append(ref_loss.detach())

    policy.train()
    total_surrogate = 0.0
    total_kl = 0.0

    for _ppo_epoch in range(config.ppo_epochs):
        optimizer.zero_grad()
        epoch_clip_loss = 0.0
        epoch_kl = 0.0

        for i, traj in enumerate(trajectories):
            adv_i = advantages[i]
            old_losses_t = old_losses_per_traj[i]
            ref_losses_t = ref_losses_per_traj[i]

            new_losses_t = _compute_fm_loss_batched(
                policy, traj, instrs_per_traj[i],
                fixed_noise[i], fixed_time[i],
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
            kl_approx = 0.5 * (log_ratio_ref ** 2).mean()
            kl_penalty = config.kl_coeff * kl_approx

            traj_loss = (clip_loss + kl_penalty) / M
            traj_loss.backward()

            epoch_clip_loss += clip_loss.item()
            epoch_kl += kl_penalty.item()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_surrogate += epoch_clip_loss
        total_kl += epoch_kl

    denom = max(config.ppo_epochs * M, 1)
    return UpdateMetrics(
        avg_loss=total_surrogate / denom,
        avg_kl=total_kl / denom,
    )
