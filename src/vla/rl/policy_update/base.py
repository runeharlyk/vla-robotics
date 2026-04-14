"""Shared helpers for SRPO policy update algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import Trajectory


def _sample_fixed_noise_time(
    traj: Trajectory,
    policy: SmolVLAPolicy,
    n_samples: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pre-sample noise and time tensors for an entire trajectory."""
    T = traj.length
    noise_list: list[torch.Tensor] = []
    time_list: list[torch.Tensor] = []
    beta = torch.distributions.Beta(1.5, 1.0)
    for _ in range(n_samples):
        noise_list.append(torch.randn(T, policy.chunk_size, policy.max_action_dim))
        time_list.append(beta.sample((T,)) * 0.999 + 0.001)
    return noise_list, time_list


def _compute_fm_loss_batched(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    fixed_noise: torch.Tensor,
    fixed_time: torch.Tensor,
    batch_size: int = 32,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute per-timestep FM loss in mini-batches."""
    T = traj.length
    return policy.compute_fm_loss_batched(
        images=traj.images[:T],
        actions=traj.actions[:T],
        states=traj.states[:T] if traj.states is not None else None,
        instruction=instruction,
        fixed_noise=fixed_noise[:T],
        fixed_time=fixed_time[:T],
        batch_size=batch_size,
        reduction=reduction,
    )


def _compute_fm_loss_multi_sample(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    noise_list: list[torch.Tensor],
    time_list: list[torch.Tensor],
    batch_size: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute per-timestep FM loss averaged over N noise/time samples."""
    T = traj.length
    return policy.compute_fm_loss_multi_sample(
        images=traj.images[:T],
        actions=traj.actions[:T],
        states=traj.states[:T] if traj.states is not None else None,
        instruction=instruction,
        noise_list=[n[:T] for n in noise_list],
        time_list=[t[:T] for t in time_list],
        batch_size=batch_size,
        reduction=reduction,
    )


@dataclass
class UpdateMetrics:
    """Metrics returned by a single policy update step."""

    avg_loss: float = 0.0
    avg_kl: float = 0.0
    avg_sft_kl: float = 0.0
    avg_shift: float = 0.0
    avg_weight: float = 0.0
    raw_kl: float = 0.0
    raw_sft_kl: float = 0.0
    mean_ratio: float = 1.0
    max_log_ratio: float = 0.0


__all__ = [
    "UpdateMetrics",
    "_compute_fm_loss_batched",
    "_compute_fm_loss_multi_sample",
    "_sample_fixed_noise_time",
]
