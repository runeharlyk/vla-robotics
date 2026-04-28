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
    full_chunk_target: bool = True,
) -> torch.Tensor:
    """Compute per-timestep FM loss in mini-batches."""
    T = traj.length
    actions_for_loss, chunk_mask = _actions_and_mask_for_loss(
        traj,
        chunk_size=policy.chunk_size,
        full_chunk_target=full_chunk_target,
    )
    return policy.compute_fm_loss_batched(
        images=traj.images[:T],
        actions=actions_for_loss,
        states=traj.states[:T] if traj.states is not None else None,
        instruction=instruction,
        fixed_noise=fixed_noise[:T],
        fixed_time=fixed_time[:T],
        batch_size=batch_size,
        reduction=reduction,
        chunk_mask=chunk_mask,
    )


def _compute_fm_loss_multi_sample(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    noise_list: list[torch.Tensor],
    time_list: list[torch.Tensor],
    batch_size: int,
    reduction: str = "mean",
    full_chunk_target: bool = True,
) -> torch.Tensor:
    """Compute per-timestep FM loss averaged over N noise/time samples."""
    T = traj.length
    actions_for_loss, chunk_mask = _actions_and_mask_for_loss(
        traj,
        chunk_size=policy.chunk_size,
        full_chunk_target=full_chunk_target,
    )
    return policy.compute_fm_loss_multi_sample(
        images=traj.images[:T],
        actions=actions_for_loss,
        states=traj.states[:T] if traj.states is not None else None,
        instruction=instruction,
        noise_list=[n[:T] for n in noise_list],
        time_list=[t[:T] for t in time_list],
        batch_size=batch_size,
        reduction=reduction,
        chunk_mask=chunk_mask,
    )


def _actions_and_mask_for_loss(
    traj: Trajectory,
    *,
    chunk_size: int,
    full_chunk_target: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return action targets for FM loss.

    ``full_chunk_target=True`` preserves the v28/SFT target semantics:
    each observation is trained against a dense sliding window of actions
    that were actually executed in the environment.  For single-step
    rollouts this is just ``traj.actions`` and the policy builds the
    sliding chunks internally.  For chunked rollouts we reconstruct the
    same target shape from ``executed_chunks``.

    ``full_chunk_target=False`` uses the direct chunk-execution targets:
    only actions inside the sampled chunk that were stepped into the env
    receive loss weight.
    """
    T = traj.length
    executed_chunks = getattr(traj, "executed_chunks", None)
    chunk_mask = getattr(traj, "chunk_mask", None)

    if executed_chunks is None or chunk_mask is None:
        return traj.actions[:T], None

    executed_chunks = executed_chunks[:T]
    chunk_mask = chunk_mask[:T]

    if not full_chunk_target:
        return executed_chunks, chunk_mask

    flat_actions = executed_chunks[chunk_mask]
    action_dim = executed_chunks.shape[-1]
    chunks = executed_chunks.new_zeros((T, chunk_size, action_dim))
    mask = torch.zeros((T, chunk_size), dtype=torch.bool, device=executed_chunks.device)

    counts = chunk_mask.sum(dim=1).to(torch.long)
    starts = torch.cumsum(counts, dim=0) - counts
    total = flat_actions.shape[0]

    for t in range(T):
        start = int(starts[t].item())
        end = min(total, start + chunk_size)
        n = end - start
        if n <= 0:
            continue
        chunks[t, :n] = flat_actions[start:end]
        mask[t, :n] = True

    return chunks, mask


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
    "_actions_and_mask_for_loss",
    "_compute_fm_loss_batched",
    "_compute_fm_loss_multi_sample",
    "_sample_fixed_noise_time",
]
