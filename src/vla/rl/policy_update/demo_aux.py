"""Supervised demonstration auxiliary update for online RL methods."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched, _resolve_minibatch_trajs


def demo_aux_update(
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run a demo-only FM/BC anchor as a separate optimizer step.

    This is intentionally not a PPO/GRPO update: demonstrations do not carry
    paths sampled by the current old policy, so they cannot provide valid
    action log-probability ratios.  The auxiliary loss only pulls the policy
    toward demonstration actions with a configurable supervised coefficient.
    """
    coeff = float(getattr(config, "demo_aux_coeff", 0.0))
    M = len(trajectories)
    if M == 0 or coeff <= 0.0:
        return UpdateMetrics()

    B = config.fm_batch_size
    reduction = getattr(config, "demo_aux_loss_reduction", "mean")
    full_chunk_target = bool(getattr(config, "fpo_full_chunk_target", True))
    minibatch_trajs = _resolve_minibatch_trajs(config.demo_aux_minibatch_trajs, M)

    policy.train()
    total_raw_loss = 0.0
    total_weighted_loss = 0.0
    total_used = 0
    num_updates = 0

    for _ in range(max(getattr(config, "demo_aux_epochs", 1), 1)):
        order = torch.randperm(M).tolist()
        minibatches = [order[start : start + minibatch_trajs] for start in range(0, M, minibatch_trajs)]

        for idxs in minibatches:
            n_idxs = len(idxs)
            if n_idxs == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            mb_raw_loss = 0.0
            used = 0

            for i in idxs:
                fm_loss_t = _compute_fm_loss_batched(
                    policy,
                    trajectories[i],
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction=reduction,
                    full_chunk_target=full_chunk_target,
                )
                fm_loss = fm_loss_t.mean()
                (coeff * fm_loss / n_idxs).backward()
                mb_raw_loss += fm_loss.item() / n_idxs
                used += 1

            if used == 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
            optimizer.step()

            total_raw_loss += mb_raw_loss
            total_weighted_loss += coeff * mb_raw_loss
            total_used += used
            num_updates += 1

    denom = max(num_updates, 1)
    return UpdateMetrics(
        avg_loss=total_weighted_loss / denom,
        avg_weight=coeff,
        raw_sft_kl=total_raw_loss / denom,
        demo_fraction=1.0 if total_used else 0.0,
    )


__all__ = ["demo_aux_update"]
