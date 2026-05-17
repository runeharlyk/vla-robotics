"""Success-filtered behavior cloning policy update."""

from __future__ import annotations

import math

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched, _resolve_minibatch_trajs


def _build_balanced_minibatches(
    trajectories: list[Trajectory],
    minibatch_trajs: int,
    demo_ratio: float,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Yield index minibatches with `demo_ratio` demos per batch (best-effort).

    Implements RLPD-style symmetric sampling (Ball et al., 2023): every
    minibatch is composed of ``ceil(K * demo_ratio)`` indices drawn from
    the demo pool and ``K - ceil(K * demo_ratio)`` from the online pool.
    Indices are sampled WITHOUT replacement within an epoch from each
    pool independently; when one pool is depleted before the other, the
    remaining minibatches are filled from the still-populated pool to
    avoid stalling the optimizer mid-epoch.
    """
    demo_idxs = [i for i, t in enumerate(trajectories) if t.is_demo]
    online_idxs = [i for i, t in enumerate(trajectories) if not t.is_demo]
    if not demo_idxs or not online_idxs:
        order = torch.randperm(len(trajectories), generator=generator).tolist()
        return [order[i : i + minibatch_trajs] for i in range(0, len(trajectories), minibatch_trajs)]

    demo_perm = torch.randperm(len(demo_idxs), generator=generator).tolist()
    online_perm = torch.randperm(len(online_idxs), generator=generator).tolist()

    demos_per_mb = max(1, math.ceil(minibatch_trajs * demo_ratio))
    online_per_mb = max(1, minibatch_trajs - demos_per_mb)

    minibatches: list[list[int]] = []
    di = oi = 0
    while di < len(demo_perm) or oi < len(online_perm):
        mb: list[int] = []
        for _ in range(demos_per_mb):
            if di < len(demo_perm):
                mb.append(demo_idxs[demo_perm[di]])
                di += 1
            elif oi < len(online_perm):
                mb.append(online_idxs[online_perm[oi]])
                oi += 1
        for _ in range(online_per_mb):
            if oi < len(online_perm):
                mb.append(online_idxs[online_perm[oi]])
                oi += 1
            elif di < len(demo_perm):
                mb.append(demo_idxs[demo_perm[di]])
                di += 1
        if mb:
            minibatches.append(mb)
        else:
            break
    return minibatches


def success_bc_update(
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    config: SRPOConfig,
    sft_policy: SmolVLAPolicy | None = None,
) -> UpdateMetrics:
    """Run standard BC/SFT on successful rollout trajectories only.

    The trainer is responsible for filtering ``trajectories`` to successes.
    This update deliberately has no advantage temperature/beta: once a
    trajectory is accepted into the success set, it receives uniform BC
    weight.  Optional SFT anchoring can still be enabled via
    ``config.sft_kl_coeff``.
    """
    B = config.fm_batch_size
    M = len(trajectories)
    if M == 0:
        return UpdateMetrics()

    reduction = getattr(config, "success_bc_loss_reduction", "mean")
    full_chunk_target = bool(getattr(config, "fpo_full_chunk_target", True))
    sft_kl_coeff = float(getattr(config, "sft_kl_coeff", 0.0))

    sft_losses_per_traj: list[torch.Tensor] = []
    if sft_kl_coeff > 0:
        if sft_policy is None:
            raise ValueError("Success-BC SFT-anchor KL requested, but sft_policy was not provided.")
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                sft_loss = _compute_fm_loss_batched(
                    sft_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction=reduction,
                    full_chunk_target=full_chunk_target,
                )
                sft_losses_per_traj.append(sft_loss.detach())

    policy.train()
    total_loss = 0.0
    total_sft_kl = 0.0
    total_raw_sft_kl = 0.0
    used_total = 0
    num_updates = 0

    minibatch_trajs = _resolve_minibatch_trajs(config.success_bc_minibatch_trajs, M)
    balanced = bool(getattr(config, "success_bc_balanced_demo_sampling", False))
    demo_ratio = float(getattr(config, "success_bc_demo_sampling_ratio", 0.5))

    n_demos = sum(1 for t in trajectories if t.is_demo)
    n_online = M - n_demos
    total_demo_idxs_used = 0
    total_online_idxs_used = 0

    for _ in range(max(getattr(config, "success_bc_epochs", 1), 1)):
        if balanced and n_demos > 0 and n_online > 0:
            minibatches = _build_balanced_minibatches(trajectories, minibatch_trajs, demo_ratio)
        else:
            order = torch.randperm(M).tolist()
            minibatches = [order[start : start + minibatch_trajs] for start in range(0, M, minibatch_trajs)]

        for idxs in minibatches:
            n_idxs = len(idxs)
            total_demo_idxs_used += sum(1 for i in idxs if trajectories[i].is_demo)
            total_online_idxs_used += sum(1 for i in idxs if not trajectories[i].is_demo)

            optimizer.zero_grad(set_to_none=True)

            mb_loss = 0.0
            mb_sft_kl = 0.0
            mb_raw_sft_kl = 0.0
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
                loss_i = fm_loss / n_idxs

                if sft_losses_per_traj:
                    sft_fm_t = sft_losses_per_traj[i].to(device=fm_loss_t.device, dtype=torch.float32)
                    fm_loss_f = fm_loss_t.float()
                    sft_kl_per_step = 0.5 * (sft_fm_t - fm_loss_f) ** 2
                    sft_kl = sft_kl_per_step.mean()
                    loss_i = loss_i + (sft_kl_coeff * sft_kl) / n_idxs
                    mb_sft_kl += (sft_kl_coeff * sft_kl).item() / n_idxs
                    mb_raw_sft_kl += sft_kl.item() / n_idxs

                loss_i.backward()
                mb_loss += fm_loss.item() / n_idxs
                used += 1

            if used == 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
            optimizer.step()

            total_loss += mb_loss
            total_sft_kl += mb_sft_kl
            total_raw_sft_kl += mb_raw_sft_kl
            used_total += used
            num_updates += 1

    denom = max(num_updates, 1)
    consumed = total_demo_idxs_used + total_online_idxs_used
    demo_fraction = total_demo_idxs_used / consumed if consumed else 0.0
    online_fraction = total_online_idxs_used / consumed if consumed else 0.0
    return UpdateMetrics(
        avg_loss=total_loss / denom,
        avg_sft_kl=total_sft_kl / denom,
        raw_sft_kl=total_raw_sft_kl / denom,
        avg_weight=1.0,
        demo_fraction=demo_fraction,
        online_fraction=online_fraction,
    )


__all__ = ["success_bc_update"]
