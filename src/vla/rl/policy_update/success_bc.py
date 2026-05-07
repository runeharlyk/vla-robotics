"""Success-filtered behavior cloning policy update."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched, _resolve_minibatch_trajs


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

    for _ in range(max(getattr(config, "success_bc_epochs", 1), 1)):
        order = torch.randperm(M).tolist()

        for start in range(0, M, minibatch_trajs):
            idxs = order[start : start + minibatch_trajs]
            n_idxs = len(idxs)

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
    return UpdateMetrics(
        avg_loss=total_loss / denom,
        avg_sft_kl=total_sft_kl / denom,
        raw_sft_kl=total_raw_sft_kl / denom,
        avg_weight=1.0,
    )


__all__ = ["success_bc_update"]
