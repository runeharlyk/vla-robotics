"""Success-filtered behavior cloning policy update."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_batched


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

    for _ in range(max(getattr(config, "success_bc_epochs", 1), 1)):
        order = torch.randperm(M).tolist()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_sft_kl = 0.0
        epoch_raw_sft_kl = 0.0
        used = 0

        for i in order:
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
            loss_i = fm_loss / M

            if sft_losses_per_traj:
                sft_fm_t = sft_losses_per_traj[i].to(device=fm_loss_t.device, dtype=torch.float32)
                fm_loss_f = fm_loss_t.float()
                sft_kl_per_step = 0.5 * (sft_fm_t - fm_loss_f) ** 2
                sft_kl = sft_kl_per_step.mean()
                loss_i = loss_i + (sft_kl_coeff * sft_kl) / M
                epoch_sft_kl += (sft_kl_coeff * sft_kl).item()
                epoch_raw_sft_kl += sft_kl.item()

            loss_i.backward()
            epoch_loss += fm_loss.item()
            used += 1

        if used == 0:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_loss += epoch_loss
        total_sft_kl += epoch_sft_kl
        total_raw_sft_kl += epoch_raw_sft_kl
        used_total += used

    denom = max(used_total, 1)
    return UpdateMetrics(
        avg_loss=total_loss / denom,
        avg_sft_kl=total_sft_kl / denom,
        raw_sft_kl=total_raw_sft_kl / denom,
        avg_weight=1.0,
    )


__all__ = ["success_bc_update"]
