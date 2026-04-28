"""FPO policy update."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics, _compute_fm_loss_multi_sample


def fpo_update(
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    fixed_noise: list[list[torch.Tensor]],
    fixed_time: list[list[torch.Tensor]],
    config: SRPOConfig,
    ref_policy: SmolVLAPolicy | None = None,
    sft_policy: SmolVLAPolicy | None = None,
) -> UpdateMetrics:
    """FPO (Flow Policy Optimization) update with PPO-clip objective."""
    B = config.fm_batch_size
    M = len(trajectories)

    if M == 0:
        return UpdateMetrics()

    device = next(policy.parameters()).device
    reduction = getattr(config, "fpo_loss_reduction", "sum")
    positive_only = bool(getattr(config, "fpo_positive_adv_only", False))
    negative_scale = float(getattr(config, "fpo_negative_adv_scale", 0.25))
    log_ratio_clip = float(getattr(config, "fpo_log_ratio_clip", 5.0))
    full_chunk_target = bool(getattr(config, "fpo_full_chunk_target", True))

    with torch.no_grad():
        old_fm_per_traj = []
        for i, traj in enumerate(trajectories):
            old_fm = _compute_fm_loss_multi_sample(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
                reduction=reduction,
                full_chunk_target=full_chunk_target,
            )
            old_fm_per_traj.append(old_fm.detach())

    use_kl = config.kl_coeff > 0
    use_ref_policy_kl = bool(getattr(config, "fpo_use_ref_policy_kl", False))

    ref_fm_per_traj: list[torch.Tensor] = []
    if use_kl and use_ref_policy_kl:
        if ref_policy is None:
            raise ValueError("FPO ref-policy KL requested, but ref_policy was not provided.")
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                ref_fm = _compute_fm_loss_multi_sample(
                    ref_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction=reduction,
                    full_chunk_target=full_chunk_target,
                )
                ref_fm_per_traj.append(ref_fm.detach())

    sft_kl_coeff = float(getattr(config, "sft_kl_coeff", 0.0))
    sft_fm_per_traj: list[torch.Tensor] = []
    if sft_kl_coeff > 0:
        if sft_policy is None:
            raise ValueError("FPO SFT-anchor KL requested, but sft_policy was not provided.")
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                sft_fm = _compute_fm_loss_multi_sample(
                    sft_policy,
                    traj,
                    instrs_per_traj[i],
                    fixed_noise[i],
                    fixed_time[i],
                    batch_size=B,
                    reduction=reduction,
                    full_chunk_target=full_chunk_target,
                )
                sft_fm_per_traj.append(sft_fm.detach())

    policy.train()

    minibatch_trajs = min(getattr(config, "ppo_minibatch_trajs", 4), M)
    total_loss = 0.0
    total_local_kl = 0.0
    total_sft_kl = 0.0
    total_shift = 0.0
    total_clip_frac = 0.0
    total_raw_kl = 0.0
    total_raw_sft_kl = 0.0
    total_mean_ratio = 0.0
    total_max_log_ratio = 0.0
    num_updates = 0

    for _ in range(max(config.ppo_epochs, 1)):
        order = torch.randperm(M).tolist()

        for start in range(0, M, minibatch_trajs):
            idxs = order[start : start + minibatch_trajs]
            n_idxs = len(idxs)

            optimizer.zero_grad(set_to_none=True)

            mb_loss = 0.0
            mb_local_kl = 0.0
            mb_sft_kl = 0.0
            mb_shift = 0.0
            mb_clip_frac = 0.0
            mb_raw_kl = 0.0
            mb_raw_sft_kl = 0.0
            mb_max_log_ratio = 0.0
            mb_sum_ratio = 0.0
            used = 0

            for i in idxs:
                adv_i = float(advantages[i])

                if adv_i <= 0.0 and positive_only:
                    continue

                if adv_i < 0.0:
                    adv_i *= negative_scale
                    if adv_i == 0.0:
                        continue

                new_fm = _compute_fm_loss_multi_sample(
                    policy,
                    traj=trajectories[i],
                    instruction=instrs_per_traj[i],
                    noise_list=fixed_noise[i],
                    time_list=fixed_time[i],
                    batch_size=B,
                    reduction=reduction,
                    full_chunk_target=full_chunk_target,
                )

                old_fm = old_fm_per_traj[i].to(device=device, dtype=torch.float32)
                new_fm_f = new_fm.float()

                log_ratio = (old_fm - new_fm_f).clamp(-log_ratio_clip, log_ratio_clip)
                ratio = torch.exp(log_ratio)

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon_high,
                )

                adv_t = torch.full_like(ratio, adv_i)
                surr1 = ratio * adv_t
                surr2 = clipped_ratio * adv_t

                loss_i = -torch.min(surr1, surr2).mean() / n_idxs

                if use_kl:
                    kl_anchor = ref_fm_per_traj[i].to(device=device, dtype=torch.float32) if ref_fm_per_traj else old_fm
                    kl_per_step = 0.5 * (kl_anchor - new_fm_f) ** 2
                    kl_penalty = config.kl_coeff * kl_per_step.mean() / n_idxs
                    loss_i = loss_i + kl_penalty
                    mb_local_kl += (config.kl_coeff * kl_per_step.mean()).item() / n_idxs

                if sft_fm_per_traj:
                    sft_anchor = sft_fm_per_traj[i].to(device=device, dtype=torch.float32)
                    sft_kl_per_step = 0.5 * (sft_anchor - new_fm_f) ** 2
                    sft_kl_penalty = sft_kl_coeff * sft_kl_per_step.mean() / n_idxs
                    loss_i = loss_i + sft_kl_penalty
                    mb_sft_kl += (sft_kl_coeff * sft_kl_per_step.mean()).item() / n_idxs

                loss_i.backward()

                with torch.no_grad():
                    det_log_ratio = log_ratio.detach()
                    det_ratio = ratio.detach()

                mb_loss += loss_i.item()
                mb_shift += (new_fm_f.detach().mean() - old_fm.mean()).abs().item() / n_idxs
                mb_clip_frac += (
                    (det_ratio < 1.0 - config.clip_epsilon) | (det_ratio > 1.0 + config.clip_epsilon_high)
                ).float().mean().item() / n_idxs
                mb_max_log_ratio = max(mb_max_log_ratio, det_log_ratio.abs().max().item())
                mb_sum_ratio += det_ratio.mean().item() / n_idxs
                if use_kl:
                    kl_anchor = ref_fm_per_traj[i].to(device=device, dtype=torch.float32) if ref_fm_per_traj else old_fm
                    mb_raw_kl += (0.5 * (kl_anchor - new_fm_f.detach()) ** 2).mean().item() / n_idxs
                if sft_fm_per_traj:
                    sft_anchor = sft_fm_per_traj[i].to(device=device, dtype=torch.float32)
                    mb_raw_sft_kl += (0.5 * (sft_anchor - new_fm_f.detach()) ** 2).mean().item() / n_idxs
                used += 1

            if used == 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
            optimizer.step()

            total_loss += mb_loss
            total_local_kl += mb_local_kl
            total_sft_kl += mb_sft_kl
            total_shift += mb_shift
            total_clip_frac += mb_clip_frac
            total_raw_kl += mb_raw_kl
            total_raw_sft_kl += mb_raw_sft_kl
            total_mean_ratio += mb_sum_ratio
            total_max_log_ratio = max(total_max_log_ratio, mb_max_log_ratio)
            num_updates += 1

    denom = max(num_updates, 1)
    return UpdateMetrics(
        avg_loss=total_loss / denom,
        avg_kl=total_local_kl / denom,
        avg_sft_kl=total_sft_kl / denom,
        avg_shift=total_shift / denom,
        avg_weight=total_clip_frac / denom,
        raw_kl=total_raw_kl / denom,
        raw_sft_kl=total_raw_sft_kl / denom,
        mean_ratio=total_mean_ratio / denom,
        max_log_ratio=total_max_log_ratio,
    )


__all__ = ["fpo_update"]
