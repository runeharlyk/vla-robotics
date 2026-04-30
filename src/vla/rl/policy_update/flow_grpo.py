"""Flow-GRPO policy update."""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory

from .base import UpdateMetrics


def _flow_path_log_probs(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    *,
    batch_size: int,
    reduction: str,
) -> torch.Tensor:
    missing = [
        name
        for name in ("flow_states", "flow_next_states", "flow_times", "flow_dts", "flow_sigmas")
        if getattr(traj, name, None) is None
    ]
    if missing:
        raise ValueError(
            "Flow-GRPO trajectories must be collected with the SDE sampler; "
            f"missing fields: {', '.join(missing)}"
        )

    T = traj.length
    chunk_mask = traj.chunk_mask[:T] if traj.chunk_mask is not None else None
    return policy.compute_flow_path_log_probs(
        images=traj.images[:T],
        instruction=instruction,
        states=traj.states[:T] if traj.states is not None else None,
        path_states=traj.flow_states[:T],
        path_next_states=traj.flow_next_states[:T],
        path_times=traj.flow_times[:T],
        path_dts=traj.flow_dts[:T],
        path_sigmas=traj.flow_sigmas[:T],
        chunk_mask=chunk_mask,
        batch_size=batch_size,
        reduction=reduction,
    )


def flow_grpo_update(
    policy: SmolVLAPolicy,
    ref_policy: SmolVLAPolicy,
    sft_policy: SmolVLAPolicy | None,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run Flow-GRPO using log-probs of stored SDE denoising transitions."""
    B = config.fm_batch_size
    M = len(trajectories)
    if M == 0:
        return UpdateMetrics()

    reduction = config.flow_grpo_logprob_reduction
    log_ratio_clip = float(config.flow_grpo_log_ratio_clip)
    positive_only = bool(config.flow_grpo_positive_adv_only)
    negative_scale = float(config.flow_grpo_negative_adv_scale)
    sft_kl_coeff = float(getattr(config, "sft_kl_coeff", 0.0))

    old_logps_per_traj: list[torch.Tensor] = []
    sft_logps_per_traj: list[torch.Tensor] = []
    with torch.no_grad():
        ref_policy.eval()
        for i, traj in enumerate(trajectories):
            old_logps = _flow_path_log_probs(
                ref_policy,
                traj,
                instrs_per_traj[i],
                batch_size=B,
                reduction=reduction,
            )
            old_logps_per_traj.append(old_logps.detach())

            if sft_kl_coeff > 0:
                if sft_policy is None:
                    raise ValueError("Flow-GRPO SFT-anchor KL requested, but sft_policy was not provided.")
                sft_logps = _flow_path_log_probs(
                    sft_policy,
                    traj,
                    instrs_per_traj[i],
                    batch_size=B,
                    reduction=reduction,
                )
                sft_logps_per_traj.append(sft_logps.detach())

    policy.train()
    device = next(policy.parameters()).device
    minibatch_trajs = min(config.ppo_minibatch_trajs, M)

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
            mb_sum_ratio = 0.0
            mb_max_log_ratio = 0.0
            used = 0

            for i in idxs:
                adv_i = float(advantages[i])
                if adv_i <= 0.0 and positive_only:
                    continue
                if adv_i < 0.0:
                    adv_i *= negative_scale
                    if adv_i == 0.0:
                        continue

                new_logps = _flow_path_log_probs(
                    policy,
                    trajectories[i],
                    instrs_per_traj[i],
                    batch_size=B,
                    reduction=reduction,
                ).float()
                old_logps = old_logps_per_traj[i].to(device=device, dtype=torch.float32)

                log_ratio = (new_logps - old_logps).clamp(-log_ratio_clip, log_ratio_clip)
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

                if config.kl_coeff > 0:
                    kl_per_step = 0.5 * (new_logps - old_logps).square()
                    kl_penalty = config.kl_coeff * kl_per_step.mean() / n_idxs
                    loss_i = loss_i + kl_penalty
                    mb_local_kl += (config.kl_coeff * kl_per_step.mean()).item() / n_idxs

                if sft_logps_per_traj:
                    sft_anchor = sft_logps_per_traj[i].to(device=device, dtype=torch.float32)
                    sft_kl_per_step = 0.5 * (new_logps - sft_anchor).square()
                    sft_kl_penalty = sft_kl_coeff * sft_kl_per_step.mean() / n_idxs
                    loss_i = loss_i + sft_kl_penalty
                    mb_sft_kl += (sft_kl_coeff * sft_kl_per_step.mean()).item() / n_idxs

                loss_i.backward()

                with torch.no_grad():
                    det_ratio = ratio.detach()
                    det_log_ratio = log_ratio.detach()
                    raw_kl = 0.5 * (new_logps.detach() - old_logps).square()

                mb_loss += loss_i.item()
                mb_shift += (new_logps.detach().mean() - old_logps.mean()).abs().item() / n_idxs
                mb_clip_frac += (
                    (det_ratio < 1.0 - config.clip_epsilon) | (det_ratio > 1.0 + config.clip_epsilon_high)
                ).float().mean().item() / n_idxs
                mb_sum_ratio += det_ratio.mean().item() / n_idxs
                mb_max_log_ratio = max(mb_max_log_ratio, det_log_ratio.abs().max().item())
                mb_raw_kl += raw_kl.mean().item() / n_idxs
                if sft_logps_per_traj:
                    sft_anchor = sft_logps_per_traj[i].to(device=device, dtype=torch.float32)
                    mb_raw_sft_kl += (0.5 * (new_logps.detach() - sft_anchor).square()).mean().item() / n_idxs
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


__all__ = ["flow_grpo_update"]
