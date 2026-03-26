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
    n_samples: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pre-sample noise and time tensors for an entire trajectory.

    When *n_samples* > 1 (used by FPO for variance reduction), multiple
    independent (noise, time) pairs are returned so the FM-loss ratio can
    be averaged across samples before exponentiation, following the FPO
    paper (Kanazawa et al., 2025).

    Returns:
        noise_list: *n_samples* tensors of shape ``(T, chunk_size, max_action_dim)``
        time_list:  *n_samples* tensors of shape ``(T,)``
    """
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
        reduction=reduction,
    )


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

    AWR weights each trajectory's FM loss by exp(advantage / β) and
    optionally adds a per-timestep KL penalty against the reference
    policy (refreshed each iteration in the training loop).

    The KL approximation follows the FPO paper (Kanazawa et al., 2025):
    the FM loss difference is a proxy for log-likelihood ratio, so the
    squared per-timestep difference provides a Taylor-expansion KL
    estimate: KL ≈ 0.5 * E[(L_ref(t) - L_θ(t))²].

    Returns:
        :class:`UpdateMetrics` with average loss, KL, and weight.
    """
    B = config.fm_batch_size

    active_indices = [
        i for i, traj in enumerate(trajectories) if traj.task_id not in skipped_task_ids
    ]
    active_count = len(active_indices)
    if active_count == 0:
        return UpdateMetrics()

    # Cache per-timestep ref FM losses for KL computation.
    # These are compared per-timestep against current-policy losses,
    # not reduced to scalar means — the old code used scalar-mean
    # comparison which collapsed to ≈0.
    # NOTE: _compute_fm_loss_batched always returns a (T,) tensor natively.
    # The `reduction="mean"` argument reduces the chunk_size dimension,
    # not the time dimension. Do not set to "none" as that throws a ValueError.
    ref_losses_per_traj: list[torch.Tensor] = []
    if config.kl_coeff > 0:
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

    policy.train()
    total_weighted_loss = 0.0
    total_kl = 0.0
    total_weight = 0.0
    used_count_total = 0

    for _ in range(config.awr_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_weight = 0.0
        used_count = 0

        for i in active_indices:
            traj = trajectories[i]
            adv_i = advantages[i]
            weight = min(math.exp(adv_i / config.awr_temperature), config.awr_weight_clip)

            # Compute per-timestep loss (returns (T,) natively).
            # The `reduction="mean"` arg reduces the action chunk dimension,
            # NOT the sequence time dimension. Setting it to "none" will crash.
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
            traj_loss = weight * fm_loss / active_count

            # Per-timestep KL approximation:
            #   KL ≈ 0.5 * E_t[(L_ref(t) - L_θ(t))²]
            # This uses the FPO paper's insight that FM loss differences
            # proxy log-likelihood ratios. Computing this per-timestep
            # (not from scalar means) gives a meaningful divergence
            # signal. The old code computed 0.5*(mean_ref - mean_cur)²
            # which was always ≈ 0 due to Jensen's inequality.
            if config.kl_coeff > 0 and ref_losses_per_traj:
                ref_fm_t = ref_losses_per_traj[i]  # (T,) per-timestep
                kl_per_step = 0.5 * (ref_fm_t - fm_loss_t) ** 2
                kl_approx = kl_per_step.mean()
                traj_loss = traj_loss + config.kl_coeff * kl_approx / active_count
                epoch_kl += (config.kl_coeff * kl_approx).item()

            traj_loss.backward()
            epoch_loss += (weight * fm_loss).item()
            epoch_weight += weight
            used_count += 1

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_weighted_loss += epoch_loss
        total_kl += epoch_kl
        total_weight += epoch_weight
        used_count_total += used_count

    denom = max(used_count_total, 1)
    return UpdateMetrics(
        avg_loss=total_weighted_loss / denom,
        avg_kl=total_kl / denom,
        avg_weight=total_weight / denom,
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
    """Compute per-timestep FM loss averaged over N noise/time samples.

    Following the FPO paper (Kanazawa et al., 2025), averaging multiple
    CFM loss estimates per timestep before exponentiation drastically
    reduces variance in the importance-sampling ratio.

    Returns:
        ``(T,)`` per-timestep loss averaged across samples.
    """
    losses = []
    for noise, time in zip(noise_list, time_list, strict=True):
        loss = _compute_fm_loss_batched(
            policy,
            traj,
            instruction,
            noise,
            time,
            batch_size=batch_size,
            reduction=reduction,
        )
        losses.append(loss)
    return torch.stack(losses).mean(dim=0)


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
) -> UpdateMetrics:
    """FPO (Flow Policy Optimization) update with PPO-clip objective.

    Computes the FPO ratio r = exp(L_old - L_new) as a proxy for the
    likelihood ratio, following Kanazawa et al., 2025. Uses asymmetric
    clipping (SimpleVLA-RL / DAPO) and optional KL regularization.

    KL regularization (controlled by ``config.kl_coeff``) prevents
    catastrophic policy collapse by penalizing deviation from the
    start-of-iteration policy. The ``old_fm`` losses (computed before
    any gradient steps) serve double duty as both the FPO ratio
    denominator and the KL anchor, since ``ref_policy`` is refreshed
    to match the current policy at the start of each iteration.

    Returns:
        :class:`UpdateMetrics` with avg surrogate loss, KL shift, and clip fraction.
    """
    B = config.fm_batch_size
    M = len(trajectories)

    if M == 0:
        return UpdateMetrics()

    device = next(policy.parameters()).device
    reduction = getattr(config, "fpo_loss_reduction", "sum")
    positive_only = bool(getattr(config, "fpo_positive_adv_only", False))
    negative_scale = float(getattr(config, "fpo_negative_adv_scale", 0.25))
    log_ratio_clip = float(getattr(config, "fpo_log_ratio_clip", 5.0))

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
            )
            old_fm_per_traj.append(old_fm.detach())

    use_kl = config.kl_coeff > 0

    policy.train()

    minibatch_trajs = min(getattr(config, "ppo_minibatch_trajs", 4), M)
    total_loss = 0.0
    total_shift = 0.0
    total_clip_frac = 0.0
    num_updates = 0

    for _ in range(max(config.ppo_epochs, 1)):
        order = torch.randperm(M).tolist()

        for start in range(0, M, minibatch_trajs):
            idxs = order[start : start + minibatch_trajs]
            n_idxs = len(idxs)

            optimizer.zero_grad(set_to_none=True)

            mb_loss = 0.0
            mb_shift = 0.0
            mb_clip_frac = 0.0
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
                    kl_per_step = 0.5 * (old_fm - new_fm_f) ** 2
                    kl_penalty = config.kl_coeff * kl_per_step.mean() / n_idxs
                    loss_i = loss_i + kl_penalty

                loss_i.backward()

                mb_loss += loss_i.item()
                mb_shift += (new_fm_f.detach().mean() - old_fm.mean()).abs().item() / n_idxs
                mb_clip_frac += (
                    (ratio.detach() < 1.0 - config.clip_epsilon) | (ratio.detach() > 1.0 + config.clip_epsilon_high)
                ).float().mean().item() / n_idxs
                used += 1

            if used == 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
            optimizer.step()

            total_loss += mb_loss
            total_shift += mb_shift
            total_clip_frac += mb_clip_frac
            num_updates += 1

    denom = max(num_updates, 1)
    return UpdateMetrics(
        avg_loss=total_loss / denom,
        avg_kl=total_shift / denom,
        avg_weight=total_clip_frac / denom,
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
    skipped_task_ids: set[str],
    config: SRPOConfig,
) -> UpdateMetrics:
    """Run PPO clipped-surrogate policy update with KL regularisation.

    Returns:
        :class:`UpdateMetrics` with average surrogate loss and KL penalty.
    """
    B = config.fm_batch_size
    active_indices = [i for i, traj in enumerate(trajectories) if traj.task_id not in skipped_task_ids]
    active_count = len(active_indices)
    if active_count == 0:
        return UpdateMetrics()

    old_losses_per_traj: dict[int, torch.Tensor] = {}
    ref_losses_per_traj: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for i in active_indices:
            traj = trajectories[i]
            old_loss = _compute_fm_loss_batched(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
            )
            old_losses_per_traj[i] = old_loss.detach()

            ref_loss = _compute_fm_loss_batched(
                ref_policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
                batch_size=B,
            )
            ref_losses_per_traj[i] = ref_loss.detach()

    policy.train()
    total_surrogate = 0.0
    total_kl = 0.0
    used_count_total = 0

    for _ppo_epoch in range(config.ppo_epochs):
        optimizer.zero_grad()
        epoch_clip_loss = 0.0
        epoch_kl = 0.0
        used_count = 0

        for i in active_indices:
            traj = trajectories[i]
            adv_i = advantages[i]
            old_losses_t = old_losses_per_traj[i]
            ref_losses_t = ref_losses_per_traj[i]

            new_losses_t = _compute_fm_loss_batched(
                policy,
                traj,
                instrs_per_traj[i],
                fixed_noise[i],
                fixed_time[i],
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
            kl_approx = 0.5 * (log_ratio_ref**2).mean()
            kl_penalty = config.kl_coeff * kl_approx

            traj_loss = (clip_loss + kl_penalty) / active_count
            traj_loss.backward()

            epoch_clip_loss += clip_loss.item()
            epoch_kl += kl_penalty.item()
            used_count += 1

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
        optimizer.step()

        total_surrogate += epoch_clip_loss
        total_kl += epoch_kl
        used_count_total += used_count

    denom = max(used_count_total, 1)
    return UpdateMetrics(
        avg_loss=total_surrogate / denom,
        avg_kl=total_kl / denom,
    )
