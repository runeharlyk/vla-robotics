"""SRPO reinforcement-learning training loop.

Implements the full SRPO algorithm (Section 3.3 of the paper):
  1. Collect M trajectories with π_θ_old.
  2. Compute world-progress trajectory rewards g_i via the world model.
  3. Compute trajectory-level advantages  = (g_i − μ_g) / σ_g.
  4. Policy update (selected via ``config.update_method``):
     **AWR** (default, recommended for flow-matching policies):
       - Weight each trajectory's FM loss by exp(advantage / β)
       - No importance-sampling ratios needed
     **PPO** (legacy, for discrete-token policies):
       - Cache FM losses under θ_old and π_ref
       - Clipped surrogate loss + KL regularisation
  5. Update θ_old ← θ, add any new successes to reference set.

The rollout engine is **simulator-agnostic**: pass in any object that
implements :class:`~vla.rl.rollout.RolloutEngine` (ManiSkill or LIBERO).
"""

from __future__ import annotations

import copy
import logging
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from vla.constants import AdvantageMode, Simulator, UpdateMethod
from vla.diagnostics.eval import EvalMetrics, evaluate_smolvla, metrics_from_trajectories, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.models.world_model import WorldModelEncoder, build_world_model
from vla.rl.advantage import leave_one_out_advantages_per_task, normalize_advantages_per_task
from vla.rl.config import SRPOConfig, TaskSpec
from vla.rl.policy_update import UpdateMetrics, _sample_fixed_noise_time, awr_update, fpo_update, ppo_update
from vla.rl.rollout import RolloutEngine, Trajectory
from vla.rl.srpo_reward import (
    MultiTaskWorldProgressReward,
    SRPORewardConfig,
)
from vla.training.checkpoint import save_best_checkpoint
from vla.training.metrics_logger import MetricsLogger
from vla.utils.tensor import to_float01

logger = logging.getLogger(__name__)

# Re-export so existing ``from vla.rl.trainer import SRPOConfig`` still works.
__all__ = [
    "SRPOConfig",
    "TaskSpec",
    "UpdateMetrics",
    "awr_update",
    "build_rollout_engine",
    "collect_all_trajectories",
    "evaluate_and_checkpoint",
    "log_training_config",
    "ppo_update",
    "fpo_update",
    "resample_uniform_reward_tasks",
    "train_srpo",
]


# ---------------------------------------------------------------------------
# Reference-policy helpers
# ---------------------------------------------------------------------------


def _freeze_policy_copy(policy: SmolVLAPolicy) -> SmolVLAPolicy:
    """Return an eval-only, non-trainable copy of a policy."""
    frozen = copy.deepcopy(policy)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


# ---------------------------------------------------------------------------
# Config logging
# ---------------------------------------------------------------------------


def log_training_config(
    config: SRPOConfig,
    task_specs: list[TaskSpec],
    trajs_per_task_per_iter: int,
) -> None:
    """Print all training parameters to the log at the start of a run.

    Produces a single structured block that makes it easy to reproduce
    the run from logs alone (mirroring the CLI lines in job logs).
    """
    sep = "=" * 72
    lines = [sep, "SRPO TRAINING CONFIGURATION", sep]

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        lines.append(f"  git_commit:            {commit}")
        lines.append(f"  git_branch:            {branch}")
        lines.append("")
    except Exception:
        pass

    lines.append(f"  mode:                  {config.mode}")
    lines.append(f"  simulator:             {config.simulator}")
    lines.append(f"  suite:                 {config.suite}")
    lines.append(f"  num_tasks:             {len(task_specs)}")
    for spec in task_specs:
        lines.append(f"    [{spec.task_id}] libero_idx={spec.libero_task_idx}  instruction={spec.instruction!r}")

    lines.append("")
    lines.append("  Policy update:")
    lines.append(f"    update_method:       {config.update_method}")
    lines.append(f"    advantage_mode:      {config.advantage_mode}")
    lines.append(f"    lr:                  {config.lr}")
    lines.append(f"    max_grad_norm:       {config.max_grad_norm}")
    lines.append(f"    clip_epsilon:        {config.clip_epsilon}")
    lines.append(f"    clip_epsilon_high:   {config.clip_epsilon_high}")
    lines.append(f"    kl_coeff:            {config.kl_coeff}")
    lines.append(f"    sft_kl_coeff:        {config.sft_kl_coeff}")
    lines.append(f"    adaptive_kl:         {config.adaptive_kl}")
    if config.adaptive_kl:
        lines.append(f"    kl_target:           {config.kl_target}")
        lines.append(f"    kl_adapt_factor:     {config.kl_adapt_factor}")
    lines.append(f"    ppo_epochs:          {config.ppo_epochs}")
    lines.append(f"    ppo_minibatch_trajs: {config.ppo_minibatch_trajs}")
    lines.append(f"    awr_epochs:          {config.awr_epochs}")
    lines.append(f"    awr_temperature:     {config.awr_temperature}")
    lines.append(f"    awr_weight_clip:     {config.awr_weight_clip}")
    lines.append(f"    adv_eps:             {config.adv_eps}")
    lines.append(f"    adv_skip_threshold:  {config.adv_skip_threshold}")

    lines.append("")
    lines.append("  FPO-specific:")
    lines.append(f"    num_fm_noise_samples:    {config.num_fm_noise_samples}")
    lines.append(f"    fpo_negative_adv_scale:  {config.fpo_negative_adv_scale}")
    lines.append(f"    fpo_positive_adv_only:   {config.fpo_positive_adv_only}")
    lines.append(f"    fpo_log_ratio_clip:      {config.fpo_log_ratio_clip}")
    lines.append(f"    fpo_loss_reduction:      {config.fpo_loss_reduction}")
    lines.append(f"    fpo_full_chunk_target:   {config.fpo_full_chunk_target}")
    lines.append(f"    fpo_use_ref_policy_kl:   {config.fpo_use_ref_policy_kl}")

    lines.append("")
    lines.append("  Rollout & eval:")
    lines.append(f"    num_iterations:      {config.num_iterations}")
    lines.append(f"    trajs_per_task:      {trajs_per_task_per_iter}")
    lines.append(f"    num_rollout_envs:    {config.num_rollout_envs}")
    lines.append(f"    num_envs:            {config.num_envs}")
    lines.append(f"    max_steps:           {config.max_steps}")
    lines.append(f"    eval_every:          {config.eval_every}")
    lines.append(f"    eval_episodes:       {config.eval_episodes}")
    lines.append(f"    eval_zero_sample:    {config.eval_zero_sample}")
    lines.append(f"    fm_batch_size:       {config.fm_batch_size}")
    lines.append(f"    seed:                {config.seed}")
    lines.append(f"    include_demos_in_update:    {config.include_demos_in_update}")
    lines.append(f"    success_replay_buffer_size: {config.success_replay_buffer_size}")
    lines.append(f"    success_replay_total_size:  {config.success_replay_total_size}")
    lines.append(f"    success_replay_alpha:       {config.success_replay_alpha}")
    lines.append(f"    success_replay_ema_decay:   {config.success_replay_ema_decay}")
    lines.append(f"    success_replay_max_ratio:   {config.success_replay_max_ratio}")
    lines.append(f"    dynamic_sampling:           {config.dynamic_sampling}")
    lines.append(f"    dynamic_sampling_max_retries: {config.dynamic_sampling_max_retries}")
    lines.append(f"    n_action_steps:             {config.n_action_steps}")

    lines.append("")
    lines.append("  SRPO reward (world-model):")
    lines.append(f"    world_model_type:    {config.world_model_type}")
    lines.append(f"    distance_metric:     {config.distance_metric}")
    lines.append(f"    subsample_every:     {config.subsample_every}")
    lines.append(f"    dbscan_eps:          {config.dbscan_eps}")
    lines.append(f"    dbscan_min_samples:  {config.dbscan_min_samples}")
    lines.append(f"    dbscan_auto_eps:     {config.dbscan_auto_eps}")
    lines.append(f"    use_failure_rewards:  {config.use_failure_rewards}")
    lines.append(f"    use_standard_scaler: {config.use_standard_scaler}")

    lines.append("")
    lines.append("  Infrastructure:")
    lines.append(f"    gradient_checkpointing: {config.gradient_checkpointing}")
    lines.append(f"    save_dir:            {config.save_dir}")

    lines.append(sep)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Rollout engine factory
# ---------------------------------------------------------------------------


def build_rollout_engine(
    config: SRPOConfig,
    spec: TaskSpec | None = None,
    num_envs: int | None = None,
) -> RolloutEngine:
    """Create a rollout engine from config and optional task spec.

    When *spec* is provided, its ``env_id`` / ``libero_task_idx`` override
    the values in *config*.  When *num_envs* is ``None``, defaults to
    ``config.num_rollout_envs``.
    """
    sim = config.simulator
    resolved_envs = num_envs if num_envs is not None else config.num_rollout_envs

    if sim is Simulator.MANISKILL:
        from vla.rl.maniskill_rollout import ManiSkillRollout

        env_id = (spec.env_id or config.env_id) if spec else config.env_id
        return ManiSkillRollout(
            env_id=env_id,
            num_envs=resolved_envs,
            max_steps=config.max_steps,
        )

    if sim is Simulator.LIBERO:
        from vla.rl.libero_rollout import LiberoRollout

        task_id = spec.libero_task_idx if spec else config.task_id
        return LiberoRollout(
            suite_name=config.suite,
            task_id=task_id,
            num_envs=resolved_envs,
            max_steps=config.max_steps,
            state_dim=config.state_dim,
        )

    raise ValueError(f"Unknown simulator {config.simulator!r}. Available: maniskill, libero")


# ---------------------------------------------------------------------------
# Decomposed training stages
# ---------------------------------------------------------------------------


_SHARED_LIBERO_KEY = "_shared_libero"


def _resolve_success_replay_capacity(config: SRPOConfig) -> int:
    """Return the global replay capacity, preserving the legacy flag as a fallback."""
    if config.success_replay_total_size > 0:
        return config.success_replay_total_size
    return config.success_replay_buffer_size


def _inverse_success_weights(
    task_ids: list[str],
    success_rate_ema: dict[str, float],
    alpha: float,
    rate_floor: float,
) -> dict[str, float]:
    """Compute bounded inverse-success weights for quota allocation."""
    bounded_alpha = max(alpha, 0.0)
    floor = max(rate_floor, 1e-6)
    weights: dict[str, float] = {}
    for tid in task_ids:
        rate = max(success_rate_ema.get(tid, 0.0), 0.0)
        weights[tid] = 1.0 / math.pow(rate + floor, bounded_alpha)
    return weights


def _allocate_integer_budget(weights: dict[str, float], budget: int) -> dict[str, int]:
    """Allocate an integer budget proportionally to the provided positive weights."""
    if budget <= 0 or not weights:
        return {tid: 0 for tid in weights}

    total_weight = sum(max(w, 0.0) for w in weights.values())
    if total_weight <= 0:
        return {tid: 0 for tid in weights}

    quotas: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    used = 0
    for tid, weight in weights.items():
        raw = budget * max(weight, 0.0) / total_weight
        base = int(math.floor(raw))
        quotas[tid] = base
        used += base
        fractional.append((raw - base, tid))

    for _, tid in sorted(fractional, reverse=True)[: max(budget - used, 0)]:
        quotas[tid] += 1

    return quotas


def _rebalance_success_buffer(
    success_buffer: dict[str, list[Trajectory]],
    success_rate_ema: dict[str, float],
    total_capacity: int,
    alpha: float,
    rate_floor: float,
) -> dict[str, int]:
    """Evict only from over-quota tasks so the retained buffer matches the target mix."""
    active_task_ids = [tid for tid, buf in success_buffer.items() if buf]
    if total_capacity <= 0 or not active_task_ids:
        success_buffer.clear()
        return {}

    quotas = _allocate_integer_budget(
        _inverse_success_weights(active_task_ids, success_rate_ema, alpha=alpha, rate_floor=rate_floor),
        total_capacity,
    )
    total_size = sum(len(success_buffer[tid]) for tid in active_task_ids)
    while total_size > total_capacity:
        overfull = [tid for tid in active_task_ids if len(success_buffer[tid]) > quotas.get(tid, 0)]
        if not overfull:
            overfull = [tid for tid in active_task_ids if success_buffer[tid]]
        evict_tid = max(
            overfull, key=lambda tid: (len(success_buffer[tid]) - quotas.get(tid, 0), len(success_buffer[tid]))
        )
        success_buffer[evict_tid].pop(0)
        total_size -= 1
        if not success_buffer[evict_tid]:
            del success_buffer[evict_tid]
            active_task_ids = [tid for tid in active_task_ids if tid != evict_tid]

    return quotas


def _sample_success_replay(
    success_buffer: dict[str, list[Trajectory]],
    success_rate_ema: dict[str, float],
    max_replay: int,
    alpha: float,
    rate_floor: float,
) -> tuple[list[Trajectory], dict[str, int]]:
    """Sample a capped replay batch using the same inverse-success balancing as retention."""
    active_task_ids = [tid for tid, buf in success_buffer.items() if buf]
    if max_replay <= 0 or not active_task_ids:
        return [], {}

    available = sum(len(success_buffer[tid]) for tid in active_task_ids)
    replay_budget = min(max_replay, available)
    quotas = _allocate_integer_budget(
        _inverse_success_weights(active_task_ids, success_rate_ema, alpha=alpha, rate_floor=rate_floor),
        replay_budget,
    )

    sampled: list[Trajectory] = []
    sampled_counts: dict[str, int] = {}
    carry = replay_budget
    remaining = active_task_ids.copy()

    while carry > 0 and remaining:
        progressed = False
        for tid in remaining.copy():
            take = min(len(success_buffer[tid]), quotas.get(tid, 0))
            if take > 0:
                sampled.extend(success_buffer[tid][-take:])
                sampled_counts[tid] = take
                carry -= take
                quotas[tid] = 0
                progressed = True
            remaining.remove(tid)

        if carry <= 0 or not progressed:
            leftovers = sorted(
                ((len(success_buffer[tid]) - sampled_counts.get(tid, 0), tid) for tid in active_task_ids),
                reverse=True,
            )
            for left, tid in leftovers:
                if carry <= 0:
                    break
                if left <= 0:
                    continue
                take = min(left, carry)
                already = sampled_counts.get(tid, 0)
                start = len(success_buffer[tid]) - already - take
                end = len(success_buffer[tid]) - already
                sampled.extend(success_buffer[tid][start:end])
                sampled_counts[tid] = already + take
                carry -= take
            break

    return sampled, sampled_counts


def _reward_std(trajs: list[Trajectory]) -> float:
    """Population std of sparse-success rewards for a task's trajectory group."""
    if len(trajs) < 2:
        return 0.0
    rewards = [1.0 if t.success else 0.0 for t in trajs]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return var**0.5


def resample_uniform_reward_tasks(
    policy: SmolVLAPolicy,
    rollout_trajectories: list[Trajectory],
    task_specs: list[TaskSpec],
    rollout_engines: dict[str, RolloutEngine],
    config: SRPOConfig,
    iteration: int,
    trajs_per_task: int,
) -> tuple[list[Trajectory], dict[str, int], dict[str, int], list[str]]:
    """DAPO-style replacement sampling for uniform-reward rollout groups.

    After the initial rollout, any task whose trajectory group has a reward
    std below ``config.adv_skip_threshold`` (i.e. all-success or all-failure)
    contributes zero advantage signal and wastes its slot in the update
    batch.  This helper re-collects ``trajs_per_task`` trajectories for each
    such task, up to ``config.dynamic_sampling_max_retries`` times, stopping
    as soon as the group becomes informative.

    The call is a no-op when ``config.dynamic_sampling`` is ``False`` or the
    retry budget is zero, so gating the feature behind a CLI flag keeps
    ablations clean.

    Returns:
        Tuple of (rebuilt trajectory list, per-task successes, retries per
        resampled task, tasks that gave up after exhausting the retry budget).
    """
    if not config.dynamic_sampling or config.dynamic_sampling_max_retries <= 0:
        per_task_successes: dict[str, int] = {}
        for spec in task_specs:
            per_task_successes[spec.task_id] = sum(
                1 for t in rollout_trajectories if t.task_id == spec.task_id and t.success
            )
        return rollout_trajectories, per_task_successes, {}, []

    trajs_by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in rollout_trajectories:
        trajs_by_task[t.task_id].append(t)

    use_vectorized = config.num_rollout_envs > 1
    chunked = config.n_action_steps > 1
    retries_per_task: dict[str, int] = {}
    gave_up_tasks: list[str] = []

    for spec in task_specs:
        tid = spec.task_id
        task_trajs = trajs_by_task.get(tid, [])
        retries = 0
        while _reward_std(task_trajs) < config.adv_skip_threshold and retries < config.dynamic_sampling_max_retries:
            retries += 1
            engine = _get_or_build_engine(rollout_engines, config, spec)
            new_trajs = engine.collect_batch(
                policy_fn=policy.predict_action,
                instruction=spec.instruction,
                num_trajectories=trajs_per_task,
                seed=config.seed + iteration * 1000 + retries * 100003,
                policy_batch_fn=policy.predict_action_batch if use_vectorized else None,
                n_action_steps=config.n_action_steps,
                policy_chunk_fn=policy.predict_action_chunk if chunked else None,
                policy_chunk_batch_fn=policy.predict_action_chunk_batch if chunked and use_vectorized else None,
            )
            for t in new_trajs:
                t.task_id = tid
            task_trajs = new_trajs
            trajs_by_task[tid] = new_trajs

        if retries > 0:
            retries_per_task[tid] = retries
            if _reward_std(task_trajs) < config.adv_skip_threshold:
                gave_up_tasks.append(tid)

    new_all: list[Trajectory] = []
    new_per_task_successes: dict[str, int] = {}
    for spec in task_specs:
        task_list = trajs_by_task.get(spec.task_id, [])
        new_all.extend(task_list)
        new_per_task_successes[spec.task_id] = sum(1 for t in task_list if t.success)

    if retries_per_task:
        total_retries = sum(retries_per_task.values())
        logger.info(
            "Iter %d: dynamic sampling recollected %d task(s) (%d total retries); gave_up=%s",
            iteration,
            len(retries_per_task),
            total_retries,
            gave_up_tasks or "none",
        )
        for tid, n in sorted(retries_per_task.items()):
            logger.info(
                "  [dynsample %s] retries=%d final_std=%.4f gave_up=%s",
                tid,
                n,
                _reward_std(trajs_by_task[tid]),
                tid in gave_up_tasks,
            )

    return new_all, new_per_task_successes, retries_per_task, gave_up_tasks


def collect_all_trajectories(
    policy: SmolVLAPolicy,
    task_specs: list[TaskSpec],
    rollout_engines: dict[str, RolloutEngine],
    config: SRPOConfig,
    iteration: int,
    trajs_per_task: int,
) -> tuple[list[Trajectory], dict[str, int]]:
    """Collect rollout trajectories from all tasks.

    For LIBERO, uses a **single shared** rollout engine that reconfigures
    between tasks (RLinf pattern), keeping only ``num_rollout_envs``
    subprocesses alive at any time regardless of the number of tasks.

    For ManiSkill, creates per-task engines lazily (GPU-batched, no
    subprocess overhead).

    Returns:
        Tuple of (all_trajectories, per_task_successes).
    """
    policy.eval()
    all_trajectories: list[Trajectory] = []
    per_task_successes: dict[str, int] = {}
    use_vectorized = config.num_rollout_envs > 1
    chunked = config.n_action_steps > 1

    for spec in task_specs:
        engine = _get_or_build_engine(rollout_engines, config, spec)
        trajs = engine.collect_batch(
            policy_fn=policy.predict_action,
            instruction=spec.instruction,
            num_trajectories=trajs_per_task,
            seed=config.seed + iteration * 1000,
            policy_batch_fn=policy.predict_action_batch if use_vectorized else None,
            n_action_steps=config.n_action_steps,
            policy_chunk_fn=policy.predict_action_chunk if chunked else None,
            policy_chunk_batch_fn=policy.predict_action_chunk_batch if chunked and use_vectorized else None,
        )
        for t in trajs:
            t.task_id = spec.task_id
        all_trajectories.extend(trajs)
        per_task_successes[spec.task_id] = sum(1 for t in trajs if t.success)

    return all_trajectories, per_task_successes


def _get_or_build_engine(
    rollout_engines: dict[str, RolloutEngine],
    config: SRPOConfig,
    spec: TaskSpec,
) -> RolloutEngine:
    """Return a rollout engine for *spec*, reusing/reconfiguring when possible.

    For LIBERO: maintains a single shared ``LiberoRollout`` under the key
    ``_SHARED_LIBERO_KEY`` and hot-swaps its task via ``reconfigure()``.
    For ManiSkill: creates per-task engines lazily (no subprocess overhead).
    """
    if config.simulator is Simulator.LIBERO:
        from vla.rl.libero_rollout import LiberoRollout

        if _SHARED_LIBERO_KEY in rollout_engines:
            engine = rollout_engines[_SHARED_LIBERO_KEY]
            assert isinstance(engine, LiberoRollout)
            engine.reconfigure(config.suite, spec.libero_task_idx)
            return engine
        logger.info("Creating shared LIBERO rollout engine (%d envs)", config.num_rollout_envs)
        engine = build_rollout_engine(config, spec=spec)
        rollout_engines[_SHARED_LIBERO_KEY] = engine
        return engine

    if spec.task_id not in rollout_engines:
        logger.info(f"Creating rollout engine for {spec.task_id}")
        rollout_engines[spec.task_id] = build_rollout_engine(config, spec=spec)
    return rollout_engines[spec.task_id]


def _evaluate_task(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    spec: TaskSpec,
    rollout_engines: dict[str, RolloutEngine] | None,
) -> EvalMetrics:
    """Evaluate a single task, reusing the shared LIBERO engine when available."""
    if config.simulator is Simulator.LIBERO and rollout_engines is not None and _SHARED_LIBERO_KEY in rollout_engines:
        engine = _get_or_build_engine(rollout_engines, config, spec)
        use_vec = config.num_rollout_envs > 1
        trajs = engine.collect_batch(
            policy_fn=policy.predict_action,
            instruction=spec.instruction,
            num_trajectories=config.eval_episodes,
            seed=config.seed + 20000,
            policy_batch_fn=policy.predict_action_batch if use_vec else None,
        )
        return metrics_from_trajectories(trajs, expected_episodes=config.eval_episodes)

    kwargs: dict[str, Any] = dict(
        instruction=spec.instruction,
        simulator=config.simulator,
        num_episodes=config.eval_episodes,
        max_steps=config.max_steps,
        seed=config.seed + 20000,
    )
    if config.simulator is Simulator.LIBERO:
        kwargs.update(suite=config.suite, task_id=spec.libero_task_idx, num_envs=config.num_envs)
    else:
        kwargs.update(env_id=spec.env_id or config.env_id)
    return evaluate_smolvla(policy, **kwargs)


def evaluate_and_checkpoint(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    task_specs: list[TaskSpec],
    iteration: int,
    save_path: Path,
    best_success: float,
    log_data: dict[str, Any],
    rollout_engines: dict[str, RolloutEngine] | None = None,
) -> float:
    """Run periodic evaluation and save the best checkpoint.

    When *rollout_engines* is provided and contains a shared LIBERO
    engine, it is reused for evaluation (reconfigured per task) instead
    of spawning new subprocess environments each time.

    Returns:
        Updated best success rate.
    """
    prev_eval_zero_sample = policy.eval_zero_sample
    policy.eval_zero_sample = False
    try:
        task_sr_sum = 0.0
        for spec in task_specs:
            metrics = _evaluate_task(policy, config, spec, rollout_engines)
            print_metrics(
                metrics,
                tag=f"{config.mode} iter {iteration} [{spec.task_id}]",
            )
            log_data[f"{config.mode}/{spec.task_id}/eval/success_rate"] = metrics.success_rate
            log_data[f"{config.mode}/{spec.task_id}/eval/mean_reward"] = metrics.mean_reward
            log_data[f"{config.mode}/{spec.task_id}/eval/mean_ep_len"] = metrics.mean_episode_length
            task_sr_sum += metrics.success_rate

        avg_sr = task_sr_sum / len(task_specs)
        log_data[f"{config.mode}/eval/success_rate"] = avg_sr
        best_success = save_best_checkpoint(
            avg_sr,
            best_success,
            lambda: policy.save_checkpoint(save_path / "best"),
            tag=config.mode,
        )
    finally:
        policy.eval_zero_sample = prev_eval_zero_sample

    return best_success


def save_best_rollout_checkpoint(
    current_successes: int,
    best_successes: int,
    save_fn: Any,
    *,
    tag: str = "",
) -> int:
    """Save a checkpoint when rollout successes improve.

    Unlike eval-based checkpointing, this metric is available every iteration
    and is useful when jobs die before the first scheduled evaluation.
    """
    if current_successes > best_successes:
        save_fn()
        logger.info(
            "New best %srollout checkpoint: %d successes",
            f"{tag} " if tag else "",
            current_successes,
        )
        return current_successes
    return best_successes


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_srpo(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    task_specs: list[TaskSpec],
    demo_trajectories: dict[str, list[Trajectory]] | None = None,
    metrics_logger: MetricsLogger | None = None,
    trajs_per_task_per_iter: int = 4,
    rollout_engines: dict[str, RolloutEngine] | None = None,
) -> SmolVLAPolicy:
    """Run SRPO training (single- or multi-task) on an SFT-initialised policy.

    Handles both single-task (``len(task_specs) == 1``) and multi-task
    training in a single code-path.  Each iteration:

      1. Collects trajectories from every task via per-task rollout engines.
      2. Computes per-task world-progress rewards (DBSCAN clustering).
      3. Normalises advantages **per task** (with 1 task this equals global).
      4. Runs the policy update (AWR or PPO or FPO).
      5. Periodically evaluates and checkpoints.

    Args:
        policy: SFT-initialised SmolVLA policy (becomes both π_θ and π_ref).
        config: SRPO hyperparameters.
        task_specs: One or more :class:`TaskSpec` describing the task(s).
        demo_trajectories: ``{task_id: [Trajectory, ...]}`` for seeding the
            per-task reward models.  Pass ``None`` to skip demo seeding.
        metrics_logger: Optional :class:`MetricsLogger` for W&B and JSONL
            persistence.  When ``None``, a JSONL-only logger is created
            from ``config.save_dir``.
        trajs_per_task_per_iter: Trajectories to collect **per task** each
            iteration.
        rollout_engines: Optional pre-built ``{task_id: RolloutEngine}`` map.
            When ``None``, engines are created from ``task_specs`` and
            ``config`` automatically.

    Returns:
        The RL-tuned policy.
    """
    if config.gradient_checkpointing:
        policy.enable_gradient_checkpointing()

    optimizer, trainable = config.build_optimizer(policy)

    # Create a reference policy for KL regularisation.
    # AWR/PPO always use it. FPO only uses it when explicitly requested
    # for debugging, since the default FPO path reuses cached old_fm
    # losses as the KL anchor.
    ref_policy: SmolVLAPolicy | None = None
    if config.update_method is not UpdateMethod.FPO or getattr(config, "fpo_use_ref_policy_kl", False):
        ref_policy = _freeze_policy_copy(policy)

    sft_policy: SmolVLAPolicy | None = None
    if config.sft_kl_coeff > 0:
        sft_policy = _freeze_policy_copy(policy)

    # -- Per-task rollout engines -----------------------------------------
    if rollout_engines is None:
        rollout_engines = {}

    if config.simulator is Simulator.LIBERO and _SHARED_LIBERO_KEY not in rollout_engines:
        _get_or_build_engine(rollout_engines, config, task_specs[0])
    elif config.simulator is Simulator.MANISKILL and config.num_rollout_envs > 1:
        # ManiSkill GPU PhysX must be enabled before any other PhysX-backed env
        # is constructed. Prebuild vectorized rollout engines before the
        # baseline evaluation step so the later eval envs do not block GPU PhysX.
        for spec in task_specs:
            _get_or_build_engine(rollout_engines, config, spec)

    spec_lookup: dict[str, TaskSpec] = {s.task_id: s for s in task_specs}

    success_buffer: dict[str, list[Trajectory]] = defaultdict(list)
    success_rate_ema: dict[str, float] = {s.task_id: 0.0 for s in task_specs}
    replay_capacity = _resolve_success_replay_capacity(config)

    # -- Per-task reward model (works for single-task too: 1 key) ---------
    world_encoder: WorldModelEncoder | None = None
    reward_model: MultiTaskWorldProgressReward | None = None

    if config.mode == "srpo":
        world_encoder = build_world_model(
            model_type=config.world_model_type,
            device=str(policy.device),
        )
        reward_cfg = SRPORewardConfig(
            subsample_every=config.subsample_every,
            dbscan_eps=config.dbscan_eps,
            dbscan_min_samples=config.dbscan_min_samples,
            distance_metric=config.distance_metric,
            dbscan_auto_eps=config.dbscan_auto_eps,
            use_failure_rewards=config.use_failure_rewards,
            use_standard_scaler=config.use_standard_scaler,
        )
        reward_model = MultiTaskWorldProgressReward(world_encoder, reward_cfg)

        if demo_trajectories:
            for _tid, demos in demo_trajectories.items():
                demo_images = []
                for dt in demos:
                    imgs = dt.images[: dt.length]
                    imgs = to_float01(imgs)
                    demo_images.append(imgs)
                reward_model.add_demo_trajectories(_tid, demo_images)

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if metrics_logger is None:
        metrics_logger = MetricsLogger(jsonl_path=save_path / "metrics.jsonl")
    best_success = -1.0
    best_rollout_successes = -1

    log_training_config(config, task_specs, trajs_per_task_per_iter)

    log_data: dict[str, Any] = {
        f"{config.mode}/iteration": 0,
        f"{config.mode}/pre_rl_eval": 1,
    }
    best_success = evaluate_and_checkpoint(  # Evaluate for iteration 0 to get a baseline for improvements
        policy,
        config,
        task_specs,
        0,
        save_path,
        best_success,
        log_data,
        rollout_engines=rollout_engines,
    )
    metrics_logger.log(log_data)

    for iteration in trange(1, config.num_iterations + 1, desc="Training iterations"):
        # -- 0. Refresh reference policy: θ_old ← θ  --------------------
        #    Per RIPT-VLA (Algorithm 1: "Set sampling policy π_ψ ← π_θ")
        #    and SimpleVLA-RL's GRPO loop, the reference/old policy must
        #    be refreshed each iteration so that KL regularisation anchors
        #    to the *previous* iteration rather than to the stale initial
        #    checkpoint.  Without this, KL ≈ 0 and the trust region is
        #    meaningless.
        if ref_policy is not None:
            ref_policy.load_state_dict(policy.state_dict())

        # -- 1. Collect trajectories from all tasks -----------------------
        rollout_trajectories, per_task_successes = collect_all_trajectories(
            policy,
            task_specs,
            rollout_engines,
            config,
            iteration,
            trajs_per_task_per_iter,
        )

        # -- 1b. Dynamic (replacement) sampling (DAPO-style) --------------
        (
            rollout_trajectories,
            per_task_successes,
            dynsample_retries,
            dynsample_gave_up,
        ) = resample_uniform_reward_tasks(
            policy,
            rollout_trajectories,
            task_specs,
            rollout_engines,
            config,
            iteration,
            trajs_per_task_per_iter,
        )

        total_rollout_successes = sum(per_task_successes.values())
        rate_floor = 1.0 / max(trajs_per_task_per_iter, 1)
        for _tid, n_success in per_task_successes.items():
            rollout_sr = n_success / max(trajs_per_task_per_iter, 1)
            if iteration == 1:
                success_rate_ema[_tid] = rollout_sr
            else:
                prev = success_rate_ema.get(_tid, rollout_sr)
                success_rate_ema[_tid] = (
                    config.success_replay_ema_decay * prev + (1.0 - config.success_replay_ema_decay) * rollout_sr
                )

        if replay_capacity > 0:
            for t in rollout_trajectories:
                if t.success:
                    success_buffer[t.task_id].append(t)
            buffer_quotas = _rebalance_success_buffer(
                success_buffer,
                success_rate_ema,
                total_capacity=replay_capacity,
                alpha=config.success_replay_alpha,
                rate_floor=rate_floor,
            )
        else:
            buffer_quotas = {}

        extra_trajectories: list[Trajectory] = []
        if config.include_demos_in_update and demo_trajectories:
            for _tid, demos in demo_trajectories.items():
                extra_trajectories.extend(demos)

        if replay_capacity > 0:
            max_replay = int(config.success_replay_max_ratio * len(rollout_trajectories))
            replay_trajs, replay_counts = _sample_success_replay(
                success_buffer,
                success_rate_ema,
                max_replay=max_replay,
                alpha=config.success_replay_alpha,
                rate_floor=rate_floor,
            )
            extra_trajectories.extend(replay_trajs)
        else:
            replay_counts = {}

        all_trajectories = rollout_trajectories + extra_trajectories

        total_successes = sum(1 for t in all_trajectories if t.success)
        logger.info(
            f"Iter {iteration}: collected {len(rollout_trajectories)} trajs, "
            f"{total_rollout_successes} rollout successes, {len(all_trajectories)} total trajs for update "
            f"({total_successes} total successes)"
        )
        best_rollout_successes = save_best_rollout_checkpoint(
            total_rollout_successes,
            best_rollout_successes,
            lambda: policy.save_checkpoint(save_path / "best_rollout"),
            tag=config.mode,
        )
        if replay_counts:
            replay_total = sum(replay_counts.values())
            logger.info(
                "Iter %d: replayed %d successes from buffer (capacity=%d, buffered=%d)",
                iteration,
                replay_total,
                replay_capacity,
                sum(len(buf) for buf in success_buffer.values()),
            )
            for _tid in sorted(replay_counts):
                logger.info(
                    "  [replay %s] sampled=%d buffered=%d quota=%d sr_ema=%.4f",
                    _tid,
                    replay_counts[_tid],
                    len(success_buffer[_tid]),
                    buffer_quotas.get(_tid, 0),
                    success_rate_ema.get(_tid, 0.0),
                )

        # -- 2. Per-task rewards g_i --------------------------------------
        if world_encoder is not None:
            world_encoder.reload(policy.device)

        if config.mode == "srpo" and reward_model is not None:
            g_values, traj_embs = reward_model.compute_trajectory_rewards(all_trajectories)
            all_diags = reward_model.get_diagnostics()
            by_task_embs: dict[str, list[torch.Tensor]] = defaultdict(list)
            for i, t in enumerate(all_trajectories):
                if t.success:
                    by_task_embs[t.task_id].append(traj_embs[i])

            for _tid, embs in by_task_embs.items():
                reward_model.add_successful_embeddings(_tid, embs)
        else:
            g_values = [1.0 if t.success else 0.0 for t in all_trajectories]
            all_diags = None

        if world_encoder is not None:
            world_encoder.offload()

        # -- 3. Per-task advantage normalization --------------------------
        task_ids = [t.task_id for t in all_trajectories]
        if config.advantage_mode == AdvantageMode.SRPO_ZSCORE:
            adv_result = normalize_advantages_per_task(
                g_values=g_values,
                task_ids=task_ids,
                eps=config.adv_eps,
                skip_threshold=config.adv_skip_threshold,
            )
        elif config.advantage_mode == AdvantageMode.LEAVE_ONE_OUT:
            adv_result = leave_one_out_advantages_per_task(
                g_values=g_values,
                task_ids=task_ids,
                update_method=config.update_method,
                skip_threshold=config.adv_skip_threshold,
            )
        else:
            raise ValueError(f"Unknown advantage_mode: {config.advantage_mode}")
        advantages = adv_result.advantages
        skipped_tasks = adv_result.skipped_tasks
        per_task_g_mean = adv_result.per_task_g_mean

        skipped_task_set = set(skipped_tasks)
        if len(skipped_tasks) == len(task_specs):
            logger.info(
                f"Iter {iteration}: skipping update - all tasks have uniform rewards (successes={total_successes})"
            )
            log_data: dict[str, Any] = {
                f"{config.mode}/skipped_update": 1,
                f"{config.mode}/total_successes": total_successes,
                f"{config.mode}/dynamic_sampling/resampled_tasks": len(dynsample_retries),
                f"{config.mode}/dynamic_sampling/total_retries": sum(dynsample_retries.values()),
                f"{config.mode}/dynamic_sampling/gave_up_tasks": len(dynsample_gave_up),
                f"{config.mode}/iteration": iteration,
            }
            for _tid in per_task_g_mean:
                log_data[f"{config.mode}/{_tid}/g_mean"] = per_task_g_mean[_tid]
            metrics_logger.log(log_data)
            continue

        if skipped_tasks:
            logger.info(
                f"Iter {iteration}: dynamic rejection - skipped {len(skipped_tasks)} tasks "
                f"with uniform rewards: {skipped_tasks}"
            )

        # -- 4. Filter skipped-task trajectories & pre-sample noise ------
        active_mask = [t.task_id not in skipped_task_set for t in all_trajectories]
        active_trajs = [t for t, keep in zip(all_trajectories, active_mask, strict=True) if keep]
        active_advs = [a for a, keep in zip(advantages, active_mask, strict=True) if keep]
        active_instrs = [spec_lookup[t.task_id].instruction for t in active_trajs]

        # FPO must process ALL trajectories, not just active ones.
        # Zero-advantage (skipped-task) trajectories produce zero surrogate
        # loss but non-zero KL penalty, which prevents catastrophic
        # forgetting on solved tasks.  AWR/PPO only need active trajectories.
        policy.eval()
        is_fpo = config.update_method is UpdateMethod.FPO
        n_noise_samples = config.num_fm_noise_samples if is_fpo else 1
        update_trajs = all_trajectories if is_fpo else active_trajs
        update_advs = advantages if is_fpo else active_advs
        update_instrs = [spec_lookup[t.task_id].instruction for t in all_trajectories] if is_fpo else active_instrs
        update_noise: list[list[torch.Tensor]] = []
        update_time: list[list[torch.Tensor]] = []
        for traj in update_trajs:
            noise_list, time_list = _sample_fixed_noise_time(traj, policy, n_samples=n_noise_samples)
            update_noise.append(noise_list)
            update_time.append(time_list)

        # -- 5. Policy update ---------------------------------------------
        if config.update_method is UpdateMethod.AWR:
            assert ref_policy is not None
            update_metrics = awr_update(
                policy,
                ref_policy,
                sft_policy,
                optimizer,
                trainable,
                update_trajs,
                update_advs,
                update_instrs,
                [nl[0] for nl in update_noise],
                [tl[0] for tl in update_time],
                config,
            )
        elif config.update_method is UpdateMethod.FPO:
            update_metrics = fpo_update(
                policy,
                optimizer,
                trainable,
                update_trajs,
                update_advs,
                update_instrs,
                update_noise,
                update_time,
                config,
                ref_policy=ref_policy,
                sft_policy=sft_policy,
            )
        elif config.update_method is UpdateMethod.PPO:
            assert ref_policy is not None
            update_metrics = ppo_update(
                policy,
                ref_policy,
                sft_policy,
                optimizer,
                trainable,
                update_trajs,
                update_advs,
                update_instrs,
                [nl[0] for nl in update_noise],
                [tl[0] for tl in update_time],
                config,
            )
        else:
            raise ValueError(f"Unknown update_method: {config.update_method}")

        update_noise.clear()
        update_time.clear()

        # -- 6. Adaptive KL adjustment ------------------------------------
        if config.adaptive_kl and config.update_method is UpdateMethod.FPO and update_metrics.raw_kl > 0:
            old_kl_coeff = config.kl_coeff
            if update_metrics.raw_kl > 2.0 * config.kl_target:
                config.kl_coeff = min(config.kl_coeff * config.kl_adapt_factor, 1.0)
            elif update_metrics.raw_kl < 0.5 * config.kl_target:
                config.kl_coeff = max(config.kl_coeff / config.kl_adapt_factor, 1e-6)
            if config.kl_coeff != old_kl_coeff:
                logger.info(
                    f"Iter {iteration}: adaptive KL adjusted kl_coeff "
                    f"{old_kl_coeff:.6f} → {config.kl_coeff:.6f} "
                    f"(raw_kl={update_metrics.raw_kl:.6f}, target={config.kl_target:.4f})"
                )

        # -- 7. Logging ---------------------------------------------------
        M = len(all_trajectories)
        if config.update_method == UpdateMethod.AWR:
            loss_key = "awr_loss"
        elif config.update_method == UpdateMethod.FPO:
            loss_key = "fpo_loss"
        elif config.update_method == UpdateMethod.PPO:
            loss_key = "ppo_loss"
        else:
            raise ValueError(f"Unknown update_method: {config.update_method}")
        log_data = {
            f"{config.mode}/{loss_key}": update_metrics.avg_loss,
            f"{config.mode}/kl_penalty": update_metrics.avg_kl + update_metrics.avg_sft_kl,
            f"{config.mode}/step_kl_penalty": update_metrics.avg_kl,
            f"{config.mode}/sft_kl_penalty": update_metrics.avg_sft_kl,
            f"{config.mode}/total_successes": total_successes,
            f"{config.mode}/rollout_successes": total_rollout_successes,
            f"{config.mode}/replay_successes": sum(replay_counts.values()),
            f"{config.mode}/success_buffer_size": sum(len(buf) for buf in success_buffer.values()),
            f"{config.mode}/skipped_tasks": len(skipped_tasks),
            f"{config.mode}/dynamic_sampling/resampled_tasks": len(dynsample_retries),
            f"{config.mode}/dynamic_sampling/total_retries": sum(dynsample_retries.values()),
            f"{config.mode}/dynamic_sampling/gave_up_tasks": len(dynsample_gave_up),
            f"{config.mode}/iteration": iteration,
        }
        for _tid, n_retries in dynsample_retries.items():
            log_data[f"{config.mode}/{_tid}/dynamic_sampling/retries"] = n_retries
            log_data[f"{config.mode}/{_tid}/dynamic_sampling/gave_up"] = int(_tid in dynsample_gave_up)
        if config.update_method == UpdateMethod.AWR:
            log_data[f"{config.mode}/mean_weight"] = update_metrics.avg_weight
        elif config.update_method == UpdateMethod.FPO:
            log_data[f"{config.mode}/clip_frac"] = update_metrics.avg_weight
            log_data[f"{config.mode}/mean_shift"] = update_metrics.avg_shift
            log_data[f"{config.mode}/raw_kl"] = update_metrics.raw_kl
            log_data[f"{config.mode}/raw_sft_kl"] = update_metrics.raw_sft_kl
            log_data[f"{config.mode}/mean_ratio"] = update_metrics.mean_ratio
            log_data[f"{config.mode}/max_log_ratio"] = update_metrics.max_log_ratio
            log_data[f"{config.mode}/kl_coeff"] = config.kl_coeff
            log_data[f"{config.mode}/sft_kl_coeff"] = config.sft_kl_coeff

        for _tid, n_succ in per_task_successes.items():
            log_data[f"{config.mode}/{_tid}/successes"] = n_succ
            log_data[f"{config.mode}/{_tid}/g_mean"] = per_task_g_mean.get(_tid, 0.0)
            log_data[f"{config.mode}/{_tid}/success_rate_ema"] = success_rate_ema.get(_tid, 0.0)
            log_data[f"{config.mode}/{_tid}/success_buffer_size"] = len(success_buffer.get(_tid, []))
            log_data[f"{config.mode}/{_tid}/success_buffer_quota"] = buffer_quotas.get(_tid, 0)
            log_data[f"{config.mode}/{_tid}/replay_successes"] = replay_counts.get(_tid, 0)
        if all_diags is not None:
            for _tid, diag in all_diags.items():
                if diag is not None:
                    log_data.update(diag.as_dict(prefix=f"{config.mode}/{_tid}/cluster"))

        ratio_info = ""
        if config.update_method is UpdateMethod.FPO:
            ratio_info = (
                f"  raw_kl={update_metrics.raw_kl:.6f}"
                f"  raw_sft_kl={update_metrics.raw_sft_kl:.6f}"
                f"  shift={update_metrics.avg_shift:.6f}"
                f"  mean_r={update_metrics.mean_ratio:.4f}"
                f"  max_lr={update_metrics.max_log_ratio:.4f}"
                f"  kl_c={config.kl_coeff:.6f}"
                f"  sft_kl_c={config.sft_kl_coeff:.6f}"
            )
        logger.info(
            f"Iter {iteration}  {loss_key}={update_metrics.avg_loss:.6f}"
            f"  step_kl={update_metrics.avg_kl:.6f}"
            f"  sft_kl={update_metrics.avg_sft_kl:.6f}"
            f"  successes={total_successes}/{M}{ratio_info}"
        )
        for _tid in per_task_successes:
            logger.info(f"  [{_tid}] successes={per_task_successes[_tid]}  g_mean={per_task_g_mean.get(_tid, 0.0):.4f}")

        # Overwrite the rolling "last" checkpoint every iteration so that
        # interrupted jobs still leave behind the most recent trainable state.
        policy.save_checkpoint(save_path / "last")

        # -- 8. Periodic evaluation ---------------------------------------
        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            best_success = evaluate_and_checkpoint(
                policy,
                config,
                task_specs,
                iteration,
                save_path,
                best_success,
                log_data,
                rollout_engines=rollout_engines,
            )

        metrics_logger.log(log_data)

    policy.save_checkpoint(save_path / "last")
    for engine in rollout_engines.values():
        engine.close()
    return policy
