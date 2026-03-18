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
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from vla.constants import AdvantageMode, Simulator, UpdateMethod
from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.models.world_model import WorldModelEncoder, build_world_model
from vla.rl.advantage import leave_one_out_advantages_per_task, normalize_advantages_per_task
from vla.rl.config import SRPOConfig, TaskSpec
from vla.rl.maniskill_rollout import ManiSkillRollout
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
    "ppo_update",
    "fpo_update",
    "train_srpo",
]


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


def collect_all_trajectories(
    policy: SmolVLAPolicy,
    task_specs: list[TaskSpec],
    rollout_engines: dict[str, RolloutEngine],
    config: SRPOConfig,
    iteration: int,
    trajs_per_task: int,
) -> tuple[list[Trajectory], dict[str, int]]:
    """Collect rollout trajectories from all tasks.

    Creates rollout engines lazily if they are missing from the map.

    Returns:
        Tuple of (all_trajectories, per_task_successes).
    """
    policy.eval()
    all_trajectories: list[Trajectory] = []
    per_task_successes: dict[str, int] = {}

    for spec in task_specs:
        if spec.task_id not in rollout_engines:
            logger.info(f"Creating rollout engine for {spec.task_id}")
            rollout_engines[spec.task_id] = build_rollout_engine(config, spec=spec)
        use_vectorized = config.num_rollout_envs > 1
        trajs = rollout_engines[spec.task_id].collect_batch(
            policy_fn=policy.predict_action,
            instruction=spec.instruction,
            num_trajectories=trajs_per_task,
            seed=config.seed + iteration * 1000,
            policy_batch_fn=policy.predict_action_batch if use_vectorized else None,
        )
        for t in trajs:
            t.task_id = spec.task_id
        all_trajectories.extend(trajs)
        per_task_successes[spec.task_id] = sum(1 for t in trajs if t.success)

    return all_trajectories, per_task_successes


def evaluate_and_checkpoint(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    task_specs: list[TaskSpec],
    iteration: int,
    save_path: Path,
    best_success: float,
    log_data: dict[str, Any],
) -> float:
    """Run periodic evaluation and save the best checkpoint.

    Returns:
        Updated best success rate.
    """
    prev_eval_zero_sample = policy.eval_zero_sample
    policy.eval_zero_sample = True
    try:
        if config.simulator == "libero":
            task_sr_sum = 0.0
            for spec in task_specs:
                metrics = evaluate_smolvla(
                    policy,
                    instruction=spec.instruction,
                    simulator="libero",
                    num_episodes=config.eval_episodes,
                    max_steps=config.max_steps,
                    seed=config.seed + 20000,
                    suite=config.suite,
                    task_id=spec.libero_task_idx,
                    num_eval_envs=config.num_eval_envs,
                )
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
        else:
            task_sr_sum = 0.0
            for spec in task_specs:
                metrics = evaluate_smolvla(
                    policy,
                    instruction=spec.instruction,
                    simulator=config.simulator,
                    env_id=spec.env_id or config.env_id,
                    num_episodes=config.eval_episodes,
                    max_steps=config.max_steps,
                    seed=config.seed + 20000,
                )
                print_metrics(
                    metrics,
                    tag=f"{config.mode} iter {iteration} [{spec.task_id}]",
                )
                log_data[f"{config.mode}/{spec.task_id}/eval/success_rate"] = metrics.success_rate
                task_sr_sum += metrics.success_rate
            avg_sr = task_sr_sum / len(task_specs)
            best_success = save_best_checkpoint(
                avg_sr,
                best_success,
                lambda: policy.save_checkpoint(save_path / "best"),
                tag=config.mode,
            )
    finally:
        policy.eval_zero_sample = prev_eval_zero_sample

    return best_success


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

    needs_ref = config.update_method is not UpdateMethod.FPO
    ref_policy: SmolVLAPolicy | None = None
    if needs_ref:
        ref_policy = copy.deepcopy(policy)
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    # -- Per-task rollout engines (created lazily to avoid OOM) ----------
    if rollout_engines is None:
        rollout_engines = {}

    spec_lookup: dict[str, TaskSpec] = {s.task_id: s for s in task_specs}

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
        )
        reward_model = MultiTaskWorldProgressReward(world_encoder, reward_cfg)

        if demo_trajectories:
            for tid, demos in demo_trajectories.items():
                demo_images = []
                for dt in demos:
                    imgs = dt.images[: dt.length]
                    imgs = to_float01(imgs)
                    demo_images.append(imgs)
                reward_model.add_demo_trajectories(tid, demo_images)

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if metrics_logger is None:
        metrics_logger = MetricsLogger(jsonl_path=save_path / "metrics.jsonl")
    best_success = -1.0

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
    )
    metrics_logger.log(log_data)

    for iteration in trange(1, config.num_iterations + 1, desc="Training iterations"):
        # -- 1. Collect trajectories from all tasks -----------------------
        all_trajectories, per_task_successes = collect_all_trajectories(
            policy,
            task_specs,
            rollout_engines,
            config,
            iteration,
            trajs_per_task_per_iter,
        )

        total_successes = sum(per_task_successes.values())
        logger.info(
            f"Iter {iteration}: collected {len(all_trajectories)} trajs across "
            f"{len(task_specs)} tasks, {total_successes} successes "
            f"({per_task_successes})"
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

            for tid, embs in by_task_embs.items():
                reward_model.add_successful_embeddings(tid, embs)
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
                f"{config.mode}/iteration": iteration,
            }
            for tid in per_task_g_mean:
                log_data[f"{config.mode}/{tid}/g_mean"] = per_task_g_mean[tid]
            metrics_logger.log(log_data)
            continue

        if skipped_tasks:
            logger.info(
                f"Iter {iteration}: dynamic rejection - skipped {len(skipped_tasks)} tasks "
                f"with uniform rewards: {skipped_tasks}"
            )

        # -- 4. Pre-sample fixed noise/time for FM loss computation ------
        policy.eval()
        instrs_per_traj: list[str] = []
        for traj in all_trajectories:
            instrs_per_traj.append(spec_lookup[traj.task_id].instruction)

        n_noise_samples = config.num_fm_noise_samples if config.update_method is UpdateMethod.FPO else 1
        fixed_noise_per_traj: list[list[torch.Tensor]] = []
        fixed_time_per_traj: list[list[torch.Tensor]] = []
        for traj in all_trajectories:
            noise_list, time_list = _sample_fixed_noise_time(traj, policy, n_samples=n_noise_samples)
            fixed_noise_per_traj.append(noise_list)
            fixed_time_per_traj.append(time_list)

        # -- 5. Policy update ---------------------------------------------
        if config.update_method is UpdateMethod.AWR:
            assert ref_policy is not None
            noise_single = [nl[0] for nl in fixed_noise_per_traj]
            time_single = [tl[0] for tl in fixed_time_per_traj]
            update_metrics = awr_update(
                policy,
                ref_policy,
                optimizer,
                trainable,
                all_trajectories,
                advantages,
                instrs_per_traj,
                noise_single,
                time_single,
                skipped_task_set,
                config,
            )
        elif config.update_method is UpdateMethod.FPO:
            update_metrics = fpo_update(
                policy,
                optimizer,
                trainable,
                all_trajectories,
                advantages,
                instrs_per_traj,
                fixed_noise_per_traj,
                fixed_time_per_traj,
                config,
            )
        elif config.update_method is UpdateMethod.PPO:
            assert ref_policy is not None
            noise_single = [nl[0] for nl in fixed_noise_per_traj]
            time_single = [tl[0] for tl in fixed_time_per_traj]
            update_metrics = ppo_update(
                policy,
                ref_policy,
                optimizer,
                trainable,
                all_trajectories,
                advantages,
                instrs_per_traj,
                noise_single,
                time_single,
                skipped_task_set,
                config,
            )
        else:
            raise ValueError(f"Unknown update_method: {config.update_method}")

        fixed_noise_per_traj.clear()
        fixed_time_per_traj.clear()

        # -- 6. Logging ---------------------------------------------------
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
            f"{config.mode}/kl_penalty": update_metrics.avg_kl,
            f"{config.mode}/total_successes": total_successes,
            f"{config.mode}/skipped_tasks": len(skipped_tasks),
            f"{config.mode}/iteration": iteration,
        }
        if config.update_method == UpdateMethod.AWR:
            log_data[f"{config.mode}/mean_weight"] = update_metrics.avg_weight
        elif config.update_method == UpdateMethod.FPO:
            log_data[f"{config.mode}/clip_frac"] = update_metrics.avg_weight

        for tid, n_succ in per_task_successes.items():
            log_data[f"{config.mode}/{tid}/successes"] = n_succ
            log_data[f"{config.mode}/{tid}/g_mean"] = per_task_g_mean.get(tid, 0.0)
        if all_diags is not None:
            for tid, diag in all_diags.items():
                if diag is not None:
                    log_data.update(diag.as_dict(prefix=f"{config.mode}/{tid}/cluster"))

        logger.info(
            f"Iter {iteration}  {loss_key}={update_metrics.avg_loss:.6f}  kl={update_metrics.avg_kl:.6f}"
            f"  successes={total_successes}/{M}"
        )
        for tid in per_task_successes:
            logger.info(f"  [{tid}] successes={per_task_successes[tid]}  g_mean={per_task_g_mean.get(tid, 0.0):.4f}")

        # -- 7. Periodic evaluation ---------------------------------------
        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            best_success = evaluate_and_checkpoint(
                policy,
                config,
                task_specs,
                iteration,
                save_path,
                best_success,
                log_data,
            )

        metrics_logger.log(log_data)

    policy.save_checkpoint(save_path / "last")
    for engine in rollout_engines.values():
        engine.close()
    return policy
