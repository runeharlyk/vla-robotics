"""Simulator-agnostic evaluation utilities for VLA policies.

Supports both ManiSkill and Libero (and any future simulator implementing
the :class:`~vla.envs.base.SimEnv` protocol) through the
:class:`~vla.envs.base.SimEnvFactory` abstraction.

LIBERO evaluation can be vectorized across multiple subprocess environments
for significantly faster wall-clock time (see ``num_envs`` in
:func:`evaluate_smolvla`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vla.envs import SimEnvFactory, make_env_factory
from vla.rl.rollout import Trajectory
from vla.utils.tensor import action_to_numpy

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics over multiple episodes."""

    success_rate: float
    mean_reward: float
    mean_episode_length: float
    median_episode_length: float
    num_episodes: int
    successes: int


# ---------------------------------------------------------------------------
# Shared metrics aggregation
# ---------------------------------------------------------------------------


def _compute_eval_metrics(
    successes: int,
    rewards: Sequence[float],
    lengths: Sequence[int],
    num_episodes: int,
) -> EvalMetrics:
    """Build :class:`EvalMetrics` from raw aggregation lists."""
    n = max(len(lengths), 1)
    lengths_sorted = sorted(lengths)
    mid = len(lengths_sorted) // 2
    if len(lengths_sorted) % 2 == 0 and len(lengths_sorted) >= 2:
        median_len = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2.0
    else:
        median_len = float(lengths_sorted[mid]) if lengths_sorted else 0.0

    return EvalMetrics(
        success_rate=successes / max(num_episodes, 1),
        mean_reward=sum(rewards) / n,
        mean_episode_length=sum(lengths) / n,
        median_episode_length=median_len,
        num_episodes=num_episodes,
        successes=successes,
    )


def metrics_from_trajectories(
    trajectories: Sequence[Trajectory],
    expected_episodes: int | None = None,
) -> EvalMetrics:
    """Compute :class:`EvalMetrics` from collected :class:`Trajectory` objects.

    This avoids reimplementing the aggregation logic every time a rollout
    engine is used for evaluation.

    Args:
        trajectories: Collected episodes.
        expected_episodes: Override for the denominator in success-rate
            computation.  Defaults to ``len(trajectories)``.
    """
    total_successes = sum(1 for t in trajectories if t.success)
    total_rewards = [float(t.rewards.sum()) for t in trajectories]
    total_lengths = [t.length for t in trajectories]
    num_ep = expected_episodes if expected_episodes is not None else len(trajectories)
    return _compute_eval_metrics(total_successes, total_rewards, total_lengths, num_ep)


# ---------------------------------------------------------------------------
# Generic sequential evaluation
# ---------------------------------------------------------------------------


def evaluate(
    policy_fn: Callable[[dict], np.ndarray | torch.Tensor],
    env_factory: SimEnvFactory,
    num_episodes: int = 100,
    seed: int = 0,
    device: torch.device | str = "cpu",
    noise_reset_fn: Callable[[int], None] | None = None,
    task_metrics_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> EvalMetrics:
    """Evaluate a policy across all tasks exposed by *env_factory*.

    Args:
        policy_fn: ``(batch_dict) -> action``  where *batch_dict* comes from
            :meth:`SimEnv.obs_to_batch` and *action* is a numpy array or
            tensor of shape ``(action_dim,)`` or ``(1, action_dim)``.
        env_factory: Factory that creates simulator environments.
        num_episodes: Episodes per task.
        seed: Base random seed.
        device: Device used by ``obs_to_batch``.

    Returns:
        Aggregated :class:`EvalMetrics`.
    """
    if isinstance(device, str):
        device = torch.device(device)

    total_successes = 0
    total_rewards: list[float] = []
    total_lengths: list[int] = []

    for task_id in range(env_factory.num_tasks):
        env = env_factory(task_id)
        task_desc = env.task_description
        max_steps = env.max_episode_steps
        task_successes = 0
        task_rewards: list[float] = []
        task_lengths: list[int] = []

        for ep in range(num_episodes):
            ep_seed = seed + task_id * num_episodes + ep
            if noise_reset_fn is not None:
                noise_reset_fn(ep_seed)
            raw_obs, info = env.reset(seed=ep_seed)
            ep_reward = 0.0
            ep_len = 0
            success = False

            for _step in range(max_steps):
                batch = env.obs_to_batch(raw_obs, device=device)
                action = policy_fn(batch)

                action_np = action_to_numpy(action)

                raw_obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += float(reward)
                ep_len += 1

                if env.is_success(info):
                    success = True
                    break
                if terminated or truncated:
                    break

            if success:
                total_successes += 1
                task_successes += 1
            total_rewards.append(ep_reward)
            total_lengths.append(ep_len)
            task_rewards.append(ep_reward)
            task_lengths.append(ep_len)

        if task_metrics_callback is not None:
            task_metrics_callback(
                task_id,
                {
                    "task_id": task_id,
                    "task_description": task_desc,
                    "num_episodes": num_episodes,
                    "successes": task_successes,
                    "success_rate": task_successes / max(num_episodes, 1),
                    "mean_reward": sum(task_rewards) / max(len(task_rewards), 1),
                    "mean_episode_length": sum(task_lengths) / max(len(task_lengths), 1),
                },
            )

        env.close()

    total_ep = env_factory.num_tasks * num_episodes
    return _compute_eval_metrics(total_successes, total_rewards, total_lengths, total_ep)


# ---------------------------------------------------------------------------
# Vectorised LIBERO evaluation (delegates to LiberoRollout)
# ---------------------------------------------------------------------------


def _evaluate_libero_vectorized(
    policy,
    suite: str,
    task_id: int | None,
    num_episodes: int,
    num_envs: int,
    seed: int,
    state_dim: int,
    image_size: int = 256,
    max_steps: int = 280,
    fixed_noise_seed: int | None = None,
    n_action_steps: int = 1,
    task_metrics_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> EvalMetrics:
    from vla.rl.libero_rollout import LiberoRollout

    env_factory = make_env_factory("libero", suite=suite, state_dim=state_dim, task_id=task_id)
    task_ids = [task_id] if task_id is not None else list(range(env_factory.num_tasks))
    resolved_task_id = task_ids[0]
    rollout = LiberoRollout(
        suite_name=suite,
        task_id=resolved_task_id,
        num_envs=num_envs,
        max_steps=max_steps,
        image_size=image_size,
        state_dim=state_dim,
    )

    def _batch_fn(images: torch.Tensor, instr: str, states: torch.Tensor) -> torch.Tensor:
        return policy.predict_action_batch(images, instr, states)

    def _single_fn(image: torch.Tensor, instr: str, state: torch.Tensor) -> torch.Tensor:
        return policy.predict_action(image, instr, state)

    def _chunk_batch_fn(images: torch.Tensor, instr: str, states: torch.Tensor) -> torch.Tensor:
        return policy.predict_action_chunk_batch(images, instr, states)

    def _chunk_single_fn(image: torch.Tensor, instr: str, state: torch.Tensor) -> torch.Tensor:
        return policy.predict_action_chunk(image, instr, state)

    try:
        total_successes = 0
        total_rewards: list[float] = []
        total_lengths: list[int] = []
        for idx, current_task_id in enumerate(task_ids):
            if idx > 0:
                rollout.reconfigure(suite, current_task_id)
            if fixed_noise_seed is not None and hasattr(policy, "reset_eval_noise"):
                policy.reset_eval_noise(fixed_noise_seed + current_task_id * num_episodes)
            instruction = rollout.task_description
            task_seed = seed + current_task_id * num_episodes
            logger.info(
                "Starting LIBERO eval task %d/%d (task_id=%d, seed=%d)",
                idx + 1,
                len(task_ids),
                current_task_id,
                task_seed,
            )
            try:
                task_trajectories = rollout.collect_batch(
                    policy_fn=_single_fn,
                    instruction=instruction,
                    num_trajectories=num_episodes,
                    seed=task_seed,
                    policy_batch_fn=_batch_fn,
                    n_action_steps=n_action_steps,
                    policy_chunk_fn=_chunk_single_fn if n_action_steps > 1 else None,
                    policy_chunk_batch_fn=_chunk_batch_fn if n_action_steps > 1 else None,
                )
            except Exception:
                logger.exception("LIBERO eval failed on task_id=%d", current_task_id)
                raise

            task_successes = sum(1 for t in task_trajectories if t.success)
            task_rewards = [float(t.rewards.sum()) for t in task_trajectories]
            task_lengths = [t.length for t in task_trajectories]
            total_successes += task_successes
            total_rewards.extend(task_rewards)
            total_lengths.extend(task_lengths)
            logger.info(
                "Finished LIBERO eval task %d/%d (task_id=%d): %.2f%% success (%d/%d)",
                idx + 1,
                len(task_ids),
                current_task_id,
                100.0 * task_successes / max(num_episodes, 1),
                task_successes,
                num_episodes,
            )
            if task_metrics_callback is not None:
                task_metrics_callback(
                    current_task_id,
                    {
                        "task_id": current_task_id,
                        "task_description": instruction,
                        "task_index": idx,
                        "tasks_total": len(task_ids),
                        "num_episodes": num_episodes,
                        "successes": task_successes,
                        "success_rate": task_successes / max(num_episodes, 1),
                        "mean_reward": sum(task_rewards) / max(len(task_rewards), 1),
                        "mean_episode_length": sum(task_lengths) / max(len(task_lengths), 1),
                    },
                )
    finally:
        rollout.close()

    expected_episodes = num_episodes * len(task_ids)
    return _compute_eval_metrics(total_successes, total_rewards, total_lengths, expected_episodes)


def evaluate_smolvla(
    policy,
    instruction: str,
    simulator: str = "maniskill",
    env_id: str = "PickCube-v1",
    num_episodes: int = 100,
    max_steps: int = 280,
    seed: int = 0,
    control_mode: str = "pd_joint_delta_pos",
    suite: str = "all",
    image_size: int = 256,
    task_id: int | None = None,
    num_envs: int = 1,
    fixed_noise_seed: int | None = None,
    n_action_steps: int = 1,
    task_metrics_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> EvalMetrics:
    """Convenience wrapper: evaluate a :class:`SmolVLAPolicy` in any simulator.

    Builds a :class:`SimEnvFactory` from the provided parameters and wraps
    ``policy.predict_action`` into the batch-dict interface expected by
    :func:`evaluate`.
    """
    sim = simulator.lower()

    if hasattr(policy, "set_eval_fixed_noise"):
        policy.set_eval_fixed_noise(fixed_noise_seed)

    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")

    if sim == "libero" and (num_envs > 1 or n_action_steps > 1):
        logger.info(
            "LIBERO rollout eval: %d envs, %d episodes, task_id=%s, n_action_steps=%d",
            num_envs,
            num_episodes,
            task_id,
            n_action_steps,
        )
        return _evaluate_libero_vectorized(
            policy,
            suite=suite,
            task_id=task_id,
            num_episodes=num_episodes,
            num_envs=num_envs,
            seed=seed,
            state_dim=policy.state_dim,
            image_size=image_size,
            max_steps=max_steps,
            fixed_noise_seed=fixed_noise_seed,
            n_action_steps=n_action_steps,
            task_metrics_callback=task_metrics_callback,
        )

    factory_kwargs: dict = {}
    if sim == "maniskill":
        factory_kwargs.update(
            env_id=env_id,
            instruction=instruction,
            max_episode_steps=max_steps,
            image_size=image_size,
            control_mode=control_mode,
        )
    elif sim == "libero":
        factory_kwargs.update(suite=suite, state_dim=policy.state_dim)
        if task_id is not None:
            factory_kwargs["task_id"] = task_id
    else:
        raise ValueError(f"Unknown simulator {simulator!r}")

    env_factory = make_env_factory(sim, **factory_kwargs)
    device = policy.device

    def _policy_fn(batch: dict) -> torch.Tensor:
        image_keys = sorted(k for k in batch if k.startswith("observation.images."))
        if not image_keys:
            raise ValueError(f"No observation.images.* in batch. Keys: {list(batch.keys())}")
        cam_views = []
        for k in image_keys:
            img = batch[k]
            if img.ndim in (4, 5):
                img = img[0]
            if img.ndim == 2:
                img = img.unsqueeze(0)
            cam_views.append(img)
        # Keep a batch dimension so multi-view LIBERO observations remain
        # multi-view all the way into ``predict_action_batch``.
        image = torch.stack(cam_views, dim=0).unsqueeze(0) if len(cam_views) > 1 else cam_views[0].unsqueeze(0)
        state = batch.get("observation.state")
        if state is not None and state.ndim == 2:
            state = state[0]
        if state is not None:
            state = state.unsqueeze(0)
        task = batch.get("task", instruction)
        if isinstance(task, (list, tuple)):
            task = task[0]
        return policy.predict_action_batch(image, task, state)[0]

    def _noise_reset(ep_seed: int) -> None:
        if fixed_noise_seed is None or not hasattr(policy, "reset_eval_noise"):
            return
        policy.reset_eval_noise(fixed_noise_seed + ep_seed)

    return evaluate(
        _policy_fn,
        env_factory,
        num_episodes=num_episodes,
        seed=seed,
        device=device,
        noise_reset_fn=_noise_reset if fixed_noise_seed is not None else None,
        task_metrics_callback=task_metrics_callback,
    )


def print_metrics(metrics: EvalMetrics, tag: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}Episodes: {metrics.num_episodes}")
    print(f"{prefix}Success rate: {metrics.success_rate:.2%} ({metrics.successes}/{metrics.num_episodes})")
    print(f"{prefix}Mean reward: {metrics.mean_reward:.4f}")
    print(f"{prefix}Mean episode length: {metrics.mean_episode_length:.1f}")
    print(f"{prefix}Median episode length: {metrics.median_episode_length:.1f}")
