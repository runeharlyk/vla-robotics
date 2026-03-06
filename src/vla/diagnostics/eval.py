"""Simulator-agnostic evaluation utilities for VLA policies.

Supports both ManiSkill and Libero (and any future simulator implementing
the :class:`~vla.envs.base.SimEnv` protocol) through the
:class:`~vla.envs.base.SimEnvFactory` abstraction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vla.envs import SimEnvFactory, make_env_factory

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


def evaluate(
    policy_fn: Callable[[dict], np.ndarray | torch.Tensor],
    env_factory: SimEnvFactory,
    num_episodes: int = 100,
    seed: int = 0,
    device: torch.device | str = "cpu",
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
        max_steps = env.max_episode_steps

        for ep in range(num_episodes):
            raw_obs, info = env.reset(seed=seed + task_id * num_episodes + ep)
            ep_reward = 0.0
            ep_len = 0
            success = False

            for _step in range(max_steps):
                batch = env.obs_to_batch(raw_obs, device=device)
                action = policy_fn(batch)

                if isinstance(action, torch.Tensor):
                    action_np = action.detach().cpu().numpy()
                else:
                    action_np = np.asarray(action, dtype=np.float32)
                action_np = action_np.flatten()

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
            total_rewards.append(ep_reward)
            total_lengths.append(ep_len)

        env.close()

    n = max(len(total_lengths), 1)
    lengths_sorted = sorted(total_lengths)
    mid = len(lengths_sorted) // 2
    if len(lengths_sorted) % 2 == 0 and len(lengths_sorted) >= 2:
        median_len = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2.0
    else:
        median_len = float(lengths_sorted[mid]) if lengths_sorted else 0.0

    total_ep = env_factory.num_tasks * num_episodes
    return EvalMetrics(
        success_rate=total_successes / max(total_ep, 1),
        mean_reward=sum(total_rewards) / n,
        mean_episode_length=sum(total_lengths) / n,
        median_episode_length=median_len,
        num_episodes=total_ep,
        successes=total_successes,
    )


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
) -> EvalMetrics:
    """Convenience wrapper: evaluate a :class:`SmolVLAPolicy` in any simulator.

    Builds a :class:`SimEnvFactory` from the provided parameters and wraps
    ``policy.predict_action`` into the batch-dict interface expected by
    :func:`evaluate`.
    """
    factory_kwargs: dict = {}
    sim = simulator.lower()
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
        image_key = next((k for k in batch if k.startswith("observation.images.")), None)
        if image_key is None:
            raise ValueError(f"No observation.images.* in batch. Keys: {list(batch.keys())}")
        image = batch[image_key]
        if image.ndim in (4, 5):
            image = image[0]
        state = batch.get("observation.state")
        if state is not None and state.ndim == 2:
            state = state[0]
        task = batch.get("task", instruction)
        if isinstance(task, (list, tuple)):
            task = task[0]
        return policy.predict_action(image, task, state)

    return evaluate(_policy_fn, env_factory, num_episodes=num_episodes, seed=seed, device=device)


def print_metrics(metrics: EvalMetrics, tag: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}Episodes: {metrics.num_episodes}")
    print(f"{prefix}Success rate: {metrics.success_rate:.2%} ({metrics.successes}/{metrics.num_episodes})")
    print(f"{prefix}Mean reward: {metrics.mean_reward:.4f}")
    print(f"{prefix}Mean episode length: {metrics.mean_episode_length:.1f}")
    print(f"{prefix}Median episode length: {metrics.median_episode_length:.1f}")
