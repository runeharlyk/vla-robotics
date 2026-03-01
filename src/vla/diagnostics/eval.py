"""Evaluation utilities for ManiSkill VLA policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vla.rl.rollout import ManiSkillRollout, Trajectory


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
    policy_fn: callable,
    instruction: str,
    env_id: str = "PickCube-v1",
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    image_size: int = 256,
) -> EvalMetrics:
    """Evaluate a policy in ManiSkill and return aggregated metrics.

    Args:
        policy_fn: Callable ``(image_tensor, instruction) -> action_tensor``.
        instruction: Language instruction for the task.
        env_id: ManiSkill environment id.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        seed: Base random seed.
        image_size: Rendered image resolution.

    Returns:
        :class:`EvalMetrics` with success rate, rewards, and episode lengths.
    """
    rollout = ManiSkillRollout(
        env_id=env_id,
        num_envs=1,
        max_steps=max_steps,
        image_size=image_size,
    )

    trajectories: list[Trajectory] = []
    for i in range(num_episodes):
        traj = rollout.collect_trajectory(policy_fn, instruction, seed=seed + i)
        trajectories.append(traj)

    rollout.close()

    successes = sum(1 for t in trajectories if t.success)
    total_rewards = [t.rewards.sum().item() for t in trajectories]
    lengths = [t.length for t in trajectories]
    lengths_sorted = sorted(lengths)
    mid = len(lengths_sorted) // 2
    if len(lengths_sorted) % 2 == 0:
        median_len = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2.0
    else:
        median_len = float(lengths_sorted[mid])

    return EvalMetrics(
        success_rate=successes / max(num_episodes, 1),
        mean_reward=sum(total_rewards) / max(num_episodes, 1),
        mean_episode_length=sum(lengths) / max(num_episodes, 1),
        median_episode_length=median_len,
        num_episodes=num_episodes,
        successes=successes,
    )


def print_metrics(metrics: EvalMetrics, tag: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}Episodes: {metrics.num_episodes}")
    print(f"{prefix}Success rate: {metrics.success_rate:.2%} ({metrics.successes}/{metrics.num_episodes})")
    print(f"{prefix}Mean reward: {metrics.mean_reward:.4f}")
    print(f"{prefix}Mean episode length: {metrics.mean_episode_length:.1f}")
    print(f"{prefix}Median episode length: {metrics.median_episode_length:.1f}")
