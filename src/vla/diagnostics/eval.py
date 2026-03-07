"""Simulator-agnostic evaluation utilities for VLA policies.

Supports both ManiSkill and Libero (and any future simulator implementing
the :class:`~vla.envs.base.SimEnv` protocol) through the
:class:`~vla.envs.base.SimEnvFactory` abstraction.

LIBERO evaluation can be vectorized across multiple subprocess environments
for significantly faster wall-clock time (see ``num_eval_envs`` in
:func:`evaluate_smolvla`).
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


def _evaluate_libero_vectorized(
    policy,
    suite: str,
    task_id: int,
    num_episodes: int,
    num_envs: int,
    seed: int,
    state_dim: int,
    image_size: int = 256,
) -> EvalMetrics:
    from vla.envs.libero import LIBERO_CAMERAS, LiberoEnvFactory
    from vla.rl.libero_rollout import LiberoVecEnv

    factory = LiberoEnvFactory(suite=suite, state_dim=state_dim, task_id=task_id)
    suite_name = factory._libero_name
    resolved_task_id = task_id if task_id is not None else 0

    vec_env = LiberoVecEnv(
        suite_name=suite_name,
        task_id=resolved_task_id,
        num_envs=num_envs,
        state_dim=state_dim,
        image_size=image_size,
        camera_name=LIBERO_CAMERAS,
    )

    max_steps = 280
    num_cameras = 2
    instruction = vec_env.task_description

    total_successes = 0
    total_rewards: list[float] = []
    total_lengths: list[int] = []

    remaining = num_episodes
    ep_seed = seed

    try:
        while remaining > 0:
            wave_n = min(num_envs, remaining)
            seeds = [ep_seed + i for i in range(num_envs)]
            obs_list = vec_env.reset(seeds)
            ep_seed += num_envs

            reward_accum = [0.0] * num_envs
            length_accum = [0] * num_envs
            success_flags = [False] * num_envs
            env_done = [i >= wave_n for i in range(num_envs)]

            for _step in range(max_steps):
                if all(env_done):
                    break

                all_imgs = []
                all_states = []
                for i in range(num_envs):
                    cam_tensors = []
                    for img_np in obs_list[i]["images"]:
                        cam_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
                    while len(cam_tensors) < num_cameras:
                        cam_tensors.append(
                            cam_tensors[-1].clone()
                            if cam_tensors
                            else torch.zeros(3, image_size, image_size, dtype=torch.uint8)
                        )
                    cam_tensors = cam_tensors[:num_cameras]
                    all_imgs.append(torch.stack(cam_tensors, dim=0))
                    all_states.append(torch.from_numpy(obs_list[i]["state"]))

                images_batch = torch.stack(all_imgs, dim=0)
                states_batch = torch.stack(all_states, dim=0)

                active_indices = [i for i in range(num_envs) if not env_done[i]]
                if not active_indices:
                    break

                active_imgs = images_batch[active_indices]
                active_states = states_batch[active_indices]

                with torch.no_grad():
                    active_actions = policy.predict_action_batch(active_imgs, instruction, active_states)

                if isinstance(active_actions, torch.Tensor):
                    active_actions_np = active_actions.detach().cpu().numpy()
                else:
                    active_actions_np = np.asarray(active_actions, dtype=np.float32)
                if active_actions_np.ndim == 1:
                    active_actions_np = active_actions_np[np.newaxis]

                action_dim = active_actions_np.shape[-1]
                actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
                for idx, env_i in enumerate(active_indices):
                    actions_np[env_i] = active_actions_np[idx]

                obs_list, rewards, terminateds, truncateds, infos = vec_env.step(actions_np)

                for env_i in active_indices:
                    reward_accum[env_i] += rewards[env_i]
                    length_accum[env_i] += 1
                    if infos[env_i].get("is_success", False):
                        success_flags[env_i] = True
                    if terminateds[env_i] or truncateds[env_i] or success_flags[env_i]:
                        env_done[env_i] = True

            for i in range(wave_n):
                if success_flags[i]:
                    total_successes += 1
                total_rewards.append(reward_accum[i])
                total_lengths.append(length_accum[i])

            remaining -= wave_n
    finally:
        vec_env.close()

    n = max(len(total_lengths), 1)
    lengths_sorted = sorted(total_lengths)
    mid = len(lengths_sorted) // 2
    if len(lengths_sorted) % 2 == 0 and len(lengths_sorted) >= 2:
        median_len = (lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2.0
    else:
        median_len = float(lengths_sorted[mid]) if lengths_sorted else 0.0

    return EvalMetrics(
        success_rate=total_successes / max(num_episodes, 1),
        mean_reward=sum(total_rewards) / n,
        mean_episode_length=sum(total_lengths) / n,
        median_episode_length=median_len,
        num_episodes=num_episodes,
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
    num_eval_envs: int = 1,
) -> EvalMetrics:
    """Convenience wrapper: evaluate a :class:`SmolVLAPolicy` in any simulator.

    Builds a :class:`SimEnvFactory` from the provided parameters and wraps
    ``policy.predict_action`` into the batch-dict interface expected by
    :func:`evaluate`.
    """
    sim = simulator.lower()

    if sim == "libero" and num_eval_envs > 1 and task_id is not None:
        logger.info(
            "Vectorized LIBERO eval: %d envs, %d episodes, task_id=%d",
            num_eval_envs,
            num_episodes,
            task_id,
        )
        return _evaluate_libero_vectorized(
            policy,
            suite=suite,
            task_id=task_id,
            num_episodes=num_episodes,
            num_envs=num_eval_envs,
            seed=seed,
            state_dim=policy.state_dim,
            image_size=image_size,
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
        image = torch.stack(cam_views, dim=0) if len(cam_views) > 1 else cam_views[0]
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
