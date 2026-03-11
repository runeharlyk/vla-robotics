"""Shared vectorised environment abstraction and wave-loop collection.

Provides a :class:`VecEnvAdapter` protocol that normalises the differences
between ManiSkill (GPU-batched) and LIBERO (subprocess-based) vectorised
environments.  The generic :func:`collect_wave` and
:func:`collect_trajectories_vectorized` functions implement the wave-stepping
loop once, eliminating duplication across rollout backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from vla.rl.rollout import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Normalised result from a single vectorised env step."""

    raw_obs: Any
    rewards: list[float]
    terminateds: list[bool]
    truncateds: list[bool]
    successes: list[bool]


@runtime_checkable
class VecEnvAdapter(Protocol):
    """Thin adapter that normalises vectorised env interfaces.

    Implementations wrap a concrete vectorised environment (ManiSkill
    batched ``gymnasium.Env`` or :class:`~vla.rl.libero_rollout.LiberoVecEnv`)
    and expose a uniform reset / observe / step API.
    """

    @property
    def num_envs(self) -> int: ...

    def reset(self, seed: int | None) -> Any:
        """Reset all environments.  Returns raw observations."""
        ...

    def extract_batch_obs(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw observations to ``(images, states)`` tensors.

        Returns:
            images: ``(N, num_cameras, C, H, W)`` uint8 tensor.
            states: ``(N, state_dim)`` float32 tensor.
        """
        ...

    def step(self, actions: np.ndarray) -> StepResult:
        """Step all environments with ``(N, action_dim)`` actions."""
        ...


def collect_wave(
    adapter: VecEnvAdapter,
    policy_batch_fn: Any,
    instruction: str,
    active_n: int,
    seed: int | None,
    max_steps: int,
) -> list[Trajectory]:
    """Run one wave of ``active_n`` parallel episodes to completion.

    This is the core vectorised collection loop shared by all rollout
    backends.  It handles per-environment bookkeeping, active-index
    masking, and trajectory assembly.

    Args:
        adapter: Vectorised environment adapter.
        policy_batch_fn: Batched callable
            ``(images_BNCHW, instruction, states_BS) -> actions_BA``.
        instruction: Language instruction forwarded to the policy.
        active_n: How many of the ``adapter.num_envs`` environments
            should actually collect trajectories (the rest idle).
        seed: Base random seed for the resets.
        max_steps: Maximum episode length.

    Returns:
        List of :class:`Trajectory` objects (up to ``active_n``).
    """
    N = adapter.num_envs

    raw_obs = adapter.reset(seed)

    img_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
    state_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
    action_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
    reward_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
    done_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
    success_flags = [False] * N
    env_done = [i >= active_n for i in range(N)]

    for _step in range(max_steps):
        if all(env_done):
            break

        images_batch, states_batch = adapter.extract_batch_obs(raw_obs)

        active_indices = [i for i in range(N) if not env_done[i]]
        if not active_indices:
            break

        active_imgs = images_batch[active_indices]
        active_states = states_batch[active_indices]

        with torch.no_grad():
            active_actions = policy_batch_fn(active_imgs, instruction, active_states)

        if isinstance(active_actions, torch.Tensor):
            active_actions_np = active_actions.detach().cpu().numpy()
        else:
            active_actions_np = np.asarray(active_actions, dtype=np.float32)
        if active_actions_np.ndim == 1:
            active_actions_np = active_actions_np[np.newaxis]

        action_dim = active_actions_np.shape[-1]
        actions_np = np.zeros((N, action_dim), dtype=np.float32)
        for idx, env_i in enumerate(active_indices):
            actions_np[env_i] = active_actions_np[idx]

        for idx, env_i in enumerate(active_indices):
            img_bufs[env_i].append(images_batch[env_i])
            state_bufs[env_i].append(states_batch[env_i])
            action_bufs[env_i].append(torch.from_numpy(active_actions_np[idx].copy()))

        result = adapter.step(actions_np)
        raw_obs = result.raw_obs

        for env_i in active_indices:
            reward_bufs[env_i].append(torch.tensor(result.rewards[env_i]))
            step_done = result.terminateds[env_i] or result.truncateds[env_i]
            done_bufs[env_i].append(torch.tensor(float(step_done)))

            if result.successes[env_i]:
                success_flags[env_i] = True

            if step_done:
                env_done[env_i] = True

    trajectories: list[Trajectory] = []
    for i in range(active_n):
        T = len(img_bufs[i])
        if T == 0:
            continue
        trajectories.append(
            Trajectory(
                images=torch.stack(img_bufs[i]),
                states=torch.stack(state_bufs[i]),
                actions=torch.stack(action_bufs[i]),
                rewards=torch.stack(reward_bufs[i]),
                dones=torch.stack(done_bufs[i]),
                success=success_flags[i],
                length=T,
            )
        )

    return trajectories


def collect_trajectories_vectorized(
    adapter: VecEnvAdapter,
    policy_batch_fn: Any,
    instruction: str,
    num_trajectories: int,
    seed: int | None,
    max_steps: int,
) -> list[Trajectory]:
    """Collect ``num_trajectories`` via multiple waves of parallel episodes.

    Reuses the ``adapter.num_envs`` environments across waves until
    enough trajectories have been gathered.
    """
    N = adapter.num_envs
    all_trajectories: list[Trajectory] = []

    remaining = num_trajectories
    wave_idx = 0

    while remaining > 0:
        active_n = min(N, remaining)
        wave_seed = (seed + wave_idx * N) if seed is not None else None
        wave_trajs = collect_wave(adapter, policy_batch_fn, instruction, active_n, wave_seed, max_steps)
        all_trajectories.extend(wave_trajs)
        remaining -= len(wave_trajs)
        wave_idx += 1

    return all_trajectories[:num_trajectories]
