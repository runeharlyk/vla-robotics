"""Shared rollout abstractions for VLA trajectory collection.

Defines the simulator-agnostic data types and protocols that all rollout
backends (ManiSkill, LIBERO, …) share:

* :class:`Trajectory` - a single episode's data.
* :class:`RolloutEngine` - the ``collect_batch`` / ``close`` protocol.
* :class:`SingleEnvAdapter` - single-env reset/step adapter.
* :func:`collect_single_episode` - shared step loop.
* :func:`collect_batch_sequential` - sequential fallback.

Simulator-specific implementations live in their own modules:

* :mod:`vla.rl.maniskill_rollout` - ManiSkill GPU-vectorised rollout.
* :mod:`vla.rl.libero_rollout` - LIBERO subprocess-vectorised rollout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from vla.utils.tensor import action_to_numpy

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A single episode trajectory.

    All tensors have shape ``(T, ...)``.
    """

    images: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    success: bool
    privileged_states: list[dict[str, float]] = field(default_factory=list)
    length: int = 0
    task_id: str = ""


@runtime_checkable
class RolloutEngine(Protocol):
    """Interface that all rollout backends must implement."""

    def collect_batch(
        self,
        policy_fn: Any,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
        policy_batch_fn: Any = None,
    ) -> list[Trajectory]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Shared single-episode collection
# ---------------------------------------------------------------------------


@dataclass
class SingleStepResult:
    """Normalised result from a single env step."""

    raw_obs: Any
    reward: float
    terminated: bool
    truncated: bool
    success: bool


@runtime_checkable
class SingleEnvAdapter(Protocol):
    """Thin adapter that normalises single-env reset/step interfaces.

    Implementations wrap a concrete environment (ManiSkill ``gymnasium.Env``
    or a :class:`~vla.rl.libero_rollout.LiberoVecEnv` with ``num_envs=1``)
    and expose a uniform API for :func:`collect_single_episode`.
    """

    def reset(self, seed: int | None) -> Any:
        """Reset the environment and return raw observations."""
        ...

    def obs_to_tensors(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw observations to ``(images, state)`` tensors.

        Returns:
            images: ``(num_cameras, C, H, W)`` uint8 tensor.
            state: ``(state_dim,)`` float32 tensor.
        """
        ...

    def step(self, action: np.ndarray) -> SingleStepResult:
        """Step the environment with a flat action array."""
        ...


def collect_single_episode(
    adapter: SingleEnvAdapter,
    policy_fn: Any,
    instruction: str,
    max_steps: int,
    seed: int | None = None,
) -> Trajectory:
    """Collect one episode using the shared step loop.

    This function eliminates duplication between ManiSkill and LIBERO
    single-episode collection.  Each backend provides a thin
    :class:`SingleEnvAdapter` that handles the env-specific details.

    Args:
        adapter: Environment adapter.
        policy_fn: ``(image, instruction, state) -> action`` callable.
        instruction: Language instruction.
        max_steps: Maximum episode length.
        seed: Random seed for the env reset.

    Returns:
        A :class:`Trajectory` for the episode.
    """
    raw_obs = adapter.reset(seed)
    images, states, actions_list, rewards_list, dones_list = [], [], [], [], []
    success = False

    for _step in range(max_steps):
        img_t, state_t = adapter.obs_to_tensors(raw_obs)
        action = policy_fn(img_t, instruction, state_t)
        action_np = action_to_numpy(action)

        images.append(img_t)
        states.append(state_t)
        actions_list.append(torch.from_numpy(action_np.copy()))

        result = adapter.step(action_np)
        raw_obs = result.raw_obs

        rewards_list.append(torch.tensor(result.reward))
        done = result.terminated or result.truncated
        dones_list.append(torch.tensor(float(done)))

        if result.success:
            success = True
        if done:
            break

    T = len(images)
    return Trajectory(
        images=torch.stack(images) if images else torch.empty(0),
        states=torch.stack(states) if states else torch.empty(0),
        actions=torch.stack(actions_list) if actions_list else torch.empty(0),
        rewards=torch.stack(rewards_list) if rewards_list else torch.empty(0),
        dones=torch.stack(dones_list) if dones_list else torch.empty(0),
        success=success,
        length=T,
    )


def collect_batch_sequential(
    collect_fn: Any,
    num_trajectories: int,
    seed: int | None,
) -> list[Trajectory]:
    """Sequential fallback for ``collect_batch`` - shared across backends.

    Args:
        collect_fn: ``(seed) -> Trajectory`` callable.
        num_trajectories: How many episodes to collect.
        seed: Base seed (incremented per episode).

    Returns:
        List of :class:`Trajectory` objects.
    """
    trajectories: list[Trajectory] = []
    for i in range(num_trajectories):
        s = (seed + i) if seed is not None else None
        trajectories.append(collect_fn(s))
    return trajectories
