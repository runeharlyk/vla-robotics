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

    All tensors have shape ``(T, ...)`` where ``T`` is the number of
    recorded transitions.  In the default single-step rollout mode
    (``n_action_steps == 1``), ``T`` equals the number of env steps.
    With chunk execution (``n_action_steps > 1``), ``T`` equals the
    number of *decision points*, each of which drove ``n_action_steps``
    env steps (fewer only if the episode terminated mid-chunk).

    When chunk execution is active, ``executed_chunks`` and
    ``chunk_mask`` are populated.  The policy-update path can either use
    them directly, or reconstruct v28-style sliding-window targets from
    the flattened stream of actions actually executed in the environment.
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
    n_action_steps: int = 1
    executed_chunks: torch.Tensor | None = None
    chunk_mask: torch.Tensor | None = None


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


def collect_single_episode_chunked(
    adapter: SingleEnvAdapter,
    policy_chunk_fn: Any,
    instruction: str,
    max_steps: int,
    n_action_steps: int,
    seed: int | None = None,
) -> Trajectory:
    """Collect one episode using action-chunk execution.

    At each decision point the policy is queried once to produce a full
    chunk of shape ``(chunk_size, action_dim)``.  The first
    ``n_action_steps`` actions of that chunk are executed against the
    environment before the policy is queried again.  Only one transition
    is recorded per decision point (not per env step).  The executed
    chunk data lets the update reconstruct dense sliding-window targets
    from real env actions, or use the direct executed chunk with a mask.

    Args:
        adapter: Environment adapter.
        policy_chunk_fn: ``(image, instruction, state) -> chunk`` callable
            returning a ``(chunk_size, action_dim)`` (denormalised) tensor.
        instruction: Language instruction.
        max_steps: Maximum env steps per episode (NOT decision points).
        n_action_steps: How many actions to execute from each sampled
            chunk before re-planning.  Must be ``>= 1``.
        seed: Random seed for the env reset.

    Returns:
        A :class:`Trajectory` whose length is the number of decision
        points, with ``executed_chunks`` and ``chunk_mask`` populated.
    """
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")

    raw_obs = adapter.reset(seed)

    decision_imgs: list[torch.Tensor] = []
    decision_states: list[torch.Tensor] = []
    decision_chunks: list[torch.Tensor] = []
    decision_chunk_masks: list[torch.Tensor] = []
    decision_first_actions: list[torch.Tensor] = []
    decision_rewards: list[torch.Tensor] = []
    decision_dones: list[torch.Tensor] = []

    success = False
    steps_taken = 0

    while steps_taken < max_steps:
        img_t, state_t = adapter.obs_to_tensors(raw_obs)
        chunk = policy_chunk_fn(img_t, instruction, state_t)
        if chunk.ndim != 2:
            raise ValueError(f"policy_chunk_fn must return (chunk_size, action_dim), got shape {tuple(chunk.shape)}")
        chunk_cpu = chunk.detach().to("cpu").float()

        max_exec = min(n_action_steps, max_steps - steps_taken)
        executed_this_dec = 0
        reward_sum = 0.0
        last_done = False

        decision_imgs.append(img_t)
        decision_states.append(state_t)

        for k in range(max_exec):
            action = chunk_cpu[k]
            action_np = action_to_numpy(action)
            result = adapter.step(action_np)
            raw_obs = result.raw_obs

            executed_this_dec += 1
            steps_taken += 1
            reward_sum += float(result.reward)

            if result.success:
                success = True

            if result.terminated or result.truncated:
                last_done = True
                break

        first_action = chunk_cpu[0].clone()
        mask = torch.zeros(n_action_steps, dtype=torch.bool)
        mask[:executed_this_dec] = True

        padded_chunk = torch.zeros(n_action_steps, chunk_cpu.shape[-1], dtype=chunk_cpu.dtype)
        padded_chunk[:executed_this_dec] = chunk_cpu[:executed_this_dec]

        decision_first_actions.append(first_action)
        decision_chunks.append(padded_chunk)
        decision_chunk_masks.append(mask)
        decision_rewards.append(torch.tensor(reward_sum))
        decision_dones.append(torch.tensor(float(last_done)))

        if last_done:
            break

    T_dec = len(decision_imgs)
    return Trajectory(
        images=torch.stack(decision_imgs) if decision_imgs else torch.empty(0),
        states=torch.stack(decision_states) if decision_states else torch.empty(0),
        actions=torch.stack(decision_first_actions) if decision_first_actions else torch.empty(0),
        rewards=torch.stack(decision_rewards) if decision_rewards else torch.empty(0),
        dones=torch.stack(decision_dones) if decision_dones else torch.empty(0),
        success=success,
        length=T_dec,
        n_action_steps=n_action_steps,
        executed_chunks=torch.stack(decision_chunks) if decision_chunks else None,
        chunk_mask=torch.stack(decision_chunk_masks) if decision_chunk_masks else None,
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
