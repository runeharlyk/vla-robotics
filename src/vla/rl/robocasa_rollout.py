"""Sequential RoboCasa rollout engine.

RoboCasa does not currently have a repo-native vectorized rollout path here,
so collection reuses a single environment sequentially. This is sufficient
for sparse-RL training and smoke testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vla.envs.robocasa import RoboCasaEnv
from vla.rl.rollout import (
    SingleStepResult,
    Trajectory,
    collect_batch_sequential,
    collect_single_episode,
)


def _batch_to_replay_obs(batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError(f"No observation.images.* keys found in RoboCasa batch: {list(batch.keys())}")

    camera_views: list[torch.Tensor] = []
    for key in image_keys:
        image = batch[key]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected tensor for {key}, got {type(image)!r}")
        if image.ndim == 4:
            image = image[0]
        camera_views.append((image.detach().cpu() * 255.0).round().clamp(0, 255).to(torch.uint8))

    state = batch.get("observation.state")
    if not isinstance(state, torch.Tensor):
        state_t = torch.zeros(1, dtype=torch.float32)
    else:
        if state.ndim == 2:
            state = state[0]
        state_t = state.detach().cpu().float()

    return torch.stack(camera_views, dim=0), state_t


class RoboCasaRollout:
    """Sequential rollout engine for RoboCasa tasks."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        max_steps: int = 300,
        image_size: int = 256,
        layout_id: int | None = None,
        style_id: int | None = None,
        split: str = "all",
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self._env = RoboCasaEnv(
            env_id=env_id,
            max_episode_steps=max_steps,
            image_size=image_size,
            layout_id=layout_id,
            style_id=style_id,
            split=split,
        )

    @property
    def task_description(self) -> str:
        return self._env.task_description

    def collect_trajectory(
        self,
        policy_fn: Any,
        instruction: str,
        seed: int | None = None,
    ) -> Trajectory:
        return collect_single_episode(
            _RoboCasaSingleAdapter(self._env),
            policy_fn,
            instruction,
            self.max_steps,
            seed,
        )

    def collect_batch(
        self,
        policy_fn: Any,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
        policy_batch_fn: Any = None,
    ) -> list[Trajectory]:
        return collect_batch_sequential(
            lambda s: self.collect_trajectory(policy_fn, instruction, seed=s),
            num_trajectories,
            seed,
        )

    def close(self) -> None:
        self._env.close()


class _RoboCasaSingleAdapter:
    """Adapts :class:`RoboCasaEnv` to the shared single-env rollout loop."""

    def __init__(self, env: RoboCasaEnv) -> None:
        self._env = env
        self._last_info: dict[str, object] = {}

    def reset(self, seed: int | None) -> Any:
        raw_obs, info = self._env.reset(seed=0 if seed is None else seed)
        self._last_info = info
        return raw_obs

    def obs_to_tensors(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._env.obs_to_batch(raw_obs)
        return _batch_to_replay_obs(batch)

    def step(self, action: np.ndarray) -> SingleStepResult:
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_info = info
        return SingleStepResult(
            raw_obs=raw_obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=self._env.is_success(info),
        )
