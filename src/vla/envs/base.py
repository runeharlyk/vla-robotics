from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SimEnv(Protocol):
    """Simulator environment interface.

    Each simulator backend (LIBERO, ManiSkill, …) implements this protocol
    so that evaluation and RL training code stay simulator-agnostic.
    """

    @property
    def task_description(self) -> str: ...

    @property
    def max_episode_steps(self) -> int: ...

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        """Reset the environment.

        Returns:
            (raw_obs, info) tuple.
        """
        ...

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one action.

        Returns:
            (raw_obs, reward, terminated, truncated, info)
        """
        ...

    def close(self) -> None: ...

    def obs_to_batch(
        self,
        raw_obs: dict,
        device: torch.device | None = None,
    ) -> dict:
        """Convert raw simulator observations into a model-ready batch dict.

        The returned dict should contain keys such as
        ``observation.images.<cam>``, ``observation.state``, and ``task``.
        """
        ...

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        """Extract an RGB frame from raw observations for video recording."""
        ...

    def is_success(self, info: dict) -> bool:
        """Check if the episode achieved success."""
        ...


class SimEnvFactory(Protocol):
    """Creates environments for a given simulator + task configuration."""

    def __call__(
        self,
        task_id: int,
        **kwargs: Any,
    ) -> SimEnv: ...

    @property
    def num_tasks(self) -> int: ...

    @property
    def suite_name(self) -> str: ...
