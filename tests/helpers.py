"""Shared test helper utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def make_fake_pt(
    path: Path,
    num_episodes: int = 3,
    T: int = 5,
    action_dim: int = 8,
    state_dim: int = 14,
    instruction: str = "pick up",
    libero_task_id: int = 0,
) -> None:
    """Write a minimal fake preprocessed ``.pt`` file for use in tests.

    Args:
        path: Destination file path.
        num_episodes: Number of episodes to generate.
        T: Timesteps per episode.
        action_dim: Action vector dimensionality.
        state_dim: State vector dimensionality.
        instruction: Task instruction string stored in metadata.
        libero_task_id: LIBERO task index stored in metadata.
    """
    episodes = [
        {
            "images": torch.randint(0, 255, (T, 2, 3, 64, 64), dtype=torch.uint8),
            "states": torch.randn(T, state_dim),
            "actions": torch.randn(T, action_dim),
        }
        for _ in range(num_episodes)
    ]
    torch.save(
        {
            "metadata": {
                "env_id": "TestEnv-v1",
                "instruction": instruction,
                "action_dim": action_dim,
                "state_dim": state_dim,
                "image_size": 64,
                "control_mode": "pd_joint_delta_pos",
                "libero_task_id": libero_task_id,
            },
            "episodes": episodes,
        },
        path,
    )
