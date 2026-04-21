from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import torch

from vla.envs.base import SimEnv
from vla.envs.robocasa_runtime import configure_robocasa_runtime
from vla.utils.tensor import to_float01

logger = logging.getLogger(__name__)

ROBOCASA_CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
ROBOCASA_STATE_KEYS = (
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
)
ROBOCASA_DEFAULT_SPLIT = "all"
ROBOCASA_DEFAULT_STATE_DIM = 16
ROBOCASA_DEFAULT_ACTION_DIM = 12


def _resolve_registered_tasks() -> list[str]:
    configure_robocasa_runtime()
    import robocasa  # noqa: F401
    from robosuite.environments.base import REGISTERED_ENVS

    return sorted(str(name) for name in REGISTERED_ENVS if str(name) != "Kitchen")


def list_robocasa_tasks() -> list[str]:
    """Return the locally registered RoboCasa task names."""
    return _resolve_registered_tasks()


def _camel_to_instruction(name: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip().lower()
    return words or "complete the kitchen task"


def _flatten_action_space(space: Any) -> int:
    if hasattr(space, "spaces"):
        return sum(_flatten_action_space(child) for child in space.spaces.values())
    shape = getattr(space, "shape", None)
    if shape:
        size = int(np.prod(shape))
        if size > 0:
            return size
    return 1


class RoboCasaEnv(SimEnv):
    """Wrap RoboCasa's gym environment to satisfy the :class:`SimEnv` protocol."""

    def __init__(
        self,
        env_id: str,
        instruction: str = "",
        max_episode_steps: int | None = None,
        image_size: int = 256,
        layout_id: int | None = None,
        style_id: int | None = None,
        split: str = ROBOCASA_DEFAULT_SPLIT,
    ) -> None:
        configure_robocasa_runtime()

        import gymnasium as gym
        import robocasa.wrappers.gym_wrapper  # noqa: F401

        env_kwargs: dict[str, Any] = {
            "split": split,
            "camera_widths": image_size,
            "camera_heights": image_size,
        }
        if layout_id is not None:
            env_kwargs["layout_ids"] = layout_id
        if style_id is not None:
            env_kwargs["style_ids"] = style_id

        self._env = gym.make(f"robocasa/{env_id}", **env_kwargs)
        self._env_id = env_id
        self._instruction_override = instruction.strip()
        self._task_description = self._instruction_override or _camel_to_instruction(env_id)
        self._image_size = image_size
        self._layout_id = layout_id
        self._style_id = style_id
        self._split = split
        self._max_steps = int(max_episode_steps or getattr(self._env.unwrapped, "horizon", 500))
        self._action_dim = _flatten_action_space(self._env.action_space)

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return ROBOCASA_DEFAULT_STATE_DIM

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        raw_obs, info = self._env.reset(seed=seed)
        task_text = str(raw_obs.get("annotation.human.task_description", "")).strip()
        if task_text:
            self._task_description = self._instruction_override or task_text
        return raw_obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action_dict = self._action_to_dict(np.asarray(action, dtype=np.float32))
        raw_obs, reward, terminated, truncated, info = self._env.step(action_dict)
        return raw_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._env.close()

    def _action_to_dict(self, action: np.ndarray) -> dict[str, np.ndarray]:
        flat = action.reshape(-1)
        if flat.shape[0] != self._action_dim:
            raise ValueError(f"Expected RoboCasa action_dim={self._action_dim}, got {flat.shape[0]}")

        return {
            "action.gripper_close": flat[0:1],
            "action.end_effector_position": flat[1:4],
            "action.end_effector_rotation": flat[4:7],
            "action.base_motion": flat[7:11],
            "action.control_mode": flat[11:12],
        }

    def _state_vector(self, raw_obs: dict) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in ROBOCASA_STATE_KEYS:
            value = raw_obs.get(key)
            if value is None:
                continue
            parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros(ROBOCASA_DEFAULT_STATE_DIM, dtype=np.float32)
        state = np.concatenate(parts).astype(np.float32, copy=False)
        if state.shape[0] < ROBOCASA_DEFAULT_STATE_DIM:
            padded = np.zeros(ROBOCASA_DEFAULT_STATE_DIM, dtype=np.float32)
            padded[: state.shape[0]] = state
            return padded
        return state[:ROBOCASA_DEFAULT_STATE_DIM]

    def obs_to_batch(
        self,
        raw_obs: dict,
        device: torch.device | None = None,
    ) -> dict:
        batch: dict[str, object] = {}

        for camera_key in ROBOCASA_CAMERA_KEYS:
            image = raw_obs.get(camera_key)
            if image is None:
                continue
            img = torch.from_numpy(np.asarray(image, dtype=np.uint8))
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2).contiguous()
            img = to_float01(img)
            if device is not None:
                img = img.to(device, non_blocking=True)
            batch[f"observation.images.{camera_key.split('.', 1)[1]}"] = img

        state = torch.from_numpy(self._state_vector(raw_obs)).float().unsqueeze(0)
        if device is not None:
            state = state.to(device, non_blocking=True)
        batch["observation.state"] = state
        batch["task"] = [self.task_description]
        return batch

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        frames = [np.asarray(raw_obs[key], dtype=np.uint8) for key in ROBOCASA_CAMERA_KEYS if key in raw_obs]
        if not frames:
            return np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        if len(frames) == 1:
            return frames[0]
        return np.concatenate(frames, axis=1)

    def is_success(self, info: dict) -> bool:
        return bool(info.get("success", False))


def probe_robocasa_task(
    env_id: str,
    *,
    layout_id: int | None = None,
    style_id: int | None = None,
    split: str = ROBOCASA_DEFAULT_SPLIT,
    image_size: int = 256,
) -> dict[str, object]:
    """Instantiate one RoboCasa task and return its effective metadata."""
    env = RoboCasaEnv(
        env_id=env_id,
        image_size=image_size,
        layout_id=layout_id,
        style_id=style_id,
        split=split,
    )
    try:
        raw_obs, _ = env.reset(seed=0)
        task_text = str(raw_obs.get("annotation.human.task_description", "")).strip()
        return {
            "env_id": env_id,
            "instruction": task_text or env.task_description,
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "layout_id": layout_id,
            "style_id": style_id,
            "split": split,
        }
    finally:
        env.close()


class RoboCasaEnvFactory:
    """Creates :class:`RoboCasaEnv` instances for one or many RoboCasa tasks."""

    def __init__(
        self,
        env_id: str | None = None,
        instruction: str = "",
        max_episode_steps: int | None = None,
        image_size: int = 256,
        layout_id: int | None = None,
        style_id: int | None = None,
        split: str = ROBOCASA_DEFAULT_SPLIT,
    ) -> None:
        self._tasks = [env_id] if env_id is not None else list_robocasa_tasks()
        self._instruction = instruction
        self._max_episode_steps = max_episode_steps
        self._image_size = image_size
        self._layout_id = layout_id
        self._style_id = style_id
        self._split = split

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    @property
    def suite_name(self) -> str:
        return "robocasa"

    def __call__(self, task_id: int = 0, **kwargs: Any) -> RoboCasaEnv:
        resolved_env_id = self._tasks[task_id]
        return RoboCasaEnv(
            env_id=resolved_env_id,
            instruction=self._instruction,
            max_episode_steps=self._max_episode_steps,
            image_size=self._image_size,
            layout_id=self._layout_id,
            style_id=self._style_id,
            split=self._split,
            **kwargs,
        )
