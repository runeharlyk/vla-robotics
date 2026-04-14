from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from vla.envs.base import SimEnv
from vla.utils.tensor import to_float01

_logger = logging.getLogger(__name__)


def _has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


# pd_ee_delta_pose gives end-effector control (needs pinocchio / `pin`).
# pd_joint_delta_pos is a safe fallback that works everywhere.
if _has_pinocchio():
    DEFAULT_CONTROL_MODE = "pd_ee_delta_pose"
else:
    DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"
    _logger.warning(
        "pinocchio (`pin`) is not installed - falling back to '%s' control mode. "
        "On Linux, `uv sync` installs it automatically. "
        "On Windows, end-effector control (pd_ee_delta_pose) is not supported.",
        DEFAULT_CONTROL_MODE,
    )


class ManiSkillEnv(SimEnv):
    """Wraps a ManiSkill gymnasium environment to satisfy the :class:`SimEnv` protocol."""

    def __init__(
        self,
        env_id: str,
        obs_mode: str = "rgb",
        control_mode: str = DEFAULT_CONTROL_MODE,
        render_mode: str = "cameras",
        max_episode_steps: int | None = None,
        instruction: str = "",
        image_size: int = 256,
        shader: str = "default",
    ):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401, I001 - must register envs before gym.make

        self._env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            shader_dir=shader,
        )
        self._instruction = instruction or env_id
        self._image_size = image_size
        self._max_steps = max_episode_steps or getattr(self._env, "_max_episode_steps", 200)

    @property
    def task_description(self) -> str:
        return self._instruction

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        obs, info = self._env.reset(seed=seed)
        return self._unwrap_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._unwrap_obs(obs), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def _unwrap_obs(self, obs: dict) -> dict:
        """Normalise ManiSkill obs into a common raw dict."""
        out: dict = {"pixels": {}, "agent_state": None}

        # Some envs use 'image' (rgb), others use 'sensor_data' (rgb+state).
        image_dict: dict = {}
        if "image" in obs:
            image_dict = obs["image"]
        elif "sensor_data" in obs:
            image_dict = obs["sensor_data"]

        for k, v in image_dict.items():
            if isinstance(v, dict) and "rgb" in v:
                out["pixels"][k] = self._to_uint8(v["rgb"])

        if "agent" in obs:
            if isinstance(obs["agent"], dict):
                parts = []
                for key in ("qpos", "qvel"):
                    if key in obs["agent"]:
                        parts.append(self._to_numpy(obs["agent"][key]).flatten())
                if parts:
                    out["agent_state"] = np.concatenate(parts)
            else:
                out["agent_state"] = self._to_numpy(obs["agent"]).flatten()

        return out

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    @classmethod
    def _to_uint8(cls, arr: Any) -> np.ndarray:
        arr = cls._to_numpy(arr)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 4:
            arr = arr[0]
        return arr

    def obs_to_batch(
        self,
        raw_obs: dict,
        device: torch.device | None = None,
    ) -> dict:
        batch: dict = {}

        for cam_key, img_np in raw_obs.get("pixels", {}).items():
            img = to_float01(torch.from_numpy(img_np))
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[-1] in (3, 4):
                img = img[..., :3].permute(0, 3, 1, 2).contiguous()
            if device is not None:
                img = img.to(device, non_blocking=True)
            batch[f"observation.images.{cam_key}"] = img

        if raw_obs.get("agent_state") is not None:
            state = torch.from_numpy(raw_obs["agent_state"]).float().unsqueeze(0)
            if device is not None:
                state = state.to(device, non_blocking=True)
            batch["observation.state"] = state

        batch["task"] = [self.task_description]
        return batch

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        cams = list(raw_obs.get("pixels", {}).values())
        if not cams:
            return np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        if len(cams) == 1:
            return cams[0]
        return np.concatenate(cams, axis=1)

    def is_success(self, info: dict) -> bool:
        return bool(info.get("success", False))


class ManiSkillEnvFactory:
    """Creates :class:`ManiSkillEnv` instances for a single ManiSkill task."""

    def __init__(
        self,
        env_id: str,
        instruction: str = "",
        max_episode_steps: int | None = None,
        image_size: int = 256,
        obs_mode: str = "rgb",
        control_mode: str = DEFAULT_CONTROL_MODE,
    ):
        self._env_id = env_id
        self._instruction = instruction
        self._max_episode_steps = max_episode_steps
        self._image_size = image_size
        self._obs_mode = obs_mode
        self._control_mode = control_mode

    @property
    def num_tasks(self) -> int:
        return 1

    @property
    def suite_name(self) -> str:
        return self._env_id

    def __call__(self, task_id: int = 0, **kwargs: Any) -> ManiSkillEnv:
        return ManiSkillEnv(
            env_id=self._env_id,
            obs_mode=self._obs_mode,
            control_mode=self._control_mode,
            max_episode_steps=self._max_episode_steps,
            instruction=self._instruction,
            image_size=self._image_size,
            **kwargs,
        )
