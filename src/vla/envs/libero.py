from __future__ import annotations

import importlib
import logging
from typing import Any

import einops
import numpy as np
import torch

from vla.constants import SUITE_MAP
from vla.envs.base import SimEnv
from vla.envs.libero_runtime import configure_libero_runtime
from vla.utils.tensor import to_float01

configure_libero_runtime()

_lerobot_libero = importlib.import_module("lerobot.envs.libero")
_processor_mod = importlib.import_module("lerobot.processor.env_processor")

_LeRobotLiberoEnv = _lerobot_libero.LiberoEnv
_get_suite = _lerobot_libero._get_suite
LiberoProcessorStep = _processor_mod.LiberoProcessorStep

logger = logging.getLogger(__name__)

_PROC = LiberoProcessorStep()

LIBERO_CAMERAS = "agentview_image,robot0_eye_in_hand_image"


class LiberoEnv(SimEnv):
    """Wraps LeRobot's LiberoEnv to satisfy the :class:`SimEnv` protocol."""

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        obs_type: str = "pixels_agent_pos",
        state_dim: int = 8,
        camera_name: str | list[str] = LIBERO_CAMERAS,
    ):
        self._suite = _get_suite(suite_name)
        if isinstance(camera_name, str):
            camera_name = sorted([c.strip() for c in camera_name.split(",") if c.strip()])

        self._env = _LeRobotLiberoEnv(
            task_suite=self._suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type=obs_type,
            camera_name=camera_name,
        )
        logger.info(
            "LiberoEnv created: suite=%s, task_id=%d, camera_name=%r",
            suite_name,
            task_id,
            camera_name,
        )
        self._state_dim = state_dim

    @property
    def task_description(self) -> str:
        return self._env.task_description

    @property
    def max_episode_steps(self) -> int:
        return self._env._max_episode_steps

    def _resolve_init_state_id(self, seed: int | None) -> int | None:
        init_states = getattr(self._env, "_init_states", None)
        if init_states is None:
            return None

        num_init_states = len(init_states)
        if num_init_states <= 0:
            return None

        if seed is None:
            current = int(getattr(self._env, "_init_state_id", 0))
            return current % num_init_states

        rng = np.random.RandomState(int(seed))
        return int(rng.randint(num_init_states))

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        init_state_id = self._resolve_init_state_id(seed)
        if init_state_id is not None and hasattr(self._env, "_init_state_id"):
            self._env._init_state_id = init_state_id

        obs, info = self._env.reset(seed=seed)
        if init_state_id is not None:
            info = dict(info)
            info["libero_init_state_id"] = init_state_id
            info["libero_num_init_states"] = len(self._env._init_states)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()

    def obs_to_batch(
        self,
        raw_obs: dict,
        device: torch.device | None = None,
    ) -> dict:
        batch: dict = {}

        if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
            for cam_key, img_np in raw_obs["pixels"].items():
                img = torch.from_numpy(img_np)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
                img = to_float01(img)
                img = torch.flip(img, dims=[2, 3])
                if device is not None:
                    img = img.to(device, non_blocking=True)
                batch[f"observation.images.{cam_key}"] = img

        if "robot_state" in raw_obs:
            rs = raw_obs["robot_state"]
            eef_pos = torch.from_numpy(rs["eef"]["pos"]).float().unsqueeze(0)
            eef_quat = torch.from_numpy(rs["eef"]["quat"]).float().unsqueeze(0)
            gripper_qpos = torch.from_numpy(rs["gripper"]["qpos"]).float().unsqueeze(0)
            eef_axisangle = _PROC._quat2axisangle(eef_quat)
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)
            if self._state_dim < state.shape[-1]:
                state = state[..., : self._state_dim]
            if device is not None:
                state = state.to(device, non_blocking=True)
            batch["observation.state"] = state

        batch["task"] = [self.task_description]
        return batch

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        if "pixels" not in raw_obs or not isinstance(raw_obs["pixels"], dict):
            return np.zeros((256, 256, 3), dtype=np.uint8)
        cams = list(raw_obs["pixels"].values())
        flipped = [np.flip(c, axis=(0, 1)).copy() for c in cams]
        if len(flipped) == 1:
            return flipped[0]
        return np.concatenate(flipped, axis=1)

    def is_success(self, info: dict) -> bool:
        return info.get("is_success", False)


class LiberoEnvFactory:
    """Creates :class:`LiberoEnv` instances for each task in a LIBERO suite."""

    def __init__(self, suite: str, state_dim: int = 8, task_id: int | None = None):
        self._suite_key = suite.lower()
        self._libero_name = SUITE_MAP.get(self._suite_key, f"libero_{self._suite_key}")
        self._benchmark = _get_suite(self._libero_name)
        self._state_dim = state_dim
        self._single_task_id = task_id

    @property
    def num_tasks(self) -> int:
        if self._single_task_id is not None:
            return 1
        return len(self._benchmark.tasks)

    @property
    def suite_name(self) -> str:
        return self._suite_key

    def __call__(self, task_id: int, **kwargs: Any) -> LiberoEnv:
        resolved_id = task_id
        if self._single_task_id is not None:
            resolved_id = self._single_task_id
        return LiberoEnv(
            suite_name=self._libero_name,
            task_id=resolved_id,
            state_dim=self._state_dim,
            **kwargs,
        )
