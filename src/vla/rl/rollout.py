"""ManiSkill rollout engine for collecting trajectories with a VLA policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401 – registers envs
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A single episode trajectory collected from ManiSkill.

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


class ManiSkillRollout:
    """Vectorised ManiSkill rollout engine.

    Args:
        env_id: ManiSkill environment id (e.g. ``PickCube-v1``).
        num_envs: Number of parallel environments.
        max_steps: Maximum steps per episode.
        image_size: Rendered image resolution.
        obs_mode: ManiSkill observation mode.
        control_mode: Robot control mode.
        sim_backend: Simulation backend (``physx_cpu`` or ``gpu``).
    """

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        num_envs: int = 1,
        max_steps: int = 200,
        image_size: int = 256,
        obs_mode: str = "rgb+state",
        control_mode: str = "pd_joint_delta_pos",
        sim_backend: str = "physx_cpu",
        num_cameras: int = 2,
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.image_size = image_size
        self.num_cameras = max(1, num_cameras)
        self._warned_camera_fallback = False
        self._warned_render_fallback = False

        render_backend = "cpu" if sim_backend == "physx_cpu" else "gpu"
        self.env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array",
            sim_backend=sim_backend,
            render_backend=render_backend,
            sensor_configs={"width": image_size, "height": image_size},
            max_episode_steps=max_steps,
        )

    @staticmethod
    def _frame_to_rgb_list(frame: Any) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        if frame is None:
            return frames
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        if isinstance(frame, dict):
            for value in frame.values():
                frames.extend(ManiSkillRollout._frame_to_rgb_list(value))
            return frames
        if isinstance(frame, (list, tuple)):
            for value in frame:
                frames.extend(ManiSkillRollout._frame_to_rgb_list(value))
            return frames

        arr = np.asarray(frame)
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                frames.append(arr.astype(np.uint8))
            elif arr.shape[0] == 3:
                frames.append(np.transpose(arr, (1, 2, 0)).astype(np.uint8))
            return frames
        if arr.ndim == 4:
            if arr.shape[-1] == 3:
                if arr.shape[0] == 1:
                    frames.append(arr[0].astype(np.uint8))
                else:
                    frames.extend([arr[i].astype(np.uint8) for i in range(arr.shape[0])])
            elif arr.shape[1] == 3:
                frames.extend([np.transpose(arr[i], (1, 2, 0)).astype(np.uint8) for i in range(arr.shape[0])])
            return frames
        if arr.ndim == 5 and arr.shape[-1] == 3:
            if arr.shape[0] == 1:
                frames.extend([arr[0, i].astype(np.uint8) for i in range(arr.shape[1])])
            return frames
        return frames

    def _render_images(self) -> list[np.ndarray]:
        """Render the current scene and return a list of ``(H, W, 3)`` uint8 images."""
        frame = self.env.render()
        frames = self._frame_to_rgb_list(frame)
        if not frames:
            raise RuntimeError("Could not extract RGB frame(s) from environment render output")
        return frames

    @staticmethod
    def _flatten_obs(obs: dict | np.ndarray) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
        if hasattr(obs, "cpu"):
            obs_np: Any = obs.cpu().numpy()
            return obs_np.flatten().astype(np.float32)
        if "state" in obs:
            x = obs["state"]
        elif "state_dict" in obs:
            parts = []
            for v in obs["state_dict"].values():
                arr = np.asarray(v).flatten()
                parts.append(arr)
            x = np.concatenate(parts)
        elif "agent" in obs:
            agent = obs["agent"]
            x = np.concatenate([np.asarray(agent["qpos"]).flatten(), np.asarray(agent["qvel"]).flatten()])
        else:
            x = np.zeros(1, dtype=np.float32)
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32).flatten()

    @staticmethod
    def _extract_sensor_rgbs(obs: dict | np.ndarray) -> list[np.ndarray]:
        if not isinstance(obs, dict) or "sensor_data" not in obs:
            return []
        sensor_data = obs["sensor_data"]
        if not isinstance(sensor_data, dict):
            return []
        frames: list[np.ndarray] = []
        for cam_data in sensor_data.values():
            if not isinstance(cam_data, dict) or "rgb" not in cam_data:
                continue
            rgb = cam_data["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            arr = np.asarray(rgb)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                frames.append(arr.astype(np.uint8))
        return frames

    def _get_camera_images(self, obs: dict | np.ndarray) -> list[np.ndarray]:
        sensor_frames = self._extract_sensor_rgbs(obs)
        if len(sensor_frames) >= self.num_cameras:
            return sensor_frames[: self.num_cameras]
        render_frames = self._render_images()
        if len(sensor_frames) < self.num_cameras and render_frames and not self._warned_render_fallback:
            logger.warning(
                "Only %d sensor camera view(s) available, but %d requested. "
                "Using render() views to fill remaining camera slots before duplication fallback.",
                len(sensor_frames),
                self.num_cameras,
            )
            self._warned_render_fallback = True
        frames = sensor_frames + render_frames
        if not frames:
            raise RuntimeError("Could not extract any camera images from sensor_data or render output")
        if len(frames) < self.num_cameras:
            if not self._warned_camera_fallback:
                logger.warning(
                    "Only %d total camera view(s) available after sensor+render extraction, but %d requested. "
                    "Falling back by duplicating the last camera view.",
                    len(frames),
                    self.num_cameras,
                )
                self._warned_camera_fallback = True
            frames.extend([frames[-1].copy() for _ in range(self.num_cameras - len(frames))])
        return frames[: self.num_cameras]

    @staticmethod
    def _extract_privileged(info: dict) -> dict[str, float]:
        """Extract privileged state features from ManiSkill info for SRPO Tier A."""
        priv: dict[str, float] = {}
        for key in ("is_grasped", "is_obj_placed", "success"):
            if key in info:
                val = info[key]
                if hasattr(val, "item"):
                    val = val.item()
                priv[key] = float(val)
        return priv

    def collect_trajectory(
        self,
        policy_fn: callable,
        instruction: str,
        seed: int | None = None,
    ) -> Trajectory:
        """Roll out one episode using ``policy_fn`` and return a :class:`Trajectory`.

        Args:
            policy_fn: Callable ``(image_tensor_CHW, instruction) -> action_tensor``.
            instruction: Language instruction for the policy.
            seed: Optional reset seed.
        """
        obs, info = self.env.reset(seed=seed)
        images, states, actions_list, rewards_list, dones_list = [], [], [], [], []
        privileged_states: list[dict[str, float]] = []
        success = False

        for _step in range(self.max_steps):
            rgbs = self._get_camera_images(obs)
            from PIL import Image as PILImage

            cam_tensors = []
            for rgb in rgbs:
                pil = PILImage.fromarray(rgb).resize((self.image_size, self.image_size), PILImage.BILINEAR)
                img_np = np.array(pil, dtype=np.uint8)
                cam_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
            img_t = torch.stack(cam_tensors, dim=0)
            state = self._flatten_obs(obs)
            state_t = torch.from_numpy(state)

            action = policy_fn(img_t, instruction, state_t)
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy().flatten()
            else:
                action_np = np.asarray(action, dtype=np.float32).flatten()

            images.append(img_t)
            states.append(torch.from_numpy(state))
            actions_list.append(torch.from_numpy(action_np.copy()))
            privileged_states.append(self._extract_privileged(info))

            action_env = action_np.reshape(1, -1)
            obs, reward, terminated, truncated, info = self.env.step(action_env)

            rew_val = float(reward.item()) if hasattr(reward, "item") else float(reward)
            rewards_list.append(torch.tensor(rew_val))
            done = bool(terminated) or bool(truncated)
            if hasattr(terminated, "item"):
                done = bool(terminated.item()) or bool(truncated.item())
            dones_list.append(torch.tensor(float(done)))

            if "success" in info:
                succ = info["success"]
                if hasattr(succ, "item"):
                    succ = succ.item()
                if bool(succ):
                    success = True

            if done:
                break

        T = len(images)
        return Trajectory(
            images=torch.stack(images),
            states=torch.stack(states),
            actions=torch.stack(actions_list),
            rewards=torch.stack(rewards_list),
            dones=torch.stack(dones_list),
            success=success,
            privileged_states=privileged_states,
            length=T,
        )

    def collect_batch(
        self,
        policy_fn: callable,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
    ) -> list[Trajectory]:
        """Collect multiple trajectories sequentially.

        Args:
            policy_fn: Callable ``(image_tensor, instruction) -> action_tensor``.
            instruction: Language instruction.
            num_trajectories: How many episodes to collect.
            seed: Base seed (each episode uses ``seed + i``).

        Returns:
            List of :class:`Trajectory` objects.
        """
        trajectories = []
        for i in range(num_trajectories):
            s = (seed + i) if seed is not None else None
            traj = self.collect_trajectory(policy_fn, instruction, seed=s)
            trajectories.append(traj)
        return trajectories

    def close(self) -> None:
        self.env.close()
