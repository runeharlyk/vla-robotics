"""ManiSkill vectorised rollout engine.

Uses ManiSkill 3's native GPU-vectorised simulation (``num_envs > 1``)
for parallel trajectory collection.  Implements the
:class:`~vla.rl.rollout.RolloutEngine` interface so that the SRPO
trainer remains simulator-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers envs
import numpy as np
import torch

from vla.rl.rollout import (
    SingleStepResult,
    Trajectory,
    collect_batch_sequential,
    collect_single_episode,
    collect_single_episode_chunked,
)
from vla.utils.camera import pad_camera_views

logger = logging.getLogger(__name__)


class ManiSkillRollout:
    """Vectorised ManiSkill rollout engine.

    When ``num_envs > 1`` ManiSkill runs all environments in parallel (CPU
    or GPU physics) and returns batched observations/rewards/dones.  This
    class exploits that to collect ``num_envs`` trajectories simultaneously
    with **batched** policy inference, giving a near-linear speedup.

    Args:
        env_id: ManiSkill environment id (e.g. ``PickCube-v1``).
        num_envs: Number of parallel environments.
        max_steps: Maximum steps per episode.
        image_size: Rendered image resolution.
        obs_mode: ManiSkill observation mode.
        control_mode: Robot control mode.
        sim_backend: Simulation backend (``physx_cpu`` or ``gpu``).
        num_cameras: Expected number of camera views.
    """

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        num_envs: int = 1,
        max_steps: int = 200,
        image_size: int = 256,
        obs_mode: str = "rgb+state",
        control_mode: str = "pd_joint_delta_pos",
        sim_backend: str = "auto",
        num_cameras: int = 2,
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.image_size = image_size
        self.num_cameras = max(1, num_cameras)
        self._warned_camera_fallback = False
        self._warned_render_fallback = False

        resolved_backend = sim_backend
        if resolved_backend == "auto":
            resolved_backend = "physx_cpu" if num_envs == 1 else "physx_cuda"

        render_backend = "cpu" if resolved_backend == "physx_cpu" else "gpu"
        self.env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array",
            sim_backend=resolved_backend,
            render_backend=render_backend,
            sensor_configs={"width": image_size, "height": image_size},
            max_episode_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Observation extraction helpers
    # ------------------------------------------------------------------

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
        if len(frames) < self.num_cameras and not self._warned_camera_fallback:
            logger.warning(
                "Only %d total camera view(s) available after sensor+render extraction, but %d requested. "
                "Falling back by duplicating the last camera view.",
                len(frames),
                self.num_cameras,
            )
            self._warned_camera_fallback = True
        return pad_camera_views(frames, self.num_cameras)

    @staticmethod
    def _extract_privileged(info: dict) -> dict[str, float]:
        priv: dict[str, float] = {}
        for key in ("is_grasped", "is_obj_placed", "success"):
            if key in info:
                val = info[key]
                if hasattr(val, "item"):
                    val = val.item()
                priv[key] = float(val)
        return priv

    # ------------------------------------------------------------------
    # Batched observation extraction (num_envs > 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sensor_rgbs_batched(obs: dict) -> list[np.ndarray]:
        """Extract per-camera RGB arrays with shape ``(N, H, W, 3)``."""
        if not isinstance(obs, dict) or "sensor_data" not in obs:
            return []
        sensor_data = obs["sensor_data"]
        if not isinstance(sensor_data, dict):
            return []
        cam_batches: list[np.ndarray] = []
        for cam_data in sensor_data.values():
            if not isinstance(cam_data, dict) or "rgb" not in cam_data:
                continue
            rgb = cam_data["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr[np.newaxis]
            if arr.ndim == 4 and arr.shape[-1] == 3:
                cam_batches.append(arr.astype(np.uint8))
        return cam_batches

    @staticmethod
    def _flatten_obs_batched(obs: dict, num_envs: int) -> np.ndarray:
        """Return ``(N, state_dim)`` float32 state array."""
        if "agent" in obs:
            agent = obs["agent"]
            if isinstance(agent, dict):
                parts = []
                for key in ("qpos", "qvel"):
                    if key in agent:
                        v = agent[key]
                        if hasattr(v, "cpu"):
                            v = v.cpu().numpy()
                        v = np.asarray(v, dtype=np.float32)
                        if v.ndim == 1:
                            v = v[np.newaxis]
                        parts.append(v)
                if parts:
                    return np.concatenate(parts, axis=-1)
        return np.zeros((num_envs, 1), dtype=np.float32)

    def _extract_batched_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract ``(N, num_cameras, C, H, W)`` images and ``(N, state_dim)`` states."""
        from PIL import Image as PILImage

        N = self.num_envs
        cam_batches = self._extract_sensor_rgbs_batched(obs)
        default_batch = np.zeros((N, self.image_size, self.image_size, 3), dtype=np.uint8)
        cam_batches = pad_camera_views(cam_batches, self.num_cameras, default=default_batch)

        all_imgs = []
        for env_i in range(N):
            cam_tensors = []
            for cam_arr in cam_batches:
                rgb = cam_arr[env_i]
                pil = PILImage.fromarray(rgb).resize((self.image_size, self.image_size), PILImage.BILINEAR)
                img_np = np.array(pil, dtype=np.uint8)
                cam_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
            all_imgs.append(torch.stack(cam_tensors, dim=0))

        images = torch.stack(all_imgs, dim=0)
        states_np = self._flatten_obs_batched(obs, N)
        states = torch.from_numpy(states_np)
        return images, states

    @staticmethod
    def _extract_success_batched(info: dict, num_envs: int) -> list[bool]:
        if "success" not in info:
            return [False] * num_envs
        s = info["success"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        s = np.asarray(s).flatten()
        return [bool(s[i]) for i in range(min(len(s), num_envs))]

    @staticmethod
    def _extract_done_batched(terminated: Any, truncated: Any, num_envs: int) -> tuple[list[bool], list[bool]]:
        if hasattr(terminated, "cpu"):
            terminated = terminated.cpu().numpy()
        if hasattr(truncated, "cpu"):
            truncated = truncated.cpu().numpy()
        t = np.asarray(terminated).flatten()
        tr = np.asarray(truncated).flatten()
        return (
            [bool(t[i]) for i in range(min(len(t), num_envs))],
            [bool(tr[i]) for i in range(min(len(tr), num_envs))],
        )

    @staticmethod
    def _extract_rewards_batched(reward: Any, num_envs: int) -> list[float]:
        if hasattr(reward, "cpu"):
            reward = reward.cpu().numpy()
        r = np.asarray(reward, dtype=np.float32).flatten()
        return [float(r[i]) for i in range(min(len(r), num_envs))]

    # ------------------------------------------------------------------
    # Single-env trajectory collection
    # ------------------------------------------------------------------

    def _make_single_adapter(self) -> _ManiSkillSingleAdapter:
        return _ManiSkillSingleAdapter(self)

    def collect_trajectory(
        self,
        policy_fn: Any,
        instruction: str,
        seed: int | None = None,
    ) -> Trajectory:
        """Roll out one episode using ``policy_fn`` and return a :class:`Trajectory`."""
        return collect_single_episode(
            self._make_single_adapter(),
            policy_fn,
            instruction,
            self.max_steps,
            seed,
        )

    # ------------------------------------------------------------------
    # Vectorised batch collection
    # ------------------------------------------------------------------

    def collect_batch(
        self,
        policy_fn: Any,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
        policy_batch_fn: Any | None = None,
        n_action_steps: int = 1,
        policy_chunk_fn: Any | None = None,
        policy_chunk_batch_fn: Any | None = None,
    ) -> list[Trajectory]:
        """Collect trajectories - vectorised when ``num_envs > 1``.

        When ``policy_batch_fn`` is provided and ``self.num_envs > 1``,
        all environments are stepped in parallel with batched policy
        inference.  Otherwise falls back to sequential collection via
        :func:`collect_batch_sequential`.  When ``n_action_steps > 1``
        the chunk-aware policy callables are required and the chunked
        rollout paths are used.
        """
        chunked = n_action_steps > 1
        if chunked and policy_chunk_fn is None and policy_chunk_batch_fn is None:
            raise ValueError("n_action_steps > 1 requires policy_chunk_fn or policy_chunk_batch_fn")

        if self.num_envs > 1 and (policy_batch_fn is not None or policy_chunk_batch_fn is not None):
            return self._collect_batch_vectorized(
                policy_batch_fn,
                instruction,
                num_trajectories,
                seed,
                n_action_steps=n_action_steps,
                policy_chunk_batch_fn=policy_chunk_batch_fn,
            )
        if chunked:
            adapter = self._make_single_adapter()
            return collect_batch_sequential(
                lambda s: collect_single_episode_chunked(
                    adapter, policy_chunk_fn, instruction, self.max_steps, n_action_steps, s
                ),
                num_trajectories,
                seed,
            )
        return collect_batch_sequential(
            lambda s: self.collect_trajectory(policy_fn, instruction, seed=s),
            num_trajectories,
            seed,
        )

    def _collect_batch_vectorized(
        self,
        policy_batch_fn: Any,
        instruction: str,
        num_trajectories: int,
        seed: int | None,
        n_action_steps: int = 1,
        policy_chunk_batch_fn: Any | None = None,
    ) -> list[Trajectory]:
        """Collect ``num_trajectories`` episodes using all ``num_envs`` in parallel.

        If ``num_trajectories > num_envs`` the environments are reused
        across multiple waves.
        """
        from vla.rl.vec_env import collect_trajectories_vectorized

        adapter = _ManiSkillVecAdapter(self)
        return collect_trajectories_vectorized(
            adapter,
            policy_batch_fn,
            instruction,
            num_trajectories,
            seed,
            self.max_steps,
            n_action_steps=n_action_steps,
            policy_chunk_batch_fn=policy_chunk_batch_fn,
        )

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------------------
# SingleEnvAdapter for the shared episode loop
# ---------------------------------------------------------------------------


class _ManiSkillSingleAdapter:
    """Adapts :class:`ManiSkillRollout` to the :class:`SingleEnvAdapter` protocol."""

    def __init__(self, rollout: ManiSkillRollout) -> None:
        self._r = rollout
        self._last_info: dict = {}

    def reset(self, seed: int | None) -> Any:
        obs, self._last_info = self._r.env.reset(seed=seed)
        return obs

    def obs_to_tensors(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image as PILImage

        rgbs = self._r._get_camera_images(raw_obs)
        cam_tensors = []
        for rgb in rgbs:
            pil = PILImage.fromarray(rgb).resize(
                (self._r.image_size, self._r.image_size),
                PILImage.Resampling.BILINEAR,  # type: ignore[attr-defined]
            )
            img_np = np.array(pil, dtype=np.uint8)
            cam_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
        img_t = torch.stack(cam_tensors, dim=0)
        state = self._r._flatten_obs(raw_obs)
        state_t = torch.from_numpy(state)
        return img_t, state_t

    def step(self, action: np.ndarray) -> SingleStepResult:
        action_env = action.reshape(1, -1)
        obs, reward, terminated, truncated, info = self._r.env.step(action_env)
        self._last_info = info

        rew_val = float(reward.item()) if hasattr(reward, "item") else float(reward)
        done_t = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        done_tr = bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated)

        succ = False
        if "success" in info:
            s = info["success"]
            succ = bool(s.item()) if hasattr(s, "item") else bool(s)

        return SingleStepResult(
            raw_obs=obs,
            reward=rew_val,
            terminated=done_t,
            truncated=done_tr,
            success=succ,
        )


# ---------------------------------------------------------------------------
# VecEnvAdapter for the shared wave-loop
# ---------------------------------------------------------------------------


class _ManiSkillVecAdapter:
    """Adapts :class:`ManiSkillRollout` to the :class:`VecEnvAdapter` protocol."""

    def __init__(self, rollout: ManiSkillRollout) -> None:
        self._r = rollout

    @property
    def num_envs(self) -> int:
        return self._r.num_envs

    def reset(self, seed: int | None) -> Any:
        obs, _info = self._r.env.reset(seed=seed)
        return obs

    def extract_batch_obs(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._r._extract_batched_obs(raw_obs)

    def step(self, actions: np.ndarray) -> Any:
        from vla.rl.vec_env import StepResult

        obs, reward, terminated, truncated, info = self._r.env.step(actions)
        N = self._r.num_envs
        term_list, trunc_list = ManiSkillRollout._extract_done_batched(terminated, truncated, N)
        return StepResult(
            raw_obs=obs,
            rewards=ManiSkillRollout._extract_rewards_batched(reward, N),
            terminateds=term_list,
            truncateds=trunc_list,
            successes=ManiSkillRollout._extract_success_batched(info, N),
        )
