"""Subprocess-vectorised LIBERO rollout engine for RL trajectory collection.

Follows the RLinf pattern (``ReconfigureSubprocEnv``):
each LIBERO environment runs in its own subprocess to bypass the GIL, and
the main process coordinates batched policy inference + action distribution.

References:
    - https://github.com/RLinf/RLinf/blob/main/rlinf/envs/libero/libero_env.py
    - https://github.com/RLinf/RLinf/blob/main/rlinf/envs/libero/venv.py
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

import numpy as np
import torch

from vla.rl.rollout import Trajectory

logger = logging.getLogger(__name__)

_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_CMD_TASK_DESC = "task_desc"


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _libero_worker(
    pipe: mp.connection.Connection,
    suite_name: str,
    task_id: int,
    obs_type: str,
    state_dim: int,
    image_size: int,
    camera_name: str | None = None,
) -> None:
    """Worker process: creates one LIBERO env and responds to commands."""
    from vla.envs.libero import LiberoEnv

    kwargs: dict = {}
    if camera_name is not None:
        kwargs["camera_name"] = camera_name

    env = LiberoEnv(
        suite_name=suite_name,
        task_id=task_id,
        obs_type=obs_type,
        state_dim=state_dim,
        **kwargs,
    )
    task_desc = env.task_description
    img_size = image_size

    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == _CMD_RESET:
                obs, info = env.reset(seed=data)
                pipe.send((_pack_obs(obs, img_size, state_dim), info))
            elif cmd == _CMD_STEP:
                obs, reward, terminated, truncated, info = env.step(data)
                pipe.send((_pack_obs(obs, img_size, state_dim), reward, terminated, truncated, info))
            elif cmd == _CMD_TASK_DESC:
                pipe.send(task_desc)
            elif cmd == _CMD_CLOSE:
                env.close()
                pipe.send(None)
                break
    except Exception:
        import traceback

        logger.exception("LIBERO worker crashed")
        pipe.send(RuntimeError(f"LIBERO worker crashed:\n{traceback.format_exc()}"))


def _pack_obs(raw_obs: dict, image_size: int, state_dim: int) -> dict:
    """Convert raw LIBERO obs into a compact dict of numpy arrays.

    Returns:
        Dict with ``images`` (list of ``(H, W, 3)`` uint8) and
        ``state`` (``(state_dim,)`` float32).
    """
    from lerobot.processor.env_processor import LiberoProcessorStep
    from PIL import Image as PILImage

    images: list[np.ndarray] = []
    if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
        for img_np in raw_obs["pixels"].values():
            flipped = np.flip(img_np, axis=(0, 1)).copy()
            pil = PILImage.fromarray(flipped).resize((image_size, image_size), PILImage.BILINEAR)
            images.append(np.array(pil, dtype=np.uint8))

    state = np.zeros(state_dim, dtype=np.float32)
    if "robot_state" in raw_obs:
        rs = raw_obs["robot_state"]
        proc = LiberoProcessorStep()
        eef_pos = np.asarray(rs["eef"]["pos"], dtype=np.float32).flatten()
        eef_quat = torch.from_numpy(np.asarray(rs["eef"]["quat"], dtype=np.float32)).float().unsqueeze(0)
        eef_aa = proc._quat2axisangle(eef_quat).squeeze(0).numpy()
        gripper = np.asarray(rs["gripper"]["qpos"], dtype=np.float32).flatten()
        raw_state = np.concatenate([eef_pos, eef_aa, gripper])
        dim = min(len(raw_state), state_dim)
        state[:dim] = raw_state[:dim]

    return {"images": images, "state": state}


# ---------------------------------------------------------------------------
# Vectorised LIBERO environment
# ---------------------------------------------------------------------------


class LiberoVecEnv:
    """Manages ``num_envs`` LIBERO environments in subprocesses.

    Each subprocess owns one ``LiberoEnv`` instance.  The main process
    communicates via :class:`multiprocessing.Pipe` connections to send
    actions and receive observations, following the same pattern as
    RLinf's ``ReconfigureSubprocEnv``.

    Args:
        suite_name: LIBERO benchmark name (e.g. ``libero_spatial``).
        task_id: Task index within the benchmark.
        num_envs: Number of parallel environments.
        obs_type: LIBERO observation type.
        state_dim: Dimensionality of the robot state.
        image_size: Target image resolution.
    """

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        num_envs: int,
        obs_type: str = "pixels_agent_pos",
        state_dim: int = 8,
        image_size: int = 256,
        camera_name: str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.image_size = image_size
        self._pipes: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_libero_worker,
                args=(child_conn, suite_name, task_id, obs_type, state_dim, image_size, camera_name),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._pipes.append(parent_conn)
            self._procs.append(p)

        self._pipes[0].send((_CMD_TASK_DESC, None))
        self.task_description: str = self._pipes[0].recv()

    def reset(self, seeds: list[int | None]) -> list[dict]:
        for pipe, seed in zip(self._pipes, seeds, strict=True):
            pipe.send((_CMD_RESET, seed))
        results = [pipe.recv() for pipe in self._pipes]
        for r in results:
            if isinstance(r, Exception):
                raise r
        obs_list = [r[0] for r in results]
        return obs_list

    def step(self, actions: np.ndarray) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        for pipe, action in zip(self._pipes, actions, strict=True):
            pipe.send((_CMD_STEP, action))

        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for pipe in self._pipes:
            result = pipe.recv()
            if isinstance(result, Exception):
                raise result
            obs, reward, terminated, truncated, info = result
            obs_list.append(obs)
            rewards.append(float(reward))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))
            infos.append(info)

        return obs_list, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send((_CMD_CLOSE, None))
                pipe.recv()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# LIBERO Rollout Engine
# ---------------------------------------------------------------------------


class LiberoRollout:
    """Vectorised LIBERO rollout engine for RL trajectory collection.

    Creates ``num_envs`` LIBERO environments in subprocesses and collects
    trajectories with batched policy inference, following the same
    pattern as :class:`~vla.rl.rollout.ManiSkillRollout`.

    Args:
        suite_name: LIBERO benchmark name (e.g. ``libero_spatial``).
        task_id: Task index within the benchmark.
        num_envs: Number of parallel environments.
        max_steps: Maximum steps per episode.
        image_size: Target image resolution.
        obs_type: LIBERO observation type.
        state_dim: Dimensionality of the robot state.
        num_cameras: Expected number of camera views.
    """

    SUITE_MAP: dict[str, str] = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "long": "libero_10",
    }

    def __init__(
        self,
        suite_name: str = "libero_spatial",
        task_id: int = 0,
        num_envs: int = 1,
        max_steps: int = 300,
        image_size: int = 256,
        obs_type: str = "pixels_agent_pos",
        state_dim: int = 8,
        num_cameras: int = 2,
        camera_name: str | None = None,
    ) -> None:
        resolved = self.SUITE_MAP.get(suite_name.lower(), suite_name)
        self.suite_name = resolved
        self.task_id = task_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.image_size = image_size
        self.num_cameras = max(1, num_cameras)
        self.state_dim = state_dim

        from vla.envs.libero import LIBERO_CAMERAS

        self.vec_env = LiberoVecEnv(
            suite_name=resolved,
            task_id=task_id,
            num_envs=num_envs,
            obs_type=obs_type,
            state_dim=state_dim,
            image_size=image_size,
            camera_name=camera_name or LIBERO_CAMERAS,
        )

    @property
    def task_description(self) -> str:
        return self.vec_env.task_description

    def _obs_to_tensors(self, packed_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a packed obs dict into image and state tensors.

        Returns:
            ``(num_cameras, C, H, W)`` uint8 image tensor and
            ``(state_dim,)`` float32 state tensor.
        """
        cam_tensors = []
        for img_np in packed_obs["images"]:
            cam_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
        while len(cam_tensors) < self.num_cameras:
            cam_tensors.append(
                cam_tensors[-1].clone()
                if cam_tensors
                else torch.zeros(3, self.image_size, self.image_size, dtype=torch.uint8)
            )
        cam_tensors = cam_tensors[: self.num_cameras]

        images = torch.stack(cam_tensors, dim=0)
        state = torch.from_numpy(packed_obs["state"])
        return images, state

    def collect_batch(
        self,
        policy_fn: Any,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
        policy_batch_fn: Any | None = None,
    ) -> list[Trajectory]:
        """Collect trajectories from LIBERO — vectorised when ``num_envs > 1``.

        Args:
            policy_fn: Single-obs callable (used when ``num_envs == 1``).
            instruction: Language instruction (overridden by env task description).
            num_trajectories: How many episodes to collect.
            seed: Base seed.
            policy_batch_fn: Batched callable
                ``(images_BNCHW, instruction, states_BS) -> actions_BA``.

        Returns:
            List of :class:`Trajectory` objects.
        """
        task_instr = self.vec_env.task_description
        if self.num_envs > 1 and policy_batch_fn is not None:
            return self._collect_vectorized(policy_batch_fn, task_instr, num_trajectories, seed)
        return self._collect_sequential(policy_fn, task_instr, num_trajectories, seed)

    def _collect_sequential(
        self, policy_fn: Any, instruction: str, num_trajectories: int, seed: int | None
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for i in range(num_trajectories):
            s = (seed + i) if seed is not None else None
            traj = self._collect_single(policy_fn, instruction, s)
            trajectories.append(traj)
        return trajectories

    def _collect_single(self, policy_fn: Any, instruction: str, seed: int | None) -> Trajectory:
        obs_list = self.vec_env.reset([seed])
        obs = obs_list[0]

        images, states, actions_list, rewards_list, dones_list = [], [], [], [], []
        success = False

        for _step in range(self.max_steps):
            img_t, state_t = self._obs_to_tensors(obs)
            action = policy_fn(img_t, instruction, state_t)
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy().flatten()
            else:
                action_np = np.asarray(action, dtype=np.float32).flatten()

            images.append(img_t)
            states.append(state_t)
            actions_list.append(torch.from_numpy(action_np.copy()))

            obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step(action_np[np.newaxis])
            obs = obs_list[0]

            rewards_list.append(torch.tensor(rewards[0]))
            done = terminateds[0] or truncateds[0]
            dones_list.append(torch.tensor(float(done)))

            if infos[0].get("is_success", False):
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

    def _collect_vectorized(
        self,
        policy_batch_fn: Any,
        instruction: str,
        num_trajectories: int,
        seed: int | None,
    ) -> list[Trajectory]:
        N = self.num_envs
        all_trajectories: list[Trajectory] = []
        remaining = num_trajectories
        wave_idx = 0

        while remaining > 0:
            active_n = min(N, remaining)
            wave_seed = (seed + wave_idx * N) if seed is not None else None
            wave_trajs = self._collect_wave(policy_batch_fn, instruction, active_n, wave_seed)
            all_trajectories.extend(wave_trajs)
            remaining -= len(wave_trajs)
            wave_idx += 1

        return all_trajectories[:num_trajectories]

    def _collect_wave(
        self,
        policy_batch_fn: Any,
        instruction: str,
        active_n: int,
        seed: int | None,
    ) -> list[Trajectory]:
        N = self.num_envs
        seeds = [(seed + i) if seed is not None else None for i in range(N)]
        obs_list = self.vec_env.reset(seeds)

        img_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
        state_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
        action_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
        reward_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
        done_bufs: list[list[torch.Tensor]] = [[] for _ in range(N)]
        success_flags = [False] * N
        env_done = [i >= active_n for i in range(N)]

        for _step in range(self.max_steps):
            if all(env_done):
                break

            all_imgs = []
            all_states = []
            for i in range(N):
                img_t, state_t = self._obs_to_tensors(obs_list[i])
                all_imgs.append(img_t)
                all_states.append(state_t)

            images_batch = torch.stack(all_imgs, dim=0)
            states_batch = torch.stack(all_states, dim=0)

            active_indices = [i for i in range(N) if not env_done[i]]
            if not active_indices:
                break

            active_imgs = images_batch[active_indices]
            active_states = states_batch[active_indices]

            with torch.no_grad():
                active_actions = policy_batch_fn(active_imgs, instruction, active_states)

            if isinstance(active_actions, torch.Tensor):
                active_actions_np = active_actions.detach().cpu().numpy()
            else:
                active_actions_np = np.asarray(active_actions, dtype=np.float32)
            if active_actions_np.ndim == 1:
                active_actions_np = active_actions_np[np.newaxis]

            action_dim = active_actions_np.shape[-1]
            actions_np = np.zeros((N, action_dim), dtype=np.float32)
            for idx, env_i in enumerate(active_indices):
                actions_np[env_i] = active_actions_np[idx]

            for idx, env_i in enumerate(active_indices):
                img_bufs[env_i].append(images_batch[env_i])
                state_bufs[env_i].append(states_batch[env_i])
                action_bufs[env_i].append(torch.from_numpy(active_actions_np[idx].copy()))

            obs_list, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_np)

            for env_i in active_indices:
                reward_bufs[env_i].append(torch.tensor(rewards[env_i]))
                step_done = terminateds[env_i] or truncateds[env_i]
                done_bufs[env_i].append(torch.tensor(float(step_done)))

                if infos[env_i].get("is_success", False):
                    success_flags[env_i] = True
                if step_done:
                    env_done[env_i] = True

        trajectories: list[Trajectory] = []
        for i in range(active_n):
            T = len(img_bufs[i])
            if T == 0:
                continue
            trajectories.append(
                Trajectory(
                    images=torch.stack(img_bufs[i]),
                    states=torch.stack(state_bufs[i]),
                    actions=torch.stack(action_bufs[i]),
                    rewards=torch.stack(reward_bufs[i]),
                    dones=torch.stack(done_bufs[i]),
                    success=success_flags[i],
                    length=T,
                )
            )

        return trajectories

    def close(self) -> None:
        self.vec_env.close()
