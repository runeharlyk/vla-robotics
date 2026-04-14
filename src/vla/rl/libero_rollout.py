"""Subprocess-vectorised LIBERO rollout engine for RL trajectory collection.

Follows the RLinf pattern (``ReconfigureSubprocEnv``):
each LIBERO environment runs in its own subprocess to bypass the GIL, and
the main process coordinates batched policy inference + action distribution.

Supports hot-swapping the LIBERO task within each subprocess via
:meth:`LiberoVecEnv.reconfigure`, avoiding costly process teardown and
respawn when switching between tasks during multi-task training.

References:
    - https://github.com/RLinf/RLinf/blob/main/rlinf/envs/libero/libero_env.py
    - https://github.com/RLinf/RLinf/blob/main/rlinf/envs/libero/venv.py
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import multiprocessing.connection
import time
from typing import Any

import numpy as np
import torch

from vla.constants import SUITE_MAP
from vla.rl.rollout import (
    SingleStepResult,
    Trajectory,
    collect_batch_sequential,
    collect_single_episode,
)

logger = logging.getLogger(__name__)

_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_CMD_TASK_DESC = "task_desc"
_CMD_RECONFIGURE = "reconfigure"
_ROBOSUITE_PATCHED = False


def _safe_close_env(env: Any) -> None:
    """Best-effort env close that also swallows interrupt-time teardown noise."""
    try:
        env.close()
    except BaseException:
        logger.debug("Ignoring LIBERO env close failure during shutdown", exc_info=True)


def _patch_robosuite() -> None:
    """Patch noisy robosuite cleanup hooks that can fail during interrupted EGL teardown."""
    global _ROBOSUITE_PATCHED
    if _ROBOSUITE_PATCHED:
        return

    try:
        from robosuite.renderers.context.egl_context import EGLGLContext
        from robosuite.utils.binding_utils import MjRenderContext
    except Exception:
        return

    orig_mj_del = getattr(MjRenderContext, "__del__", None)
    if callable(orig_mj_del):

        def _safe_mj_del(self: object) -> None:
            if not hasattr(self, "con"):
                return
            with contextlib.suppress(Exception):
                orig_mj_del(self)

        MjRenderContext.__del__ = _safe_mj_del

    orig_egl_free = getattr(EGLGLContext, "free", None)
    if callable(orig_egl_free):

        def _safe_egl_free(self: object) -> None:
            with contextlib.suppress(Exception):
                orig_egl_free(self)

        EGLGLContext.free = _safe_egl_free

    orig_egl_del = getattr(EGLGLContext, "__del__", None)
    if callable(orig_egl_del):

        def _safe_egl_del(self: object) -> None:
            with contextlib.suppress(Exception):
                orig_egl_del(self)

        EGLGLContext.__del__ = _safe_egl_del

    _ROBOSUITE_PATCHED = True


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _libero_worker(
    pipe: multiprocessing.connection.Connection,
    suite_name: str,
    task_id: int,
    obs_type: str,
    state_dim: int,
    image_size: int,
    camera_name: str | None = None,
) -> None:
    """Worker process: creates one LIBERO env and responds to commands.

    Supports ``_CMD_RECONFIGURE`` to hot-swap the task without restarting
    the process, following the RLinf ``ReconfigureSubprocEnv`` pattern.
    """
    _patch_robosuite()

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
            elif cmd == _CMD_RECONFIGURE:
                new_suite, new_task_id = data
                _safe_close_env(env)
                env = LiberoEnv(
                    suite_name=new_suite,
                    task_id=new_task_id,
                    obs_type=obs_type,
                    state_dim=state_dim,
                    **kwargs,
                )
                task_desc = env.task_description
                pipe.send(task_desc)
            elif cmd == _CMD_CLOSE:
                _safe_close_env(env)
                pipe.send(None)
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        _safe_close_env(env)
        return
    except Exception:
        import traceback

        logger.exception("LIBERO worker crashed")
        _safe_close_env(env)
        with contextlib.suppress(BrokenPipeError, OSError):
            pipe.send(RuntimeError(f"LIBERO worker crashed:\n{traceback.format_exc()}"))


_cached_libero_processor: object | None = None


def _get_libero_processor() -> object:
    global _cached_libero_processor
    if _cached_libero_processor is None:
        from lerobot.processor.env_processor import LiberoProcessorStep

        _cached_libero_processor = LiberoProcessorStep()
    return _cached_libero_processor


def _pack_obs(raw_obs: dict, image_size: int, state_dim: int) -> dict:
    """Convert raw LIBERO obs into a compact dict of numpy arrays.

    Returns:
        Dict with ``images`` (list of ``(H, W, 3)`` uint8) and
        ``state`` (``(state_dim,)`` float32).
    """
    import cv2

    images: list[np.ndarray] = []
    if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
        for img_np in raw_obs["pixels"].values():
            flipped = np.flip(img_np, axis=(0, 1)).copy()
            resized = cv2.resize(flipped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(resized.astype(np.uint8))

    state = np.zeros(state_dim, dtype=np.float32)
    if "robot_state" in raw_obs:
        rs = raw_obs["robot_state"]
        proc = _get_libero_processor()
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
        self.suite_name = suite_name
        self.task_id = task_id
        self._obs_type = obs_type
        self._state_dim = state_dim
        self._camera_name = camera_name
        self._pipes: list[multiprocessing.connection.Connection] = []
        self._procs: list[mp.Process] = []

        t0 = time.monotonic()
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
        elapsed = time.monotonic() - t0
        logger.info(
            "LiberoVecEnv created: %d envs for task %d in %.1fs",
            num_envs,
            task_id,
            elapsed,
        )

    def reconfigure(self, suite_name: str, task_id: int) -> None:
        """Hot-swap all workers to a new task without restarting processes.

        Follows the RLinf ``ReconfigureSubprocEnv`` pattern: each subprocess
        closes its current ``LiberoEnv`` and creates a new one for the
        requested task.  This avoids the cost of process teardown/spawn.
        """
        if suite_name == self.suite_name and task_id == self.task_id:
            return

        t0 = time.monotonic()
        for pipe in self._pipes:
            pipe.send((_CMD_RECONFIGURE, (suite_name, task_id)))
        results = [pipe.recv() for pipe in self._pipes]
        for r in results:
            if isinstance(r, Exception):
                raise r
        self.suite_name = suite_name
        self.task_id = task_id
        self.task_description = results[0]
        elapsed = time.monotonic() - t0
        logger.info(
            "LiberoVecEnv reconfigured: %d envs to task %d in %.1fs",
            self.num_envs,
            task_id,
            elapsed,
        )

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
        resolved = SUITE_MAP.get(suite_name.lower(), suite_name)
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

    def reconfigure(self, suite_name: str, task_id: int) -> None:
        """Hot-swap all workers to a new task without restarting processes."""
        resolved = SUITE_MAP.get(suite_name.lower(), suite_name)
        self.vec_env.reconfigure(resolved, task_id)
        self.suite_name = resolved
        self.task_id = task_id

    def _obs_to_tensors(self, packed_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a packed obs dict into image and state tensors.

        Returns:
            ``(num_cameras, C, H, W)`` uint8 image tensor and
            ``(state_dim,)`` float32 state tensor.
        """
        cam_tensors = [torch.from_numpy(img_np).permute(2, 0, 1) for img_np in packed_obs["images"]]
        default = torch.zeros(3, self.image_size, self.image_size, dtype=torch.uint8)
        while len(cam_tensors) < self.num_cameras:
            cam_tensors.append(cam_tensors[-1].clone() if cam_tensors else default.clone())
        cam_tensors = cam_tensors[: self.num_cameras]

        images = torch.stack(cam_tensors, dim=0)
        state = torch.from_numpy(packed_obs["state"])
        return images, state

    def _make_single_adapter(self) -> _LiberoSingleAdapter:
        return _LiberoSingleAdapter(self)

    def collect_batch(
        self,
        policy_fn: Any,
        instruction: str,
        num_trajectories: int = 16,
        seed: int | None = None,
        policy_batch_fn: Any | None = None,
    ) -> list[Trajectory]:
        """Collect trajectories from LIBERO - vectorised when ``num_envs > 1``.

        Falls back to :func:`collect_batch_sequential` with the shared
        single-episode loop when vectorised collection is not available.
        """
        task_instr = self.vec_env.task_description
        if self.num_envs > 1 and policy_batch_fn is not None:
            return self._collect_vectorized(policy_batch_fn, task_instr, num_trajectories, seed)
        adapter = self._make_single_adapter()
        return collect_batch_sequential(
            lambda s: collect_single_episode(adapter, policy_fn, task_instr, self.max_steps, s),
            num_trajectories,
            seed,
        )

    def _collect_vectorized(
        self,
        policy_batch_fn: Any,
        instruction: str,
        num_trajectories: int,
        seed: int | None,
    ) -> list[Trajectory]:
        from vla.rl.vec_env import collect_trajectories_vectorized

        adapter = _LiberoVecAdapter(self)
        return collect_trajectories_vectorized(
            adapter, policy_batch_fn, instruction, num_trajectories, seed, self.max_steps
        )

    def close(self) -> None:
        self.vec_env.close()


# ---------------------------------------------------------------------------
# SingleEnvAdapter for the shared episode loop
# ---------------------------------------------------------------------------


class _LiberoSingleAdapter:
    """Adapts :class:`LiberoRollout` to the :class:`SingleEnvAdapter` protocol."""

    def __init__(self, rollout: LiberoRollout) -> None:
        self._r = rollout

    def reset(self, seed: int | None) -> dict:
        obs_list = self._r.vec_env.reset([seed])
        return obs_list[0]

    def obs_to_tensors(self, raw_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._r._obs_to_tensors(raw_obs)

    def step(self, action: np.ndarray) -> SingleStepResult:
        obs_list, rewards, terminateds, truncateds, infos = self._r.vec_env.step(
            action[np.newaxis],
        )
        return SingleStepResult(
            raw_obs=obs_list[0],
            reward=rewards[0],
            terminated=terminateds[0],
            truncated=truncateds[0],
            success=infos[0].get("is_success", False),
        )


# ---------------------------------------------------------------------------
# VecEnvAdapter for the shared wave-loop
# ---------------------------------------------------------------------------


class _LiberoVecAdapter:
    """Adapts :class:`LiberoRollout` to the :class:`VecEnvAdapter` protocol."""

    def __init__(self, rollout: LiberoRollout) -> None:
        self._r = rollout

    @property
    def num_envs(self) -> int:
        return self._r.num_envs

    def reset(self, seed: int | None) -> list[dict]:
        N = self._r.num_envs
        seeds = [(seed + i) if seed is not None else None for i in range(N)]
        return self._r.vec_env.reset(seeds)

    def extract_batch_obs(self, raw_obs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        all_imgs = []
        all_states = []
        for i in range(self._r.num_envs):
            img_t, state_t = self._r._obs_to_tensors(raw_obs[i])
            all_imgs.append(img_t)
            all_states.append(state_t)
        return torch.stack(all_imgs, dim=0), torch.stack(all_states, dim=0)

    def step(self, actions: np.ndarray) -> Any:
        from vla.rl.vec_env import StepResult

        obs_list, rewards, terminateds, truncateds, infos = self._r.vec_env.step(actions)
        successes = [info.get("is_success", False) for info in infos]
        return StepResult(
            raw_obs=obs_list,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            successes=successes,
        )
