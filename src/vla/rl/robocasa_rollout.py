"""Subprocess-vectorized RoboCasa rollout engine.

RoboCasa does not provide GPU-batched simulation like ManiSkill, but it can
still parallelize collection by running one environment per subprocess. This
matches the style used by RL-INF for MuJoCo / robosuite-backed rollouts.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import multiprocessing.connection
from typing import Any

import numpy as np
import torch

from vla.envs.robocasa import (
    ROBOCASA_CAMERA_KEYS,
    ROBOCASA_DEFAULT_STATE_DIM,
    ROBOCASA_STATE_KEYS,
    RoboCasaEnv,
)
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


def _raw_obs_to_tensors(raw_obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    frames: list[torch.Tensor] = []
    for key in ROBOCASA_CAMERA_KEYS:
        if key not in raw_obs:
            continue
        image = np.asarray(raw_obs[key], dtype=np.uint8)
        frames.append(torch.from_numpy(image).permute(2, 0, 1).contiguous())

    if not frames:
        raise ValueError(f"No RoboCasa camera frames found in observation keys: {sorted(raw_obs.keys())}")

    state_parts: list[np.ndarray] = []
    for key in ROBOCASA_STATE_KEYS:
        value = raw_obs.get(key)
        if value is None:
            continue
        state_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))

    if state_parts:
        state_np = np.concatenate(state_parts).astype(np.float32, copy=False)
    else:
        state_np = np.zeros(ROBOCASA_DEFAULT_STATE_DIM, dtype=np.float32)

    if state_np.shape[0] < ROBOCASA_DEFAULT_STATE_DIM:
        padded = np.zeros(ROBOCASA_DEFAULT_STATE_DIM, dtype=np.float32)
        padded[: state_np.shape[0]] = state_np
        state_np = padded
    else:
        state_np = state_np[:ROBOCASA_DEFAULT_STATE_DIM]

    return torch.stack(frames, dim=0), torch.from_numpy(state_np)


def _robocasa_worker(
    pipe: multiprocessing.connection.Connection,
    env_id: str,
    max_steps: int,
    image_size: int,
    layout_id: int | None,
    style_id: int | None,
    split: str,
    instruction: str,
) -> None:
    env = RoboCasaEnv(
        env_id=env_id,
        instruction=instruction,
        max_episode_steps=max_steps,
        image_size=image_size,
        layout_id=layout_id,
        style_id=style_id,
        split=split,
    )
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == _CMD_RESET:
                obs, info = env.reset(seed=0 if data is None else int(data))
                pipe.send((obs, info))
            elif cmd == _CMD_STEP:
                obs, reward, terminated, truncated, info = env.step(data)
                pipe.send((obs, reward, terminated, truncated, info))
            elif cmd == _CMD_TASK_DESC:
                pipe.send(env.task_description)
            elif cmd == _CMD_CLOSE:
                env.close()
                pipe.send(None)
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        with contextlib.suppress(Exception):
            env.close()
        return
    except Exception:
        import traceback

        logger.exception("RoboCasa worker crashed")
        with contextlib.suppress(Exception):
            env.close()
        with contextlib.suppress(BrokenPipeError, OSError):
            pipe.send(RuntimeError(f"RoboCasa worker crashed:\n{traceback.format_exc()}"))


class RoboCasaVecEnv:
    """Manage one RoboCasa environment per subprocess."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        max_steps: int,
        image_size: int,
        layout_id: int | None,
        style_id: int | None,
        split: str,
        instruction: str,
    ) -> None:
        self.num_envs = num_envs
        self.env_id = env_id
        self.max_steps = max_steps
        self.image_size = image_size
        self.layout_id = layout_id
        self.style_id = style_id
        self.split = split
        self.instruction = instruction
        self._pipes: list[multiprocessing.connection.Connection] = []
        self._procs: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_robocasa_worker,
                args=(
                    child_conn,
                    env_id,
                    max_steps,
                    image_size,
                    layout_id,
                    style_id,
                    split,
                    instruction,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._pipes.append(parent_conn)
            self._procs.append(proc)

        self._pipes[0].send((_CMD_TASK_DESC, None))
        self.task_description = self._pipes[0].recv()

    def reset(self, seeds: list[int | None]) -> list[dict]:
        for pipe, seed in zip(self._pipes, seeds, strict=True):
            pipe.send((_CMD_RESET, seed))
        results = [pipe.recv() for pipe in self._pipes]
        obs_list: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            obs, _info = result
            obs_list.append(obs)
        return obs_list

    def step(self, actions: np.ndarray) -> tuple[list[dict], list[float], list[bool], list[bool], list[dict]]:
        for pipe, action in zip(self._pipes, actions, strict=True):
            pipe.send((_CMD_STEP, action))

        obs_list: list[dict] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict] = []
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
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class RoboCasaRollout:
    """RoboCasa rollout engine with subprocess parallelism."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        max_steps: int = 300,
        image_size: int = 256,
        layout_id: int | None = None,
        style_id: int | None = None,
        split: str = "all",
        instruction: str = "",
    ) -> None:
        self.env_id = env_id
        self.num_envs = max(1, num_envs)
        self.max_steps = max_steps
        self.image_size = image_size
        self.layout_id = layout_id
        self.style_id = style_id
        self.split = split
        self.instruction = instruction

        if self.num_envs == 1:
            self._env = RoboCasaEnv(
                env_id=env_id,
                instruction=instruction,
                max_episode_steps=max_steps,
                image_size=image_size,
                layout_id=layout_id,
                style_id=style_id,
                split=split,
            )
            self.vec_env: RoboCasaVecEnv | None = None
        else:
            self._env = None
            self.vec_env = RoboCasaVecEnv(
                env_id=env_id,
                num_envs=self.num_envs,
                max_steps=max_steps,
                image_size=image_size,
                layout_id=layout_id,
                style_id=style_id,
                split=split,
                instruction=instruction,
            )

    @property
    def task_description(self) -> str:
        if self._env is not None:
            return self._env.task_description
        assert self.vec_env is not None
        return self.vec_env.task_description

    def collect_trajectory(
        self,
        policy_fn: Any,
        instruction: str,
        seed: int | None = None,
    ) -> Trajectory:
        if self._env is None:
            raise RuntimeError("collect_trajectory requires a single-env RoboCasa rollout")
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
        if self.num_envs > 1 and self.vec_env is not None and policy_batch_fn is not None:
            return self._collect_vectorized(policy_batch_fn, instruction, num_trajectories, seed)

        return collect_batch_sequential(
            lambda s: self.collect_trajectory(policy_fn, instruction, seed=s),
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

        assert self.vec_env is not None
        adapter = _RoboCasaVecAdapter(self.vec_env)
        return collect_trajectories_vectorized(
            adapter,
            policy_batch_fn,
            instruction,
            num_trajectories,
            seed,
            self.max_steps,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        if self.vec_env is not None:
            self.vec_env.close()


class _RoboCasaSingleAdapter:
    """Adapts :class:`RoboCasaEnv` to the shared single-env rollout loop."""

    def __init__(self, env: RoboCasaEnv) -> None:
        self._env = env

    def reset(self, seed: int | None) -> Any:
        raw_obs, _info = self._env.reset(seed=0 if seed is None else seed)
        return raw_obs

    def obs_to_tensors(self, raw_obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return _raw_obs_to_tensors(raw_obs)

    def step(self, action: np.ndarray) -> SingleStepResult:
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        return SingleStepResult(
            raw_obs=raw_obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=self._env.is_success(info),
        )


class _RoboCasaVecAdapter:
    """Adapts :class:`RoboCasaVecEnv` to the shared wave-loop interface."""

    def __init__(self, vec_env: RoboCasaVecEnv) -> None:
        self._vec_env = vec_env

    @property
    def num_envs(self) -> int:
        return self._vec_env.num_envs

    def reset(self, seed: int | None) -> list[dict]:
        seeds = [(seed + i) if seed is not None else None for i in range(self.num_envs)]
        return self._vec_env.reset(seeds)

    def extract_batch_obs(self, raw_obs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for obs in raw_obs:
            img_t, state_t = _raw_obs_to_tensors(obs)
            images.append(img_t)
            states.append(state_t)
        return torch.stack(images, dim=0), torch.stack(states, dim=0)

    def step(self, actions: np.ndarray) -> Any:
        from vla.rl.vec_env import StepResult

        obs_list, rewards, terminateds, truncateds, infos = self._vec_env.step(actions)
        successes = [bool(info.get("success", False)) for info in infos]
        return StepResult(
            raw_obs=obs_list,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            successes=successes,
        )
