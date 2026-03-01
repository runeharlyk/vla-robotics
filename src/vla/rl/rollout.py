"""ManiSkill rollout engine for collecting trajectories with a VLA policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 – registers envs


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
        obs_mode: str = "state",
        control_mode: str = "pd_joint_pos",
        sim_backend: str = "physx_cpu",
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.image_size = image_size

        render_backend = "cpu" if sim_backend == "physx_cpu" else "gpu"
        self.env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array",
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=max_steps,
        )

    def _render_image(self) -> np.ndarray:
        """Render the current scene and return ``(H, W, 3)`` uint8 array."""
        frame = self.env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        if frame.ndim == 4:
            frame = frame[0]
        return np.asarray(frame, dtype=np.uint8)

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
            rgb = self._render_image()
            from PIL import Image as PILImage

            pil = PILImage.fromarray(rgb).resize((self.image_size, self.image_size), PILImage.BILINEAR)
            img_np = np.array(pil, dtype=np.uint8)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)
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
