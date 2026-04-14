"""Shared helpers for replaying recorded demos through simulator environments.

This is used when training needs trajectories with live simulator observations
while still following recorded demonstration actions.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

from vla.constants import OUTPUTS_DIR, LiberoSuite, Simulator
from vla.envs import SimEnv, make_env_factory
from vla.rl.config import TaskSpec
from vla.rl.rollout import Trajectory

logger = logging.getLogger(__name__)


def _batch_to_replay_obs(
    batch: dict[str, object],
    fallback_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError(f"No observation.images.* keys found in replay batch: {list(batch.keys())}")

    cam_views: list[torch.Tensor] = []
    for key in image_keys:
        img = batch[key]
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected tensor for {key}, got {type(img)!r}")
        if img.ndim == 4:
            img = img[0]
        cam_views.append(img.detach().cpu())

    state = batch.get("observation.state")
    if isinstance(state, torch.Tensor):
        if state.ndim == 2:
            state = state[0]
        state_t = state.detach().cpu().float()
    elif fallback_state is not None:
        state_t = fallback_state.detach().cpu().float()
    else:
        state_t = torch.zeros(1, dtype=torch.float32)

    return torch.stack(cam_views, dim=0), state_t


def _replay_single_demo(env: SimEnv, demo: Trajectory, seed: int) -> Trajectory:
    raw_obs, _info = env.reset(seed=seed)

    images: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    success = False

    T = int(demo.length or demo.actions.shape[0])
    for t in range(T):
        fallback_state = demo.states[t] if demo.states.ndim >= 2 and t < demo.states.shape[0] else None
        batch = env.obs_to_batch(raw_obs)
        img_t, state_t = _batch_to_replay_obs(batch, fallback_state=fallback_state)
        action_t = demo.actions[t].detach().cpu().float()

        images.append(img_t)
        states.append(state_t)
        actions.append(action_t)

        action_np = action_t.numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze()

        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        done = bool(terminated or truncated)
        rewards.append(torch.tensor(float(reward), dtype=torch.float32))
        dones.append(torch.tensor(float(done), dtype=torch.float32))

        if env.is_success(info):
            success = True
        if done:
            break

    used = len(actions)
    return Trajectory(
        images=torch.stack(images) if images else torch.empty(0),
        states=torch.stack(states).float() if states else torch.empty(0),
        actions=torch.stack(actions).float() if actions else torch.empty(0),
        rewards=torch.stack(rewards) if rewards else torch.empty(0),
        dones=torch.stack(dones) if dones else torch.empty(0),
        success=success,
        length=used,
        task_id=demo.task_id,
    )


def _replay_cache_path(
    *,
    cache_dir: Path,
    spec: TaskSpec,
    demos: list[Trajectory],
    simulator: Simulator,
    suite: LiberoSuite,
    max_steps: int,
    state_dim: int,
) -> Path:
    key = hashlib.sha1()
    key.update(str(simulator).encode("utf-8"))
    key.update(str(suite).encode("utf-8"))
    key.update(spec.task_id.encode("utf-8"))
    key.update(spec.env_id.encode("utf-8"))
    key.update(spec.instruction.encode("utf-8"))
    key.update(str(spec.libero_task_idx).encode("utf-8"))
    key.update(str(max_steps).encode("utf-8"))
    key.update(str(state_dim).encode("utf-8"))
    for demo in demos:
        key.update(str(int(demo.length or demo.actions.shape[0])).encode("utf-8"))
        key.update(demo.actions.detach().cpu().numpy().tobytes())
    return cache_dir / f"{spec.task_id}_{key.hexdigest()[:16]}.pt"


def replay_demo_rollouts(
    *,
    task_specs: list[TaskSpec],
    demo_trajectories: dict[str, list[Trajectory]] | None,
    simulator: Simulator,
    suite: LiberoSuite,
    max_steps: int,
    seed: int,
    state_dim: int,
    cache_dir: Path | None = None,
) -> dict[str, list[Trajectory]] | None:
    """Replay demo actions in the simulator and return observation-aligned trajectories."""
    if not demo_trajectories:
        return demo_trajectories

    resolved_cache_dir = cache_dir or (OUTPUTS_DIR / "demo_replay_cache")
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    replayed: dict[str, list[Trajectory]] = {}
    for spec_idx, spec in enumerate(task_specs):
        demos = demo_trajectories.get(spec.task_id, [])
        if not demos:
            replayed[spec.task_id] = []
            continue

        cache_path = _replay_cache_path(
            cache_dir=resolved_cache_dir,
            spec=spec,
            demos=demos,
            simulator=simulator,
            suite=suite,
            max_steps=max_steps,
            state_dim=state_dim,
        )
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_trajs = cached["trajectories"] if isinstance(cached, dict) else cached
            for traj in cached_trajs:
                traj.task_id = spec.task_id
            replayed[spec.task_id] = cached_trajs
            logger.info(
                "Loaded %d cached replayed demo trajectory/trajectories for %s from %s.",
                len(cached_trajs),
                spec.task_id,
                cache_path,
            )
            continue

        if simulator is Simulator.LIBERO:
            factory = make_env_factory("libero", suite=str(suite), state_dim=state_dim, task_id=spec.libero_task_idx)
        elif simulator is Simulator.MANISKILL:
            factory = make_env_factory(
                "maniskill",
                env_id=spec.env_id,
                instruction=spec.instruction,
                max_episode_steps=max_steps,
            )
        else:
            raise ValueError(f"Unsupported simulator for demo replay: {simulator!r}")

        env = factory(0)
        try:
            replayed_trajs: list[Trajectory] = []
            success_count = 0
            for demo_idx, demo in enumerate(demos):
                traj = _replay_single_demo(env, demo, seed=seed + spec_idx * 1000 + demo_idx)
                traj.task_id = spec.task_id
                replayed_trajs.append(traj)
                success_count += int(traj.success)
            replayed[spec.task_id] = replayed_trajs
            logger.info(
                "Replayed %d demo trajectory/trajectories for %s (%d successes) using simulator observations.",
                len(replayed_trajs),
                spec.task_id,
                success_count,
            )
            torch.save(
                {
                    "task_id": spec.task_id,
                    "simulator": str(simulator),
                    "suite": str(suite),
                    "trajectories": replayed_trajs,
                },
                cache_path,
            )
            logger.info("Saved replayed demo cache for %s to %s.", spec.task_id, cache_path)
        finally:
            env.close()

    return replayed
