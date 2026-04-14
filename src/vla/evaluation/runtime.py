"""Shared runtime setup for checkpoint-based evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import typer

from vla.constants import MANISKILL_TASKS
from vla.env_metadata import EnvMetadata
from vla.envs import SimEnvFactory, make_env_factory
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device


@dataclass(slots=True)
class EvalConfig:
    checkpoint_dir: Path | None = None
    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    simulator: str = "maniskill"
    env_id: str | None = None
    suite: str = "all"
    num_episodes: int = 100
    max_steps: int | None = None
    seed: int = 0
    num_envs: int = 4
    task_id: int | None = None
    fixed_noise_seed: int | None = None
    use_wandb: bool = False
    wandb_project: str = "vla-eval"
    wandb_name: str | None = None
    wandb_entity: str | None = None
    instruction: str | None = None
    control_mode: str | None = None
    action_dim: int = 7
    state_dim: int = 8
    device: str = "cuda"


@dataclass(slots=True)
class EvalRuntime:
    config: EvalConfig
    policy: SmolVLAPolicy
    device: torch.device
    env_meta: EnvMetadata
    action_dim: int
    state_dim: int
    env_id: str
    instruction: str
    control_mode: str
    max_steps: int
    checkpoint_tag: str


def make_eval_env_factory(
    simulator: str,
    suite: str | None = None,
    env_id: str | None = None,
    state_dim: int = 8,
    instruction: str = "",
    max_episode_steps: int | None = None,
    control_mode: str | None = None,
) -> SimEnvFactory:
    sim = simulator.lower()
    kwargs: dict[str, Any] = {}

    if sim == "libero":
        kwargs["suite"] = suite or "all"
        kwargs["state_dim"] = state_dim
    elif sim == "maniskill":
        if not env_id:
            raise typer.BadParameter("--env/--env-id is required for ManiSkill evaluation")
        task_meta = MANISKILL_TASKS.get(env_id, {})
        kwargs["env_id"] = env_id
        kwargs["instruction"] = instruction or task_meta.get("instruction", env_id)
        kwargs["max_episode_steps"] = max_episode_steps or task_meta.get("max_episode_steps")
        if control_mode is not None:
            kwargs["control_mode"] = control_mode
    else:
        raise typer.BadParameter(f"Unknown simulator: {simulator!r}")

    return make_env_factory(sim, **kwargs)


def predict_action_from_batch(
    policy: SmolVLAPolicy,
    batch: dict[str, Any],
    default_instruction: str,
) -> torch.Tensor:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError(f"No observation.images.* in batch. Keys: {list(batch.keys())}")

    cam_views = []
    for key in image_keys:
        image = batch[key]
        if image.ndim in (4, 5):
            image = image[0]
        if image.ndim == 2:
            image = image.unsqueeze(0)
        cam_views.append(image)

    image_batch = torch.stack(cam_views, dim=0).unsqueeze(0) if len(cam_views) > 1 else cam_views[0].unsqueeze(0)

    state = batch.get("observation.state")
    if state is not None and state.ndim == 2:
        state = state[0]
    if state is not None:
        state = state.unsqueeze(0)

    task = batch.get("task", default_instruction)
    if isinstance(task, (list, tuple)):
        task = task[0]

    return policy.predict_action_batch(image_batch, task, state)[0]


def make_runtime_env_factory(runtime: EvalRuntime) -> SimEnvFactory:
    return make_eval_env_factory(
        runtime.config.simulator,
        suite=runtime.config.suite,
        env_id=runtime.env_id,
        state_dim=runtime.state_dim,
        instruction=runtime.instruction,
        max_episode_steps=runtime.max_steps,
        control_mode=runtime.control_mode,
    )


def resolve_eval_runtime(config: EvalConfig) -> EvalRuntime:
    device_obj = get_device(config.device)

    action_dim = config.action_dim
    state_dim = config.state_dim
    if config.checkpoint_dir is not None:
        ckpt_data = torch.load(config.checkpoint_dir / "policy.pt", map_location="cpu", weights_only=False)
        action_dim = ckpt_data.get("action_dim", action_dim)
        state_dim = ckpt_data.get("state_dim", state_dim)

    policy = SmolVLAPolicy(
        checkpoint=config.checkpoint,
        action_dim=action_dim,
        state_dim=state_dim,
        device=str(device_obj),
    )

    env_meta = policy.load_checkpoint(config.checkpoint_dir) if config.checkpoint_dir is not None else EnvMetadata()

    resolved_env_id = config.env_id or env_meta.env_id
    resolved_instruction = config.instruction or env_meta.instruction
    resolved_control_mode = config.control_mode or env_meta.control_mode

    if config.checkpoint_dir is None:
        if config.simulator.lower() == "libero":
            resolved_env_id = config.env_id or f"libero_{config.suite}"
            resolved_instruction = config.instruction or "follow the task instruction"
            resolved_control_mode = config.control_mode or "relative"
        elif config.simulator.lower() == "maniskill":
            resolved_env_id = config.env_id or "PickCube-v1"
            resolved_instruction = config.instruction or "pick up the cube"
            resolved_control_mode = config.control_mode or "pd_joint_delta_pos"

    checkpoint_tag = config.checkpoint_dir.name if config.checkpoint_dir else config.checkpoint.split("/")[-1]

    return EvalRuntime(
        config=config,
        policy=policy,
        device=device_obj,
        env_meta=env_meta,
        action_dim=policy.action_dim,
        state_dim=policy.state_dim,
        env_id=resolved_env_id,
        instruction=resolved_instruction,
        control_mode=resolved_control_mode,
        max_steps=config.max_steps or 220,
        checkpoint_tag=checkpoint_tag,
    )
