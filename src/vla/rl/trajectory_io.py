"""Utilities for serialising rollout trajectories into SFT datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vla.data.dataset import build_action_chunk_targets, pad_action_chunk_targets
from vla.rl.rollout import Trajectory


def _trajectory_action_targets(
    traj: Trajectory,
    action_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(actions, action_chunks, action_masks)`` for one trajectory."""
    T = int(traj.length)
    if T <= 0:
        raise ValueError("Cannot serialise an empty trajectory")

    executed_chunks = getattr(traj, "executed_chunks", None)
    chunk_mask = getattr(traj, "chunk_mask", None)
    if executed_chunks is not None and chunk_mask is not None:
        chunks, masks = pad_action_chunk_targets(
            executed_chunks[:T].float(),
            chunk_mask[:T].bool(),
            action_chunk_size,
        )
        actions = chunks[:, 0]
        return actions, chunks, masks

    actions = traj.actions[:T].float()
    chunks, masks = build_action_chunk_targets(actions, action_chunk_size)
    return actions, chunks, masks


def save_trajectories_as_sft_pt(
    trajectories: list[Trajectory],
    output: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    default_instruction: str = "complete the manipulation task",
    instructions_by_task: dict[str, str] | None = None,
    only_successful: bool = True,
    action_chunk_size: int = 50,
) -> Path:
    """Save rollout trajectories as a ``.pt`` file consumable by ``FewDemoDataset``.

    Chunked rollouts do not store intermediate observations for every action
    inside the executed chunk. For those trajectories, this writer stores one
    sample per decision point and preserves the actually executed chunk via
    ``action_chunks`` / ``action_masks``.
    """
    if action_chunk_size <= 0:
        raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")

    selected = [t for t in trajectories if t.success] if only_successful else list(trajectories)
    if not selected:
        raise ValueError("No trajectories to save")

    instructions_by_task = instructions_by_task or {}
    episodes: list[dict[str, Any]] = []
    action_dim = int(selected[0].actions.shape[-1])
    state_dim = int(selected[0].states.shape[-1]) if selected[0].states is not None else 0
    image_size = int(selected[0].images.shape[-1])

    for i, traj in enumerate(selected):
        T = int(traj.length)
        if T <= 0:
            continue
        actions, action_chunks, action_masks = _trajectory_action_targets(traj, action_chunk_size)
        states = traj.states[:T].float() if traj.states is not None else torch.zeros(T, 0, dtype=torch.float32)
        task_id = traj.task_id or f"trajectory_{i}"
        instruction = instructions_by_task.get(task_id, default_instruction)
        episodes.append(
            {
                "images": traj.images[:T].detach().cpu(),
                "states": states.detach().cpu(),
                "actions": actions.detach().cpu(),
                "action_chunks": action_chunks.detach().cpu(),
                "action_masks": action_masks.detach().cpu(),
                "instruction": instruction,
                "task_id": task_id,
                "success": bool(traj.success),
                "reset_seed": getattr(traj, "reset_seed", None),
            }
        )

    if not episodes:
        raise ValueError("No non-empty trajectories to save")

    unique_instructions = list(dict.fromkeys(str(ep["instruction"]) for ep in episodes))
    resolved_metadata: dict[str, Any] = {
        "instruction": unique_instructions[0] if len(unique_instructions) == 1 else default_instruction,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "image_size": image_size,
        "num_episodes": len(episodes),
        "action_chunk_size": action_chunk_size,
    }
    if metadata:
        resolved_metadata.update(metadata)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": resolved_metadata, "episodes": episodes}, output_path)
    return output_path


__all__ = ["save_trajectories_as_sft_pt"]
