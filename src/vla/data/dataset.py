"""Few-demonstration dataset for VLA behavior cloning.

Loads preprocessed .pt files produced by scripts/preprocess_data.py and
supports subsampling to N episodes for few-shot learning experiments.

Computes per-dimension mean/std normalization statistics for actions and
states so the flow-matching model operates in a well-scaled space.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vla.rl.rollout import Trajectory


@dataclass
class NormStats:
    """Mean / std normalization statistics."""

    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor


def build_action_chunk_targets(actions: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sliding-window action chunks and masks from a flat episode.

    Args:
        actions: ``(T, action_dim)`` action tensor.
        chunk_size: Number of action positions expected by SmolVLA.

    Returns:
        ``(chunks, mask)`` where chunks is ``(T, chunk_size, action_dim)``
        and mask is ``(T, chunk_size)``.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape {tuple(actions.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    T, action_dim = actions.shape
    chunks = actions.new_zeros((T, chunk_size, action_dim))
    mask = torch.zeros((T, chunk_size), dtype=torch.bool)
    for t in range(T):
        end = min(T, t + chunk_size)
        n = end - t
        if n <= 0:
            continue
        chunks[t, :n] = actions[t:end]
        mask[t, :n] = True
    return chunks, mask


def pad_action_chunk_targets(
    chunks: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate precomputed action chunk targets to ``chunk_size``."""
    if chunks.ndim != 3:
        raise ValueError(f"chunks must be 3D, got shape {tuple(chunks.shape)}")
    if mask.shape != chunks.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match chunks[:, :2] {tuple(chunks.shape[:2])}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    T, H, action_dim = chunks.shape
    out_chunks = chunks.new_zeros((T, chunk_size, action_dim))
    out_mask = torch.zeros((T, chunk_size), dtype=torch.bool)
    h_eff = min(H, chunk_size)
    out_chunks[:, :h_eff] = chunks[:, :h_eff]
    out_mask[:, :h_eff] = mask[:, :h_eff].to(dtype=torch.bool)
    return out_chunks, out_mask


def _episodes_to_trajectories(episodes: list[dict]) -> list[Trajectory]:
    """Convert raw episode dicts to :class:`Trajectory` objects for SRPO seeding.

    Args:
        episodes: List of dicts with ``images``, ``states``, ``actions`` keys.

    Returns:
        One :class:`Trajectory` per episode, marked as successful demos.
    """
    trajs: list[Trajectory] = []
    for ep in episodes:
        T = ep["actions"].shape[0]
        trajs.append(
            Trajectory(
                images=ep["images"][:T],
                states=ep["states"][:T],
                actions=ep["actions"][:T],
                rewards=torch.ones(T),
                dones=torch.zeros(T),
                success=True,
                length=T,
            )
        )
    return trajs


def norm_stats_from_tensors(actions: torch.Tensor, states: torch.Tensor) -> NormStats:
    """Compute per-dimension mean/std normalization statistics.

    Args:
        actions: ``(N, action_dim)`` float tensor of all action samples.
        states: ``(N, state_dim)`` float tensor of all state samples.

    Returns:
        :class:`NormStats` with per-dimension mean and std (std clamped to ≥ 1e-8).
    """
    return NormStats(
        action_mean=actions.mean(dim=0),
        action_std=actions.std(dim=0).clamp(min=1e-8),
        state_mean=states.mean(dim=0),
        state_std=states.std(dim=0).clamp(min=1e-8),
    )


class FewDemoDataset(Dataset):
    """Dataset that flattens episode timesteps into individual (image, state, action) samples.

    Args:
        pt_path: Path to the preprocessed .pt file.
        num_demos: Number of episodes to use. ``None`` means use all.
        seed: Random seed used when subsampling episodes.
    """

    def __init__(
        self,
        pt_path: str | Path,
        num_demos: int | None = None,
        seed: int = 42,
        action_chunk_size: int = 50,
    ) -> None:
        # weights_only=False: file contains Python dicts and lists alongside tensors
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.metadata: dict = data["metadata"]
        episodes: list[dict] = data["episodes"]

        if num_demos is not None and num_demos < len(episodes):
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(episodes), generator=gen)[:num_demos].tolist()
            episodes = [episodes[i] for i in sorted(indices)]

        self._episodes = episodes
        self.images: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.action_chunks: list[torch.Tensor] = []
        self.action_masks: list[torch.Tensor] = []
        self.instruction: str = self.metadata.get("instruction", "complete the manipulation task")
        self._instructions: list[str] = []
        self.episode_boundaries: list[tuple[int, int]] = []
        self.action_chunk_size = int(self.metadata.get("action_chunk_size", action_chunk_size))

        offset = 0
        for ep in episodes:
            T = ep["actions"].shape[0]
            img = ep["images"][:T]
            if img.ndim == 4:
                img = img.unsqueeze(1)
            st = ep["states"][:T]
            act = ep["actions"][:T]
            if "action_chunks" in ep and "action_masks" in ep:
                chunks, masks = pad_action_chunk_targets(
                    ep["action_chunks"][:T].float(),
                    ep["action_masks"][:T].bool(),
                    self.action_chunk_size,
                )
            else:
                chunks, masks = build_action_chunk_targets(act.float(), self.action_chunk_size)
            self.images.append(img)
            self.states.append(st)
            self.actions.append(act)
            self.action_chunks.append(chunks)
            self.action_masks.append(masks)
            self.episode_boundaries.append((offset, offset + T))
            self._instructions.extend([str(ep.get("instruction", self.instruction))] * T)
            offset += T

        # Keep images as uint8 to avoid 4× memory blow-up from float32.
        # Conversion to float happens lazily in __getitem__.
        self.images_cat = torch.cat(self.images, dim=0)
        self.states_cat = torch.cat(self.states, dim=0).float()
        self.actions_cat = torch.cat(self.actions, dim=0).float()
        self.action_chunks_cat = torch.cat(self.action_chunks, dim=0).float()
        self.action_masks_cat = torch.cat(self.action_masks, dim=0).bool()
        del self.images, self.states, self.actions, self.action_chunks, self.action_masks

        self.num_episodes = len(episodes)
        self.action_dim = int(self.metadata["action_dim"])
        self.state_dim = int(self.metadata["state_dim"])
        self.control_mode: str = self.metadata.get("control_mode", "pd_joint_delta_pos")

        self.norm_stats = norm_stats_from_tensors(self.actions_cat, self.states_cat)

    def __len__(self) -> int:
        return self.images_cat.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.images_cat[idx].float(),
            "state": self.states_cat[idx],
            "action": self.actions_cat[idx],
            "action_chunk": self.action_chunks_cat[idx],
            "action_mask": self.action_masks_cat[idx],
            "instruction": self._instructions[idx],
        }

    @property
    def image_size(self) -> int:
        return int(self.metadata["image_size"])

    def episodes_as_trajectories(self) -> list[Trajectory]:
        """Convert stored episodes to :class:`Trajectory` objects for SRPO seeding."""
        return _episodes_to_trajectories(self._episodes)


class ConcatFewDemoDataset(Dataset):
    """Combines multiple preprocessed ``.pt`` files into a single dataset.

    All source files must share the same ``action_dim`` and ``state_dim``
    (i.e. same robot / control mode).  Instructions may differ across
    files, enabling multi-task training.

    Args:
        pt_paths: Paths to preprocessed ``.pt`` files.
        num_demos: Per-file cap on episodes (``None`` = use all).
        seed: Random seed for episode subsampling.
    """

    def __init__(
        self,
        pt_paths: list[str | Path],
        num_demos: int | None = None,
        seed: int = 42,
        action_chunk_size: int = 50,
    ) -> None:
        if not pt_paths:
            raise ValueError("pt_paths must contain at least one path")

        subs = [
            FewDemoDataset(p, num_demos=num_demos, seed=seed, action_chunk_size=action_chunk_size) for p in pt_paths
        ]

        action_dims = {d.action_dim for d in subs}
        state_dims = {d.state_dim for d in subs}
        if len(action_dims) > 1:
            raise ValueError(f"Cannot combine datasets with different action_dim: {action_dims}")
        if len(state_dims) > 1:
            raise ValueError(f"Cannot combine datasets with different state_dim: {state_dims}")

        self.action_dim: int = subs[0].action_dim
        self.state_dim: int = subs[0].state_dim
        self.num_episodes: int = sum(d.num_episodes for d in subs)

        self.images_cat = torch.cat([d.images_cat for d in subs])
        self.states_cat = torch.cat([d.states_cat for d in subs])
        self.actions_cat = torch.cat([d.actions_cat for d in subs])
        self.action_chunks_cat = torch.cat([d.action_chunks_cat for d in subs])
        self.action_masks_cat = torch.cat([d.action_masks_cat for d in subs])

        self._instructions: list[str] = []
        for d in subs:
            self._instructions.extend(d._instructions)

        unique_instrs = list(dict.fromkeys(d.instruction for d in subs))
        self.instruction: str = unique_instrs[0]

        self.norm_stats = norm_stats_from_tensors(self.actions_cat, self.states_cat)

        self.metadata: dict = subs[0].metadata.copy()
        if len(unique_instrs) > 1:
            self.metadata["instructions"] = unique_instrs
        self.control_mode: str = subs[0].control_mode

        self._episodes: list[dict] = []
        for d in subs:
            self._episodes.extend(d._episodes)

    def __len__(self) -> int:
        return self.images_cat.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.images_cat[idx].float(),
            "state": self.states_cat[idx],
            "action": self.actions_cat[idx],
            "action_chunk": self.action_chunks_cat[idx],
            "action_mask": self.action_masks_cat[idx],
            "instruction": self._instructions[idx],
        }

    @property
    def image_size(self) -> int:
        return int(self.images_cat.shape[-1])

    def episodes_as_trajectories(self) -> list[Trajectory]:
        """Convert stored episodes to :class:`Trajectory` objects for SRPO seeding."""
        return _episodes_to_trajectories(self._episodes)

def find_episode_index(boundaries: list[tuple[int, int]], idx: int) -> int:
    """Return the episode index containing a flattened sample index."""
    starts = [start for start, _ in boundaries]
    ep_idx = bisect.bisect_right(starts, idx) - 1
    if ep_idx < 0 or idx >= boundaries[ep_idx][1]:
        raise IndexError(idx)
    return ep_idx
