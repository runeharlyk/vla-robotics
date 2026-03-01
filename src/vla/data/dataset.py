"""Few-demonstration dataset for VLA behavior cloning.

Loads preprocessed .pt files produced by scripts/preprocess_data.py and
supports subsampling to N episodes for few-shot learning experiments.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class FewDemoDataset(Dataset):
    """Dataset that flattens episode timesteps into individual (image, state, action) samples.

    Args:
        pt_path: Path to the preprocessed .pt file.
        num_demos: Number of episodes to use. ``None`` means use all.
        seed: Random seed used when subsampling episodes.
    """

    def __init__(self, pt_path: str | Path, num_demos: int | None = None, seed: int = 42) -> None:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.metadata: dict = data["metadata"]
        episodes: list[dict] = data["episodes"]

        if num_demos is not None and num_demos < len(episodes):
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(episodes), generator=gen)[:num_demos].tolist()
            episodes = [episodes[i] for i in sorted(indices)]

        self.images: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.instruction: str = self.metadata["instruction"]
        self.episode_boundaries: list[tuple[int, int]] = []

        offset = 0
        for ep in episodes:
            T = ep["actions"].shape[0]
            img = ep["images"][:T]
            st = ep["states"][:T]
            act = ep["actions"][:T]
            self.images.append(img)
            self.states.append(st)
            self.actions.append(act)
            self.episode_boundaries.append((offset, offset + T))
            offset += T

        self.images_cat = torch.cat(self.images, dim=0).float()
        self.states_cat = torch.cat(self.states, dim=0).float()
        self.actions_cat = torch.cat(self.actions, dim=0).float()

        self.num_episodes = len(episodes)
        self.action_dim = int(self.metadata["action_dim"])
        self.state_dim = int(self.metadata["state_dim"])

    def __len__(self) -> int:
        return self.images_cat.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.images_cat[idx],
            "state": self.states_cat[idx],
            "action": self.actions_cat[idx],
            "instruction": self.instruction,
        }

    @property
    def image_size(self) -> int:
        return int(self.metadata["image_size"])
