from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ManiSkillDataset(Dataset):
    """Dataset for preprocessed ManiSkill demonstrations.

    Expects a ``.pt`` file produced by the preprocessing pipeline containing
    a list of episode dicts, each with keys ``images``, ``actions``, and
    optionally ``states`` and ``instruction``.
    """

    def __init__(
        self,
        data_path: str | Path,
        instruction: str = "",
        action_dim: int = 8,
    ):
        self._path = Path(data_path)
        self._instruction = instruction
        self._action_dim = action_dim

        if self._path.suffix == ".pt":
            self._samples = torch.load(self._path, weights_only=False)
        elif self._path.suffix in (".h5", ".hdf5"):
            self._samples = self._load_hdf5(self._path)
        else:
            raise ValueError(f"Unsupported file format: {self._path.suffix}")

    @staticmethod
    def _load_hdf5(path: Path) -> list[dict]:
        import h5py

        samples: list[dict] = []
        with h5py.File(path, "r") as f:
            for ep_key in sorted(f.keys()):
                ep = f[ep_key]
                sample: dict = {}
                if "images" in ep:
                    sample["images"] = torch.from_numpy(np.array(ep["images"]))
                if "actions" in ep:
                    sample["actions"] = torch.from_numpy(np.array(ep["actions"]))
                if "states" in ep:
                    sample["states"] = torch.from_numpy(np.array(ep["states"]))
                if "instruction" in ep.attrs:
                    sample["instruction"] = ep.attrs["instruction"]
                samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]

        images = sample["images"]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        actions = sample["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        state = sample.get("states", torch.zeros(self._action_dim))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        instruction = sample.get("instruction", self._instruction)

        return {
            "images": images.float(),
            "state": state.float(),
            "actions": actions.float(),
            "instruction": instruction,
        }


def load_maniskill_dataset(
    data_path: str | Path,
    instruction: str = "",
    action_dim: int = 8,
) -> ManiSkillDataset:
    return ManiSkillDataset(data_path, instruction=instruction, action_dim=action_dim)
