import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from vla.constants import ACTION_DIM, LIBERO_SUITES


class LiberoDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        image_key: str = "observation.images.image",
        action_key: str = "action",
        state_key: str = "observation.state",
        delta_timestamps: dict[str, list[float]] | None = None,
        episodes: list[int] | None = None,
        revision: str | None = "main",
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.image_key = image_key
        self.action_key = action_key
        self.state_key = state_key

        self._ds = LeRobotDataset(
            repo_id,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=revision,
        )

        self._task_map: dict[int, str] = {}
        if hasattr(self._ds, "meta") and hasattr(self._ds.meta, "tasks"):
            self._task_map = self._ds.meta.tasks
        elif hasattr(self._ds, "tasks"):
            self._task_map = self._ds.tasks

    @property
    def lerobot_dataset(self):
        return self._ds

    def __len__(self) -> int:
        return len(self._ds)

    def _get_instruction(self, sample: dict) -> str:
        if "task" in sample:
            return str(sample["task"])
        task_idx = sample.get("task_index", 0)
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        return self._task_map.get(int(task_idx), "")

    def __getitem__(self, idx: int) -> dict:
        sample = self._ds[idx]

        image = sample[self.image_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        state = sample.get(self.state_key, torch.zeros(ACTION_DIM))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        action = sample[self.action_key]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        instruction = self._get_instruction(sample)

        return {
            "images": image.float(),
            "state": state.float(),
            "actions": action.float(),
            "instruction": instruction,
        }


def load_libero_suite(
    suite: str,
    delta_timestamps: dict[str, list[float]] | None = None,
    episodes: list[int] | None = None,
) -> LiberoDataset:
    repo_id = LIBERO_SUITES.get(suite.lower(), suite)
    return LiberoDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes)


def load_libero_all(
    suites: list[str] | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
    episodes: list[int] | None = None,
) -> ConcatDataset:
    if suites is None:
        suites = list(LIBERO_SUITES.keys())

    datasets = [load_libero_suite(s, delta_timestamps=delta_timestamps, episodes=episodes) for s in suites]
    return ConcatDataset(datasets)
