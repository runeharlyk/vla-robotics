"""
Dataset utilities for LIBERO training.

Wraps LeRobot's LeRobotDataset to load LIBERO task suites from HuggingFace
and provide batches compatible with our custom training loop.
"""

from typing import Optional
import constants

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


class LiberoDataset(Dataset):
    """Wraps a LeRobotDataset for use with our training loop.

    Converts LeRobot sample format to the dict format expected by our trainers:
    ``{"images": Tensor, "state": Tensor, "actions": Tensor, "instruction": str}``

    Args:
        repo_id: HuggingFace dataset repo id (e.g. "lerobot/libero_spatial_image")
        image_key: Key for the camera image in the LeRobot dataset
        action_key: Key for the action in the LeRobot dataset
        state_key: Key for the robot state in the LeRobot dataset
        delta_timestamps: Dict mapping keys to lists of relative timestamps for chunking
        episodes: Optional list of episode indices to load
    """

    def __init__(
        self,
        repo_id: str,
        image_key: str = "observation.images.image",
        action_key: str = "action",
        state_key: str = "observation.state",
        delta_timestamps: Optional[dict[str, list[float]]] = None,
        episodes: Optional[list[int]] = None,
        revision: Optional[str] = "main",
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
    delta_timestamps: Optional[dict[str, list[float]]] = None,
    episodes: Optional[list[int]] = None,
) -> LiberoDataset:
    """Load a single LIBERO task suite.

    Args:
        suite: One of "spatial", "object", "goal", "long" or a full HF repo id
        delta_timestamps: Delta timestamp config for action chunking
        episodes: Optional list of episode indices to load

    Returns:
        LiberoDataset wrapping the requested suite
    """
    repo_id = LIBERO_SUITES.get(suite.lower(), suite)
    return LiberoDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes)


def load_libero_all(
    suites: Optional[list[str]] = None,
    delta_timestamps: Optional[dict[str, list[float]]] = None,
    episodes: Optional[list[int]] = None,
) -> ConcatDataset:
    """Load multiple LIBERO suites as a single ConcatDataset.

    Args:
        suites: List of suite names. Defaults to all four.
        delta_timestamps: Delta timestamp config
        episodes: Optional list of episode indices to load (applied per suite)

    Returns:
        ConcatDataset over the requested suites
    """
    if suites is None:
        suites = list(LIBERO_SUITES.keys())

    datasets = [load_libero_suite(s, delta_timestamps=delta_timestamps, episodes=episodes) for s in suites]
    return ConcatDataset(datasets)


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a dataset into train/val by random indices.

    Args:
        dataset: Source dataset
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_subset, val_subset)
    """
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for training.

    Args:
        dataset: Source dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: DataLoader workers
        prefetch_factor: Prefetch factor per worker
        drop_last: Drop the last incomplete batch

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
    )
