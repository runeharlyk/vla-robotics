from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset

from vla.data.dataset import FewDemoDataset
from vla.data.libero import LiberoDataset, load_libero_all, load_libero_suite
from vla.data.maniskill import ManiSkillDataset, load_maniskill_dataset


def load_dataset(
    simulator: str,
    *,
    suite: str | None = None,
    data_path: str | Path | None = None,
    instruction: str = "",
    action_dim: int = 7,
    **kwargs,
) -> Dataset:
    """Unified dataset loader.

    Args:
        simulator: ``"libero"`` or ``"maniskill"``.
        suite: LIBERO suite name(s), comma-separated or ``"all"``.
        data_path: Path to preprocessed data (ManiSkill).
        instruction: Task instruction (ManiSkill).
        action_dim: Action dimensionality.
    """
    sim = simulator.lower()

    if sim == "libero":
        from vla.constants import resolve_suites

        suites = resolve_suites(suite or "all")
        if len(suites) == 1:
            return load_libero_suite(suites[0], **kwargs)
        return load_libero_all(suites, **kwargs)

    if sim == "maniskill":
        if data_path is None:
            raise ValueError("data_path is required for ManiSkill datasets")
        return load_maniskill_dataset(data_path, instruction=instruction, action_dim=action_dim)

    raise ValueError(f"Unknown simulator {simulator!r}. Available: libero, maniskill")


def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    import numpy as np

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


__all__ = [
    "FewDemoDataset",
    "LiberoDataset",
    "ManiSkillDataset",
    "load_dataset",
    "load_libero_suite",
    "load_libero_all",
    "load_maniskill_dataset",
    "split_dataset",
    "make_dataloader",
]
