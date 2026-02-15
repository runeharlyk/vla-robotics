import torch
from torch.utils.data import Subset, TensorDataset

from vla.data import LIBERO_SUITES, make_dataloader, split_dataset


class TestLiberoSuites:
    def test_all_suites_defined(self):
        assert "spatial" in LIBERO_SUITES
        assert "object" in LIBERO_SUITES
        assert "goal" in LIBERO_SUITES
        assert "long" in LIBERO_SUITES
        assert len(LIBERO_SUITES) == 4


class TestSplitDataset:
    def test_split_sizes(self):
        data = list(range(100))
        train, val = split_dataset(data, val_ratio=0.1, seed=42)
        assert isinstance(train, Subset)
        assert isinstance(val, Subset)
        assert len(train) + len(val) == 100
        assert len(val) == 10

    def test_split_no_overlap(self):
        data = list(range(50))
        train, val = split_dataset(data, val_ratio=0.2, seed=0)
        train_indices = set(train.indices)
        val_indices = set(val.indices)
        assert train_indices.isdisjoint(val_indices)

    def test_split_deterministic(self):
        data = list(range(100))
        train1, val1 = split_dataset(data, val_ratio=0.1, seed=42)
        train2, val2 = split_dataset(data, val_ratio=0.1, seed=42)
        assert train1.indices == train2.indices
        assert val1.indices == val2.indices

    def test_split_min_one_val(self):
        data = list(range(5))
        _, val = split_dataset(data, val_ratio=0.01, seed=0)
        assert len(val) >= 1


class TestMakeDataloader:
    def test_creates_dataloader(self):
        dataset = TensorDataset(torch.randn(20, 3))
        loader = make_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)
        assert len(loader) == 5

    def test_drop_last(self):
        dataset = TensorDataset(torch.randn(10, 3))
        loader = make_dataloader(dataset, batch_size=3, num_workers=0, drop_last=True)
        assert len(loader) == 3
