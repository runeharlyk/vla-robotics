import torch
from torch.utils.data import Subset, TensorDataset

from vla.constants import LIBERO_SUITES
from vla.data import make_dataloader, split_dataset
from vla.data.dataset import ConcatFewDemoDataset, FewDemoDataset

from tests.helpers import make_fake_pt


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


class TestFewDemoDataset:
    def test_load_single(self, tmp_path):
        pt = tmp_path / "test.pt"
        make_fake_pt(pt, num_episodes=4, T=3)
        ds = FewDemoDataset(pt)
        assert ds.num_episodes == 4
        assert len(ds) == 12
        assert ds.action_dim == 8
        sample = ds[0]
        assert "image" in sample and "action" in sample and "instruction" in sample

    def test_subsample(self, tmp_path):
        pt = tmp_path / "test.pt"
        make_fake_pt(pt, num_episodes=10, T=2)
        ds = FewDemoDataset(pt, num_demos=3, seed=0)
        assert ds.num_episodes == 3
        assert len(ds) == 6


class TestConcatFewDemoDataset:
    def test_combine_two(self, tmp_path):
        pt1 = tmp_path / "a.pt"
        pt2 = tmp_path / "b.pt"
        make_fake_pt(pt1, num_episodes=2, T=3, instruction="pick up")
        make_fake_pt(pt2, num_episodes=3, T=4, instruction="stack cubes")
        ds = ConcatFewDemoDataset([pt1, pt2])
        assert ds.num_episodes == 5
        assert len(ds) == 2 * 3 + 3 * 4
        assert ds.action_dim == 8
        sample = ds[0]
        assert sample["instruction"] == "pick up"
        sample_last = ds[len(ds) - 1]
        assert sample_last["instruction"] == "stack cubes"

    def test_rejects_mismatched_dims(self, tmp_path):
        pt1 = tmp_path / "a.pt"
        pt2 = tmp_path / "b.pt"
        make_fake_pt(pt1, action_dim=8)
        make_fake_pt(pt2, action_dim=7)
        import pytest

        with pytest.raises(ValueError):
            ConcatFewDemoDataset([pt1, pt2])

    def test_norm_stats_computed(self, tmp_path):
        pt1 = tmp_path / "a.pt"
        pt2 = tmp_path / "b.pt"
        make_fake_pt(pt1, num_episodes=2, T=5)
        make_fake_pt(pt2, num_episodes=2, T=5)
        ds = ConcatFewDemoDataset([pt1, pt2])
        assert ds.norm_stats.action_mean.shape == (8,)
        assert ds.norm_stats.action_std.shape == (8,)


class TestMakeDataloader:
    def test_creates_dataloader(self):
        dataset = TensorDataset(torch.randn(20, 3))
        loader = make_dataloader(dataset, batch_size=4, num_workers=0, drop_last=False)
        assert len(loader) == 5

    def test_drop_last(self):
        dataset = TensorDataset(torch.randn(10, 3))
        loader = make_dataloader(dataset, batch_size=3, num_workers=0, drop_last=True)
        assert len(loader) == 3
