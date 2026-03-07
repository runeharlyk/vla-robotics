from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from vla.constants import ACTION_DIM, LIBERO_SUITES
from vla.data.dataset import NormStats

logger = logging.getLogger(__name__)


def _load_libero_v2_from_hf(
    repo_id: str,
    num_demos: int | None,
    seed: int,
    task_id: int | None = None,
) -> tuple[object, dict[int, str], dict, int]:
    """Load a LIBERO v2.0 dataset from HuggingFace, bypassing LeRobotDataset.

    The official ``lerobot/libero_*_image`` repos are v2.0 format which is
    incompatible with LeRobot >= 0.4 (expects v3.0).  This loader downloads
    the metadata files directly and uses ``datasets.load_dataset`` to read the
    per-episode parquet files.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    def _dl(path: str) -> Path:
        return Path(hf_hub_download(repo_id, path, repo_type="dataset"))

    with open(_dl("meta/info.json")) as f:
        json.load(f)
    with open(_dl("meta/stats.json")) as f:
        stats_raw = json.load(f)
    task_map: dict[int, str] = {}
    with open(_dl("meta/tasks.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            task_map[int(entry["task_index"])] = entry["task"]

    episodes_meta: list[dict] = []
    with open(_dl("meta/episodes.jsonl")) as f:
        for line in f:
            episodes_meta.append(json.loads(line))

    if task_id is not None:
        task_name = task_map.get(task_id, "")
        filtered = []
        for e in episodes_meta:
            ep_task = e.get("task_index")
            if ep_task is not None and int(ep_task) == task_id:
                filtered.append(e)
            elif task_name and task_name in e.get("tasks", []):
                filtered.append(e)
        if filtered:
            episodes_meta = filtered

    total_eps = len(episodes_meta)
    if num_demos is not None and num_demos < total_eps:
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_eps, generator=gen)[:num_demos].tolist()
        episodes_meta = [episodes_meta[i] for i in sorted(indices)]

    ep_indices = [e["episode_index"] for e in episodes_meta]
    data_files = [f"data/chunk-000/episode_{i:06d}.parquet" for i in ep_indices]
    ds = load_dataset(repo_id, data_files=data_files, split="train")

    def _cast_stats(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        s = stats_raw.get(key, {})
        mean = np.array(s.get("mean", [0.0])).flatten()
        std = np.array(s.get("std", [1.0])).flatten()
        return (
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32).clamp(min=1e-8),
        )

    act_mean, act_std = _cast_stats("action")
    st_mean, st_std = _cast_stats("observation.state")
    if st_mean.numel() == 0:
        st_mean = torch.zeros(1)
        st_std = torch.ones(1)

    stats = {
        "action": {"mean": act_mean, "std": act_std},
        "observation.state": {"mean": st_mean, "std": st_std},
    }

    episode_data_index = {"from": [], "to": []}
    offset = 0
    for ep in episodes_meta:
        length = ep["length"]
        episode_data_index["from"].append(offset)
        episode_data_index["to"].append(offset + length)
        offset += length

    class _Wrapped:
        """Thin wrapper giving a dict-of-tensors interface over a HF Dataset."""

        def __init__(self, hf_ds, ep_index, tm):
            self._ds = hf_ds
            self.episode_data_index = {
                "from": torch.tensor(ep_index["from"]),
                "to": torch.tensor(ep_index["to"]),
            }
            self._task_map = tm

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, idx):
            row = self._ds[idx]
            out: dict = {}
            for k, v in row.items():
                if v is None:
                    out[k] = None
                elif isinstance(v, (list, tuple)):
                    out[k] = torch.tensor(v, dtype=torch.float32)
                elif isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v)
                elif hasattr(v, "convert"):
                    arr = np.array(v.convert("RGB")).transpose(2, 0, 1)
                    out[k] = torch.from_numpy(arr).float() / 255.0
                else:
                    out[k] = v
            return out

    wrapped = _Wrapped(ds, episode_data_index, task_map)
    return wrapped, task_map, stats, len(episodes_meta)


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


class LiberoSFTDataset(Dataset):
    """Wraps a LeRobot Libero dataset for use with the SFT training loop.

    Returns items with the same keys and shapes as :class:`FewDemoDataset`::

        {"image": (C, H, W), "state": (state_dim,), "action": (action_dim,), "instruction": str}

    Normalization statistics come from the precomputed ``meta.stats`` of the
    LeRobot dataset (no extra iteration needed).

    Args:
        suite: Libero suite name (``"spatial"``, ``"object"``, ``"goal"``, ``"long"``).
        num_demos: Maximum number of episodes to use (``None`` = all).
        seed: Random seed for episode subsampling.
    """

    def __init__(self, suite: str, num_demos: int | None = None, seed: int = 42, task_id: int | None = None) -> None:
        repo_id = LIBERO_SUITES.get(suite.lower())
        if not repo_id:
            raise ValueError(f"Unknown Libero suite {suite!r}. Available: {list(LIBERO_SUITES)}")

        self._ds, self._task_map, stats_dict, num_eps = _load_libero_v2_from_hf(
            repo_id, num_demos, seed, task_id=task_id
        )
        act_mean = stats_dict["action"]["mean"]
        act_std = stats_dict["action"]["std"]
        st_mean = stats_dict["observation.state"]["mean"]
        st_std = stats_dict["observation.state"]["std"]
        if act_mean.ndim > 1:
            act_mean = act_mean.squeeze()
            act_std = act_std.squeeze()
        if st_mean.ndim > 1:
            st_mean = st_mean.squeeze()
            st_std = st_std.squeeze()
        self.state_dim: int = int(st_mean.shape[0]) if st_mean.numel() > 0 else 0

        self.norm_stats = NormStats(
            action_mean=act_mean,
            action_std=act_std,
            state_mean=st_mean,
            state_std=st_std,
        )

        self.action_dim: int = int(act_mean.shape[0])
        self.num_episodes: int = num_eps
        self.instruction: str = next(iter(self._task_map.values()), "complete the manipulation task")
        self.control_mode: str = "libero_default"

        self.metadata: dict = {
            "env_id": f"libero_{suite}",
            "instruction": self.instruction,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "image_size": 256,
            "control_mode": self.control_mode,
            "simulator": "libero",
            "suite": suite,
            "source_repo": repo_id,
        }

        logger.info(
            "LiberoSFTDataset(%s): %d episodes, %d timesteps, action_dim=%d, state_dim=%d",
            suite,
            self.num_episodes,
            len(self._ds),
            self.action_dim,
            self.state_dim,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self._ds[idx]

        image = sample["observation.images.image"]
        if image.ndim == 4:
            image = image.squeeze(0)

        state = sample.get("observation.state", torch.zeros(max(self.state_dim, 1)))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.ndim > 1:
            state = state.squeeze(0)

        action = sample["action"]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if action.ndim > 1:
            action = action.squeeze(0)

        task_idx = sample.get("task_index", 0)
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        instr = self._task_map.get(int(task_idx), self.instruction)

        return {
            "image": image.float(),
            "state": state.float(),
            "action": action.float(),
            "instruction": instr,
        }

    @property
    def image_size(self) -> int:
        return int(self.metadata["image_size"])

    def episodes_as_trajectories(self, task_id: int | None = None) -> list:
        """Convert stored LeRobot episodes to :class:`Trajectory` objects for SRPO seeding.

        Args:
            task_id: If given, only return episodes whose ``task_index`` matches.
                     Each LIBERO suite has 10 tasks (indices 0-9).
        """
        from vla.rl.rollout import Trajectory

        ep_index = self._ds.episode_data_index
        num_episodes = len(ep_index["from"])
        trajs: list[Trajectory] = []

        for ep in range(num_episodes):
            start = ep_index["from"][ep].item()
            end = ep_index["to"][ep].item()

            if task_id is not None:
                first_sample = self._ds[start]
                ti = first_sample.get("task_index", 0)
                if isinstance(ti, torch.Tensor):
                    ti = ti.item()
                if int(ti) != task_id:
                    continue

            images_list: list[torch.Tensor] = []
            states_list: list[torch.Tensor] = []
            actions_list: list[torch.Tensor] = []

            for i in range(start, end):
                sample = self._ds[i]

                img = sample["observation.images.image"]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if img.ndim == 4:
                    img = img.squeeze(0)
                images_list.append(img)

                state = sample.get("observation.state", torch.zeros(max(self.state_dim, 1)))
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state)
                if state.ndim > 1:
                    state = state.squeeze(0)
                states_list.append(state)

                action = sample["action"]
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                if action.ndim > 1:
                    action = action.squeeze(0)
                actions_list.append(action)

            T = len(actions_list)
            trajs.append(
                Trajectory(
                    images=torch.stack(images_list),
                    states=torch.stack(states_list).float(),
                    actions=torch.stack(actions_list).float(),
                    rewards=torch.ones(T),
                    dones=torch.zeros(T),
                    success=True,
                    length=T,
                )
            )

        logger.info(
            "Built %d demo trajectories from LeRobot (task_id=%s)", len(trajs), task_id
        )
        return trajs


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
