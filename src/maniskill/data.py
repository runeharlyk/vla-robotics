"""
Dataset utilities for VLA training on ManiSkill.

Provides PyTorch Dataset classes for loading preprocessed HDF5 demonstrations
with lazy image decoding for low memory usage.
"""

import io
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else DATA_DIR / "preprocessed"


def get_skill_filename(env_id: str) -> str:
    return env_id.replace("-v1", "").replace("-", "_").lower() + ".h5"


def get_preprocessed_path(env_id: str) -> Path:
    return PREPROCESSED_DIR / get_skill_filename(env_id)


class PreprocessedDataset(Dataset):
    """
    Dataset that lazily loads from HDF5 files with JPEG-compressed images.

    Only a sample index is built in __init__ (fast, no images loaded).
    Images are decoded on demand in __getitem__.

    Args:
        data_path: Path to preprocessed .h5 file
        image_size: Output image size (224 for pretrained backbone, 256 for scratch)
        sequence_length: Number of frames per sample (for temporal models)
        action_horizon: Number of future actions to predict
        augment: Whether to apply data augmentation during training
        episode_indices: If provided, only use these episode indices (for train/val splits)
    """

    def __init__(
        self,
        data_path: Path,
        image_size: int = 256,
        sequence_length: int = 1,
        action_horizon: int = 1,
        augment: bool = False,
        episode_indices: Optional[list[int]] = None,
    ):
        self.data_path = str(data_path)
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.augment = augment
        self._h5: Optional[h5py.File] = None

        if augment:
            self.train_transform = T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                    T.RandomAffine(degrees=3, translate=(0.03, 0.03)),
                ]
            )
        else:
            self.train_transform = None

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading preprocessed data from {data_path}")
        with h5py.File(data_path, "r") as f:
            self.metadata = {k: v for k, v in f.attrs.items()}
            self.num_episodes = int(self.metadata["num_episodes"])
            self._episode_lengths: list[int] = []
            for i in range(self.num_episodes):
                ep_len = f[f"episode_{i}/actions"].shape[0]
                self._episode_lengths.append(ep_len)

        self._used_episodes = episode_indices if episode_indices is not None else list(range(self.num_episodes))

        self._index: list[tuple[int, int]] = []
        for ep_idx in self._used_episodes:
            ep_len = self._episode_lengths[ep_idx]
            for t in range(ep_len - action_horizon + 1):
                self._index.append((ep_idx, t))

        used_transitions = sum(self._episode_lengths[i] for i in self._used_episodes)
        print(
            f"Indexed {len(self._index)} samples from {len(self._used_episodes)} episodes ({used_transitions} transitions)"
        )
        print(f"  Action dim: {self.metadata['action_dim']}")
        print(f"  State dim: {self.metadata['state_dim']}")
        print(f"  Image size: {image_size}")
        print(f"  Augmentation: {augment}")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.data_path, "r")
        return self._h5

    def _decode_jpeg(self, jpeg_bytes: np.ndarray) -> torch.Tensor:
        img = Image.open(io.BytesIO(jpeg_bytes.tobytes()))
        if img.size[0] != self.image_size or img.size[1] != self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self._index[idx]
        grp = self.h5[f"episode_{ep_idx}"]

        start_idx = max(0, t - self.sequence_length + 1)
        img_indices = list(range(start_idx, t + 1))
        pad_count = self.sequence_length - len(img_indices)

        frames = []
        for i in img_indices:
            frames.append(self._decode_jpeg(grp["images"][i]))
        if pad_count > 0:
            first_frame = frames[0]
            frames = [first_frame] * pad_count + frames
        images = torch.stack(frames, dim=0)

        if self.train_transform is not None:
            seed = torch.randint(0, 2**31, (1,)).item()
            augmented = []
            for f in images:
                torch.manual_seed(seed)
                augmented.append(self.train_transform(f))
            images = torch.stack(augmented, dim=0)

        state = torch.from_numpy(grp["states"][t].astype(np.float32))
        actions = torch.from_numpy(grp["actions"][t : t + self.action_horizon].astype(np.float32))

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "instruction": str(self.metadata["instruction"]),
        }

    def get_all_actions(self) -> torch.Tensor:
        parts = []
        for ep_idx in range(self.num_episodes):
            actions = self.h5[f"episode_{ep_idx}/actions"][:]
            parts.append(torch.from_numpy(actions.astype(np.float32)))
        return torch.cat(parts, dim=0)

    @property
    def action_dim(self) -> int:
        return int(self.metadata["action_dim"])

    @property
    def state_dim(self) -> int:
        return int(self.metadata["state_dim"])

    @property
    def instruction(self) -> str:
        return str(self.metadata["instruction"])


class RawH5Dataset(Dataset):
    """
    Fallback dataset that loads directly from h5 files (uses dummy images).

    Args:
        demo_path: Path to demo directory
        env_id: Environment ID
        max_samples: Maximum number of samples to load
    """

    def __init__(self, demo_path: Path, env_id: str, max_samples: Optional[int] = None):
        self.episodes = []
        demo_file = self._find_h5_file(demo_path, env_id)

        print(f"Loading demos from: {demo_file}")

        with h5py.File(demo_file, "r") as f:
            for ep_key in f.keys():
                if not ep_key.startswith("traj"):
                    continue
                ep = f[ep_key]
                actions = ep["actions"][:]

                obs = ep["obs"]
                if "agent" in obs and len(list(obs["agent"].keys())) > 0:
                    qpos = obs["agent"]["qpos"][:]
                    qvel = obs["agent"]["qvel"][:]
                    state = np.concatenate([qpos, qvel], axis=-1)
                elif "state" in obs:
                    state = obs["state"][:]
                else:
                    env_states = ep["env_states"]
                    articulations = env_states["articulations"]
                    robot_key = list(articulations.keys())[0]
                    state = articulations[robot_key][:]

                for i in range(len(actions)):
                    self.episodes.append(
                        {
                            "state": state[i],
                            "action": actions[i],
                        }
                    )

        if max_samples and len(self.episodes) > max_samples:
            indices = np.random.choice(len(self.episodes), max_samples, replace=False)
            self.episodes = [self.episodes[i] for i in indices]

        print(f"Loaded {len(self.episodes)} transitions")

    def _find_h5_file(self, demo_path: Path, env_id: str) -> Path:
        candidates = [
            demo_path / env_id / "motionplanning" / "trajectory.h5",
            demo_path / env_id / "trajectory.h5",
        ]
        for c in candidates:
            if c.exists():
                return c

        env_dir = demo_path / env_id
        if env_dir.exists():
            for subdir in env_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "trajectory.h5"
                    if candidate.exists():
                        return candidate

        raise FileNotFoundError(f"Could not find trajectory.h5 in {demo_path / env_id}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        dummy_image = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        return {
            "images": dummy_image,
            "state": torch.tensor(ep["state"], dtype=torch.float32),
            "actions": torch.tensor(ep["action"], dtype=torch.float32).unsqueeze(0),
            "instruction": "pick up the cube",
        }


def load_dataset(
    env_id: str,
    sequence_length: int = 1,
    action_horizon: int = 1,
    image_size: int = 256,
    augment: bool = False,
) -> PreprocessedDataset:
    """
    Load a preprocessed dataset for the given environment.

    Args:
        env_id: Environment ID (e.g., "PickCube-v1")
        sequence_length: Number of frames per sample
        action_horizon: Number of future actions to predict
        image_size: Output image size (224 for pretrained backbone, 256 for scratch)
        augment: Whether to apply data augmentation

    Returns:
        PreprocessedDataset instance

    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
    """
    data_path = get_preprocessed_path(env_id)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}\n"
            f"Run preprocessing first: uv run invoke preprocess-data --skill {env_id}"
        )
    return PreprocessedDataset(
        data_path,
        image_size=image_size,
        sequence_length=sequence_length,
        action_horizon=action_horizon,
        augment=augment,
    )


def load_dataset_with_split(
    env_id: str,
    sequence_length: int = 1,
    action_horizon: int = 1,
    image_size: int = 256,
    augment: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[PreprocessedDataset, PreprocessedDataset]:
    """Load preprocessed dataset split into train and validation sets by episode.

    Args:
        env_id: Environment ID (e.g., "PickCube-v1")
        sequence_length: Number of frames per sample
        action_horizon: Number of future actions to predict
        image_size: Output image size
        augment: Whether to apply data augmentation (only applied to train split)
        val_ratio: Fraction of episodes to hold out for validation
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_path = get_preprocessed_path(env_id)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}\n"
            f"Run preprocessing first: uv run invoke preprocess-data --skill {env_id}"
        )

    with h5py.File(data_path, "r") as f:
        num_episodes = int(f.attrs["num_episodes"])

    rng = np.random.RandomState(seed)
    all_indices = list(range(num_episodes))
    rng.shuffle(all_indices)

    n_val = max(1, int(num_episodes * val_ratio))
    val_indices = sorted(all_indices[:n_val])
    train_indices = sorted(all_indices[n_val:])

    print(f"Split: {len(train_indices)} train episodes, {len(val_indices)} val episodes")

    train_ds = PreprocessedDataset(
        data_path,
        image_size=image_size,
        sequence_length=sequence_length,
        action_horizon=action_horizon,
        augment=augment,
        episode_indices=train_indices,
    )
    val_ds = PreprocessedDataset(
        data_path,
        image_size=image_size,
        sequence_length=sequence_length,
        action_horizon=action_horizon,
        augment=False,
        episode_indices=val_indices,
    )
    return train_ds, val_ds


def discover_available_envs() -> list[str]:
    reverse_map: dict[str, str] = {}
    envs_file = PROJECT_ROOT / "all_envs.txt"
    if envs_file.exists():
        for line in envs_file.read_text().strip().splitlines():
            env_id = line.strip()
            if env_id:
                reverse_map[get_skill_filename(env_id)] = env_id

    h5_files = sorted(PREPROCESSED_DIR.glob("*.h5"))
    result = []
    for f in h5_files:
        if f.name in reverse_map:
            result.append(reverse_map[f.name])
        else:
            result.append(f.stem)
    return result


def _read_action_dim(data_path: Path) -> int:
    with h5py.File(data_path, "r") as f:
        return int(f.attrs["action_dim"])


def load_multi_dataset_with_split(
    env_ids: list[str],
    sequence_length: int = 1,
    action_horizon: int = 1,
    image_size: int = 256,
    augment: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[ConcatDataset, ConcatDataset, list[PreprocessedDataset], list[PreprocessedDataset]]:
    paths: list[tuple[str, Path]] = []
    for env_id in env_ids:
        data_path = get_preprocessed_path(env_id)
        if not data_path.exists():
            print(f"Skipping {env_id}: {data_path} not found")
            continue
        paths.append((env_id, data_path))

    if not paths:
        raise FileNotFoundError("No preprocessed data found for any environment")

    ref_dim = _read_action_dim(paths[0][1])
    compatible = []
    for env_id, data_path in paths:
        dim = _read_action_dim(data_path)
        if dim != ref_dim:
            print(f"Skipping {env_id}: action_dim={dim} (expected {ref_dim})")
        else:
            compatible.append(env_id)

    if not compatible:
        raise FileNotFoundError("No environments with matching action_dim found")

    train_parts: list[PreprocessedDataset] = []
    val_parts: list[PreprocessedDataset] = []
    for env_id in compatible:
        train_ds, val_ds = load_dataset_with_split(
            env_id,
            sequence_length,
            action_horizon,
            image_size,
            augment,
            val_ratio,
            seed,
        )
        train_parts.append(train_ds)
        val_parts.append(val_ds)

    return ConcatDataset(train_parts), ConcatDataset(val_parts), train_parts, val_parts, compatible
