"""
Dataset utilities for VLA training.

Provides PyTorch Dataset classes for loading preprocessed demonstrations.
"""
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"


def get_skill_filename(env_id: str) -> str:
    return env_id.replace("-v1", "").replace("-", "_").lower() + ".pt"


def get_preprocessed_path(env_id: str) -> Path:
    return PREPROCESSED_DIR / get_skill_filename(env_id)


class PreprocessedDataset(Dataset):
    """
    Dataset that loads preprocessed .pt files with images, states, and actions.
    
    Each preprocessed file contains episodes with:
        - images: (T, 3, H, W) tensor
        - states: (T, state_dim) tensor
        - actions: (T, action_dim) tensor
        - instruction: str
    
    Args:
        data_path: Path to preprocessed .pt file
        image_size: Expected image size (for validation)
        sequence_length: Number of frames per sample (for temporal models)
        action_horizon: Number of future actions to predict
    """

    def __init__(
        self,
        data_path: Path,
        image_size: int = 256,
        sequence_length: int = 1,
        action_horizon: int = 1,
    ):
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.samples = []

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading preprocessed data from {data_path}")
        data = torch.load(data_path, weights_only=False)
        self.metadata = data["metadata"]
        episodes = data["episodes"]

        for ep in episodes:
            images = ep["images"]
            states = ep["states"]
            actions = ep["actions"]
            instruction = ep["instruction"]
            T = len(actions)

            for t in range(T - action_horizon + 1):
                start_idx = max(0, t - sequence_length + 1)
                img_seq = images[start_idx : t + 1]

                if img_seq.shape[0] < sequence_length:
                    pad_size = sequence_length - img_seq.shape[0]
                    padding = img_seq[0:1].repeat(pad_size, 1, 1, 1)
                    img_seq = torch.cat([padding, img_seq], dim=0)

                action_seq = actions[t : t + action_horizon]

                self.samples.append({
                    "images": img_seq,
                    "state": states[t],
                    "actions": action_seq,
                    "instruction": instruction,
                })

        print(f"Loaded {len(self.samples)} samples from {len(episodes)} episodes")
        print(f"  Action dim: {self.metadata['action_dim']}")
        print(f"  State dim: {self.metadata['state_dim']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        images = sample["images"].float() / 255.0
        return {
            "images": images,
            "state": sample["state"],
            "actions": sample["actions"],
            "instruction": sample["instruction"],
        }

    @property
    def action_dim(self) -> int:
        return self.metadata["action_dim"]

    @property
    def state_dim(self) -> int:
        return self.metadata["state_dim"]

    @property
    def instruction(self) -> str:
        return self.metadata["instruction"]


class RawH5Dataset(Dataset):
    """
    Fallback dataset that loads directly from h5 files (uses dummy images).
    Use PreprocessedDataset when possible for real images.
    
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
                    self.episodes.append({
                        "state": state[i],
                        "action": actions[i],
                    })

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
) -> PreprocessedDataset:
    """
    Load a preprocessed dataset for the given environment.
    
    Args:
        env_id: Environment ID (e.g., "PickCube-v1")
        sequence_length: Number of frames per sample
        action_horizon: Number of future actions to predict
    
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
        sequence_length=sequence_length,
        action_horizon=action_horizon,
    )
