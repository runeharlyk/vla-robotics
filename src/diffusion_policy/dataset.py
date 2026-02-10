"""Dataset classes for loading ManiSkill demonstration data."""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset


class ManiSkillStateDataset(Dataset):
    """Dataset that loads env_states and renders RGB on-the-fly.

    This dataset loads environment states from ManiSkill demos and renders
    RGB images when requested, avoiding the need for pre-rendered RGB demos.
    """

    def __init__(
        self,
        demo_dir: str | Path,
        env_id: str,
        image_size: tuple[int, int] = (224, 224),
        max_demos: Optional[int] = None,
        task_description: str = "complete the task",
        max_trajectories: int = 100,
    ):
        """Initialize the dataset.

        Args:
            demo_dir: Directory containing demo files.
            env_id: Environment ID.
            image_size: Target image size for resizing.
            max_demos: Maximum number of demo files to load.
            task_description: Text description of the task.
            max_trajectories: Maximum trajectories per file.
        """
        self.demo_dir = Path(demo_dir) / env_id
        self.env_id = env_id
        self.image_size = image_size
        self.task_description = task_description

        self.env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            render_mode="rgb_array",
        )

        self.transitions = []
        self._load_demos(max_demos, max_trajectories)

    def _load_demos(self, max_demos: Optional[int], max_trajectories: int) -> None:
        """Load demo file references (not the actual data)."""
        demo_files = sorted(self.demo_dir.glob("**/*.h5"))

        if max_demos is not None:
            demo_files = demo_files[:max_demos]

        print(f"Loading transitions from {len(demo_files)} demo files...")

        for demo_file in demo_files:
            self._load_single_demo(demo_file, max_trajectories)

        print(f"Loaded {len(self.transitions)} total transitions")

    def _load_single_demo(self, demo_file: Path, max_trajectories: int) -> None:
        """Load transitions from a single demo file."""
        with h5py.File(demo_file, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj")][:max_trajectories]

            for traj_key in traj_keys:
                traj = f[traj_key]
                actions = traj["actions"][:]
                env_states = traj["env_states"]

                actor_keys = list(env_states["actors"].keys())
                if not actor_keys:
                    continue

                num_steps = len(actions)
                for i in range(num_steps):
                    state_dict = {"actors": {}}
                    for actor_name in env_states["actors"].keys():
                        state_dict["actors"][actor_name] = env_states["actors"][actor_name][i]

                    if "articulations" in env_states:
                        state_dict["articulations"] = {}
                        for art_name in env_states["articulations"].keys():
                            state_dict["articulations"][art_name] = env_states["articulations"][art_name][i]

                    self.transitions.append(
                        {
                            "state_dict": state_dict,
                            "action": actions[i],
                        }
                    )

    def __len__(self) -> int:
        """Return the number of transitions."""
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a single transition with rendered RGB."""
        item = self.transitions[idx]

        self.env.reset()
        try:
            self.env.unwrapped.set_state_dict(item["state_dict"])
        except Exception:
            pass

        rgb = self.env.render()
        if rgb is None or not isinstance(rgb, np.ndarray):
            rgb = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        if rgb.ndim == 4:
            rgb = rgb[0]

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if rgb.shape[-1] == 3:
            rgb = np.transpose(rgb, (2, 0, 1))

        rgb_tensor = torch.from_numpy(rgb).float()
        rgb_tensor = torch.nn.functional.interpolate(
            rgb_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        action = torch.from_numpy(item["action"]).float()

        return {
            "image": rgb_tensor,
            "action": action,
            "text": self.task_description,
        }

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


class ManiSkillDemoDataset(Dataset):
    """Dataset for loading ManiSkill HDF5 demonstration files.

    This dataset loads RGB observations and actions from ManiSkill demo files
    for behavioral cloning training.
    """

    def __init__(
        self,
        demo_dir: str | Path,
        env_id: str,
        image_size: tuple[int, int] = (224, 224),
        max_demos: Optional[int] = None,
        task_description: str = "complete the task",
    ):
        """Initialize the dataset.

        Args:
            demo_dir: Directory containing demo files (usually ~/.maniskill/demos).
            env_id: Environment ID (e.g., "PegInsertionSide-v1").
            image_size: Target image size for resizing (H, W).
            max_demos: Maximum number of demos to load (None for all).
            task_description: Text description of the task for CLIP conditioning.
        """
        self.demo_dir = Path(demo_dir) / env_id
        self.env_id = env_id
        self.image_size = image_size
        self.task_description = task_description

        self.data = []
        self._load_demos(max_demos)

    def _load_demos(self, max_demos: Optional[int] = None) -> None:
        """Load all demonstration files."""
        rgb_files = sorted(self.demo_dir.glob("**/trajectory_rgb.h5"))
        if rgb_files:
            print(f"Found {len(rgb_files)} RGB trajectory files")
            demo_files = rgb_files
        else:
            demo_files = sorted(self.demo_dir.glob("**/*.h5"))

        if max_demos is not None:
            demo_files = demo_files[:max_demos]

        print(f"Loading {len(demo_files)} demo files from {self.demo_dir}")

        for demo_file in demo_files:
            self._load_single_demo(demo_file)

        print(f"Loaded {len(self.data)} total transitions")

    def _load_single_demo(self, demo_file: Path) -> None:
        """Load a single demo file and extract transitions."""
        with h5py.File(demo_file, "r") as f:
            for traj_key in f.keys():
                if not traj_key.startswith("traj"):
                    continue

                traj = f[traj_key]
                actions = traj["actions"][:]

                if "rgb" in traj:
                    rgb_data = traj["rgb"][:]
                elif "obs" in traj:
                    obs = traj["obs"]
                    if "sensor_data" in obs:
                        if "base_camera" in obs["sensor_data"]:
                            rgb_data = obs["sensor_data"]["base_camera"]["rgb"][:]
                        elif "hand_camera" in obs["sensor_data"]:
                            rgb_data = obs["sensor_data"]["hand_camera"]["rgb"][:]
                        else:
                            camera_keys = list(obs["sensor_data"].keys())
                            if camera_keys:
                                rgb_data = obs["sensor_data"][camera_keys[0]]["rgb"][:]
                            else:
                                continue
                    elif "image" in obs:
                        rgb_data = obs["image"][:]
                    else:
                        continue
                else:
                    continue

                num_steps = min(len(rgb_data), len(actions))
                for i in range(num_steps):
                    self.data.append(
                        {
                            "rgb": rgb_data[i],
                            "action": actions[i],
                        }
                    )

    def __len__(self) -> int:
        """Return the number of transitions."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a single transition.

        Args:
            idx: Index of the transition.

        Returns:
            Dictionary containing:
                - image: Preprocessed RGB image tensor (3, H, W)
                - action: Action tensor (action_dim,)
                - text: Task description string
        """
        item = self.data[idx]

        rgb = item["rgb"]
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if rgb.shape[-1] == 3:
            rgb = np.transpose(rgb, (2, 0, 1))

        rgb_tensor = torch.from_numpy(rgb).float()
        rgb_tensor = torch.nn.functional.interpolate(
            rgb_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        action = torch.from_numpy(item["action"]).float()

        return {
            "image": rgb_tensor,
            "action": action,
            "text": self.task_description,
        }


def create_dataloader(
    env_id: str,
    demo_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    task_description: str = "complete the task",
    train_split: float = 0.9,
    max_demos: Optional[int] = None,
    max_trajectories: int = 50,
    use_state_rendering: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.

    Args:
        env_id: Environment ID.
        demo_dir: Demo directory (defaults to ~/.maniskill/demos).
        batch_size: Batch size for training.
        num_workers: Number of dataloader workers (0 for state rendering).
        task_description: Task description for CLIP conditioning.
        train_split: Fraction of data to use for training.
        max_demos: Maximum number of demo files to load.
        max_trajectories: Maximum trajectories per file.
        use_state_rendering: If True, render RGB from env_states on-the-fly.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    if demo_dir is None:
        demo_dir = Path.home() / ".maniskill" / "demos"

    if use_state_rendering:
        dataset = ManiSkillStateDataset(
            demo_dir=demo_dir,
            env_id=env_id,
            task_description=task_description,
            max_demos=max_demos,
            max_trajectories=max_trajectories,
        )
    else:
        dataset = ManiSkillDemoDataset(
            demo_dir=demo_dir,
            env_id=env_id,
            task_description=task_description,
            max_demos=max_demos,
        )

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Not enough data: {len(dataset)} samples, need at least 2")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class ManiSkillStateOnlyDataset(Dataset):
    """Dataset that uses only proprioceptive state (no RGB needed).

    This dataset extracts robot joint states from articulations in the demos
    and pairs them with actions, allowing training without RGB rendering.
    """

    def __init__(
        self,
        demo_dir: str | Path,
        env_id: str,
        max_demos: Optional[int] = None,
        max_trajectories: int = 100,
    ):
        """Initialize the dataset.

        Args:
            demo_dir: Directory containing demo files.
            env_id: Environment ID.
            max_demos: Maximum number of demo files to load.
            max_trajectories: Maximum trajectories per file.
        """
        self.demo_dir = Path(demo_dir) / env_id
        self.env_id = env_id
        self.transitions = []
        self._load_demos(max_demos, max_trajectories)

    def _load_demos(self, max_demos: Optional[int], max_trajectories: int) -> None:
        """Load demo files and extract state/action pairs."""
        demo_files = sorted(self.demo_dir.glob("**/*.h5"))

        rgb_files = [f for f in demo_files if "trajectory_rgb" in f.name]
        demo_files = [f for f in demo_files if "trajectory_rgb" not in f.name]

        if max_demos is not None:
            demo_files = demo_files[:max_demos]

        print(f"Loading state data from {len(demo_files)} demo files...")

        for demo_file in demo_files:
            self._load_single_demo(demo_file, max_trajectories)

        print(f"Loaded {len(self.transitions)} total state-action pairs")

    def _load_single_demo(self, demo_file: Path, max_trajectories: int) -> None:
        """Load state-action pairs from a single demo file."""
        with h5py.File(demo_file, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj")][:max_trajectories]

            for traj_key in traj_keys:
                traj = f[traj_key]
                actions = traj["actions"][:]
                num_steps = len(actions)

                if "env_states" not in traj:
                    continue

                env_states = traj["env_states"]

                if "articulations" not in env_states:
                    continue

                art_keys = list(env_states["articulations"].keys())
                if not art_keys:
                    continue

                for i in range(num_steps):
                    states = []
                    for art_name in art_keys:
                        art_state = env_states["articulations"][art_name][i]
                        states.append(art_state)

                    state = np.concatenate(states) if len(states) > 1 else states[0]

                    self.transitions.append({
                        "state": state.astype(np.float32),
                        "action": actions[i].astype(np.float32),
                    })

    def __len__(self) -> int:
        """Return the number of transitions."""
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single state-action pair.

        Args:
            idx: Index of the transition.

        Returns:
            Dictionary containing:
                - state: State tensor
                - action: Action tensor
        """
        item = self.transitions[idx]
        return {
            "state": torch.from_numpy(item["state"]).float(),
            "action": torch.from_numpy(item["action"]).float(),
        }


def create_state_dataloader(
    env_id: str,
    demo_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: float = 0.9,
    max_demos: Optional[int] = None,
    max_trajectories: int = 100,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int]:
    """Create train and validation dataloaders for state-only training.

    Args:
        env_id: Environment ID.
        demo_dir: Demo directory (defaults to ~/.maniskill/demos).
        batch_size: Batch size for training.
        num_workers: Number of dataloader workers.
        train_split: Fraction of data to use for training.
        max_demos: Maximum number of demo files to load.
        max_trajectories: Maximum trajectories per file.

    Returns:
        Tuple of (train_dataloader, val_dataloader, state_dim, action_dim).
    """
    if demo_dir is None:
        demo_dir = Path.home() / ".maniskill" / "demos"

    dataset = ManiSkillStateOnlyDataset(
        demo_dir=demo_dir,
        env_id=env_id,
        max_demos=max_demos,
        max_trajectories=max_trajectories,
    )

    if len(dataset) == 0:
        raise ValueError("No state-action pairs found in demos")

    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    action_dim = sample["action"].shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Not enough data: {len(dataset)} samples")

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, state_dim, action_dim
