"""SRPO configuration and task specification data types."""

from __future__ import annotations

from dataclasses import dataclass

from vla.base_config import BaseTrainingConfig
from vla.constants import DistanceMetrics, Mode, UpdateMethods, WorldModelTypes


@dataclass
class SRPOConfig(BaseTrainingConfig):
    """Hyperparameters for SRPO RL training.

    Inherits common fields from :class:`BaseTrainingConfig`.
    """

    lr: float = 1e-5
    num_iterations: int = 100
    trajectories_per_iter: int = 16
    update_method: str = UpdateMethods.awr
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    clip_epsilon_high: float = 0.2
    awr_epochs: int = 2
    awr_temperature: float = 5.0
    awr_weight_clip: float = 20.0
    kl_coeff: float = 0.01
    save_dir: str = "checkpoints/srpo"
    mode: str = Mode.srpo
    world_model_type: str = WorldModelTypes.vjepa2
    distance_metric: str = DistanceMetrics.normalized_l2
    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = False
    num_fm_noise_samples: int = 4
    suite: str = "spatial"
    task_id: int = 0
    state_dim: int = 0
    num_rollout_envs: int = 1
    num_eval_envs: int = 1
    fm_batch_size: int = 32
    gradient_checkpointing: bool = False


@dataclass
class TaskSpec:
    """Describes a single task for multi-task SRPO training.

    Args:
        task_id: Unique string key (used to index per-task reward models).
        instruction: Language instruction for the task.
        env_id: ManiSkill env id (ignored for LIBERO).
        libero_task_idx: Task index within a LIBERO suite.
        data_path: Path to preprocessed ``.pt`` file with demos for this task.
    """

    task_id: str
    instruction: str
    env_id: str = ""
    libero_task_idx: int = 0
    data_path: str = ""
