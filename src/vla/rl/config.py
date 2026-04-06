"""SRPO configuration and task specification data types."""

from __future__ import annotations

from dataclasses import dataclass

from vla.base_config import BaseTrainingConfig
from vla.constants import AdvantageMode, DistanceMetric, LiberoSuite, Mode, UpdateMethod, WorldModelType


@dataclass
class SRPOConfig(BaseTrainingConfig):
    """Hyperparameters for SRPO RL training.

    Inherits common fields from :class:`BaseTrainingConfig`.
    """

    lr: float = 1e-5
    num_iterations: int = 100
    update_method: UpdateMethod = UpdateMethod.AWR
    advantage_mode: AdvantageMode = AdvantageMode.LEAVE_ONE_OUT
    adv_eps: float = 1e-8
    adv_skip_threshold: float = 1e-6
    ppo_epochs: int = 4
    ppo_minibatch_trajs: int = 4
    # FPO paper ablation (Kanazawa et al., 2025) shows ε=0.05 achieves
    # 759.3 avg reward (best) vs ε=0.2 at 526.4 (worst, 31% degradation).
    # Asymmetric high clip (SimpleVLA-RL / DAPO) uses a small margin above.
    clip_epsilon: float = 0.05
    clip_epsilon_high: float = 0.08
    awr_epochs: int = 2
    awr_temperature: float = 5.0
    awr_weight_clip: float = 20.0
    kl_coeff: float = 0.01
    save_dir: str = "checkpoints/srpo"
    mode: Mode = Mode.SRPO
    world_model_type: WorldModelType = WorldModelType.VJEPA2
    distance_metric: DistanceMetric = DistanceMetric.NORMALIZED_L2
    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = False
    num_fm_noise_samples: int = 4
    suite: LiberoSuite = LiberoSuite.SPATIAL
    task_id: int = 0
    state_dim: int = 0
    num_rollout_envs: int = 1
    num_envs: int = 1
    fm_batch_size: int = 32
    gradient_checkpointing: bool = False
    use_failure_rewards: bool = True
    use_standard_scaler: bool = False
    fpo_full_chunk_target: bool = True
    fpo_loss_reduction: str = "sum"
    fpo_positive_adv_only: bool = False
    fpo_negative_adv_scale: float = 0.25
    fpo_log_ratio_clip: float = 5.0
    fpo_use_ref_policy_kl: bool = False
    eval_zero_sample: bool = True
    adaptive_kl: bool = False
    kl_target: float = 0.01
    kl_adapt_factor: float = 1.5
    include_demos_in_update: bool = False
    success_replay_buffer_size: int = 0
    success_replay_total_size: int = 0
    success_replay_alpha: float = 1.0
    success_replay_ema_decay: float = 0.8
    success_replay_max_ratio: float = 1.0


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
