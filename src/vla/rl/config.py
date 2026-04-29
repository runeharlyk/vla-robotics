"""SRPO configuration and task specification data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from vla.base_config import BaseTrainingConfig
from vla.constants import AdvantageMode, DistanceMetric, LiberoSuite, Mode, UpdateMethod, WorldModelType


@dataclass
class AdvantageConfig:
    mode: AdvantageMode = AdvantageMode.LEAVE_ONE_OUT
    eps: float = 1e-8
    skip_threshold: float = 1e-6


@dataclass
class PPOConfig:
    epochs: int = 4
    minibatch_trajs: int = 4
    # FPO paper ablation (Kanazawa et al., 2025) shows epsilon=0.05 achieves
    # 759.3 avg reward (best) vs epsilon=0.2 at 526.4 (worst, 31% degradation).
    # Asymmetric high clip (SimpleVLA-RL / DAPO) uses a small margin above.
    clip_epsilon: float = 0.05
    clip_epsilon_high: float = 0.08


@dataclass
class AWRConfig:
    epochs: int = 2
    temperature: float = 5.0
    weight_clip: float = 20.0


@dataclass
class FPOConfig:
    num_fm_noise_samples: int = 4
    full_chunk_target: bool = True
    loss_reduction: str = "sum"
    positive_adv_only: bool = False
    negative_adv_scale: float = 0.25
    log_ratio_clip: float = 5.0
    use_ref_policy_kl: bool = False


@dataclass
class SuccessBCConfig:
    epochs: int = 1
    loss_reduction: str = "mean"


@dataclass
class KLConfig:
    coeff: float = 0.01
    sft_coeff: float = 0.0
    adaptive: bool = False
    target: float = 0.01
    adapt_factor: float = 1.5


@dataclass
class ReplayConfig:
    include_demos_in_update: bool = False
    success_buffer_size: int = 0
    success_total_size: int = 0
    success_alpha: float = 1.0
    success_ema_decay: float = 0.8
    success_max_ratio: float = 1.0


@dataclass
class DynamicSamplingConfig:
    enabled: bool = False
    max_retries: int = 2


@dataclass
class RolloutConfig:
    num_envs: int = 1
    eval_num_envs: int = 1
    fm_batch_size: int = 32
    gradient_checkpointing: bool = False
    eval_zero_sample: bool = True
    n_action_steps: int = 1


@dataclass
class SRPOConfig(BaseTrainingConfig):
    """Hyperparameters for SRPO RL training.

    Inherits common fields from :class:`BaseTrainingConfig`.
    """

    lr: float = 1e-5
    num_iterations: int = 100
    update_method: UpdateMethod = UpdateMethod.AWR
    save_dir: str = "checkpoints/srpo"
    mode: Mode = Mode.SRPO
    world_model_type: WorldModelType = WorldModelType.VJEPA2
    distance_metric: DistanceMetric = DistanceMetric.NORMALIZED_L2
    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = False
    suite: LiberoSuite = LiberoSuite.SPATIAL
    task_id: int = 0
    state_dim: int = 0
    use_failure_rewards: bool = True
    use_standard_scaler: bool = False

    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    awr: AWRConfig = field(default_factory=AWRConfig)
    fpo: FPOConfig = field(default_factory=FPOConfig)
    success_bc: SuccessBCConfig = field(default_factory=SuccessBCConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize nested configs to a plain dict for W&B/results metadata."""
        return asdict(self)

    @property
    def advantage_mode(self) -> AdvantageMode:
        return self.advantage.mode

    @advantage_mode.setter
    def advantage_mode(self, value: AdvantageMode) -> None:
        self.advantage.mode = value

    @property
    def adv_eps(self) -> float:
        return self.advantage.eps

    @adv_eps.setter
    def adv_eps(self, value: float) -> None:
        self.advantage.eps = value

    @property
    def adv_skip_threshold(self) -> float:
        return self.advantage.skip_threshold

    @adv_skip_threshold.setter
    def adv_skip_threshold(self, value: float) -> None:
        self.advantage.skip_threshold = value

    @property
    def ppo_epochs(self) -> int:
        return self.ppo.epochs

    @property
    def ppo_minibatch_trajs(self) -> int:
        return self.ppo.minibatch_trajs

    @property
    def clip_epsilon(self) -> float:
        return self.ppo.clip_epsilon

    @property
    def clip_epsilon_high(self) -> float:
        return self.ppo.clip_epsilon_high

    @property
    def awr_epochs(self) -> int:
        return self.awr.epochs

    @property
    def awr_temperature(self) -> float:
        return self.awr.temperature

    @property
    def awr_weight_clip(self) -> float:
        return self.awr.weight_clip

    @property
    def success_bc_epochs(self) -> int:
        return self.success_bc.epochs

    @property
    def success_bc_loss_reduction(self) -> str:
        return self.success_bc.loss_reduction

    @property
    def kl_coeff(self) -> float:
        return self.kl.coeff

    @kl_coeff.setter
    def kl_coeff(self, value: float) -> None:
        self.kl.coeff = value

    @property
    def sft_kl_coeff(self) -> float:
        return self.kl.sft_coeff

    @property
    def adaptive_kl(self) -> bool:
        return self.kl.adaptive

    @property
    def kl_target(self) -> float:
        return self.kl.target

    @property
    def kl_adapt_factor(self) -> float:
        return self.kl.adapt_factor

    @property
    def num_fm_noise_samples(self) -> int:
        return self.fpo.num_fm_noise_samples

    @property
    def fpo_full_chunk_target(self) -> bool:
        return self.fpo.full_chunk_target

    @property
    def fpo_loss_reduction(self) -> str:
        return self.fpo.loss_reduction

    @property
    def fpo_positive_adv_only(self) -> bool:
        return self.fpo.positive_adv_only

    @property
    def fpo_negative_adv_scale(self) -> float:
        return self.fpo.negative_adv_scale

    @property
    def fpo_log_ratio_clip(self) -> float:
        return self.fpo.log_ratio_clip

    @property
    def fpo_use_ref_policy_kl(self) -> bool:
        return self.fpo.use_ref_policy_kl

    @property
    def include_demos_in_update(self) -> bool:
        return self.replay.include_demos_in_update

    @property
    def success_replay_buffer_size(self) -> int:
        return self.replay.success_buffer_size

    @property
    def success_replay_total_size(self) -> int:
        return self.replay.success_total_size

    @property
    def success_replay_alpha(self) -> float:
        return self.replay.success_alpha

    @property
    def success_replay_ema_decay(self) -> float:
        return self.replay.success_ema_decay

    @property
    def success_replay_max_ratio(self) -> float:
        return self.replay.success_max_ratio

    @property
    def dynamic_sampling(self) -> bool:
        return self.sampling.enabled

    @property
    def dynamic_sampling_max_retries(self) -> int:
        return self.sampling.max_retries

    @property
    def num_rollout_envs(self) -> int:
        return self.rollout.num_envs

    @property
    def num_envs(self) -> int:
        return self.rollout.eval_num_envs

    @property
    def fm_batch_size(self) -> int:
        return self.rollout.fm_batch_size

    @property
    def gradient_checkpointing(self) -> bool:
        return self.rollout.gradient_checkpointing

    @property
    def eval_zero_sample(self) -> bool:
        return self.rollout.eval_zero_sample

    @property
    def n_action_steps(self) -> int:
        return self.rollout.n_action_steps


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
