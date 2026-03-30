from vla.rl.advantage import normalize_advantages_per_task  # noqa: F401
from vla.rl.config import SRPOConfig, TaskSpec  # noqa: F401
from vla.rl.policy_update import (  # noqa: F401
    UpdateMetrics,
    awr_update,
    ppo_update,
)
from vla.rl.rollout import (  # noqa: F401
    RolloutEngine,
    SingleEnvAdapter,
    SingleStepResult,
    Trajectory,
    collect_batch_sequential,
    collect_single_episode,
)
from vla.rl.srpo_reward import WorldProgressReward  # noqa: F401
from vla.rl.vec_env import (  # noqa: F401
    StepResult,
    VecEnvAdapter,
    collect_trajectories_vectorized,
    collect_wave,
)
from vla.training.checkpoint import save_best_checkpoint  # noqa: F401


def __getattr__(name: str):
    _trainer_names = {
        "build_rollout_engine",
        "collect_all_trajectories",
        "evaluate_and_checkpoint",
        "train_srpo",
    }
    if name in _trainer_names:
        from vla.rl import trainer
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
