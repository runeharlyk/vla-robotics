import math
import torch
import torch.nn as nn

from vla.rl.policy_update import ppo_update
from vla.rl.rollout import Trajectory
from vla.rl.config import SRPOConfig
from vla.constants import UpdateMethod


class FakePolicy(nn.Module):
    """Mock policy that dispatches FM losses by trajectory identity and grad phase.

    Trajectory identity is determined by the mean pixel value of the images
    tensor (rounded to 1 decimal place), so each trajectory must be given a
    distinct fill value (e.g. 0.0 and 1.0).

    During the no_grad caching phase (old losses), torch.is_grad_enabled() is
    False.  During the update phase (new losses), it is True.  The mock uses
    this to return old_fm vs new_fm without relying on call order.
    """

    def __init__(
        self,
        old_fm: dict[float, torch.Tensor],
        new_fm: dict[float, torch.Tensor],
    ):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))
        self.chunk_size = 2
        self.max_action_dim = 4
        self._old_fm = old_fm
        self._new_fm = new_fm

    def compute_fm_loss_batched(
        self, images, actions, states, instruction, fixed_noise, fixed_time, batch_size=32
    ):
        key = round(images.float().mean().item(), 1)
        loss = self._new_fm[key] if torch.is_grad_enabled() else self._old_fm[key]
        # Multiply by (dummy * 0) so the returned tensor is part of the graph
        # without disturbing the loss value.
        return loss + self.dummy * 0.0


def _make_traj(T: int, fill: float, task_id: str = "task_a") -> Trajectory:
    """Helper: trajectory whose images are filled with a constant value."""
    return Trajectory(
        images=torch.full((T, 3, 8, 8), fill),
        states=torch.zeros(T, 4),
        actions=torch.zeros(T, 4),
        rewards=torch.zeros(T),
        dones=torch.zeros(T),
        success=False,
        length=T,
        task_id=task_id,
    )


def _make_config(**overrides) -> SRPOConfig:
    """Helper: SRPOConfig suitable for unit tests (single epoch, small batches)."""
    defaults = dict(
        update_method=UpdateMethod.PPO,
        ppo_epochs=1,
        clip_epsilon=0.2,
        clip_epsilon_high=0.28,
        max_grad_norm=10.0,
        fm_batch_size=32,
        num_fm_noise_samples=1,
        kl_coeff=0.1, # Set a nonzero KL coeff to test that it is included in the loss
    )
    defaults.update(overrides)
    return SRPOConfig(**defaults)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ppo_update():
    """Test that PPO update runs and produces expected loss values on a simple
    synthetic dataset.

    The test constructs two trajectories with distinct "task ids" (determined by
    the fill value of their images), and sets up a FakePolicy that returns
    predetermined FM losses for each trajectory and grad phase.  The test then
    runs fpo_update and checks that the returned losses match the expected
    values based on the FakePolicy's dispatch logic.

    This test does not verify any learning dynamics or parameter updates, only
    that the correct FM losses are computed and returned by fpo_update given
    the trajectories, config, and policy.  It serves as a sanity check for the
    integration of these components. As our model is somehow getting worse when 
    this is written, we want to make sure that the loss values are correct, even 
    if they are increasing.
    
    PPO_update takes
    
    policy: SmolVLAPolicy,
    ref_policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    trajectories: list[Trajectory],
    advantages: list[float],
    instrs_per_traj: list[str],
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
    config: SRPOConfig,
    """

    T = 3
    key0, key1 = 0.0, 1.0

    policy = FakePolicy(
        old_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.5)},
        new_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.7)},
    )
    optimizer = torch.optim.AdamW([policy.dummy], lr=1e-3)

    trajs = [_make_traj(T, key0), _make_traj(T, key1)]
    advantages = [2.0, -1.0]
    noise = [[torch.randn(T, 2, 4)] for _ in range(2)]
    time = [[torch.rand(T)] for _ in range(2)]

    config = _make_config()

    metrics = ppo_update(
        policy=policy,
        ref_policy=policy,
        optimizer=optimizer,
        trainable=[policy.dummy],
        trajectories=trajs,
        advantages=advantages,
        instrs_per_traj=["task", "task"],
        fixed_noise=noise,
        fixed_time=time,
        config=config,
    )

    # --- Trajectory 0 ---
    log_ratio0 = 0.5 - 0.5   # old_fm - new_fm
    ratio0 = math.exp(log_ratio0) # 1.0
    c_ratio0 = max(1 - config.clip_epsilon, min(ratio0, 1 + config.clip_epsilon_high))
    loss_0 = -min(ratio0 * 2.0, c_ratio0 * 2.0)
    kl_0 = config.kl_coeff * (0.5 * (log_ratio0 ** 2))

    # --- Trajectory 1 ---
    log_ratio1 = 0.5 - 0.7   # old_fm - new_fm
    ratio1 = math.exp(log_ratio1) # exp(-0.2)
    c_ratio1 = max(1 - config.clip_epsilon, min(ratio1, 1 + config.clip_epsilon_high))
    loss_1 = -min(ratio1 * (-1.0), c_ratio1 * (-1.0))
    kl_1 = config.kl_coeff * (0.5 * (log_ratio1 ** 2))

    # --- Combined averages ---
    expected_loss = (loss_0 + loss_1) / 2.0
    expected_kl = (kl_0 + kl_1) / 2.0

    assert abs(metrics.avg_loss - expected_loss) < 1e-4, (
        f"Expected loss {expected_loss:.6f}, got {metrics.avg_loss:.6f}"
    )
    assert abs(metrics.avg_kl - expected_kl) < 1e-4, (
        f"Expected kl {expected_kl:.6f}, got {metrics.avg_kl:.6f}"
    )   


if __name__ == "__main__":
    test_ppo_update()
    print("All tests passed.")

