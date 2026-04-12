import math
import torch
import torch.nn as nn

from vla.rl.policy_update import fpo_update
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
        update_method=UpdateMethod.FPO,
        ppo_epochs=1,
        clip_epsilon=0.2,
        clip_epsilon_high=0.28,
        max_grad_norm=10.0,
        fm_batch_size=32,
        num_fm_noise_samples=1,
    )
    defaults.update(overrides)
    return SRPOConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fpo_basic_no_clipping():
    """Unclipped FPO loss matches hand-computed surrogate for two trajectories.

    Setup
    -----
    Traj 0 (fill=0.0, advantage=+2.0):
        old_fm = new_fm = 0.5  →  log_ratio = 0.0 → ratio = 1.0
        surr = 1.0 * 2.0 = 2.0  →  loss_0 = -2.0

    Traj 1 (fill=1.0, advantage=-1.0):
        old_fm = 0.5, new_fm = 0.7  →  log_ratio = -0.2 → ratio ≈ 0.8187
        clipped_ratio = clamp(0.8187, 0.8, 1.28) = 0.8187  (not clipped)
        surr1 = surr2 = 0.8187 * -1.0 = -0.8187
        loss_1 = -(-0.8187) = 0.8187

    Expected batch loss = mean([-2.0, 0.8187]) ≈ -0.5906
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

    metrics = fpo_update(
        policy=policy,
        optimizer=optimizer,
        trainable=[policy.dummy],
        trajectories=trajs,
        advantages=advantages,
        instrs_per_traj=["task", "task"],
        fixed_noise=noise,
        fixed_time=time,
        config=_make_config(),
    )

    r0 = math.exp(0.5 - 0.5)   # 1.0
    cr0 = max(1 - 0.2, min(r0, 1 + 0.28))
    loss_0 = -min(r0 * 2.0, cr0 * 2.0)

    r1 = math.exp(0.5 - 0.7)   # exp(-0.2)
    cr1 = max(1 - 0.2, min(r1, 1 + 0.28))
    loss_1 = -min(r1 * (-1.0), cr1 * (-1.0))

    expected = (loss_0 + loss_1) / 2.0

    assert abs(metrics.avg_loss - expected) < 1e-4, (
        f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
    )


def test_fpo_upper_clip_activates():
    """Clipping activates on the upper bound when ratio > 1 + clip_epsilon_high.

    Traj 0 (fill=0.0, advantage=+3.0):
        old_fm=0.5, new_fm=0.1  →  log_ratio=0.4  →  ratio=exp(0.4)≈1.492
        clipped_ratio = clamp(1.492, 0.8, 1.28) = 1.28  ← upper clip
        surr1 = 1.492 * 3.0 = 4.476
        surr2 = 1.28  * 3.0 = 3.84
        loss = -min(4.476, 3.84) = -3.84
    """
    T = 2
    key0 = 0.0

    policy = FakePolicy(
        old_fm={key0: torch.full((T,), 0.5)},
        new_fm={key0: torch.full((T,), 0.1)},
    )
    optimizer = torch.optim.AdamW([policy.dummy], lr=1e-3)

    metrics = fpo_update(
        policy=policy,
        optimizer=optimizer,
        trainable=[policy.dummy],
        trajectories=[_make_traj(T, key0)],
        advantages=[3.0],
        instrs_per_traj=["task"],
        fixed_noise=[[torch.randn(T, 2, 4)]],
        fixed_time=[[torch.rand(T)]],
        config=_make_config(),
    )

    ratio = math.exp(0.5 - 0.1)         # ≈ 1.492
    clipped = min(ratio, 1 + 0.28)      # = 1.28, upper clip
    expected = -min(ratio * 3.0, clipped * 3.0)

    assert abs(metrics.avg_loss - expected) < 1e-4, (
        f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
    )


def test_fpo_lower_clip_activates():
    """Clipping activates on the lower bound: harmful action held back with negative advantage.

    Traj 0 (fill=0.0, advantage=-2.0):
        old_fm=0.5, new_fm=0.9  →  log_ratio=-0.4  →  ratio=exp(-0.4)≈0.670
        clipped_ratio = clamp(0.670, 0.8, 1.28) = 0.8  ← lower clip
        surr1 = 0.670 * -2.0 = -1.340
        surr2 = 0.8   * -2.0 = -1.6
        loss = -min(-1.340, -1.600) = -(-1.600) = 1.600
    """
    T = 2
    key0 = 0.0

    policy = FakePolicy(
        old_fm={key0: torch.full((T,), 0.5)},
        new_fm={key0: torch.full((T,), 0.9)},
    )
    optimizer = torch.optim.AdamW([policy.dummy], lr=1e-3)

    metrics = fpo_update(
        policy=policy,
        optimizer=optimizer,
        trainable=[policy.dummy],
        trajectories=[_make_traj(T, key0)],
        advantages=[-2.0],
        instrs_per_traj=["task"],
        fixed_noise=[[torch.randn(T, 2, 4)]],
        fixed_time=[[torch.rand(T)]],
        config=_make_config(),
    )

    ratio = math.exp(0.5 - 0.9)              # ≈ 0.670
    clipped = max(ratio, 1 - 0.2)            # = 0.8, lower clip
    expected = -min(ratio * (-2.0), clipped * (-2.0))

    assert abs(metrics.avg_loss - expected) < 1e-4, (
        f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
    )


def test_fpo_empty_trajectories_returns_zero_metrics():
    """fpo_update with an empty trajectory list returns zero metrics without crashing."""
    policy = FakePolicy(old_fm={}, new_fm={})
    optimizer = torch.optim.AdamW([policy.dummy], lr=1e-3)

    metrics = fpo_update(
        policy=policy,
        optimizer=optimizer,
        trainable=[policy.dummy],
        trajectories=[],
        advantages=[],
        instrs_per_traj=[],
        fixed_noise=[],
        fixed_time=[],
        config=_make_config(),
    )

    assert metrics.avg_loss == 0.0
    assert metrics.avg_kl == 0.0
    assert metrics.avg_weight == 0.0


if __name__ == "__main__":
    test_fpo_basic_no_clipping()
    test_fpo_upper_clip_activates()
    test_fpo_lower_clip_activates()
    test_fpo_empty_trajectories_returns_zero_metrics()
    print("All tests passed!")