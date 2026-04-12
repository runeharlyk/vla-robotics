"""Thorough tests for ppo_update and fpo_update.

Addresses five issues present in the original test_ppo.py / test_fpo.py:

1. Type mismatch: ppo_update expects list[Tensor] for noise/time but old
   tests passed list[list[Tensor]].  FakePolicy ignored the params, masking
   the bug.
2. is_grad_enabled() dispatch: fragile under gradient checkpointing.
3. Zero-gradient dummy parameter: optimizer step was a no-op.
4. ref_policy = policy collapsed the KL test so ref != old was never tested.
5. Image-mean key identity: acknowledged, mitigated by Strategy A tests that
   avoid the mechanism entirely.

Two complementary FakePolicy variants are used:

* GradientCheckPolicy (Strategy A) — real trainable weight, type/shape
  assertions on fixed_noise/fixed_time, no is_grad_enabled() hack.
* ArithmeticFakePolicy (Strategy B) — predetermined losses for exact
  arithmetic verification, but with real gradient flow via a ``scale``
  parameter and type assertions on inputs.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from vla.constants import UpdateMethod
from vla.rl.config import SRPOConfig
from vla.rl.policy_update import fpo_update, ppo_update
from vla.rl.rollout import Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 2
MAX_ACTION_DIM = 4


# ---------------------------------------------------------------------------
# FakePolicy variant 1 — type / gradient validation
# ---------------------------------------------------------------------------


class GradientCheckPolicy(nn.Module):
    """Policy with a real trainable weight that validates input types.

    The loss is deterministic given fixed_noise/fixed_time and differentiable
    w.r.t. ``self.weight``, so we can assert that a gradient step actually
    changes the parameter.  No reliance on ``torch.is_grad_enabled()``.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, max_action_dim: int = MAX_ACTION_DIM):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.weight = nn.Parameter(torch.tensor(0.5))

    def compute_fm_loss_batched(
        self, images, actions, states, instruction,
        fixed_noise, fixed_time, batch_size=32,
    ):
        T = images.shape[0]

        # ---- type validation (catches Issue 1) ----
        assert isinstance(fixed_noise, torch.Tensor), (
            f"fixed_noise must be Tensor, got {type(fixed_noise).__name__}"
        )
        assert isinstance(fixed_time, torch.Tensor), (
            f"fixed_time must be Tensor, got {type(fixed_time).__name__}"
        )

        # ---- shape validation ----
        assert fixed_noise.shape == (T, self.chunk_size, self.max_action_dim), (
            f"fixed_noise shape {fixed_noise.shape} != "
            f"({T}, {self.chunk_size}, {self.max_action_dim})"
        )
        assert fixed_time.shape == (T,), (
            f"fixed_time shape {fixed_time.shape} != ({T},)"
        )

        # ---- deterministic, differentiable loss ----
        signal = fixed_noise.mean(dim=(1, 2)) * fixed_time  # (T,)
        return signal.abs() * self.weight                    # (T,)


# ---------------------------------------------------------------------------
# FakePolicy variant 2 — exact arithmetic with real gradients
# ---------------------------------------------------------------------------


class ArithmeticFakePolicy(nn.Module):
    """Predetermined FM losses with real gradient flow and type validation.

    ``scale`` is initialised to 1.0 so ``base_loss * scale == base_loss``
    before the first optimizer step.  After backward, ``scale.grad`` is
    nonzero, so the step actually modifies the parameter.
    """

    def __init__(
        self,
        old_fm: dict[float, torch.Tensor],
        new_fm: dict[float, torch.Tensor],
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = CHUNK_SIZE
        self.max_action_dim = MAX_ACTION_DIM
        self._old_fm = old_fm
        self._new_fm = new_fm

    def compute_fm_loss_batched(
        self, images, actions, states, instruction,
        fixed_noise, fixed_time, batch_size=32,
    ):
        # ---- type validation ----
        assert isinstance(fixed_noise, torch.Tensor), (
            f"fixed_noise must be Tensor, got {type(fixed_noise).__name__}"
        )
        assert isinstance(fixed_time, torch.Tensor), (
            f"fixed_time must be Tensor, got {type(fixed_time).__name__}"
        )

        key = round(images.float().mean().item(), 1)
        base = self._new_fm[key] if torch.is_grad_enabled() else self._old_fm[key]
        return base * self.scale


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_traj(T: int, fill: float, task_id: str = "task_a") -> Trajectory:
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


def _make_ppo_config(**overrides) -> SRPOConfig:
    defaults = dict(
        update_method=UpdateMethod.PPO,
        ppo_epochs=1,
        clip_epsilon=0.2,
        clip_epsilon_high=0.28,
        max_grad_norm=10.0,
        fm_batch_size=32,
        num_fm_noise_samples=1,
        kl_coeff=0.1,
    )
    defaults.update(overrides)
    return SRPOConfig(**defaults)


def _make_fpo_config(**overrides) -> SRPOConfig:
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


def _make_ppo_noise_time(
    M: int, T: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Correct types for ppo_update: list[Tensor]."""
    noise = [torch.randn(T, CHUNK_SIZE, MAX_ACTION_DIM) for _ in range(M)]
    time = [torch.rand(T) for _ in range(M)]
    return noise, time


def _make_fpo_noise_time(
    M: int, T: int, n_samples: int = 1,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Correct types for fpo_update: list[list[Tensor]]."""
    noise = [
        [torch.randn(T, CHUNK_SIZE, MAX_ACTION_DIM) for _ in range(n_samples)]
        for _ in range(M)
    ]
    time = [
        [torch.rand(T) for _ in range(n_samples)]
        for _ in range(M)
    ]
    return noise, time


# ===================================================================
# PPO — Strategy A: type & gradient validation
# ===================================================================


class TestPPOTypeAndGradient:
    """Verify ppo_update receives correct types and produces real gradients."""

    def test_ppo_correct_types_no_crash(self):
        """ppo_update runs without error when given correct list[Tensor] types."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        ref_policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-3)
        noise, time = _make_ppo_noise_time(M, T)

        metrics = ppo_update(
            policy=policy,
            ref_policy=ref_policy,
            optimizer=optimizer,
            trainable=[policy.weight],
            trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_ppo_config(),
        )

        assert math.isfinite(metrics.avg_loss)
        assert math.isfinite(metrics.avg_kl)

    def test_ppo_wrong_noise_type_raises(self):
        """list[list[Tensor]] (the original test_ppo.py bug) is caught."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        ref_policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-3)

        # Wrong type: list[list[Tensor]] instead of list[Tensor]
        bad_noise = [[torch.randn(T, CHUNK_SIZE, MAX_ACTION_DIM)] for _ in range(M)]
        bad_time = [[torch.rand(T)] for _ in range(M)]

        with pytest.raises(AssertionError, match="fixed_noise must be Tensor"):
            ppo_update(
                policy=policy,
                ref_policy=ref_policy,
                optimizer=optimizer,
                trainable=[policy.weight],
                trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
                advantages=[2.0, -1.0],
                instrs_per_traj=["task", "task"],
                fixed_noise=bad_noise,
                fixed_time=bad_time,
                config=_make_ppo_config(),
            )

    def test_ppo_gradients_flow(self):
        """Optimizer actually updates the parameter (not a zero-gradient no-op)."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        ref_policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-2)
        noise, time = _make_ppo_noise_time(M, T)

        old_weight = policy.weight.data.clone()

        ppo_update(
            policy=policy,
            ref_policy=ref_policy,
            optimizer=optimizer,
            trainable=[policy.weight],
            trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_ppo_config(),
        )

        assert not torch.equal(policy.weight.data, old_weight), (
            "Parameter unchanged after optimizer step — gradient was zero"
        )


# ===================================================================
# PPO — Strategy B: exact arithmetic
# ===================================================================


class TestPPOArithmetic:
    """Exact arithmetic verification of surrogate loss, clipping, and KL."""

    def test_ppo_arithmetic_basic(self):
        """Matches hand-computed surrogate + KL for two trajectories.

        Traj 0 (fill=0.0, adv=+2.0): old_fm=new_fm=0.5 → ratio=1.0
        Traj 1 (fill=1.0, adv=-1.0): old_fm=0.5, new_fm=0.7 → ratio=exp(-0.2)
        ref_policy = policy so ref_losses == old_losses.
        """
        T = 3
        key0, key1 = 0.0, 1.0

        policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.5)},
            new_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.7)},
        )
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)
        noise, time = _make_ppo_noise_time(M=2, T=T)
        config = _make_ppo_config()

        metrics = ppo_update(
            policy=policy,
            ref_policy=policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[_make_traj(T, key0), _make_traj(T, key1)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=config,
        )

        # Trajectory 0: ratio=1.0, no clipping
        lr0 = 0.5 - 0.5
        r0 = math.exp(lr0)
        cr0 = max(1 - config.clip_epsilon, min(r0, 1 + config.clip_epsilon_high))
        loss_0 = -min(r0 * 2.0, cr0 * 2.0)
        kl_0 = config.kl_coeff * (0.5 * lr0 ** 2)

        # Trajectory 1: ratio=exp(-0.2), not clipped (0.818 > 0.8)
        lr1 = 0.5 - 0.7
        r1 = math.exp(lr1)
        cr1 = max(1 - config.clip_epsilon, min(r1, 1 + config.clip_epsilon_high))
        loss_1 = -min(r1 * (-1.0), cr1 * (-1.0))
        kl_1 = config.kl_coeff * (0.5 * lr1 ** 2)

        expected_loss = (loss_0 + loss_1) / 2.0
        expected_kl = (kl_0 + kl_1) / 2.0

        assert abs(metrics.avg_loss - expected_loss) < 1e-4, (
            f"avg_loss: expected {expected_loss:.6f}, got {metrics.avg_loss:.6f}"
        )
        assert abs(metrics.avg_kl - expected_kl) < 1e-4, (
            f"avg_kl: expected {expected_kl:.6f}, got {metrics.avg_kl:.6f}"
        )

        # Gradient actually flowed through scale
        assert policy.scale.item() != 1.0, "scale unchanged — gradient was zero"

    def test_ppo_independent_ref_policy(self):
        """Separate ref_policy produces a different KL than ref=policy.

        policy:     old_fm=0.5, new_fm=0.7
        ref_policy: old_fm=0.3

        log_ratio     = old - new = 0.5 - 0.7 = -0.2  (surrogate)
        log_ratio_ref = ref - new = 0.3 - 0.7 = -0.4  (KL, different!)
        """
        T = 3
        key0 = 0.0

        policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.5)},
            new_fm={key0: torch.full((T,), 0.7)},
        )
        ref_policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.3)},
            new_fm={key0: torch.full((T,), 0.3)},  # unused (ref only called under no_grad)
        )
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)
        noise, time = _make_ppo_noise_time(M=1, T=T)
        config = _make_ppo_config()

        metrics = ppo_update(
            policy=policy,
            ref_policy=ref_policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[_make_traj(T, key0)],
            advantages=[2.0],
            instrs_per_traj=["task"],
            fixed_noise=noise,
            fixed_time=time,
            config=config,
        )

        # Surrogate: log_ratio = 0.5 - 0.7 = -0.2
        lr = 0.5 - 0.7
        r = math.exp(lr)
        cr = max(1 - config.clip_epsilon, min(r, 1 + config.clip_epsilon_high))
        expected_loss = -min(r * 2.0, cr * 2.0)

        # KL with independent ref: log_ratio_ref = 0.3 - 0.7 = -0.4
        lr_ref = 0.3 - 0.7
        expected_kl = config.kl_coeff * (0.5 * lr_ref ** 2)

        # The collapsed case (ref=policy) would give kl = 0.1 * 0.5 * 0.04 = 0.002
        collapsed_kl = config.kl_coeff * (0.5 * lr ** 2)
        assert abs(expected_kl - collapsed_kl) > 1e-4, "test setup: KLs should differ"

        assert abs(metrics.avg_loss - expected_loss) < 1e-4, (
            f"avg_loss: expected {expected_loss:.6f}, got {metrics.avg_loss:.6f}"
        )
        assert abs(metrics.avg_kl - expected_kl) < 1e-4, (
            f"avg_kl: expected {expected_kl:.6f}, got {metrics.avg_kl:.6f}"
        )

    def test_ppo_empty_trajectories(self):
        """Empty trajectory list returns zero metrics without crashing."""
        policy = ArithmeticFakePolicy(old_fm={}, new_fm={})
        ref_policy = ArithmeticFakePolicy(old_fm={}, new_fm={})
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)

        metrics = ppo_update(
            policy=policy,
            ref_policy=ref_policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[],
            advantages=[],
            instrs_per_traj=[],
            fixed_noise=[],
            fixed_time=[],
            config=_make_ppo_config(),
        )

        assert metrics.avg_loss == 0.0
        assert metrics.avg_kl == 0.0


# ===================================================================
# FPO — Strategy A: type & gradient validation
# ===================================================================


class TestFPOTypeAndGradient:
    """Verify fpo_update receives correct types and produces real gradients."""

    def test_fpo_correct_types_no_crash(self):
        """fpo_update runs without error with correct list[list[Tensor]] types."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-3)
        noise, time = _make_fpo_noise_time(M, T, n_samples=1)

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.weight],
            trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_fpo_config(),
        )

        assert math.isfinite(metrics.avg_loss)
        assert math.isfinite(metrics.avg_kl)

    def test_fpo_gradients_flow(self):
        """Optimizer actually updates the parameter."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-2)
        noise, time = _make_fpo_noise_time(M, T, n_samples=1)

        old_weight = policy.weight.data.clone()

        fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.weight],
            trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_fpo_config(),
        )

        assert not torch.equal(policy.weight.data, old_weight), (
            "Parameter unchanged after optimizer step — gradient was zero"
        )

    def test_fpo_multi_sample_noise(self):
        """n_samples > 1 exercises _compute_fm_loss_multi_sample averaging."""
        T, M = 3, 2
        policy = GradientCheckPolicy()
        optimizer = torch.optim.AdamW([policy.weight], lr=1e-2)
        noise, time = _make_fpo_noise_time(M, T, n_samples=3)

        old_weight = policy.weight.data.clone()

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.weight],
            trajectories=[_make_traj(T, 0.0), _make_traj(T, 1.0)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_fpo_config(),
        )

        assert math.isfinite(metrics.avg_loss)
        assert not torch.equal(policy.weight.data, old_weight), (
            "Parameter unchanged after multi-sample update"
        )


# ===================================================================
# FPO — Strategy B: exact arithmetic
# ===================================================================


class TestFPOArithmetic:
    """Exact arithmetic verification of FPO surrogate and clipping."""

    def test_fpo_arithmetic_no_clipping(self):
        """Unclipped surrogate matches hand-computed values for two trajectories.

        Traj 0 (fill=0.0, adv=+2.0): old_fm=new_fm=0.5 → ratio=1.0
            loss_0 = -min(1.0*2.0, 1.0*2.0) = -2.0

        Traj 1 (fill=1.0, adv=-1.0): old_fm=0.5, new_fm=0.7 → ratio=exp(-0.2)
            clamp(0.818, 0.8, 1.28) = 0.818 (not clipped)
            loss_1 = -min(0.818*-1.0, 0.818*-1.0) = 0.818

        avg_loss = mean([-2.0, 0.818]) ≈ -0.591
        """
        T = 3
        key0, key1 = 0.0, 1.0

        policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.5)},
            new_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.7)},
        )
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)
        noise, time = _make_fpo_noise_time(M=2, T=T)
        config = _make_fpo_config()

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[_make_traj(T, key0), _make_traj(T, key1)],
            advantages=[2.0, -1.0],
            instrs_per_traj=["task", "task"],
            fixed_noise=noise,
            fixed_time=time,
            config=config,
        )

        r0 = math.exp(0.5 - 0.5)
        cr0 = max(1 - 0.2, min(r0, 1 + 0.28))
        loss_0 = -min(r0 * 2.0, cr0 * 2.0)

        r1 = math.exp(0.5 - 0.7)
        cr1 = max(1 - 0.2, min(r1, 1 + 0.28))
        loss_1 = -min(r1 * (-1.0), cr1 * (-1.0))

        expected = (loss_0 + loss_1) / 2.0

        assert abs(metrics.avg_loss - expected) < 1e-4, (
            f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
        )
        assert policy.scale.item() != 1.0, "scale unchanged — gradient was zero"

    def test_fpo_upper_clip_activates(self):
        """Upper clip: ratio=exp(0.4)≈1.492 clamped to 1.28.

        old_fm=0.5, new_fm=0.1, adv=+3.0
        surr1 = 1.492 * 3.0 = 4.476
        surr2 = 1.28  * 3.0 = 3.84   ← min
        loss = -3.84
        """
        T = 2
        key0 = 0.0

        policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.5)},
            new_fm={key0: torch.full((T,), 0.1)},
        )
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)
        noise, time = _make_fpo_noise_time(M=1, T=T)

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[_make_traj(T, key0)],
            advantages=[3.0],
            instrs_per_traj=["task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_fpo_config(),
        )

        ratio = math.exp(0.5 - 0.1)
        clipped = min(ratio, 1 + 0.28)  # 1.28, upper clip
        expected = -min(ratio * 3.0, clipped * 3.0)

        assert abs(metrics.avg_loss - expected) < 1e-4, (
            f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
        )

    def test_fpo_lower_clip_activates(self):
        """Lower clip: ratio=exp(-0.4)≈0.670 clamped to 0.8.

        old_fm=0.5, new_fm=0.9, adv=-2.0
        surr1 = 0.670 * -2.0 = -1.340
        surr2 = 0.800 * -2.0 = -1.600   ← min
        loss = -min(-1.340, -1.600) = -(-1.600) = 1.600
        """
        T = 2
        key0 = 0.0

        policy = ArithmeticFakePolicy(
            old_fm={key0: torch.full((T,), 0.5)},
            new_fm={key0: torch.full((T,), 0.9)},
        )
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)
        noise, time = _make_fpo_noise_time(M=1, T=T)

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[_make_traj(T, key0)],
            advantages=[-2.0],
            instrs_per_traj=["task"],
            fixed_noise=noise,
            fixed_time=time,
            config=_make_fpo_config(),
        )

        ratio = math.exp(0.5 - 0.9)
        clipped = max(ratio, 1 - 0.2)  # 0.8, lower clip
        expected = -min(ratio * (-2.0), clipped * (-2.0))

        assert abs(metrics.avg_loss - expected) < 1e-4, (
            f"Expected {expected:.6f}, got {metrics.avg_loss:.6f}"
        )

    def test_fpo_empty_trajectories(self):
        """Empty trajectory list returns zero metrics without crashing."""
        policy = ArithmeticFakePolicy(old_fm={}, new_fm={})
        optimizer = torch.optim.AdamW([policy.scale], lr=1e-3)

        metrics = fpo_update(
            policy=policy,
            optimizer=optimizer,
            trainable=[policy.scale],
            trajectories=[],
            advantages=[],
            instrs_per_traj=[],
            fixed_noise=[],
            fixed_time=[],
            config=_make_fpo_config(),
        )

        assert metrics.avg_loss == 0.0
        assert metrics.avg_kl == 0.0
        assert metrics.avg_weight == 0.0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
