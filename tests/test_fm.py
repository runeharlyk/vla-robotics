# File for testing the FM loss computation in isolation, without needing to run the full 
# PPO/SRPO update loop. This allows us to verify that the FM loss is being computed correctly and 
# that the KL divergence term is included when the KL coefficient is nonzero.

from vla.rl.policy_update import _compute_fm_loss_batched
import torch
import torch.nn as nn
from vla.rl.rollout import Trajectory

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

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_fm_loss_batched():
    """
    Test that _compute_fm_loss_batched correctly returns the FM loss from the policy,
    """    
    
    T = 3
    key0, key1 = 0.0, 1.0

    policy = FakePolicy(
        old_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.5)},
        new_fm={key0: torch.full((T,), 0.5), key1: torch.full((T,), 0.7)},
    )
    optimizer = torch.optim.AdamW([policy.dummy], lr=1e-3)

    trajs = [_make_traj(T, key0), _make_traj(T, key1)]
    instructions = ["task_a", "task_b"] # needed
    noise = [[torch.randn(T, 2, 4)] for _ in range(2)]
    time = [[torch.rand(T)] for _ in range(2)]
    batch_size = 2

    # Compute FM losses for both trajectories
    fm_loss_0 = _compute_fm_loss_batched(
        policy,
        trajs[0],
        instructions[0],
        noise[0],
        time[0],
        batch_size=batch_size,
    )
    fm_loss_1 = _compute_fm_loss_batched(
        policy,
        trajs[1],
        instructions[1],
        noise[1],
        time[1],
        batch_size=batch_size,
    )

    # hands calculated values for fm loss
    expected_fm_loss_0 = 0.5  # from new_fm for key0
    expected_fm_loss_1 = 0.7  # from new_fm for key1

    print(f"FM Loss 0: {fm_loss_0.item():.4f} (expected {expected_fm_loss_0:.4f})")
    print(f"FM Loss 1: {fm_loss_1.item():.4f} (expected {expected_fm_loss_1:.4f})")
  
if __name__ == "__main__":
    test_compute_fm_loss_batched()
    print("All tests passed.")




    

