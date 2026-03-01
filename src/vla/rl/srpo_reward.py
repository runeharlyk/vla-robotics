"""SRPO Tier A: state-based self-referential progress rewards.

Uses privileged ManiSkill simulator state features (cube height, gripper-to-cube
distance, etc.) to compute progress-wise rewards by comparing failed trajectories
against successful ones from the same batch.
"""

from __future__ import annotations

import torch

from vla.rl.rollout import Trajectory


def extract_state_features(states: torch.Tensor) -> torch.Tensor:
    """Extract a compact feature vector from raw state observations.

    The ManiSkill ``PickCube-v1`` state vector typically contains robot qpos/qvel
    followed by object pose.  We use the full state vector as the feature for
    trajectory comparison; this is Tier A (privileged state).

    Args:
        states: ``(T, state_dim)`` state tensor from a trajectory.

    Returns:
        ``(T, state_dim)`` normalised feature tensor.
    """
    mean = states.mean(dim=0, keepdim=True)
    std = states.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (states - mean) / std


def _align_and_distance(
    failed_feats: torch.Tensor,
    success_feats: torch.Tensor,
) -> torch.Tensor:
    """Compute per-timestep distance from a failed trajectory to a successful one.

    Trajectories may have different lengths; we linearly interpolate the successful
    trajectory to match the failed one's length.

    Args:
        failed_feats: ``(T_fail, D)`` features from the failed trajectory.
        success_feats: ``(T_succ, D)`` features from the successful trajectory.

    Returns:
        ``(T_fail,)`` L2 distances at each timestep.
    """
    T_fail = failed_feats.shape[0]
    T_succ = success_feats.shape[0]
    if T_succ == T_fail:
        aligned = success_feats
    else:
        aligned = torch.nn.functional.interpolate(
            success_feats.unsqueeze(0).permute(0, 2, 1),
            size=T_fail,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1).squeeze(0)
    return torch.norm(failed_feats - aligned, dim=-1)


def compute_srpo_rewards(
    trajectories: list[Trajectory],
    gamma: float = 0.99,
    reward_scale: float = 1.0,
) -> list[torch.Tensor]:
    """Compute SRPO Tier A progress-wise rewards for a batch of trajectories.

    Successful trajectories receive their original environment rewards.  Failed
    trajectories are compared against the *best* successful trajectory in the batch
    (smallest average distance in state-feature space) and assigned a shaped
    progress reward equal to the negative distance to that reference at each step.

    If the batch contains **no** successful trajectories the original environment
    rewards are returned unchanged.

    Args:
        trajectories: List of :class:`Trajectory` objects from the same batch.
        gamma: Discount factor (unused in Tier A, reserved for future extensions).
        reward_scale: Multiplier for the shaped progress reward.

    Returns:
        List of ``(T_i,)`` reward tensors, one per trajectory.
    """
    successes = [t for t in trajectories if t.success]
    failures = [t for t in trajectories if not t.success]

    if not successes:
        return [t.rewards.clone() for t in trajectories]

    success_feats = [extract_state_features(t.states[: t.length]) for t in successes]

    shaped_rewards: dict[int, torch.Tensor] = {}

    for idx, traj in enumerate(trajectories):
        if traj.success:
            shaped_rewards[idx] = traj.rewards[: traj.length].clone()
            continue

        fail_feats = extract_state_features(traj.states[: traj.length])
        best_dist = None
        for sf in success_feats:
            dist = _align_and_distance(fail_feats, sf)
            avg = dist.mean()
            if best_dist is None or avg < best_dist.mean():
                best_dist = dist

        progress = -best_dist * reward_scale
        progress = progress - progress.mean()
        progress_std = progress.std().clamp(min=1e-6)
        progress = progress / progress_std

        shaped_rewards[idx] = progress

    return [shaped_rewards[i] for i in range(len(trajectories))]


def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns for a single trajectory.

    Args:
        rewards: ``(T,)`` reward tensor.
        gamma: Discount factor.

    Returns:
        ``(T,)`` return tensor.
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns
