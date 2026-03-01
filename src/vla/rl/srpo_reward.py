"""SRPO world-progress reward model.

Implements the reward shaping mechanism from the SRPO paper (Section 3.2):
1. Encode trajectory observations with a frozen world-model encoder.
2. DBSCAN-cluster successful trajectory embeddings to obtain centres.
3. Compute distance-based progress rewards for failed trajectories.
4. Normalise rewards with batch statistics.

Supports a **demo-seeded** reference set: 5 (or more) demonstration
trajectories are always included as "successful" references, eliminating
the cold-start bootstrapping problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from sklearn.cluster import DBSCAN

from vla.models.world_model import WorldModelEncoder
from vla.rl.rollout import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class SRPORewardConfig:
    """Hyperparameters for the SRPO world-progress reward model."""

    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    activation: str = "sigmoid"
    eps: float = 1e-8


class WorldProgressReward:
    """Computes SRPO-style progress rewards using world-model embeddings.

    Args:
        encoder: A frozen :class:`WorldModelEncoder` (DINOv2 or V-JEPA 2).
        config: Reward model hyperparameters.
    """

    def __init__(self, encoder: WorldModelEncoder, config: SRPORewardConfig | None = None) -> None:
        self.encoder = encoder
        self.cfg = config or SRPORewardConfig()
        self.reference_embeddings: list[torch.Tensor] = []
        self.cluster_centers: torch.Tensor | None = None

    def add_demo_trajectories(self, demo_images: list[torch.Tensor]) -> None:
        """Encode demonstration trajectory images and add to the reference set.

        Args:
            demo_images: List of ``(T_i, C, H, W)`` image tensors (one per demo).
        """
        for imgs in demo_images:
            emb = self.encoder.encode_trajectory(imgs, self.cfg.subsample_every)
            self.reference_embeddings.append(emb)
        logger.info("Reference set seeded with %d demo trajectories", len(demo_images))
        self._refit_clusters()

    def add_successful_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Encode and add newly successful rollout trajectories to the reference set.

        Args:
            trajectories: List of successful :class:`Trajectory` objects.
        """
        added = 0
        for traj in trajectories:
            if not traj.success:
                continue
            imgs = (
                traj.images[: traj.length].float() / 255.0
                if traj.images.dtype == torch.uint8
                else traj.images[: traj.length].float()
            )
            emb = self.encoder.encode_trajectory(imgs, self.cfg.subsample_every)
            self.reference_embeddings.append(emb)
            added += 1
        if added > 0:
            logger.info(
                "Added %d in-batch successes to reference set (total=%d)", added, len(self.reference_embeddings)
            )
            self._refit_clusters()

    def _refit_clusters(self) -> None:
        """Run DBSCAN on the current reference embeddings to update cluster centres."""
        if len(self.reference_embeddings) < self.cfg.dbscan_min_samples:
            self.cluster_centers = torch.stack(self.reference_embeddings, dim=0)
            logger.info("Too few references for DBSCAN (%d); using all as centres.", len(self.reference_embeddings))
            return

        ref_matrix = torch.stack(self.reference_embeddings, dim=0)
        X = ref_matrix.cpu().numpy()
        db = DBSCAN(eps=self.cfg.dbscan_eps, min_samples=self.cfg.dbscan_min_samples).fit(X)
        labels = db.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) == 0:
            self.cluster_centers = ref_matrix
            logger.info(
                "DBSCAN found no clusters (all noise); using all %d references as centres.",
                len(self.reference_embeddings),
            )
            return

        centres = []
        for label in sorted(unique_labels):
            mask = labels == label
            centres.append(ref_matrix[mask].mean(dim=0))
        self.cluster_centers = torch.stack(centres, dim=0)
        logger.info("DBSCAN fitted: %d clusters from %d references", len(centres), len(self.reference_embeddings))

    def _encode_trajectory(self, traj: Trajectory) -> torch.Tensor:
        """Encode a trajectory's images into a trajectory-level embedding."""
        imgs = traj.images[: traj.length]
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        else:
            imgs = imgs.float()
        return self.encoder.encode_trajectory(imgs, self.cfg.subsample_every)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function φ(·) mapping to (0, 1)."""
        if self.cfg.activation == "sigmoid":
            return torch.sigmoid(-x)
        return torch.sigmoid(-x)

    def compute_trajectory_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> list[float]:
        """Compute per-trajectory SRPO rewards g_i.

        Following Section 3.2 of the paper:
        - Successful trajectories: g_i = 1.0
        - Failed trajectories: g_i = φ((d_i - d̄) / σ_d)

        Args:
            trajectories: Batch of trajectories from the current iteration.

        Returns:
            List of scalar rewards, one per trajectory.
        """
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            return [1.0 if t.success else 0.0 for t in trajectories]

        traj_embeddings = []
        for traj in trajectories:
            emb = self._encode_trajectory(traj)
            traj_embeddings.append(emb)

        centres = self.cluster_centers.to(traj_embeddings[0].device)
        rewards: list[float] = []
        failed_distances: list[torch.Tensor] = []
        failed_indices: list[int] = []

        for i, (traj, emb) in enumerate(zip(trajectories, traj_embeddings)):
            if traj.success:
                rewards.append(1.0)
            else:
                dists = torch.norm(centres - emb.unsqueeze(0), dim=-1)
                d_i = dists.min()
                failed_distances.append(d_i)
                failed_indices.append(i)
                rewards.append(0.0)

        if failed_distances:
            d_all = torch.stack(failed_distances)
            d_mean = d_all.mean()
            d_std = d_all.std().clamp(min=self.cfg.eps)
            normalised = (d_all - d_mean) / d_std
            activated = self._activation(normalised)
            for idx, fi in enumerate(failed_indices):
                rewards[fi] = activated[idx].item()

        return rewards


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
