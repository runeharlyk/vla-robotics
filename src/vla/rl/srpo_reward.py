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
from collections import defaultdict
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
    alpha: float = 0.8
    eps: float = 1e-8


@dataclass
class ClusterDiagnostics:
    """Snapshot of cluster quality metrics for encoder comparison."""

    num_references: int = 0
    num_clusters: int = 0
    num_noise_points: int = 0
    mean_intra_cluster_dist: float = 0.0
    mean_inter_cluster_dist: float = 0.0
    silhouette_ratio: float = 0.0
    mean_failed_distance: float = 0.0
    std_failed_distance: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    def as_dict(self, prefix: str = "cluster") -> dict[str, float]:
        return {f"{prefix}/{k}": v for k, v in self.__dict__.items()}


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
        self._last_labels: list[int] | None = None
        self._last_diagnostics: ClusterDiagnostics | None = None

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

        Prefer :meth:`add_successful_embeddings` when embeddings have already
        been computed (e.g. from :meth:`compute_trajectory_rewards`) to avoid
        redundant encoder forward passes.

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

    def add_successful_embeddings(self, embeddings: list[torch.Tensor]) -> None:
        """Add pre-computed embeddings of successful trajectories to the reference set.

        This avoids re-encoding trajectories that were already encoded
        during :meth:`compute_trajectory_rewards`.

        Args:
            embeddings: List of ``(D,)`` trajectory embeddings.
        """
        if not embeddings:
            return
        self.reference_embeddings.extend(embeddings)
        logger.info(
            "Added %d in-batch successes to reference set (total=%d)",
            len(embeddings),
            len(self.reference_embeddings),
        )
        self._refit_clusters()

    def _refit_clusters(self) -> None:
        """Run DBSCAN on the current reference embeddings to update cluster centres."""
        if len(self.reference_embeddings) < self.cfg.dbscan_min_samples:
            self.cluster_centers = torch.stack(self.reference_embeddings, dim=0)
            self._last_labels = [0] * len(self.reference_embeddings)
            logger.info("Too few references for DBSCAN (%d); using all as centres.", len(self.reference_embeddings))
            return

        ref_matrix = torch.stack(self.reference_embeddings, dim=0)
        X = ref_matrix.cpu().numpy()
        db = DBSCAN(eps=self.cfg.dbscan_eps, min_samples=self.cfg.dbscan_min_samples).fit(X)
        labels = db.labels_
        self._last_labels = labels.tolist()
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
        imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
        return self.encoder.encode_trajectory(imgs, self.cfg.subsample_every)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function φ(·) mapping to (0, 1)."""
        if self.cfg.activation == "sigmoid":
            return torch.sigmoid(-x)
        return torch.sigmoid(-x)

    def compute_trajectory_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Compute per-trajectory SRPO rewards g_i.

        Following Section 3.2 of the paper:
        - Successful trajectories: g_i = 1.0
        - Failed trajectories: g_i = α · φ((d_i - d̄) / σ_d)  (α=0.8 by default)

        Args:
            trajectories: Batch of trajectories from the current iteration.

        Returns:
            Tuple of (rewards, embeddings) where rewards is a list of scalar
            rewards and embeddings is the list of trajectory-level embeddings
            (one per trajectory), allowing the caller to pass successful
            embeddings to :meth:`add_successful_embeddings` without
            re-encoding.
        """
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            traj_embeddings = [self._encode_trajectory(t) for t in trajectories]
            rewards = [1.0 if t.success else 0.0 for t in trajectories]
            return rewards, traj_embeddings

        traj_embeddings = []
        for traj in trajectories:
            emb = self._encode_trajectory(traj)
            traj_embeddings.append(emb)

        centres = self.cluster_centers.to(traj_embeddings[0].device)
        rewards: list[float] = []
        failed_distances: list[torch.Tensor] = []
        failed_indices: list[int] = []

        for i, (traj, emb) in enumerate(zip(trajectories, traj_embeddings, strict=True)):
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
            d_std = d_all.std(correction=0).clamp(min=self.cfg.eps)
            normalised = (d_all - d_mean) / d_std
            activated = self._activation(normalised) * self.cfg.alpha
            for idx, fi in enumerate(failed_indices):
                rewards[fi] = activated[idx].item()

        self._last_diagnostics = self._build_diagnostics(rewards, failed_distances)
        return rewards, traj_embeddings

    def _build_diagnostics(
        self,
        rewards: list[float],
        failed_distances: list[torch.Tensor],
    ) -> ClusterDiagnostics:
        diag = ClusterDiagnostics(
            num_references=len(self.reference_embeddings),
            num_clusters=self.cluster_centers.shape[0] if self.cluster_centers is not None else 0,
        )

        if self._last_labels is not None:
            diag.num_noise_points = sum(1 for lb in self._last_labels if lb == -1)

        if len(self.reference_embeddings) >= 2 and self._last_labels is not None:
            ref_matrix = torch.stack(self.reference_embeddings, dim=0)
            labels_t = torch.tensor(self._last_labels)
            unique = set(self._last_labels)
            unique.discard(-1)

            intra_dists: list[float] = []
            for lb in sorted(unique):
                mask = labels_t == lb
                members = ref_matrix[mask]
                if members.shape[0] < 2:
                    continue
                pw = torch.cdist(members.unsqueeze(0), members.unsqueeze(0)).squeeze(0)
                n = members.shape[0]
                intra_dists.append(pw.sum().item() / max(n * (n - 1), 1))
            if intra_dists:
                diag.mean_intra_cluster_dist = sum(intra_dists) / len(intra_dists)

            if self.cluster_centers is not None and self.cluster_centers.shape[0] >= 2:
                cc = self.cluster_centers
                pw_cc = torch.cdist(cc.unsqueeze(0), cc.unsqueeze(0)).squeeze(0)
                n_c = cc.shape[0]
                diag.mean_inter_cluster_dist = pw_cc.sum().item() / max(n_c * (n_c - 1), 1)

            if diag.mean_intra_cluster_dist > 0:
                diag.silhouette_ratio = diag.mean_inter_cluster_dist / diag.mean_intra_cluster_dist

        if failed_distances:
            d_all = torch.stack(failed_distances)
            diag.mean_failed_distance = d_all.mean().item()
            diag.std_failed_distance = d_all.std().item() if len(failed_distances) > 1 else 0.0

        r_t = torch.tensor(rewards)
        diag.reward_mean = r_t.mean().item()
        diag.reward_std = r_t.std().item() if len(rewards) > 1 else 0.0
        diag.reward_min = r_t.min().item()
        diag.reward_max = r_t.max().item()

        return diag

    def get_diagnostics(self) -> ClusterDiagnostics | None:
        """Return the diagnostics from the most recent ``compute_trajectory_rewards`` call."""
        return self._last_diagnostics


class MultiTaskWorldProgressReward:
    """Per-task SRPO reward models sharing a single frozen encoder.

    Maintains a separate :class:`WorldProgressReward` instance for each
    task so that reference embeddings, DBSCAN clusters, and reward
    normalisation are all task-isolated.

    Args:
        encoder: A frozen :class:`WorldModelEncoder` (shared across tasks).
        config: Reward model hyperparameters (shared across tasks).
    """

    def __init__(self, encoder: WorldModelEncoder, config: SRPORewardConfig | None = None) -> None:
        self.encoder = encoder
        self.cfg = config or SRPORewardConfig()
        self._per_task: dict[str, WorldProgressReward] = {}

    def _get_or_create(self, task_id: str) -> WorldProgressReward:
        if task_id not in self._per_task:
            self._per_task[task_id] = WorldProgressReward(self.encoder, self.cfg)
        return self._per_task[task_id]

    @property
    def task_ids(self) -> list[str]:
        return list(self._per_task.keys())

    def add_demo_trajectories(self, task_id: str, demo_images: list[torch.Tensor]) -> None:
        self._get_or_create(task_id).add_demo_trajectories(demo_images)

    def add_successful_embeddings(self, task_id: str, embeddings: list[torch.Tensor]) -> None:
        self._get_or_create(task_id).add_successful_embeddings(embeddings)

    def compute_trajectory_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Compute rewards with per-task clustering and normalisation.

        Trajectories are grouped by ``task_id``, each group is scored
        against its own task's cluster centres, and failed-distance
        z-scoring is computed within each task independently.

        Returns:
            Tuple of (rewards, embeddings) aligned with the input order.
        """
        rewards = [0.0] * len(trajectories)
        embeddings: list[torch.Tensor | None] = [None] * len(trajectories)

        by_task: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(trajectories):
            by_task[t.task_id].append(i)

        for tid, indices in by_task.items():
            task_trajs = [trajectories[i] for i in indices]
            task_rewards, task_embs = self._get_or_create(tid).compute_trajectory_rewards(task_trajs)
            for j, idx in enumerate(indices):
                rewards[idx] = task_rewards[j]
                embeddings[idx] = task_embs[j]

        return rewards, embeddings  # type: ignore[return-value]

    def get_diagnostics(self) -> dict[str, ClusterDiagnostics | None]:
        return {tid: rm.get_diagnostics() for tid, rm in self._per_task.items()}


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
