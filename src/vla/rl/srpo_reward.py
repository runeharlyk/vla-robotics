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

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from vla.constants import DistanceMetrics
from vla.models.world_model import WorldModelEncoder
from vla.rl.rollout import Trajectory
from vla.utils import to_float01

logger = logging.getLogger(__name__)


@dataclass
class SRPORewardConfig:
    """Hyperparameters for the SRPO world-progress reward model."""

    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = True
    dbscan_percentile: int = 25
    activation: str = "sigmoid"
    alpha: float = 0.8
    eps: float = 1e-8
    max_references: int = 200
    ref_demo_ratio: float = 0.5
    distance_metric: str = DistanceMetrics.normalized_l2


@dataclass
class ClusterDiagnostics:
    """Snapshot of cluster quality metrics for encoder comparison."""

    num_references: int = 0
    num_demo_refs: int = 0
    num_online_refs: int = 0
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
        self._demo_embeddings: list[torch.Tensor] = []
        self._online_embeddings: list[torch.Tensor] = []
        self.cluster_centers: torch.Tensor | None = None
        self._last_labels: list[int] | None = None
        self._last_diagnostics: ClusterDiagnostics | None = None

    @property
    def reference_embeddings(self) -> list[torch.Tensor]:
        return self._demo_embeddings + self._online_embeddings

    @property
    def _online_capacity(self) -> int:
        demo_slots = max(len(self._demo_embeddings), int(self.cfg.max_references * self.cfg.ref_demo_ratio))
        return max(self.cfg.max_references - demo_slots, 0)

    def add_demo_trajectories(self, demo_images: list[torch.Tensor]) -> None:
        """Encode demonstration trajectory images and add to the reference set.

        Demo embeddings are permanent and never evicted.

        Args:
            demo_images: List of ``(T_i, C, H, W)`` image tensors (one per demo).
        """
        if demo_images:
            embs = self.encoder.encode_trajectories(demo_images, self.cfg.subsample_every)
            self._demo_embeddings.extend(embs.unbind(0))
        logger.info(
            "Reference set seeded with %d demo trajectories (demo_slots=%d)",
            len(demo_images),
            len(self._demo_embeddings),
        )
        self._refit_clusters()

    def add_successful_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Encode and add newly successful rollout trajectories to the reference set.

        Prefer :meth:`add_successful_embeddings` when embeddings have already
        been computed (e.g. from :meth:`compute_trajectory_rewards`) to avoid
        redundant encoder forward passes.

        Args:
            trajectories: List of successful :class:`Trajectory` objects.
        """
        new_embs = []
        for traj in trajectories:
            if not traj.success:
                continue
            imgs = to_float01(traj.images[: traj.length])
            new_embs.append(self.encoder.encode_trajectory(imgs, self.cfg.subsample_every))
        if new_embs:
            self._insert_online(new_embs)

    def add_successful_embeddings(self, embeddings: list[torch.Tensor]) -> None:
        """Add pre-computed embeddings of successful trajectories to the reference set.

        This avoids re-encoding trajectories that were already encoded
        during :meth:`compute_trajectory_rewards`.

        Args:
            embeddings: List of ``(D,)`` trajectory embeddings.
        """
        if not embeddings:
            return
        self._insert_online(embeddings)

    def _insert_online(self, embeddings: list[torch.Tensor]) -> None:
        """Insert online embeddings, evicting oldest when the online slot is full."""
        self._online_embeddings.extend(embeddings)
        cap = self._online_capacity
        if len(self._online_embeddings) > cap:
            evicted = len(self._online_embeddings) - cap
            self._online_embeddings = self._online_embeddings[-cap:]
            logger.info(
                "Online slot full — evicted %d oldest (kept %d, demo=%d, total=%d)",
                evicted,
                cap,
                len(self._demo_embeddings),
                len(self._demo_embeddings) + cap,
            )
        logger.info(
            "Added %d online successes (demo=%d, online=%d, total=%d)",
            len(embeddings),
            len(self._demo_embeddings),
            len(self._online_embeddings),
            len(self._demo_embeddings) + len(self._online_embeddings),
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

        k = self.cfg.dbscan_min_samples
        if self.cfg.dbscan_auto_eps and len(X) > k:
            kth_dists = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()[0][:, -1]
            eps = float(np.percentile(kth_dists, self.cfg.dbscan_percentile))
        else:
            eps = self.cfg.dbscan_eps

        db = DBSCAN(eps=eps, min_samples=k, metric="euclidean").fit(X)
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
        logger.info(
            "DBSCAN fitted: %d clusters from %d references (eps=%.4f)",
            len(centres),
            len(self.reference_embeddings),
            eps,
        )

    def _encode_trajectories_batched(self, trajectories: list[Trajectory]) -> list[torch.Tensor]:
        """Encode all trajectories in a single batched pass."""
        all_imgs = []
        for traj in trajectories:
            all_imgs.append(to_float01(traj.images[: traj.length]))
        embs = self.encoder.encode_trajectories(all_imgs, self.cfg.subsample_every)
        return list(embs.unbind(0))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function φ(·) mapping to (0, 1)."""
        if self.cfg.activation == "sigmoid":
            return torch.sigmoid(-x)
        logger.warning(
            "Unknown activation %r — falling back to 'sigmoid'. Choose from: 'sigmoid'.",
            self.cfg.activation,
        )
        return torch.sigmoid(-x)

    def _distances_to_centres(self, emb: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
        """Compute distances from a single embedding to all cluster centres.

        Args:
            emb: ``(D,)`` trajectory embedding.
            centres: ``(K, D)`` cluster centre embeddings.

        Returns:
            ``(K,)`` distance tensor — lower means closer to a success cluster.
        """
        metric = self.cfg.distance_metric
        if metric == DistanceMetrics.cosine:
            sims = torch.nn.functional.cosine_similarity(centres, emb.unsqueeze(0).expand_as(centres), dim=-1)
            return 1.0 - sims
        if metric == DistanceMetrics.normalized_l2:
            e = torch.nn.functional.normalize(emb.unsqueeze(0), dim=-1)
            c = torch.nn.functional.normalize(centres, dim=-1)
            return torch.norm(c - e, dim=-1)
        if metric == DistanceMetrics.l2:
            return torch.norm(centres - emb.unsqueeze(0), dim=-1)
        raise ValueError(
            f"Unknown distance_metric {metric!r}. "
            f"Choose from: {DistanceMetrics.cosine}, {DistanceMetrics.normalized_l2}, {DistanceMetrics.l2}."
        )

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
            traj_embeddings = self._encode_trajectories_batched(trajectories)
            rewards = [1.0 if t.success else 0.0 for t in trajectories]
            return rewards, traj_embeddings

        traj_embeddings = self._encode_trajectories_batched(trajectories)

        centres = self.cluster_centers.to(traj_embeddings[0].device)
        rewards: list[float] = []
        failed_distances: list[torch.Tensor] = []
        failed_indices: list[int] = []

        for i, (traj, emb) in enumerate(zip(trajectories, traj_embeddings, strict=True)):
            if traj.success:
                rewards.append(1.0)
            else:
                d_i = self._distances_to_centres(emb, centres).min()
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
            num_demo_refs=len(self._demo_embeddings),
            num_online_refs=len(self._online_embeddings),
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
