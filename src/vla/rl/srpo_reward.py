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
from sklearn.preprocessing import StandardScaler

from vla.constants import DistanceMetric
from vla.models.world_model import WorldModelEncoder
from vla.rl.rollout import Trajectory
from vla.utils import to_float01

logger = logging.getLogger(__name__)


@dataclass
class SRPORewardConfig:
    """Hyperparameters for the SRPO world-progress reward model."""

    subsample_every: int = 1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = True
    dbscan_percentile: int = 25
    activation: str = "sigmoid"
    alpha: float = 0.8
    eps: float = 1e-8
    max_references: int = 200
    ref_demo_ratio: float = 0.5
    distance_metric: DistanceMetric = DistanceMetric.NORMALIZED_L2
    use_failure_rewards: bool = True
    use_standard_scaler: bool = False


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
        self._scaler: StandardScaler | None = None
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
        """Encode demonstration trajectories as video clips and add to the reference set.

        Per the SRPO paper (§3.2), each trajectory is encoded as a full
        video clip h_i = W(o_{0:T}) so the world model captures temporal
        dynamics.  Demo embeddings are permanent and never evicted.

        Args:
            demo_images: List of ``(T_i, C, H, W)`` image tensors (one per demo).
        """
        if demo_images:
            clip_embs = self._encode_trajectory_clips(demo_images)
            self._demo_embeddings.extend(clip_embs)
        logger.info(
            "Reference set seeded with %d demo trajectories (%d trajectory embeddings)",
            len(demo_images),
            len(self._demo_embeddings),
        )
        self._refit_clusters()

    def add_successful_trajectories(self, trajectories: list[Trajectory]) -> None:
        """Encode and add newly successful rollout trajectory embeddings to the reference set.

        Prefer :meth:`add_successful_embeddings` when embeddings have already
        been computed (e.g. from :meth:`compute_trajectory_rewards`) to avoid
        redundant encoder forward passes.

        Args:
            trajectories: List of successful :class:`Trajectory` objects.
        """
        success_imgs = []
        for traj in trajectories:
            if not traj.success:
                continue
            success_imgs.append(to_float01(traj.images[: traj.length]))
        if success_imgs:
            clip_embs = self._encode_trajectory_clips(success_imgs)
            self._insert_online(clip_embs)

    def add_successful_embeddings(self, embeddings: list[torch.Tensor]) -> None:
        """Add pre-computed trajectory-level embeddings of successful trajectories.

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
                "Online slot full - evicted %d oldest (kept %d, demo=%d, total=%d)",
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
        """Run DBSCAN on the current reference embeddings to update cluster centres.

        When ``use_standard_scaler`` is enabled in the config, embeddings are
        standardised before DBSCAN (matching the siiRL production code) and
        cluster centres are inverse-transformed back to original space.
        The fitted scaler is stored for consistent distance queries.
        """
        self._scaler = None

        if len(self.reference_embeddings) < self.cfg.dbscan_min_samples:
            self.cluster_centers = torch.stack(self.reference_embeddings, dim=0)
            self._last_labels = [0] * len(self.reference_embeddings)
            logger.info("Too few references for DBSCAN (%d); using all as centres.", len(self.reference_embeddings))
            return

        ref_matrix = torch.stack(self.reference_embeddings, dim=0)
        X = ref_matrix.cpu().numpy()

        if self.cfg.use_standard_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self._scaler = scaler
        else:
            X_scaled = X

        k = self.cfg.dbscan_min_samples
        if self.cfg.dbscan_auto_eps and len(X_scaled) > k:
            kth_dists = NearestNeighbors(n_neighbors=k).fit(X_scaled).kneighbors()[0][:, -1]
            eps = float(np.percentile(kth_dists, self.cfg.dbscan_percentile))
        else:
            eps = self.cfg.dbscan_eps

        db = DBSCAN(eps=eps, min_samples=k, metric="euclidean").fit(X_scaled)
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

        if self.cfg.use_standard_scaler and self._scaler is not None:
            centres = []
            for label in sorted(unique_labels):
                mask = labels == label
                center_scaled = X_scaled[mask].mean(axis=0, keepdims=True)
                center = self._scaler.inverse_transform(center_scaled).flatten()
                centres.append(torch.from_numpy(center).float())
        else:
            centres = []
            for label in sorted(unique_labels):
                mask = labels == label
                centres.append(ref_matrix[mask].mean(dim=0))

        self.cluster_centers = torch.stack(centres, dim=0)
        logger.info(
            "DBSCAN fitted: %d clusters from %d references (eps=%.4f, scaler=%s)",
            len(centres),
            len(self.reference_embeddings),
            eps,
            "on" if self.cfg.use_standard_scaler else "off",
        )

    def _encode_trajectory_clips(self, image_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode each trajectory as a video clip, returning one ``(D,)`` embedding per trajectory.

        Uses ``encoder.encode_trajectory`` so that V-JEPA 2's temporal
        modeling captures the full observation sequence ``o_{0:T}``, matching
        the paper's ``h_i = W(o_{0:T}^{(i)})``.
        """
        embs: list[torch.Tensor] = []
        for imgs in image_sequences:
            emb = self.encoder.encode_trajectory(to_float01(imgs), self.cfg.subsample_every)
            embs.append(emb)
        return embs

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function φ(·) mapping to (0, 1)."""
        if self.cfg.activation == "sigmoid":
            return torch.sigmoid(-x)
        logger.warning(
            "Unknown activation %r - falling back to 'sigmoid'. Choose from: 'sigmoid'.",
            self.cfg.activation,
        )
        return torch.sigmoid(-x)

    def _distances_to_centres(self, emb: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
        """Compute distances from a single embedding to all cluster centres.

        When ``use_standard_scaler`` is active, both the query embedding and
        cluster centres are transformed into the scaler's standardised space
        before computing distances.  This matches the siiRL production pipeline
        where DBSCAN operates in scaled space.

        Args:
            emb: ``(D,)`` trajectory embedding.
            centres: ``(K, D)`` cluster centre embeddings.

        Returns:
            ``(K,)`` distance tensor - lower means closer to a success cluster.
        """
        if self._scaler is not None:
            emb_np = emb.detach().cpu().numpy().reshape(1, -1)
            centres_np = centres.detach().cpu().numpy()
            emb_scaled = torch.from_numpy(self._scaler.transform(emb_np)).float().squeeze(0).to(emb.device)
            centres_scaled = torch.from_numpy(self._scaler.transform(centres_np)).float().to(centres.device)
            emb = emb_scaled
            centres = centres_scaled

        metric = self.cfg.distance_metric
        if metric is DistanceMetric.COSINE:
            sims = torch.nn.functional.cosine_similarity(centres, emb.unsqueeze(0).expand_as(centres), dim=-1)
            return 1.0 - sims
        if metric == DistanceMetric.NORMALIZED_L2:
            e = torch.nn.functional.normalize(emb.unsqueeze(0), dim=-1)
            c = torch.nn.functional.normalize(centres, dim=-1)
            return torch.norm(c - e, dim=-1)
        if metric == DistanceMetric.L2:
            return torch.norm(centres - emb.unsqueeze(0), dim=-1)
        raise ValueError(f"Unknown distance_metric {metric!r}. Choose from: l2, normalized_l2, cosine.")

    def compute_trajectory_rewards(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Compute per-trajectory SRPO rewards g_i using trajectory-level embeddings.

        Following Section 3.2 of the paper:
        - Each trajectory is encoded as a full video clip: h_i = W(o_{0:T})
        - Successful trajectories: g_i = 1.0
        - Failed trajectories: g_i = α · φ((d_i - d̄) / σ_d)
          where d_i = min distance from h_i to nearest cluster center.

        Args:
            trajectories: Batch of trajectories from the current iteration.

        Returns:
            Tuple of (rewards, trajectory_embeddings) where rewards is a list
            of scalar rewards and trajectory_embeddings is a list of ``(D,)``
            tensors (one per trajectory), allowing the caller to pass
            successful embeddings to :meth:`add_successful_embeddings`
            without re-encoding.
        """
        traj_images = [to_float01(t.images[: t.length]) for t in trajectories]
        traj_embs = self._encode_trajectory_clips(traj_images)

        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            rewards = [1.0 if t.success else 0.0 for t in trajectories]
            return rewards, traj_embs

        centres = self.cluster_centers.to(traj_embs[0].device)
        rewards: list[float] = []
        failed_distances: list[torch.Tensor] = []
        failed_indices: list[int] = []

        for i, (traj, emb) in enumerate(zip(trajectories, traj_embs, strict=True)):
            if traj.success:
                rewards.append(1.0)
            else:
                d_i = self._distances_to_centres(emb, centres).min()
                failed_distances.append(d_i)
                failed_indices.append(i)
                rewards.append(0.0)

        if failed_distances and self.cfg.use_failure_rewards:
            d_all = torch.stack(failed_distances)
            d_mean = d_all.mean()
            d_std = d_all.std(correction=0).clamp(min=self.cfg.eps)
            normalised = (d_all - d_mean) / d_std
            activated = self._activation(normalised) * self.cfg.alpha
            for idx, fi in enumerate(failed_indices):
                rewards[fi] = activated[idx].item()

        self._last_diagnostics = self._build_diagnostics(rewards, failed_distances)
        return rewards, traj_embs

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
            Tuple of (rewards, trajectory_embeddings) aligned with the input order.
        """
        rewards = [0.0] * len(trajectories)
        traj_embs: list[torch.Tensor] = [torch.empty(0)] * len(trajectories)

        by_task: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(trajectories):
            by_task[t.task_id].append(i)

        for tid, indices in by_task.items():
            task_trajs = [trajectories[i] for i in indices]
            task_rewards, task_embs = self._get_or_create(tid).compute_trajectory_rewards(task_trajs)
            for j, idx in enumerate(indices):
                rewards[idx] = task_rewards[j]
                traj_embs[idx] = task_embs[j]

        return rewards, traj_embs

    def get_diagnostics(self) -> dict[str, ClusterDiagnostics | None]:
        return {tid: rm.get_diagnostics() for tid, rm in self._per_task.items()}
