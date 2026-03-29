"""Clustering diagnostics: trajectory embedding analysis and visualisation.

Pure analysis module — takes pre-collected trajectories and produces:

  Panel A — UMAP coloured by DBSCAN cluster (noise → light gray)
  Panel B — Same UMAP coords coloured by trajectory source
  Panel C — Per-cluster composition bar chart
  Panel D — Cluster prototype frames (medoids)

Data collection is handled separately by
:mod:`vla.diagnostics.collect_trajectories`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None  # type: ignore[assignment]

from vla.models.world_model import WorldModelEncoder
from vla.rl.rollout import Trajectory
from vla.utils.tensor import to_float01

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ClusteringConfig:
    """Configuration for the clustering analysis (not collection)."""

    # Embedding
    subsample_every: int = 1

    # DBSCAN
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_auto_eps: bool = True
    dbscan_percentile: int = 50

    # UMAP
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Display
    seed: int = 42
    cache_dir: Path = field(default_factory=lambda: Path("notebooks/cache"))


# ──────────────────────────────────────────────────────────────────────
# Source labels
# ──────────────────────────────────────────────────────────────────────

SOURCE_DEMO = "success demo"
SOURCE_RANDOM = "random-action rollout"
SOURCE_FAILED = "failed rollout (SFT)"
SOURCE_SFT_SUCCESS = "success rollout (SFT)"

SOURCE_COLORS = {
    SOURCE_DEMO: "#2ecc71",
    SOURCE_RANDOM: "#e74c3c",
    SOURCE_FAILED: "#f39c12",
    SOURCE_SFT_SUCCESS: "#3498db",
}

SOURCE_MARKERS = {
    SOURCE_DEMO: "o",
    SOURCE_RANDOM: "^",
    SOURCE_FAILED: "s",
    SOURCE_SFT_SUCCESS: "D",
}


# ──────────────────────────────────────────────────────────────────────
# Embedding cache helpers
# ──────────────────────────────────────────────────────────────────────


def save_embeddings(embs: torch.Tensor, path: Path) -> None:
    """Save embeddings to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, path)
    logger.info("Saved embeddings %s → %s", embs.shape, path)


def load_embeddings(path: Path) -> torch.Tensor | None:
    """Load cached embeddings, or return ``None``."""
    if path.exists():
        embs = torch.load(path, weights_only=False)
        logger.info("Loaded embeddings %s from %s", embs.shape, path)
        return embs
    return None


# ──────────────────────────────────────────────────────────────────────
# Embedding computation
# ──────────────────────────────────────────────────────────────────────


def compute_embeddings(
    trajs: list[Trajectory],
    encoder: WorldModelEncoder,
    subsample_every: int = 5,
) -> torch.Tensor:
    """Encode trajectories into a ``(N, D)`` embedding matrix."""
    all_imgs = [to_float01(t.images[: t.length]) for t in trajs]
    return encoder.encode_trajectories(all_imgs, subsample_every)


def get_or_compute_embeddings(
    trajs: list[Trajectory],
    encoder: WorldModelEncoder,
    cache_path: Path,
    subsample_every: int = 5,
) -> torch.Tensor:
    """Load cached trajectory embeddings or compute + cache them."""
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        if isinstance(data, torch.Tensor) and data.shape[0] == len(trajs):
            logger.info("Loaded embeddings %s from %s", data.shape, cache_path)
            return data
        logger.info("Cache %s has old/mismatched format, recomputing...", cache_path)
    embs = compute_embeddings(trajs, encoder, subsample_every)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, cache_path)
    logger.info("Saved embeddings %s → %s", embs.shape, cache_path)
    return embs


# ──────────────────────────────────────────────────────────────────────
# DBSCAN + UMAP
# ──────────────────────────────────────────────────────────────────────


def fit_dbscan(X: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """Run DBSCAN, optionally auto-tuning eps via k-NN distances."""
    k = cfg.dbscan_min_samples
    if cfg.dbscan_auto_eps and len(X) > k:
        kth_dists = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()[0][:, -1]
        eps = float(np.percentile(kth_dists, cfg.dbscan_percentile))
    else:
        eps = cfg.dbscan_eps
    logger.info("DBSCAN eps=%.4f, min_samples=%d", eps, k)
    db = DBSCAN(eps=eps, min_samples=k, metric="euclidean").fit(X)
    labels = db.labels_
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info("DBSCAN → %d clusters, %d noise points out of %d", n_clusters, n_noise, len(X))
    return labels


def fit_dbscan_reference(
    X_all: np.ndarray,
    reference_mask: np.ndarray,
    cfg: ClusteringConfig,
) -> np.ndarray:
    """Cluster reference embeddings only, assign others by nearest cluster center.

    This matches the SRPO repo approach: clusters represent successful
    trajectories. Failed/random points are assigned to the nearest cluster.

    Args:
        X_all: ``(N, D)`` full embedding matrix.
        reference_mask: Boolean array indicating reference points.
        cfg: Clustering configuration.

    Returns:
        ``(N,)`` array of cluster labels for all points.
    """
    X_ref = X_all[reference_mask]
    ref_labels = fit_dbscan(X_ref, cfg)

    # Compute cluster centers from reference points
    unique_clusters = sorted(set(ref_labels) - {-1})
    if not unique_clusters:
        logger.warning("No clusters found in reference set — returning all noise")
        return np.full(len(X_all), -1, dtype=int)

    centers = np.array([X_ref[ref_labels == cl].mean(axis=0) for cl in unique_clusters])

    # Assign all points to nearest cluster center
    all_labels = np.full(len(X_all), -1, dtype=int)
    
    # Keep original reference labels (including their noise assignments)
    ref_indices = np.where(reference_mask)[0]
    for i, idx in enumerate(ref_indices):
        all_labels[idx] = ref_labels[i]

    # Assign non-reference points to nearest cluster center
    non_ref_mask = ~reference_mask
    X_rest = X_all[non_ref_mask]
    if len(X_rest) > 0:
        dists_to_centers = np.linalg.norm(
            X_rest[:, None, :] - centers[None, :, :], axis=2,
        )  # (N_rest, n_clusters)
        nearest_idx = dists_to_centers.argmin(axis=1)
        
        non_ref_indices = np.where(non_ref_mask)[0]
        for i, idx in enumerate(non_ref_indices):
            all_labels[idx] = unique_clusters[nearest_idx[i]]

    n_clusters = len(unique_clusters)
    n_noise = int((all_labels == -1).sum())
    logger.info(
        "Reference clustering → %d clusters, %d noise out of %d total",
        n_clusters, n_noise, len(X_all),
    )
    return all_labels


def print_composition_table(
    labels: np.ndarray,
    sources: list[str],
) -> None:
    """Print a numeric composition table (cluster × source) to the logger."""
    unique_clusters = sorted(set(labels))
    src_types = [SOURCE_DEMO, SOURCE_SFT_SUCCESS, SOURCE_FAILED, SOURCE_RANDOM]
    src_short = {SOURCE_DEMO: "Demo", SOURCE_SFT_SUCCESS: "SFT✓", SOURCE_FAILED: "SFT✗", SOURCE_RANDOM: "Random"}

    header = f"{'Cluster':>10s}" + "".join(f"{src_short[s]:>10s}" for s in src_types) + f"{'Total':>10s}"
    logger.info("\n" + header)
    logger.info("-" * len(header))

    for cl in unique_clusters:
        mask = labels == cl
        cl_sources = [sources[i] for i in range(len(sources)) if mask[i]]
        name = "noise" if cl == -1 else f"C{cl}"
        counts = [sum(1 for x in cl_sources if x == s) for s in src_types]
        total = sum(counts)
        row = f"{name:>10s}" + "".join(f"{c:>10d}" for c in counts) + f"{total:>10d}"
        logger.info(row)

    # Totals row
    totals = [sum(1 for s in sources if s == st) for st in src_types]
    total_row = f"{'TOTAL':>10s}" + "".join(f"{c:>10d}" for c in totals) + f"{sum(totals):>10d}"
    logger.info("-" * len(header))
    logger.info(total_row)


def fit_umap(X: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """Reduce to 2D via UMAP (falls back to PCA if umap not installed)."""
    if umap is None:
        from sklearn.decomposition import PCA

        logger.warning("umap not installed, falling back to PCA")
        return PCA(n_components=2).fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.seed,
        metric="euclidean",
    )
    return reducer.fit_transform(X)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


def _get_frame(traj: Trajectory, frame_idx: int | None = None) -> np.ndarray:
    """Extract a single (H, W, 3) uint8 frame from a trajectory."""
    if frame_idx is None:
        frame_idx = traj.length // 2
    img = traj.images[frame_idx]
    if img.ndim == 4:
        img = img[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def plot_panel_a(ax, xy: np.ndarray, labels: np.ndarray) -> None:
    """Panel A: UMAP coloured by DBSCAN cluster ID, noise in light gray."""
    unique = sorted(set(labels))
    n_clusters = len([l for l in unique if l >= 0])
    base_cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    for label in unique:
        mask = labels == label
        if label == -1:
            ax.scatter(xy[mask, 0], xy[mask, 1], c="#d5d5d5", s=8, alpha=0.4, label="noise", zorder=1)
        else:
            color = base_cmap(label % 20)
            ax.scatter(
                xy[mask, 0], xy[mask, 1], c=[color], s=18, alpha=0.7,
                edgecolors="white", linewidths=0.3, label=f"cluster {label}", zorder=2,
            )

    ax.set_title("A — UMAP by DBSCAN Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) <= 15:
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.7)


def plot_panel_b(ax, xy: np.ndarray, sources: list[str]) -> None:
    """Panel B: same UMAP coords coloured by trajectory source."""
    for src in [SOURCE_RANDOM, SOURCE_FAILED, SOURCE_SFT_SUCCESS, SOURCE_DEMO]:
        mask = np.array([s == src for s in sources])
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=SOURCE_COLORS[src], marker=SOURCE_MARKERS[src],
            s=18, alpha=0.7, edgecolors="white", linewidths=0.3,
            label=src, zorder=3 if src == SOURCE_DEMO else 2,
        )
    ax.set_title("B — UMAP by Trajectory Source", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, loc="best", framealpha=0.7)


def plot_panel_c(ax, labels: np.ndarray, sources: list[str]) -> None:
    """Panel C: per-cluster composition stacked bar chart."""
    unique_clusters = sorted(set(labels))
    src_types = [s for s in [SOURCE_DEMO, SOURCE_SFT_SUCCESS, SOURCE_RANDOM, SOURCE_FAILED] if s in sources]

    cluster_names = []
    counts: dict[str, list[int]] = {s: [] for s in src_types}

    for cl in unique_clusters:
        mask = labels == cl
        cl_sources = [sources[i] for i in range(len(sources)) if mask[i]]
        cluster_names.append("noise" if cl == -1 else f"C{cl}")
        for s in src_types:
            counts[s].append(sum(1 for x in cl_sources if x == s))

    x = np.arange(len(cluster_names))
    bottom = np.zeros(len(cluster_names))

    for src in src_types:
        vals = np.array(counts[src])
        ax.bar(x, vals, 0.6, bottom=bottom, label=src, color=SOURCE_COLORS[src], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cluster_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("C — Per-Cluster Composition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="best", framealpha=0.7)


def plot_panel_d(
    ax,
    labels: np.ndarray,
    X: np.ndarray,
    all_trajs: list[Trajectory],
    traj_indices: list[int],
    max_clusters: int = 6,
    frames_per_cluster: int = 6,
) -> None:
    """Panel D: cluster prototype frames (medoids)."""
    unique_clusters = sorted(set(labels) - {-1})
    cluster_sizes = [(cl, int((labels == cl).sum())) for cl in unique_clusters]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    top_clusters = [cl for cl, _ in cluster_sizes[:max_clusters]]

    if not top_clusters:
        ax.text(0.5, 0.5, "No clusters found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return

    thumb_h, thumb_w = 64, 64
    n_rows = len(top_clusters)
    n_cols = frames_per_cluster
    grid = np.ones((n_rows * (thumb_h + 2) + 2, n_cols * (thumb_w + 2) + 2, 3), dtype=np.uint8) * 240

    for row_i, cl in enumerate(top_clusters):
        mask = labels == cl
        cl_indices = np.where(mask)[0]
        cl_embs = X[cl_indices]
        centroid = cl_embs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(cl_embs - centroid, axis=1)
        nearest = np.argsort(dists)[:frames_per_cluster]

        for col_j, idx in enumerate(nearest):
            traj_idx = traj_indices[cl_indices[idx]]
            frame = _get_frame(all_trajs[traj_idx])
            frame = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            y0 = row_i * (thumb_h + 2) + 1
            x0 = col_j * (thumb_w + 2) + 1
            grid[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = frame

    ax.imshow(grid)
    for row_i, cl in enumerate(top_clusters):
        y_center = row_i * (thumb_h + 2) + thumb_h // 2 + 1
        ax.text(-5, y_center, f"C{cl}", ha="right", va="center", fontsize=8, fontweight="bold")
    ax.set_title("D — Cluster Prototype Frames (Medoids)", fontsize=13, fontweight="bold")
    ax.axis("off")


def plot_clustering_figure(
    xy: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    X: np.ndarray,
    all_trajs: list[Trajectory],
    traj_indices: list[int],
    title: str = "Trajectory Embedding Analysis",
    save_path: Path | None = None,
) -> plt.Figure:
    """Generate the full 4-panel clustering figure.

    Args:
        xy: ``(N, 2)`` UMAP coordinates.
        labels: ``(N,)`` DBSCAN cluster labels.
        sources: Per-point trajectory source strings.
        X: ``(N, D)`` raw embedding matrix (for medoid computation).
        all_trajs: All trajectories (for frame extraction).
        traj_indices: Mapping from embedding index → trajectory index.
        title: Figure title.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Figure.
    """
    n_demo = sum(1 for s in sources if s == SOURCE_DEMO)
    n_sft_ok = sum(1 for s in sources if s == SOURCE_SFT_SUCCESS)
    n_sft_fail = sum(1 for s in sources if s == SOURCE_FAILED)
    n_random = sum(1 for s in sources if s == SOURCE_RANDOM)

    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(
        f"{title}\n({n_demo} demos, {n_sft_ok} SFT✓, {n_sft_fail} SFT✗, {n_random} random)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    ax_a = fig.add_subplot(2, 2, 1)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)

    plot_panel_a(ax_a, xy, labels)
    plot_panel_b(ax_b, xy, sources)
    plot_panel_c(ax_c, labels, sources)
    plot_panel_d(ax_d, labels, X, all_trajs, traj_indices)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)

    return fig


def plot_distance_figure(
    X: np.ndarray,
    sources: list[str],
    cfg: ClusteringConfig,
    title: str = "Reward Distance Distributions",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot KDE distributions of distances to cluster centers.
    
    Creates a 2-panel figure:
      1. Distances to Demo-only cluster centers
      2. Distances to Demo + SFT✓ cluster centers
      
    This shows how the reward landscape changes when including successful
    rollouts in the reference set.
    """
    import seaborn as sns
    from sklearn.cluster import DBSCAN

    # 1. Identify Demo-only centers
    demo_mask = np.array([s == SOURCE_DEMO for s in sources])
    X_demo = X[demo_mask]
    db_demo = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(X_demo)
    demo_centers = []
    for c in set(db_demo.labels_):
        if c != -1:
            demo_centers.append(X_demo[db_demo.labels_ == c].mean(axis=0))
    demo_centers = np.array(demo_centers)

    # 2. Identify Demo + SFT✓ centers
    ok_mask = np.array([s in (SOURCE_DEMO, SOURCE_SFT_SUCCESS) for s in sources])
    X_ok = X[ok_mask]
    db_ok = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(X_ok)
    ok_centers = []
    for c in set(db_ok.labels_):
        if c != -1:
            ok_centers.append(X_ok[db_ok.labels_ == c].mean(axis=0))
    ok_centers = np.array(ok_centers)

    # Calculate distances for each frame to nearest center
    def get_dists(centers: np.ndarray) -> dict[str, list[float]]:
        if len(centers) == 0:
            return {s: [] for s in set(sources)}
        
        # (N, C) distance matrix
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        min_dists = dists.min(axis=1)  # (N,)
        
        dist_dict: dict[str, list[float]] = {s: [] for s in set(sources)}
        for d, s in zip(min_dists, sources):
            dist_dict[s].append(d)
        return dist_dict

    dists_demo = get_dists(demo_centers)
    dists_ok = get_dists(ok_centers)

    fig = plt.figure(figsize=(16, 6), facecolor="white")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    def plot_kde(ax, dist_dict, subtitle):
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        for src in [SOURCE_DEMO, SOURCE_SFT_SUCCESS, SOURCE_FAILED, SOURCE_RANDOM]:
            if src in dist_dict and dist_dict[src]:
                sns.kdeplot(
                    dist_dict[src], 
                    ax=ax, 
                    label=src, 
                    color=SOURCE_COLORS.get(src, "black"),
                    fill=True,
                    alpha=0.2,
                    linewidth=2
                )
        ax.set_xlabel("L2 Distance to Nearest Center")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_kde(ax1, dists_demo, "Reference: Demo-Only Clusters")
    if len(ok_centers) > 0:
        plot_kde(ax2, dists_ok, "Reference: Demo + SFT✓ Clusters")
    else:
        ax2.text(0.5, 0.5, "No clusters found", ha="center", va="center")
        ax2.set_title("Reference: Demo + SFT✓ Clusters")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)

    return fig
