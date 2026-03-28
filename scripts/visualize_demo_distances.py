"""Faithful visualizer for the public siiRL SRPO implementation.

Matches the 25-point checklist for:
1. Preprocessing (Uniform 64-frame resampling/padding)
2. Clustering (StandardScaler + DBSCAN + Mean Fallback)
3. Reward Shaping (Min-Max normalization + 0.6 capped sigmoid)
4. Distance (Euclidean to nearest success cluster center)

Usage:
    uv run python scripts/visualize_demo_distances.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from matplotlib.gridspec import GridSpec
from scipy import special
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Source labels & colours ──
SOURCE_DEMO = "Success Demo"
SOURCE_SFT_SUCCESS = "Success Rollout (SFT)"
SOURCE_SFT_FAILED = "Failed Rollout (SFT)"
SOURCE_RANDOM = "Random-action Rollout"

COLORS = {
    SOURCE_DEMO: "#2ecc71",        # Green
    SOURCE_SFT_SUCCESS: "#3498db",   # Blue
    SOURCE_SFT_FAILED: "#f39c12",    # Orange
    SOURCE_RANDOM: "#e74c3c",       # Red
}

# ── siiRL Logic ──────────────────────────────────────────────────────

def _compute_cluster_centers_siirl(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """Matches siirl/utils/reward_score/embodied.py: _compute_cluster_centers.
    
    1. StandardScaler on success embeddings
    2. DBSCAN(eps=0.5, min_samples=2)
    3. Ignore -1 (noise)
    4. Return inverse-transformed means of clusters.
    5. Fallback: plain mean if no clusters.
    """
    if len(embeddings) == 0:
        return np.array([np.zeros(embeddings.shape[1])])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    centers = []
    for label in unique_labels:
        cluster_points = scaled[labels == label]
        center_scaled = cluster_points.mean(axis=0, keepdims=True)
        center = scaler.inverse_transform(center_scaled).flatten()
        centers.append(center)

    if not centers:
        logger.info("DBSCAN found no clusters — falling back to plain mean of success embeddings")
        return np.array([embeddings.mean(axis=0)])

    return np.array(centers)


def _compute_siirl_rewards(distances: np.ndarray) -> np.ndarray:
    """Matches siirl/utils/reward_score/embodied.py: embodied reward shaping.
    
    1. Min-Max normalization over the failed trajectories in the batch.
    2. sigmoid = 10.0 * (0.5 - normalized_dist)
    3. reward = 0.6 * expit(sigmoid)
    """
    if len(distances) == 0:
        return np.array([])
    
    min_d, max_d = distances.min(), distances.max()
    if max_d - min_d < 1e-6:
        normalized = np.full_like(distances, 0.5)
    else:
        normalized = (distances - min_d) / (max_d - min_d)

    sigmoid_inputs = 10.0 * (0.5 - normalized)
    return 0.6 * special.expit(sigmoid_inputs)


# ── Helpers ──────────────────────────────────────────────────────────

def _load_embs(cache_dir: Path, prefix: str, name: str) -> torch.Tensor:
    path = cache_dir / f"{prefix}_{name}_embs.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached embeddings: {path}")
    embs = torch.load(path, weights_only=False)
    return embs


def _distances_to_nearest_center(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Always Euclidean per siiRL spec
    return cdist(X, centers, "euclidean").min(axis=1)


# ── Plotting ─────────────────────────────────────────────────────────

def _plot_faithful_summary(
    dist_groups: dict[str, np.ndarray],
    reward_groups: dict[str, np.ndarray],
    title: str,
    save_path: Path | None,
):
    import seaborn as sns
    order = [SOURCE_DEMO, SOURCE_SFT_SUCCESS, SOURCE_SFT_FAILED, SOURCE_RANDOM]
    fig = plt.figure(figsize=(22, 18), facecolor="white")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # A: Pipeline Diagram (Textual/Schematic representation)
    ax_pipe = fig.add_subplot(gs[0, :])
    ax_pipe.axis("off")
    pipeline_text = (
        "Public siiRL embodied SRPO Pipeline:\n"
        "Full Rollout -> Resample/Pad to 64 frames (Uniform np.linspace) -> Preprocess (Resize short-side 438, Center-Crop 384, Normalise) ->\n"
        "V-JEPA 2 Encoder (ViT-Giant) -> Temporal Mean Pool (1 per rollout) -> \n"
        "Per-Task Success Clustering (StandardScaler + DBSCAN eps=0.5) -> Nearest Success Cluster Distance for Failures -> \n"
        "Min-Max Normalization (per batch) -> Sigmoid Mapping (Max 0.6 reward)"
    )
    ax_pipe.text(0.5, 0.5, pipeline_text, ha="center", va="center", fontsize=14, 
                 bbox=dict(boxstyle="round,pad=1", fc="#fdf6e3", ec="#93a1a1"))

    # B: Distance Distributions (KDE)
    ax_dkde = fig.add_subplot(gs[1, 0])
    for src in order:
        if src not in dist_groups: continue
        sns.kdeplot(dist_groups[src], ax=ax_dkde, label=src, color=COLORS[src], fill=True, alpha=0.1)
    ax_dkde.set_title("B — Distance to Nearest Success Center (Euclidean)", fontweight="bold")
    ax_dkde.legend()

    # C: Reward distributions
    ax_rkde = fig.add_subplot(gs[1, 1])
    for src in order:
        if src not in reward_groups: continue
        sns.kdeplot(reward_groups[src], ax=ax_rkde, label=src, color=COLORS[src], fill=True, alpha=0.1)
    ax_rkde.set_title("C — Resulting shaped Rewards (Capped at 0.6 for failures)", fontweight="bold")

    # D: Distance Violin
    ax_dviol = fig.add_subplot(gs[2, 0])
    d_data = [dist_groups[s] for s in order if s in dist_groups]
    v1 = ax_dviol.violinplot(d_data, showmeans=True)
    for i, pc in enumerate(v1["bodies"]): pc.set_facecolor(COLORS[order[i]])
    ax_dviol.set_xticks(range(1, len(d_data) + 1))
    ax_dviol.set_xticklabels([s for s in order if s in dist_groups], rotation=15)
    ax_dviol.set_title("D — Distance Distributions", fontweight="bold")

    # E: Reward Transfer Curve
    ax_curve = fig.add_subplot(gs[2, 1])
    x = np.linspace(0, 1, 100)
    y = 0.6 * special.expit(10.0 * (0.5 - x))
    ax_curve.plot(x, y, color="black", lw=2, label="siiRL Sigmoid (max 0.6)")
    ax_curve.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax_curve.set_xlabel("Min-Max Normalized Distance")
    ax_curve.set_ylabel("Reward")
    ax_curve.set_title("E — Reward Transfer Curve (Public Implementation)", fontweight="bold")
    ax_curve.legend()

    # F: Statistics Table
    ax_stat = fig.add_subplot(gs[3, :])
    ax_stat.axis("off")
    rows = []
    for s in order:
        if s not in dist_groups: continue
        d, r = dist_groups[s], reward_groups[s]
        rows.append([s, len(d), f"{d.mean():.3f}", f"{r.mean():.3f}", f"{r.max():.3f}"])
    tbl = ax_stat.table(cellText=rows, colLabels=["Source", "N", "Mean Dist", "Mean Reward", "Max Reward"], 
                        loc="center", cellLoc="center")
    tbl.scale(1, 2)
    tbl.set_fontsize(12)
    ax_stat.set_title("F — Quantitative Performance Summary", pad=20, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure → %s", save_path)


# ── Main ─────────────────────────────────────────────────────────────

def main(
    suite: str = typer.Option("spatial", "--suite"),
    task_id: int = typer.Option(5, "--task-id"),
    cache_dir: Path = typer.Option(Path("notebooks/cache"), "--cache-dir"),
    no_show: bool = typer.Option(False, "--no-show"),
) -> None:
    prefix = f"{suite}_task{task_id}"
    logger.info("Visualising Faithful siiRL Implementation for %s", prefix)

    # 1. Load data
    raw_sources = {
        SOURCE_DEMO: "demos",
        SOURCE_SFT_SUCCESS: "sft_success",
        SOURCE_SFT_FAILED: "sft_failed",
        SOURCE_RANDOM: "random_failed",
    }
    raw_embs = {k: _load_embs(cache_dir, prefix, v).cpu().numpy() for k, v in raw_sources.items()}

    # 2. Clustering on Demos (The "Expert" set)
    success_centers = _compute_cluster_centers_siirl(raw_embs[SOURCE_DEMO])
    logger.info("Identified %d expert centers (including fallback if applicable)", len(success_centers))

    # 3. Compute Distances to nearest Success Center
    dist_groups = {k: _distances_to_nearest_center(raw_embs[k], success_centers) for k in raw_embs}

    # 4. Reward Shaping (Min-Max Normalized per batch of failures)
    # Combining failures for the normalization batch (SFT Failed + Random)
    failures_combined = np.concatenate([dist_groups[SOURCE_SFT_FAILED], dist_groups[SOURCE_RANDOM]])
    
    # We apply the same min-max baseline to all failures for fair comparison
    min_d, max_d = failures_combined.min(), failures_combined.max()
    logger.info("Batch Norm: min_dist=%.3f, max_dist=%.3f", min_d, max_d)

    def _apply_batch_norm(dists: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
        if max_d - min_d < 1e-6:
            norm = np.full_like(dists, 0.5)
        else:
            norm = (dists - min_d) / (max_d - min_d)
        return 0.6 * special.expit(10.0 * (0.5 - norm))

    reward_groups = {}
    reward_groups[SOURCE_DEMO] = np.ones(len(dist_groups[SOURCE_DEMO])) # Demos = 1.0 (they are successes)
    reward_groups[SOURCE_SFT_SUCCESS] = np.ones(len(dist_groups[SOURCE_SFT_SUCCESS])) # Successes = 1.0
    reward_groups[SOURCE_SFT_FAILED] = _apply_batch_norm(dist_groups[SOURCE_SFT_FAILED], min_d, max_d)
    reward_groups[SOURCE_RANDOM] = _apply_batch_norm(dist_groups[SOURCE_RANDOM], min_d, max_d)

    # 5. Output summary
    print("\n" + "=" * 80)
    print("FAITHFUL siiRL PUBLIC IMPLEMENTATION SUMMARY")
    print("=" * 80)
    print(f"{'Source':<25} {'N':>5} {'Mean Dist':>12} {'Mean Reward':>12}")
    print("-" * 80)
    for s in [SOURCE_DEMO, SOURCE_SFT_SUCCESS, SOURCE_SFT_FAILED, SOURCE_RANDOM]:
        print(f"{s:<25} {len(dist_groups[s]):>5} {dist_groups[s].mean():>12.3f} {reward_groups[s].mean():>12.3f}")
    print("=" * 80 + "\n")

    # 6. Plotting
    save_path = cache_dir / f"{prefix}_siirl_faithful_analysis.png"
    _plot_faithful_summary(dist_groups, reward_groups, title=f"Faithful siiRL Analysis — {suite} task {task_id}", save_path=save_path)

    if not no_show:
        plt.show()

if __name__ == "__main__":
    typer.run(main)
