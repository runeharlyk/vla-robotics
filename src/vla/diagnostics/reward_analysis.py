"""Reusable reward analysis functions for SRPO diagnostic studies.

Provides stateless, pure functions for:
- Clustering trajectory embeddings (siiRL StandardScaler method and raw DBSCAN)
- Computing distances to cluster centers
- Computing shaped rewards (siiRL min-max + 0.6 cap, and z-score + alpha)
- Building summary DataFrames
- Plotting distance/reward distributions and progress curves

All functions accept numpy arrays and return results — no state, no caching.
Caching is handled by callers (notebook cells or CLI scripts).
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_cluster_centers_siirl(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2,
) -> tuple[np.ndarray, StandardScaler]:
    """Cluster success embeddings using the siiRL production method.

    1. StandardScaler on embeddings
    2. DBSCAN(eps, min_samples) on scaled space
    3. Inverse-transform cluster means back to original space
    4. Fallback: plain mean if DBSCAN finds no clusters

    Returns (centers, fitted_scaler) so the scaler can be reused for
    distance queries.
    """
    if len(embeddings) == 0:
        scaler = StandardScaler()
        return np.zeros((1, 1)), scaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    centers = []
    for label in sorted(unique_labels):
        cluster_points = scaled[labels == label]
        center_scaled = cluster_points.mean(axis=0, keepdims=True)
        center = scaler.inverse_transform(center_scaled).flatten()
        centers.append(center)

    if not centers:
        return embeddings.mean(axis=0, keepdims=True), scaler

    return np.array(centers), scaler


def compute_cluster_centers_raw(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2,
    auto_eps: bool = True,
    percentile: int = 25,
) -> np.ndarray:
    """Cluster success embeddings using raw DBSCAN (no scaler).

    Optionally auto-tunes eps from k-NN distances.
    Fallback: mean of all embeddings if no clusters found.
    """
    if len(embeddings) == 0:
        return np.zeros((1, embeddings.shape[1] if embeddings.ndim == 2 else 1))

    if auto_eps and len(embeddings) > min_samples:
        kth_dists = NearestNeighbors(n_neighbors=min_samples).fit(embeddings).kneighbors()[0][:, -1]
        eps = float(np.percentile(kth_dists, percentile))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = db.labels_
    unique_labels = set(labels) - {-1}

    centers = []
    for label in sorted(unique_labels):
        centers.append(embeddings[labels == label].mean(axis=0))

    if not centers:
        return embeddings.mean(axis=0, keepdims=True)

    return np.array(centers)


def distances_to_nearest_center(
    embeddings: np.ndarray,
    centers: np.ndarray,
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """Euclidean distance from each embedding to the nearest cluster center.

    When *scaler* is provided (siiRL method), both embeddings and centers
    are transformed into the scaled space before computing distances.
    This ensures consistent behavior with how DBSCAN saw the data.
    """
    if scaler is not None:
        embeddings = scaler.transform(embeddings)
        centers = scaler.transform(centers)

    return cdist(embeddings, centers, "euclidean").min(axis=1)


def compute_siirl_rewards(distances: np.ndarray) -> np.ndarray:
    """siiRL production reward: min-max normalization + sigmoid * 0.6 cap.

    Matches ``siirl/utils/reward_score/embodied.py``.
    """
    if len(distances) == 0:
        return np.array([])

    min_d, max_d = distances.min(), distances.max()
    if max_d - min_d < 1e-6:
        normalized = np.full_like(distances, 0.5)
    else:
        normalized = (distances - min_d) / (max_d - min_d)

    return 0.6 * special.expit(10.0 * (0.5 - normalized))


def compute_zscore_rewards(
    distances: np.ndarray,
    alpha: float = 0.8,
    eps: float = 1e-8,
) -> np.ndarray:
    """Paper formula: z-score normalization + sigmoid(-z) * alpha.

    Matches our ``srpo_reward.py`` implementation.
    """
    if len(distances) == 0:
        return np.array([])

    d_mean = distances.mean()
    d_std = max(distances.std(), eps)
    z = (distances - d_mean) / d_std
    return alpha * special.expit(-z)


def build_distance_table(
    groups: dict[str, np.ndarray],
    reward_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a summary DataFrame of distances (and optionally rewards) per group."""
    rows = []
    for name, dists in groups.items():
        row = {
            "Group": name,
            "N": len(dists),
            "Dist Mean": dists.mean(),
            "Dist Std": dists.std() if len(dists) > 1 else 0.0,
            "Dist Min": dists.min() if len(dists) > 0 else 0.0,
            "Dist Max": dists.max() if len(dists) > 0 else 0.0,
        }
        if reward_fn is not None:
            rewards = reward_fn(dists)
            row["Reward Mean"] = rewards.mean()
            row["Reward Std"] = rewards.std() if len(rewards) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_progress_table(
    progress_levels: list[float],
    distances_per_level: dict[float, np.ndarray],
    lengths_per_level: dict[float, list[int]] | None = None,
) -> pd.DataFrame:
    """Build a summary DataFrame for the progress monotonicity experiment."""
    rows = []
    for level in progress_levels:
        dists = distances_per_level.get(level, np.array([]))
        siirl_r = compute_siirl_rewards(dists)
        zscore_r = compute_zscore_rewards(dists)
        row = {
            "Progress %": f"{level * 100:.0f}%",
            "N": len(dists),
            "Dist Mean": dists.mean() if len(dists) > 0 else 0.0,
            "Dist Std": dists.std() if len(dists) > 1 else 0.0,
            "siiRL Reward": siirl_r.mean() if len(siirl_r) > 0 else 0.0,
            "z-score Reward": zscore_r.mean() if len(zscore_r) > 0 else 0.0,
        }
        if lengths_per_level is not None:
            lens = lengths_per_level.get(level, [])
            row["Mean Ep Len"] = np.mean(lens) if lens else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_progress_correlation(
    progress_levels: list[float],
    distances_per_level: dict[float, np.ndarray],
) -> tuple[float, float]:
    """Spearman correlation between progress level and mean distance.

    Returns (correlation, p_value).
    Expected: negative correlation (higher progress = lower distance).
    """
    levels = []
    mean_dists = []
    for level in sorted(progress_levels):
        dists = distances_per_level.get(level)
        if dists is not None and len(dists) > 0:
            levels.append(level)
            mean_dists.append(dists.mean())

    if len(levels) < 3:
        return 0.0, 1.0

    corr, pval = spearmanr(levels, mean_dists)
    return float(corr), float(pval)


def plot_distance_kde(
    groups: dict[str, np.ndarray],
    title: str = "Distance to Nearest Cluster Center",
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """KDE plot of distances per group."""
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    default_colors = {
        "Demo": "#2ecc71",
        "SFT Success": "#3498db",
        "SFT Failed": "#f39c12",
        "Random": "#e74c3c",
    }
    colors = colors or default_colors

    for name, dists in groups.items():
        if len(dists) == 0:
            continue
        color = colors.get(name, None)
        sns.kdeplot(dists, ax=ax, label=name, color=color, fill=True, alpha=0.15, linewidth=2)

    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_reward_kde(
    groups: dict[str, np.ndarray],
    reward_fn: Callable[[np.ndarray], np.ndarray] = compute_siirl_rewards,
    title: str = "Shaped Rewards",
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """KDE plot of computed rewards per group."""
    reward_groups = {k: reward_fn(v) for k, v in groups.items() if len(v) > 0}
    return plot_distance_kde(reward_groups, title=title, colors=colors, ax=ax)


def plot_progress_curve(
    progress_levels: list[float],
    distances_per_level: dict[float, np.ndarray],
    title: str = "Distance vs Task Progress",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line plot: progress % on x-axis, mean distance with error bars on y-axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    levels_sorted = sorted(progress_levels)
    means = []
    stds = []
    for level in levels_sorted:
        dists = distances_per_level.get(level, np.array([]))
        means.append(dists.mean() if len(dists) > 0 else 0.0)
        stds.append(dists.std() if len(dists) > 1 else 0.0)

    x_pct = [l * 100 for l in levels_sorted]
    ax.errorbar(x_pct, means, yerr=stds, marker="o", capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Task Progress (%)", fontsize=12)
    ax.set_ylabel("Mean Distance to Success Centers", fontsize=12)
    ax.set_title(title, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    corr, pval = compute_progress_correlation(progress_levels, distances_per_level)
    ax.text(
        0.02, 0.98,
        f"Spearman r={corr:.3f} (p={pval:.4f})",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )
    return ax


def plot_cosine_similarity_matrix(
    groups: dict[str, np.ndarray],
    title: str = "Mean Cosine Similarity Between Groups",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of mean pairwise cosine similarity between trajectory groups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    names = list(groups.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        embs_i = groups[name_i]
        if embs_i.ndim == 1:
            embs_i = embs_i.reshape(1, -1)
        mean_i = embs_i.mean(axis=0)
        norm_i = np.linalg.norm(mean_i)

        for j, name_j in enumerate(names):
            embs_j = groups[name_j]
            if embs_j.ndim == 1:
                embs_j = embs_j.reshape(1, -1)
            mean_j = embs_j.mean(axis=0)
            norm_j = np.linalg.norm(mean_j)

            if norm_i > 0 and norm_j > 0:
                sim_matrix[i, j] = np.dot(mean_i, mean_j) / (norm_i * norm_j)

    im = ax.imshow(sim_matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(title, fontweight="bold")
    return ax
