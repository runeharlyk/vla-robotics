"""Reusable reward analysis functions for SRPO diagnostic studies.

Provides stateless, pure functions for:
- Clustering trajectory embeddings (siiRL StandardScaler method and raw DBSCAN)
- Computing distances to cluster centers
- Computing shaped rewards (siiRL min-max + 0.6 cap, and z-score + alpha)
- Building summary DataFrames
- Plotting distance/reward distributions and progress curves
- Embedding dimensionality analysis (PCA explained variance)
- k-NN elbow plots for DBSCAN eps selection
- Per-frame temporal evolution within trajectories
- Chunked (sliding-window) encoding for dense progress rewards

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


def compute_siirl_rewards_global(
    distances: np.ndarray,
    all_distances: np.ndarray,
) -> np.ndarray:
    """siiRL reward with global min-max normalization across all groups.

    Unlike ``compute_siirl_rewards`` which normalizes within the passed
    array, this uses ``all_distances`` for min-max bounds so rewards are
    comparable across trajectory groups.
    """
    if len(distances) == 0:
        return np.array([])

    min_d, max_d = all_distances.min(), all_distances.max()
    if max_d - min_d < 1e-6:
        normalized = np.full_like(distances, 0.5)
    else:
        normalized = (distances - min_d) / (max_d - min_d)

    return 0.6 * special.expit(10.0 * (0.5 - normalized))


def compute_zscore_rewards_global(
    distances: np.ndarray,
    all_distances: np.ndarray,
    alpha: float = 0.8,
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-score reward with global mean/std normalization across all groups."""
    if len(distances) == 0:
        return np.array([])

    d_mean = all_distances.mean()
    d_std = max(all_distances.std(), eps)
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


def build_distance_table_global(
    groups: dict[str, np.ndarray],
    reward_fn_global: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build summary DataFrame with globally-normalized rewards.

    Unlike ``build_distance_table``, the reward function receives both
    the per-group distances and all distances concatenated, ensuring
    normalization is consistent across groups.
    """
    all_dists = np.concatenate(list(groups.values()))
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
        if reward_fn_global is not None:
            rewards = reward_fn_global(dists, all_dists)
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


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for Spearman correlation.

    Returns ``(correlation, ci_lower, ci_upper)``.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    corr, _ = spearmanr(x, y)

    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        c, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(c):
            boot_corrs.append(c)

    if not boot_corrs:
        return float(corr), float(corr), float(corr)

    alpha_half = (1 - ci) / 2
    lower = np.percentile(boot_corrs, alpha_half * 100)
    upper = np.percentile(boot_corrs, (1 - alpha_half) * 100)
    return float(corr), float(lower), float(upper)


def compute_per_trajectory_progress_correlation(
    progress_levels: list[float],
    distances_per_level: dict[float, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Spearman correlation at per-trajectory level (not per-level mean).

    Each trajectory contributes one ``(progress, distance)`` pair,
    giving far more statistical power than correlating level means.
    Returns ``(correlation, p_value, ci_lower, ci_upper)``.
    """
    all_progress: list[float] = []
    all_distances: list[float] = []
    for level in sorted(progress_levels):
        dists = distances_per_level.get(level)
        if dists is not None and len(dists) > 0:
            for d in dists:
                all_progress.append(level)
                all_distances.append(float(d))

    if len(all_progress) < 3:
        return 0.0, 1.0, 0.0, 0.0

    x = np.array(all_progress)
    y = np.array(all_distances)
    corr, pval = spearmanr(x, y)
    _, ci_lower, ci_upper = bootstrap_spearman_ci(x, y, n_bootstrap, seed=seed)
    return float(corr), float(pval), ci_lower, ci_upper


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


def plot_reward_kde_global(
    groups: dict[str, np.ndarray],
    reward_fn_global: Callable[[np.ndarray, np.ndarray], np.ndarray],
    title: str = "Shaped Rewards (global normalization)",
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """KDE plot with globally-normalized rewards.

    Concatenates all group distances for normalization before computing
    per-group rewards, ensuring cross-group comparability.
    """
    all_dists = np.concatenate([v for v in groups.values() if len(v) > 0])
    reward_groups = {
        k: reward_fn_global(v, all_dists)
        for k, v in groups.items() if len(v) > 0
    }
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


# ──────────────────────────────────────────────────────────────────────
# Separation metrics
# ──────────────────────────────────────────────────────────────────────


def compute_separation_metrics(
    success_distances: np.ndarray,
    failure_distances: np.ndarray,
) -> dict[str, float]:
    """Quantitative separation between two distance distributions.

    Returns Cohen's d (positive = failure farther from centers),
    AUROC (lower distance = positive/success class), and histogram
    overlap coefficient (0 = no overlap, 1 = identical).
    """
    from sklearn.metrics import roc_auc_score

    pooled_std = np.sqrt(
        (success_distances.std() ** 2 + failure_distances.std() ** 2) / 2
    )
    cohens_d = (failure_distances.mean() - success_distances.mean()) / max(pooled_std, 1e-8)

    labels = np.concatenate([
        np.ones(len(success_distances)),
        np.zeros(len(failure_distances)),
    ])
    scores = np.concatenate([-success_distances, -failure_distances])
    auroc = float(roc_auc_score(labels, scores))

    lo = min(success_distances.min(), failure_distances.min())
    hi = max(success_distances.max(), failure_distances.max())
    bins = np.linspace(lo, hi, 100)
    hist_s, _ = np.histogram(success_distances, bins=bins, density=True)
    hist_f, _ = np.histogram(failure_distances, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(hist_s, hist_f)) * bin_width)

    return {"Cohen's d": cohens_d, "AUROC": auroc, "Overlap": overlap}


def build_separation_table(
    groups: dict[str, np.ndarray],
    success_groups: list[str] | None = None,
    failure_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Separation metrics table: success vs each failure group and combined."""
    if success_groups is None:
        success_groups = ["Demo", "SFT Success"]
    if failure_groups is None:
        failure_groups = ["SFT Failed", "Random"]

    success_dists = np.concatenate([
        groups[g] for g in success_groups if g in groups and len(groups[g]) > 0
    ])

    rows = []
    for fg in failure_groups:
        if fg not in groups or len(groups[fg]) == 0:
            continue
        metrics = compute_separation_metrics(success_dists, groups[fg])
        metrics["Comparison"] = f"Success vs {fg}"
        rows.append(metrics)

    failure_dists = np.concatenate([
        groups[g] for g in failure_groups if g in groups and len(groups[g]) > 0
    ])
    if len(failure_dists) > 0:
        metrics = compute_separation_metrics(success_dists, failure_dists)
        metrics["Comparison"] = "Success vs All Failure"
        rows.append(metrics)

    return pd.DataFrame(rows)[["Comparison", "Cohen's d", "AUROC", "Overlap"]]


# ──────────────────────────────────────────────────────────────────────
# Experiment 5: Per-frame vs clip encoding comparison
# ──────────────────────────────────────────────────────────────────────


def encode_trajectories_per_frame(
    trajectories_images: list,
    encoder,
    subsample_every: int = 5,
):
    """Encode trajectories using per-frame mean-pooling (no temporal context).

    This calls ``encode_frames`` on subsampled frames and averages,
    regardless of the encoder's native video-clip capability.

    Returns ``(N, D)`` numpy array.
    """
    import torch
    from vla.utils.tensor import to_float01

    all_frames = []
    traj_sizes = []
    for imgs in trajectories_images:
        indices = list(range(0, imgs.shape[0], subsample_every))
        frames = imgs[indices]
        if frames.ndim == 5:
            t, v, c, h, w = frames.shape
            frames = frames.reshape(t * v, c, h, w)
        frames = to_float01(frames)
        all_frames.append(frames)
        traj_sizes.append(frames.shape[0])

    mega = torch.cat(all_frames, dim=0)
    all_embs = encoder.encode_frames(mega)

    results = []
    offset = 0
    for sz in traj_sizes:
        results.append(all_embs[offset : offset + sz].mean(dim=0))
        offset += sz
    return torch.stack(results, dim=0).cpu().numpy()


def encode_trajectories_clip(
    trajectories_images: list,
    encoder,
    subsample_every: int = 5,
):
    """Encode trajectories using the encoder's native video-clip mode.

    For V-JEPA 2 this sends all subsampled frames as a single 64-frame
    clip, capturing temporal context.  Falls back to per-frame if the
    encoder has no clip mode.

    Returns ``(N, D)`` numpy array.
    """
    import torch
    from vla.utils.tensor import to_float01

    imgs_list = [to_float01(imgs) for imgs in trajectories_images]
    return encoder.encode_trajectories(imgs_list, subsample_every).cpu().numpy()


def encode_trajectories_siirl_faithful(
    trajectories_images: list,
    encoder,
    target_frames: int = 64,
) -> np.ndarray:
    """Encode trajectories matching the siiRL production pipeline.

    For each trajectory, evenly samples exactly ``target_frames`` frames
    (with repetition if shorter), then encodes the clip as a single
    video through the encoder — matching siiRL's preprocessing.
    Returns ``(N, D)`` numpy array.
    """
    import torch
    from vla.utils.tensor import to_float01

    results = []
    for imgs in trajectories_images:
        imgs = to_float01(imgs)
        T = imgs.shape[0]
        if imgs.ndim == 5:
            t, v, c, h, w = imgs.shape
            imgs = imgs.reshape(t * v, c, h, w)
            T = imgs.shape[0]

        if T >= target_frames:
            indices = np.linspace(0, T - 1, num=target_frames, dtype=int)
        else:
            indices = np.resize(np.arange(T), target_frames)

        sampled = imgs[indices]
        emb = encoder.encode_trajectory(sampled, subsample_every=1)
        results.append(emb)

    return torch.stack(results, dim=0).cpu().numpy()


def compare_encoding_methods(
    per_frame_embs: np.ndarray,
    clip_embs: np.ndarray,
    reference_per_frame: np.ndarray,
    reference_clip: np.ndarray,
    group_labels: list[str],
) -> pd.DataFrame:
    """Compare distance separation between per-frame and clip encoding."""
    centers_pf = reference_per_frame.mean(axis=0, keepdims=True)
    centers_clip = reference_clip.mean(axis=0, keepdims=True)

    dists_pf = cdist(per_frame_embs, centers_pf, "euclidean").flatten()
    dists_clip = cdist(clip_embs, centers_clip, "euclidean").flatten()

    rows = []
    unique_groups = sorted(set(group_labels))
    for grp in unique_groups:
        mask = np.array([g == grp for g in group_labels])
        rows.append({
            "Group": grp,
            "N": int(mask.sum()),
            "Per-Frame Dist Mean": dists_pf[mask].mean(),
            "Per-Frame Dist Std": dists_pf[mask].std(),
            "Clip Dist Mean": dists_clip[mask].mean(),
            "Clip Dist Std": dists_clip[mask].std(),
        })
    return pd.DataFrame(rows)


def generate_null_embeddings(
    n_samples: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Random Gaussian embeddings as a null-model baseline.

    If V-JEPA 2 barely outperforms this, the embedding does not
    capture task-relevant structure.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, dim).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Experiment 6: Embedding dimensionality analysis
# ──────────────────────────────────────────────────────────────────────


def pca_explained_variance(
    embeddings: np.ndarray,
    max_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA explained variance ratio.

    Returns (cumulative_variance, per_component_variance).
    """
    from sklearn.decomposition import PCA

    n_components = min(
        embeddings.shape[0], embeddings.shape[1],
        max_components or embeddings.shape[1],
    )
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return cum, pca.explained_variance_ratio_


def plot_pca_explained_variance(
    embeddings: np.ndarray,
    title: str = "PCA Explained Variance",
    ax: plt.Axes | None = None,
    max_components: int = 100,
) -> plt.Axes:
    """Plot cumulative explained variance with 90%/95%/99% thresholds."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    cum, per_comp = pca_explained_variance(embeddings, max_components)
    x = np.arange(1, len(cum) + 1)

    ax.plot(x, cum, linewidth=2, color="#3498db", label="Cumulative")
    ax.bar(x, per_comp, alpha=0.3, color="#3498db", width=0.8)

    for threshold, color, ls in [(0.90, "#2ecc71", "--"), (0.95, "#f39c12", "--"), (0.99, "#e74c3c", "--")]:
        idx = np.searchsorted(cum, threshold)
        if idx < len(cum):
            ax.axhline(threshold, color=color, linestyle=ls, alpha=0.7, label=f"{threshold*100:.0f}% ({idx+1} dims)")
            ax.axvline(idx + 1, color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(max_components, len(cum)) + 1)
    return ax


def embedding_dimension_stats(embeddings: np.ndarray) -> pd.DataFrame:
    """Per-dimension statistics: mean, std, min, max, range."""
    rows = []
    for d in range(embeddings.shape[1]):
        col = embeddings[:, d]
        rows.append({
            "Dim": d,
            "Mean": col.mean(),
            "Std": col.std(),
            "Min": col.min(),
            "Max": col.max(),
            "Range": col.max() - col.min(),
        })
    return pd.DataFrame(rows)


def plot_dimension_variance(
    embeddings: np.ndarray,
    title: str = "Per-Dimension Std Dev (sorted)",
    top_n: int = 50,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar plot of per-dimension standard deviations (sorted descending)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    stds = embeddings.std(axis=0)
    sorted_idx = np.argsort(stds)[::-1][:top_n]

    ax.bar(range(len(sorted_idx)), stds[sorted_idx], color="#3498db", alpha=0.7)
    ax.set_xlabel("Dimension (sorted by std)")
    ax.set_ylabel("Std Dev")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ratio = stds[sorted_idx[0]] / (stds[sorted_idx[-1]] + 1e-10)
    ax.text(
        0.98, 0.95, f"Max/Min std ratio: {ratio:.1f}x",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )
    return ax


# ──────────────────────────────────────────────────────────────────────
# Experiment 7: k-NN elbow plot
# ──────────────────────────────────────────────────────────────────────


def knn_distance_curve(
    embeddings: np.ndarray,
    k: int = 2,
) -> np.ndarray:
    """Sorted k-th nearest neighbor distances (for DBSCAN eps selection)."""
    kth_dists = NearestNeighbors(n_neighbors=k).fit(embeddings).kneighbors()[0][:, -1]
    return np.sort(kth_dists)


def plot_knn_elbow(
    embeddings: np.ndarray,
    k_values: list[int] | None = None,
    title: str = "k-NN Distance Curve (DBSCAN eps diagnostic)",
    ax: plt.Axes | None = None,
    show_percentiles: bool = True,
) -> plt.Axes:
    """Plot sorted k-NN distances for multiple k values.

    The "elbow" in this curve indicates a natural eps threshold for DBSCAN.
    A smooth curve with no elbow suggests no clear cluster structure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if k_values is None:
        k_values = [2, 3, 5, 10]

    for k in k_values:
        if k >= len(embeddings):
            continue
        dists = knn_distance_curve(embeddings, k)
        ax.plot(range(len(dists)), dists, linewidth=1.5, label=f"k={k}")

        if show_percentiles and k == k_values[0]:
            for pct in [25, 50, 75]:
                val = np.percentile(dists, pct)
                ax.axhline(val, color="gray", linestyle=":", alpha=0.4)
                ax.text(
                    len(dists) * 0.02, val,
                    f"p{pct}={val:.3f}", fontsize=8, va="bottom", color="gray",
                )

    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel(f"Distance to k-th Nearest Neighbor")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_knn_elbow_comparison(
    emb_groups: dict[str, np.ndarray],
    k: int = 2,
    title: str = "k-NN Distance Curves by Group",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay k-NN distance curves for different trajectory groups."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    default_colors = {
        "Demo": "#2ecc71",
        "SFT Success": "#3498db",
        "SFT Failed": "#f39c12",
        "Random": "#e74c3c",
        "All References": "#9b59b6",
    }

    for name, embs in emb_groups.items():
        if len(embs) <= k:
            continue
        dists = knn_distance_curve(embs, k)
        color = default_colors.get(name, None)
        ax.plot(range(len(dists)), dists, linewidth=1.5, label=name, color=color)

    ax.set_xlabel("Points (sorted)")
    ax.set_ylabel(f"Distance to {k}-NN")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ──────────────────────────────────────────────────────────────────────
# Experiment 5b: Chunked (sliding-window) encoding
# ──────────────────────────────────────────────────────────────────────


def chunk_trajectory_images(
    images,
    window_size: int = 32,
    stride: int = 16,
) -> tuple[list, list[float]]:
    """Split trajectory frames into overlapping sliding windows.

    Args:
        images: ``(T, [V,] C, H, W)`` tensor of frames.
        window_size: Number of frames per chunk.
        stride: Step between consecutive windows.

    Returns:
        ``(chunks, center_timesteps)`` where chunks is a list of
        ``(window_size, C, H, W)`` tensors and center_timesteps is the
        normalised [0, 1] position of each chunk's center within the
        trajectory.
    """
    import torch

    T = images.shape[0]
    if images.ndim == 5:
        t, v, c, h, w = images.shape
        images = images.reshape(t * v, c, h, w)
        T = images.shape[0]

    if T <= window_size:
        return [images], [0.5]

    chunks = []
    centers = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        chunks.append(images[start:end])
        centers.append((start + end - 1) / 2.0 / (T - 1))

    if not chunks:
        chunks.append(images[:window_size])
        centers.append(0.5)

    return chunks, centers


def encode_chunks(
    chunks: list,
    encoder,
) -> np.ndarray:
    """Encode a list of frame chunks as V-JEPA 2 video clips.

    Each chunk is passed through ``encoder.encode_trajectory`` which
    accepts variable frame counts. Returns ``(N_chunks, D)`` numpy array.
    """
    import torch

    embs = []
    for chunk in chunks:
        emb = encoder.encode_trajectory(chunk, subsample_every=1)
        embs.append(emb)
    return torch.stack(embs, dim=0).cpu().numpy()


def cluster_reference_chunks(
    demo_images_list: list,
    encoder,
    window_size: int = 32,
    stride: int = 16,
    use_standard_scaler: bool = True,
    eps: float = 0.5,
    min_samples: int = 2,
) -> tuple[np.ndarray, StandardScaler | None]:
    """Chunk all reference trajectories, encode, and cluster.

    Returns ``(centers, scaler)`` where scaler is ``None`` when
    ``use_standard_scaler`` is False.
    """
    all_chunk_embs = []
    for imgs in demo_images_list:
        chunks, _ = chunk_trajectory_images(imgs, window_size, stride)
        embs = encode_chunks(chunks, encoder)
        all_chunk_embs.append(embs)

    X = np.concatenate(all_chunk_embs, axis=0)

    if use_standard_scaler:
        centers, scaler = compute_cluster_centers_siirl(X, eps, min_samples)
        return centers, scaler

    centers = compute_cluster_centers_raw(X, eps, min_samples)
    return centers, None


def compute_chunked_progress_curve(
    trajectory_images,
    encoder,
    centers: np.ndarray,
    scaler: StandardScaler | None = None,
    window_size: int = 32,
    stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-chunk distances for a single trajectory.

    Returns ``(timesteps, distances)`` where timesteps is in [0, 1]
    and distances is the Euclidean distance to the nearest cluster center
    for each chunk.
    """
    chunks, center_ts = chunk_trajectory_images(
        trajectory_images, window_size, stride,
    )
    chunk_embs = encode_chunks(chunks, encoder)
    dists = distances_to_nearest_center(chunk_embs, centers, scaler)
    return np.array(center_ts), dists


def compute_chunked_intra_trajectory_correlation(
    trajectories_images: list,
    encoder,
    centers: np.ndarray,
    scaler: StandardScaler | None = None,
    window_size: int = 32,
    stride: int = 16,
) -> tuple[float, float]:
    """Spearman correlation between chunk timestep and distance, pooled across trajectories.

    For successful trajectories, we expect negative correlation: later
    chunks (closer to task completion) should be nearer to success
    cluster centers.

    Returns ``(mean_correlation, mean_pvalue)`` averaged across
    trajectories with at least 3 chunks.
    """
    corrs = []
    pvals = []
    for imgs in trajectories_images:
        ts, dists = compute_chunked_progress_curve(
            imgs, encoder, centers, scaler, window_size, stride,
        )
        if len(ts) >= 3:
            c, p = spearmanr(ts, dists)
            corrs.append(float(c))
            pvals.append(float(p))

    if not corrs:
        return 0.0, 1.0
    return float(np.mean(corrs)), float(np.mean(pvals))


def compute_chunked_intra_trajectory_correlations(
    trajectories_images: list,
    encoder,
    centers: np.ndarray,
    scaler: StandardScaler | None = None,
    window_size: int = 32,
    stride: int = 16,
) -> list[float]:
    """Per-trajectory Spearman correlations (chunk timestep vs distance).

    Returns a list of correlations, one per trajectory with >= 3 chunks.
    Use for bootstrap confidence intervals on the mean.
    """
    corrs: list[float] = []
    for imgs in trajectories_images:
        ts, dists = compute_chunked_progress_curve(
            imgs, encoder, centers, scaler, window_size, stride,
        )
        if len(ts) >= 3:
            c, _ = spearmanr(ts, dists)
            if not np.isnan(c):
                corrs.append(float(c))
    return corrs


def plot_chunked_progress_curves(
    curves_by_group: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    title: str = "Chunked Encoding: Per-Chunk Distance Over Time",
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot per-chunk distance curves, overlaid by trajectory group.

    Args:
        curves_by_group: ``{label: [(timesteps, distances), ...]}``
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    default_colors = {
        "Demo": "#2ecc71",
        "SFT Success": "#3498db",
        "SFT Failed": "#f39c12",
        "Random": "#e74c3c",
    }
    colors = colors or default_colors

    for label, curves in curves_by_group.items():
        color = colors.get(label, None)
        for ts, dists in curves:
            x_pct = ts * 100
            alpha = 0.12 if len(curves) > 5 else 0.35
            ax.plot(x_pct, dists, color=color, alpha=alpha, linewidth=0.8)

        all_dists = [d for _, d in curves if len(d) > 0]
        if all_dists:
            max_len = max(len(d) for d in all_dists)
            interpolated = np.array([
                np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(d)), d)
                for d in all_dists
            ])
            mean_curve = interpolated.mean(axis=0)
            x = np.linspace(0, 100, max_len)
            ax.plot(x, mean_curve, color=color, linewidth=2.5, label=f"{label} (mean)")

    ax.set_xlabel("Episode Progress (%)", fontsize=12)
    ax.set_ylabel("Distance to Nearest Chunk Center", fontsize=12)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ──────────────────────────────────────────────────────────────────────
# Experiment 8: Per-frame temporal evolution + progress correlation
# ──────────────────────────────────────────────────────────────────────


def compute_per_frame_distances(
    trajectory_images,
    encoder,
    centers: np.ndarray,
    scaler: StandardScaler | None = None,
    subsample_every: int = 1,
) -> np.ndarray:
    """Compute distance to nearest cluster center for each frame in a trajectory.

    Returns ``(T,)`` array of distances at each (subsampled) timestep.
    """
    import torch
    from vla.utils.tensor import to_float01

    imgs = to_float01(trajectory_images)
    indices = list(range(0, imgs.shape[0], subsample_every))
    frames = imgs[indices]
    if frames.ndim == 5:
        t, v, c, h, w = frames.shape
        frames = frames.reshape(t * v, c, h, w)

    frame_embs = encoder.encode_frames(frames).cpu().numpy()
    return distances_to_nearest_center(frame_embs, centers, scaler)


def plot_per_frame_evolution(
    trajectories_distances: dict[str, list[np.ndarray]],
    title: str = "Per-Frame Distance Evolution",
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot per-frame distance to cluster centers over time for multiple trajectories.

    Args:
        trajectories_distances: ``{label: [dist_array_per_traj, ...]}``
        title: Figure title.
        colors: Color mapping per label.
        ax: Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    default_colors = {
        "Demo": "#2ecc71",
        "SFT Success": "#3498db",
        "SFT Failed": "#f39c12",
        "Random": "#e74c3c",
    }
    colors = colors or default_colors

    for label, dist_arrays in trajectories_distances.items():
        color = colors.get(label, None)
        for i, dists in enumerate(dist_arrays):
            x = np.linspace(0, 100, len(dists))
            alpha = 0.15 if len(dist_arrays) > 5 else 0.4
            ax.plot(x, dists, color=color, alpha=alpha, linewidth=0.8)

        all_dists = [d for d in dist_arrays if len(d) > 0]
        if all_dists:
            max_len = max(len(d) for d in all_dists)
            interpolated = np.array([
                np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(d)), d)
                for d in all_dists
            ])
            mean_curve = interpolated.mean(axis=0)
            x = np.linspace(0, 100, max_len)
            ax.plot(x, mean_curve, color=color, linewidth=2.5, label=f"{label} (mean)")

    ax.set_xlabel("Episode Progress (%)", fontsize=12)
    ax.set_ylabel("Distance to Nearest Center", fontsize=12)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def build_progress_correlation_table(
    methods: dict[str, tuple],
    top_rewarder_ref: float = 0.95,
) -> pd.DataFrame:
    """Build a comparison table of progress correlation across methods.

    Args:
        methods: ``{name: (r, p)}`` or ``{name: (r, p, ci_lo, ci_hi)}``.
        top_rewarder_ref: Published Top-Rewarder Spearman correlation
            (default 0.95, from the paper) included as a reference row.
    """
    rows = []
    for name, vals in methods.items():
        corr, pval = vals[0], vals[1]
        ci_lo = vals[2] if len(vals) > 2 else None
        ci_hi = vals[3] if len(vals) > 3 else None
        row: dict = {
            "Method": name,
            "Spearman r": corr,
            "p-value": pval,
            "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None else "—",
            "Monotonic": abs(corr) > 0.8 and pval < 0.05,
        }
        rows.append(row)

    rows.append({
        "Method": "Top-Rewarder (published reference)",
        "Spearman r": top_rewarder_ref,
        "p-value": 0.0,
        "95% CI": "—",
        "Monotonic": True,
    })

    return pd.DataFrame(rows)


def build_multi_task_summary(
    task_results: dict[str, dict],
) -> pd.DataFrame:
    """Aggregate key metrics across multiple tasks.

    Args:
        task_results: ``{task_key: {"sep_auroc": ..., "progress_corr": ..., ...}}``.
    """
    rows = []
    for task_key, metrics in task_results.items():
        row = {"Task": task_key}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
