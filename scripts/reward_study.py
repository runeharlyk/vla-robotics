"""Batch SRPO reward diagnostics with cached artifacts and saved reports."""

from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from sklearn.preprocessing import StandardScaler

from vla.constants import OUTPUTS_DIR, LiberoSuite, WorldModelType
from vla.diagnostics.clustering import (
    SOURCE_DEMO,
    SOURCE_FAILED,
    SOURCE_RANDOM,
    SOURCE_SFT_SUCCESS,
    ClusteringConfig,
    fit_umap,
    get_or_compute_embeddings,
    plot_panel_b,
)
from vla.diagnostics.collect_trajectories import (
    CollectionConfig,
    collect_demo_trajectories,
    collect_progress_trajectories,
    collect_rollouts,
    load_trajectories,
)
from vla.diagnostics.reward_analysis import (
    build_distance_table_global,
    build_multi_task_summary,
    build_progress_correlation_table,
    build_progress_table,
    build_separation_table,
    cluster_reference_chunks,
    compare_encoding_methods,
    compute_chunked_intra_trajectory_correlations,
    compute_chunked_progress_curve,
    compute_cluster_centers_raw,
    compute_cluster_centers_siirl,
    compute_per_frame_distances,
    compute_per_trajectory_progress_correlation,
    compute_progress_correlation,
    compute_siirl_rewards,
    compute_siirl_rewards_global,
    compute_zscore_rewards,
    compute_zscore_rewards_global,
    distances_to_nearest_center,
    embedding_dimension_stats,
    encode_trajectories_clip,
    encode_trajectories_per_frame,
    encode_trajectories_siirl_faithful,
    generate_null_embeddings,
    pca_explained_variance,
    plot_chunked_progress_curves,
    plot_cosine_similarity_matrix,
    plot_dimension_variance,
    plot_distance_kde,
    plot_knn_elbow,
    plot_knn_elbow_comparison,
    plot_pca_explained_variance,
    plot_per_frame_evolution,
    plot_progress_curve,
    plot_reward_kde_global,
)
from vla.models.world_model import build_world_model
from vla.utils import get_device, seed_everything
from vla.utils.tensor import to_float01

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

DEFAULT_CACHE_ROOT = OUTPUTS_DIR / "reward_study" / "cache"
DEFAULT_OUTPUT_ROOT = OUTPUTS_DIR / "reward_study"
DEFAULT_TASKS = ("spatial:2", "spatial:5", "object:0")
PROGRESS_LEVELS = [round(x * 0.1, 1) for x in range(11)][::-1]
WINDOW_SIZE = 32
STRIDE = 16


def _task_key(suite: str, task_id: int) -> str:
    return f"{suite}_task_{task_id}"


def _task_cache_dir(cache_root: Path, suite: str, task_id: int) -> Path:
    return cache_root / _task_key(suite, task_id)


def _task_output_dir(output_root: Path, suite: str, task_id: int) -> Path:
    return output_root / "tasks" / _task_key(suite, task_id)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _safe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_numpy(path: Path) -> np.ndarray:
    data = torch.load(path, weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return np.asarray(data)


def _chunk_corr_ci(values: list[float], seed: int, n_bootstrap: int = 1000) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))

    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _phase2_required_files() -> list[str]:
    files = [
        "emb_demos.pt",
        "emb_sft_success.pt",
        "emb_sft_failed.pt",
        "emb_random.pt",
        "emb_per_frame_all.pt",
        "emb_clip_all.pt",
        "emb_siirl_faithful_all.pt",
        "chunked_analysis.pt",
        "per_frame_dists.pt",
    ]
    files.extend(f"emb_progress_{int(level * 100)}.pt" for level in PROGRESS_LEVELS)
    return files


def _load_cached_trajectories(cfg: CollectionConfig, name: str) -> list:
    trajs = load_trajectories(cfg, name)
    if trajs is None:
        raise FileNotFoundError(f"Missing cached trajectories for {name} in {cfg.cache_dir}")
    return trajs


def _encode_grouped_cache(
    *,
    cfg: CollectionConfig,
    cache_path: Path,
    label: str,
    group_names: list[str],
    encode_fn,
) -> None:
    if cache_path.exists():
        return

    parts: list[torch.Tensor] = []
    for group_name in group_names:
        trajs = _load_cached_trajectories(cfg, group_name)
        logger.info("Encoding %s from cached %s trajectories (%d)", label, group_name, len(trajs))
        images = [to_float01(t.images[: t.length]) for t in trajs]
        parts.append(torch.from_numpy(encode_fn(images)))
        del trajs, images
        gc.collect()
        _safe_empty_cuda_cache()

    result = torch.cat(parts, dim=0)
    torch.save(result, cache_path)
    logger.info("Encoded %s -> %s", label, tuple(result.shape))


def _ensure_phase2_caches(
    *,
    collection_cfg: CollectionConfig,
    cache_dir: Path,
    cluster_cfg: ClusteringConfig,
    device: torch.device,
    world_model: WorldModelType,
    encoder_batch_size: int,
) -> None:
    missing = [name for name in _phase2_required_files() if not (cache_dir / name).exists()]
    if not missing:
        logger.info("All phase-2 caches already exist for %s", cache_dir)
        return

    logger.info("Missing %d phase-2 caches, loading %s encoder", len(missing), world_model)
    encoder = build_world_model(model_type=world_model, device=str(device), batch_size=encoder_batch_size)

    try:
        for name, cache_name, label in [
            ("demos", "emb_demos.pt", "demos"),
            ("sft_success", "emb_sft_success.pt", "sft_success"),
            ("sft_failed", "emb_sft_failed.pt", "sft_failed"),
            ("random_failed", "emb_random.pt", "random"),
        ]:
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                continue
            trajs = _load_cached_trajectories(collection_cfg, name)
            emb = get_or_compute_embeddings(trajs, encoder, cache_path, cluster_cfg.subsample_every)
            logger.info("Encoded %s -> %s", label, tuple(emb.shape))
            del trajs, emb
            gc.collect()
            _safe_empty_cuda_cache()

        for level in PROGRESS_LEVELS:
            pct = int(level * 100)
            cache_name = f"emb_progress_{pct}.pt"
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                continue
            logger.info("Encoding progress trajectories at %d%%", pct)
            progress_trajs = _load_cached_trajectories(collection_cfg, f"progress_{pct}_from_demos")
            get_or_compute_embeddings(
                progress_trajs,
                encoder,
                cache_path,
                cluster_cfg.subsample_every,
            )
            del progress_trajs
            gc.collect()
            _safe_empty_cuda_cache()

        _encode_grouped_cache(
            cfg=collection_cfg,
            cache_path=cache_dir / "emb_per_frame_all.pt",
            label="per-frame",
            group_names=["demos", "sft_success", "sft_failed", "random_failed"],
            encode_fn=lambda images: encode_trajectories_per_frame(images, encoder, cluster_cfg.subsample_every),
        )
        _encode_grouped_cache(
            cfg=collection_cfg,
            cache_path=cache_dir / "emb_clip_all.pt",
            label="clip",
            group_names=["demos", "sft_success", "sft_failed", "random_failed"],
            encode_fn=lambda images: encode_trajectories_clip(images, encoder, cluster_cfg.subsample_every),
        )
        _encode_grouped_cache(
            cfg=collection_cfg,
            cache_path=cache_dir / "emb_siirl_faithful_all.pt",
            label="siirl-faithful",
            group_names=["demos", "sft_success", "sft_failed", "random_failed"],
            encode_fn=lambda images: encode_trajectories_siirl_faithful(images, encoder),
        )

        chunked_path = cache_dir / "chunked_analysis.pt"
        if not chunked_path.exists():
            demos = _load_cached_trajectories(collection_cfg, "demos")
            demo_imgs = [to_float01(t.images[: t.length]) for t in demos]
            centers, scaler = cluster_reference_chunks(
                demo_imgs,
                encoder,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                use_standard_scaler=True,
            )
            curves_by_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            for label, group_name in [
                ("Demo", "demos"),
                ("SFT Success", "sft_success"),
                ("SFT Failed", "sft_failed"),
                ("Random", "random_failed"),
            ]:
                trajs = _load_cached_trajectories(collection_cfg, group_name)[:10]
                curves = []
                for traj in trajs:
                    ts, dists = compute_chunked_progress_curve(
                        to_float01(traj.images[: traj.length]),
                        encoder,
                        centers,
                        scaler,
                        window_size=WINDOW_SIZE,
                        stride=STRIDE,
                    )
                    curves.append((ts, dists))
                curves_by_group[label] = curves
                del trajs
                gc.collect()
                _safe_empty_cuda_cache()

            demo_imgs_sample = [to_float01(t.images[: t.length]) for t in demos[:20]]
            per_traj_corrs = compute_chunked_intra_trajectory_correlations(
                demo_imgs_sample,
                encoder,
                centers,
                scaler,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
            torch.save(
                {
                    "centers": centers,
                    "scaler": scaler,
                    "curves": curves_by_group,
                    "per_traj_corrs": per_traj_corrs,
                },
                chunked_path,
            )
            logger.info("Computed chunked analysis cache")
            del demos, demo_imgs, demo_imgs_sample
            gc.collect()
            _safe_empty_cuda_cache()

        per_frame_path = cache_dir / "per_frame_dists.pt"
        if not per_frame_path.exists():
            emb_demo = torch.load(cache_dir / "emb_demos.pt", weights_only=False)
            emb_sft_ok = torch.load(cache_dir / "emb_sft_success.pt", weights_only=False)
            ref = np.concatenate([emb_demo.cpu().numpy(), emb_sft_ok.cpu().numpy()])
            centers, scaler = compute_cluster_centers_siirl(ref)
            per_frame_dists: dict[str, list[np.ndarray]] = {}
            for label, group_name in [
                ("Demo", "demos"),
                ("SFT Success", "sft_success"),
                ("SFT Failed", "sft_failed"),
                ("Random", "random_failed"),
            ]:
                trajs = _load_cached_trajectories(collection_cfg, group_name)[:10]
                dist_list = []
                for traj in trajs:
                    dist_list.append(
                        compute_per_frame_distances(
                            traj.images[: traj.length],
                            encoder,
                            centers,
                            scaler,
                            subsample_every=1,
                        )
                    )
                per_frame_dists[label] = dist_list
                del trajs
                gc.collect()
                _safe_empty_cuda_cache()
            torch.save(per_frame_dists, per_frame_path)
            logger.info("Computed per-frame distance cache")
            del emb_demo, emb_sft_ok, ref, per_frame_dists
            gc.collect()
            _safe_empty_cuda_cache()
    finally:
        del encoder
        gc.collect()
        _safe_empty_cuda_cache()


def _load_cached_analysis(cache_dir: Path) -> dict[str, object]:
    x_demo = _load_numpy(cache_dir / "emb_demos.pt")
    x_sft_ok = _load_numpy(cache_dir / "emb_sft_success.pt")
    x_sft_fail = _load_numpy(cache_dir / "emb_sft_failed.pt")
    x_random = _load_numpy(cache_dir / "emb_random.pt")

    n_demo = len(x_demo)
    n_ok = len(x_sft_ok)
    n_fail = len(x_sft_fail)
    n_random = len(x_random)

    x_all = np.concatenate([x_demo, x_sft_ok, x_sft_fail, x_random], axis=0)
    x_reference = np.concatenate([x_demo, x_sft_ok], axis=0)
    sources = (
        [SOURCE_DEMO] * n_demo + [SOURCE_SFT_SUCCESS] * n_ok + [SOURCE_FAILED] * n_fail + [SOURCE_RANDOM] * n_random
    )
    group_labels = ["Demo"] * n_demo + ["SFT Success"] * n_ok + ["SFT Failed"] * n_fail + ["Random"] * n_random
    ref_mask = np.array([True] * (n_demo + n_ok) + [False] * (n_fail + n_random))

    progress_embs = {level: _load_numpy(cache_dir / f"emb_progress_{int(level * 100)}.pt") for level in PROGRESS_LEVELS}

    chunk_data = torch.load(cache_dir / "chunked_analysis.pt", weights_only=False)
    per_frame_dists = torch.load(cache_dir / "per_frame_dists.pt", weights_only=False)

    return {
        "x_demo": x_demo,
        "x_sft_ok": x_sft_ok,
        "x_sft_fail": x_sft_fail,
        "x_random": x_random,
        "x_all": x_all,
        "x_reference": x_reference,
        "sources": sources,
        "group_labels": group_labels,
        "ref_mask": ref_mask,
        "progress_embs": progress_embs,
        "x_pf": _load_numpy(cache_dir / "emb_per_frame_all.pt"),
        "x_clip": _load_numpy(cache_dir / "emb_clip_all.pt"),
        "x_siirl": _load_numpy(cache_dir / "emb_siirl_faithful_all.pt"),
        "chunk_data": chunk_data,
        "per_frame_dists": per_frame_dists,
        "n_demo": n_demo,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_random": n_random,
    }


def _write_multi_task_summary(output_root: Path) -> None:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    task_results: dict[str, dict] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        task_results[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    if not task_results:
        return

    summary = build_multi_task_summary(task_results)
    _save_dataframe(summary, output_root / "multi_task_summary.csv")
    _save_json({"tasks": task_results}, output_root / "multi_task_summary.json")


def run_task(
    *,
    checkpoint: str,
    suite: str,
    task_id: int,
    num_demos: int,
    num_rollouts: int,
    num_envs: int,
    max_steps: int,
    seed: int,
    subsample_every: int,
    dbscan_min_samples: int,
    dbscan_percentile: int,
    encoder_batch_size: int,
    cache_root: Path,
    output_root: Path,
    world_model: WorldModelType,
) -> dict[str, float]:
    task_key = _task_key(suite, task_id)
    cache_dir = _task_cache_dir(cache_root, suite, task_id)
    task_output_dir = _task_output_dir(output_root, suite, task_id)
    figures_dir = task_output_dir / "figures"
    tables_dir = task_output_dir / "tables"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = get_device()
    logger.info("Running reward study for %s on device %s", task_key, device)

    collection_cfg = CollectionConfig(
        checkpoint=checkpoint,
        libero_suite=suite,
        task_id=task_id,
        num_demos=num_demos,
        num_rollouts=num_rollouts,
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
        cache_dir=cache_dir,
    )
    cluster_cfg = ClusteringConfig(
        subsample_every=subsample_every,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_auto_eps=True,
        dbscan_percentile=dbscan_percentile,
        seed=seed,
        cache_dir=cache_dir,
    )

    _save_json(
        {
            "task_key": task_key,
            "checkpoint": checkpoint,
            "suite": suite,
            "task_id": task_id,
            "num_demos": num_demos,
            "num_rollouts": num_rollouts,
            "num_envs": num_envs,
            "max_steps": max_steps,
            "seed": seed,
            "subsample_every": subsample_every,
            "dbscan_min_samples": dbscan_min_samples,
            "dbscan_percentile": dbscan_percentile,
            "encoder_batch_size": encoder_batch_size,
            "cache_dir": str(cache_dir),
            "output_dir": str(task_output_dir),
        },
        task_output_dir / "run_config.json",
    )

    logger.info("Phase 1: collecting trajectories")
    demos = collect_demo_trajectories(collection_cfg)
    sft_success, sft_failed, random_trajs = collect_rollouts(collection_cfg, device)
    progress_trajs = collect_progress_trajectories(
        cfg=collection_cfg,
        reference_trajs=demos[:20],
        progress_levels=PROGRESS_LEVELS,
        source_name="demos",
    )

    logger.info(
        "Collected trajectories: demos=%d sft_success=%d sft_failed=%d random=%d",
        len(demos),
        len(sft_success),
        len(sft_failed),
        len(random_trajs),
    )

    logger.info("Phase 2: encoding and secondary caches")
    demos_count = len(demos)
    sft_success_count = len(sft_success)
    sft_failed_count = len(sft_failed)
    random_count = len(random_trajs)
    progress_count = sum(len(trajs) for trajs in progress_trajs.values())
    del demos, sft_success, sft_failed, random_trajs, progress_trajs
    gc.collect()
    _safe_empty_cuda_cache()
    logger.info("Released phase-1 trajectory buffers before loading encoder (%d progress trajectories)", progress_count)
    _ensure_phase2_caches(
        collection_cfg=collection_cfg,
        cache_dir=cache_dir,
        cluster_cfg=cluster_cfg,
        device=device,
        world_model=world_model,
        encoder_batch_size=encoder_batch_size,
    )

    logger.info(
        "Phase 2 completed from cached trajectories: demos=%d sft_success=%d sft_failed=%d random=%d",
        demos_count,
        sft_success_count,
        sft_failed_count,
        random_count,
    )

    logger.info("Phase 3: analysis and report generation")
    cached = _load_cached_analysis(cache_dir)

    x_demo = cached["x_demo"]
    x_sft_ok = cached["x_sft_ok"]
    x_sft_fail = cached["x_sft_fail"]
    x_random = cached["x_random"]
    x_all = cached["x_all"]
    x_reference = cached["x_reference"]
    sources = cached["sources"]
    group_labels = cached["group_labels"]
    ref_mask = cached["ref_mask"]
    progress_embs = cached["progress_embs"]
    x_pf = cached["x_pf"]
    x_clip = cached["x_clip"]
    x_siirl = cached["x_siirl"]
    chunk_data = cached["chunk_data"]
    per_frame_dists = cached["per_frame_dists"]
    n_demo = int(cached["n_demo"])
    n_ok = int(cached["n_ok"])
    n_fail = int(cached["n_fail"])
    n_random = int(cached["n_random"])

    chunk_centers = chunk_data["centers"]
    chunked_curves = chunk_data["curves"]
    per_traj_chunk_corrs = chunk_data["per_traj_corrs"]

    emb_groups = {
        "Demo": x_demo,
        "SFT Success": x_sft_ok,
        "SFT Failed": x_sft_fail,
        "Random": x_random,
    }

    centers_raw = compute_cluster_centers_raw(x_reference, min_samples=dbscan_min_samples, percentile=dbscan_percentile)
    dist_groups_raw = {
        "Demo": distances_to_nearest_center(x_demo, centers_raw),
        "SFT Success": distances_to_nearest_center(x_sft_ok, centers_raw),
        "SFT Failed": distances_to_nearest_center(x_sft_fail, centers_raw),
        "Random": distances_to_nearest_center(x_random, centers_raw),
    }
    table_raw = build_distance_table_global(dist_groups_raw, reward_fn_global=compute_siirl_rewards_global)
    sep_table_raw = build_separation_table(dist_groups_raw)

    null_dim = x_all.shape[1]
    null_demo = generate_null_embeddings(len(x_demo), null_dim, seed=seed)
    null_sft_ok = generate_null_embeddings(len(x_sft_ok), null_dim, seed=seed + 1)
    null_sft_fail = generate_null_embeddings(len(x_sft_fail), null_dim, seed=seed + 2)
    null_random = generate_null_embeddings(len(x_random), null_dim, seed=seed + 3)
    null_ref = np.concatenate([null_demo, null_sft_ok], axis=0)
    null_centers = compute_cluster_centers_raw(null_ref, min_samples=dbscan_min_samples, percentile=dbscan_percentile)
    null_dist_groups = {
        "Demo": distances_to_nearest_center(null_demo, null_centers),
        "SFT Success": distances_to_nearest_center(null_sft_ok, null_centers),
        "SFT Failed": distances_to_nearest_center(null_sft_fail, null_centers),
        "Random": distances_to_nearest_center(null_random, null_centers),
    }
    null_sep = build_separation_table(null_dist_groups)

    centers_siirl, scaler = compute_cluster_centers_siirl(x_reference, min_samples=dbscan_min_samples)
    dist_groups_siirl = {
        "Demo": distances_to_nearest_center(x_demo, centers_siirl, scaler),
        "SFT Success": distances_to_nearest_center(x_sft_ok, centers_siirl, scaler),
        "SFT Failed": distances_to_nearest_center(x_sft_fail, centers_siirl, scaler),
        "Random": distances_to_nearest_center(x_random, centers_siirl, scaler),
    }
    table_siirl = build_distance_table_global(dist_groups_siirl, reward_fn_global=compute_siirl_rewards_global)
    sep_table_siirl = build_separation_table(dist_groups_siirl)
    comparison = pd.concat(
        [
            table_raw.assign(Method="Raw DBSCAN"),
            table_siirl.assign(Method="siiRL (StandardScaler)"),
        ],
        ignore_index=True,
    )

    all_dists_siirl = np.concatenate(list(dist_groups_siirl.values()))
    formula_rows = []
    for name, dists in dist_groups_siirl.items():
        r_siirl = compute_siirl_rewards_global(dists, all_dists_siirl)
        r_zscore = compute_zscore_rewards_global(dists, all_dists_siirl, alpha=0.8)
        formula_rows.append(
            {
                "Group": name,
                "N": len(dists),
                "siiRL Mean": r_siirl.mean(),
                "siiRL Std": r_siirl.std(),
                "z-score Mean": r_zscore.mean(),
                "z-score Std": r_zscore.std(),
                "siiRL Range": r_siirl.max() - r_siirl.min(),
                "z-score Range": r_zscore.max() - r_zscore.min(),
            }
        )
    formula_df = pd.DataFrame(formula_rows)

    distances_per_level_raw = {
        level: distances_to_nearest_center(progress_embs[level], centers_raw) for level in PROGRESS_LEVELS
    }
    distances_per_level_siirl = {
        level: distances_to_nearest_center(progress_embs[level], centers_siirl, scaler) for level in PROGRESS_LEVELS
    }
    progress_df = build_progress_table(PROGRESS_LEVELS, distances_per_level_siirl)
    corr_raw, pval_raw = compute_progress_correlation(PROGRESS_LEVELS, distances_per_level_raw)
    corr_siirl, pval_siirl = compute_progress_correlation(PROGRESS_LEVELS, distances_per_level_siirl)
    ptr_corr_raw, ptr_pval_raw, ptr_ci_lo_raw, ptr_ci_hi_raw = compute_per_trajectory_progress_correlation(
        PROGRESS_LEVELS,
        distances_per_level_raw,
        seed=seed,
    )
    ptr_corr, ptr_pval, ptr_ci_lo, ptr_ci_hi = compute_per_trajectory_progress_correlation(
        PROGRESS_LEVELS,
        distances_per_level_siirl,
        seed=seed,
    )

    centers_pf_raw = compute_cluster_centers_raw(
        x_pf[ref_mask], min_samples=dbscan_min_samples, percentile=dbscan_percentile
    )
    centers_clip_raw = compute_cluster_centers_raw(
        x_clip[ref_mask],
        min_samples=dbscan_min_samples,
        percentile=dbscan_percentile,
    )
    centers_siirl_raw = compute_cluster_centers_raw(
        x_siirl[ref_mask],
        min_samples=dbscan_min_samples,
        percentile=dbscan_percentile,
    )
    centers_pf_siirl, scaler_pf = compute_cluster_centers_siirl(x_pf[ref_mask], min_samples=dbscan_min_samples)
    centers_clip_siirl, scaler_clip = compute_cluster_centers_siirl(x_clip[ref_mask], min_samples=dbscan_min_samples)
    centers_siirl_siirl, scaler_siirl = compute_cluster_centers_siirl(
        x_siirl[ref_mask],
        min_samples=dbscan_min_samples,
    )
    enc_df = compare_encoding_methods(
        per_frame_embs=x_pf,
        clip_embs=x_clip,
        reference_per_frame=x_pf[ref_mask],
        reference_clip=x_clip[ref_mask],
        group_labels=group_labels,
    )

    enc_sep_rows = []
    for enc_name, x_enc, centers_enc, scaler_enc in [
        ("Per-Frame", x_pf, centers_pf_siirl, scaler_pf),
        ("64-Frame Clip", x_clip, centers_clip_siirl, scaler_clip),
        ("siiRL-Faithful", x_siirl, centers_siirl_siirl, scaler_siirl),
    ]:
        enc_dist_groups = {}
        offset = 0
        for group_name, count in [
            ("Demo", n_demo),
            ("SFT Success", n_ok),
            ("SFT Failed", n_fail),
            ("Random", n_random),
        ]:
            enc_dist_groups[group_name] = distances_to_nearest_center(
                x_enc[offset : offset + count],
                centers_enc,
                scaler_enc,
            )
            offset += count
        sep = build_separation_table(enc_dist_groups)
        combined = sep[sep["Comparison"] == "Success vs All Failure"].iloc[0]
        enc_sep_rows.append(
            {
                "Encoding": enc_name,
                "Cohen's d": combined["Cohen's d"],
                "AUROC": combined["AUROC"],
                "Overlap": combined["Overlap"],
            }
        )
    enc_sep_df = pd.DataFrame(enc_sep_rows)

    chunk_corr, chunk_ci_lo, chunk_ci_hi = _chunk_corr_ci(per_traj_chunk_corrs, seed=seed)
    chunk_pval = 0.0

    cum_var, _ = pca_explained_variance(x_all)
    dim_stats = embedding_dimension_stats(x_all)

    corr_full_traj, pval_full_traj = compute_progress_correlation(PROGRESS_LEVELS, distances_per_level_siirl)
    correlation_df = build_progress_correlation_table(
        {
            "SRPO full-traj (level-mean)": (corr_full_traj, pval_full_traj),
            "SRPO full-traj (per-traj)": (ptr_corr, ptr_pval, ptr_ci_lo, ptr_ci_hi),
            "SRPO chunked (window=32)": (chunk_corr, chunk_pval, chunk_ci_lo, chunk_ci_hi),
        },
        top_rewarder_ref=0.95,
    )

    _save_dataframe(table_raw, tables_dir / "exp1_distance_table_raw.csv")
    _save_dataframe(sep_table_raw, tables_dir / "exp1_separation_raw.csv")
    _save_dataframe(null_sep, tables_dir / "exp1_null_baseline.csv")
    _save_dataframe(table_siirl, tables_dir / "exp2_distance_table_siirl.csv")
    _save_dataframe(sep_table_siirl, tables_dir / "exp2_separation_siirl.csv")
    _save_dataframe(comparison, tables_dir / "exp2_raw_vs_siirl.csv")
    _save_dataframe(formula_df, tables_dir / "exp3_reward_formula.csv")
    _save_dataframe(progress_df, tables_dir / "exp4_progress_table.csv")
    _save_dataframe(enc_df, tables_dir / "exp5_encoding_comparison.csv")
    _save_dataframe(enc_sep_df, tables_dir / "exp5_encoding_separation.csv")
    _save_dataframe(dim_stats.nlargest(10, "Std"), tables_dir / "exp6_top10_dimension_std.csv")
    _save_dataframe(correlation_df, tables_dir / "exp8_progress_correlations.csv")

    xy = fit_umap(x_all, cluster_cfg)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_distance_kde(dist_groups_raw, title="Exp 1 - Distance to Nearest Center (raw DBSCAN)", ax=axes[0])
    plot_reward_kde_global(
        dist_groups_raw,
        reward_fn_global=compute_siirl_rewards_global,
        title="Exp 1 - siiRL Rewards (raw DBSCAN, global norm)",
        ax=axes[1],
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp1_distance_reward_raw.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_panel_b(ax, xy, sources)
    ax.set_title("UMAP by Trajectory Source", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp1_umap_sources.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_cosine_similarity_matrix(emb_groups, ax=ax)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp1_cosine_similarity.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_distance_kde(dist_groups_siirl, title="Exp 2 - Distance (StandardScaler + DBSCAN)", ax=axes[0])
    plot_reward_kde_global(
        dist_groups_siirl,
        reward_fn_global=compute_siirl_rewards_global,
        title="Exp 2 - siiRL Rewards (StandardScaler + DBSCAN, global norm)",
        ax=axes[1],
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp2_distance_reward_siirl.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(all_dists_siirl, compute_siirl_rewards(all_dists_siirl), alpha=0.3, s=10)
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("siiRL: 0.6 * sigmoid(10 * (0.5 - norm_dist))")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(all_dists_siirl, compute_zscore_rewards(all_dists_siirl, alpha=0.8), alpha=0.3, s=10)
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Ours: 0.8 * sigmoid(-z_score)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Exp 3 - Reward Formula Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp3_reward_formula.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_progress_curve(
        PROGRESS_LEVELS, distances_per_level_raw, title="Exp 4 - Distance vs Progress (raw)", ax=axes[0]
    )
    plot_progress_curve(
        PROGRESS_LEVELS,
        distances_per_level_siirl,
        title="Exp 4 - Distance vs Progress (siiRL method)",
        ax=axes[1],
    )
    fig.suptitle("Progress Monotonicity Test", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp4_progress_monotonicity.png")

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    rows = [
        ("Raw DBSCAN", centers_pf_raw, centers_clip_raw, centers_siirl_raw, None, None, None),
        (
            "siiRL (StandardScaler)",
            centers_pf_siirl,
            centers_clip_siirl,
            centers_siirl_siirl,
            scaler_pf,
            scaler_clip,
            scaler_siirl,
        ),
    ]
    for row_idx, (label, c_pf, c_clip, c_siirl, s_pf, s_clip, s_siirl) in enumerate(rows):
        for col_idx, (enc_name, x_enc, centers_enc, scaler_enc) in enumerate(
            [
                ("Per-Frame", x_pf, c_pf, s_pf),
                ("64-Frame Clip", x_clip, c_clip, s_clip),
                ("siiRL-Faithful", x_siirl, c_siirl, s_siirl),
            ]
        ):
            dist_groups = {}
            offset = 0
            for group_name, count in [
                ("Demo", n_demo),
                ("SFT Success", n_ok),
                ("SFT Failed", n_fail),
                ("Random", n_random),
            ]:
                dist_groups[group_name] = distances_to_nearest_center(
                    x_enc[offset : offset + count],
                    centers_enc,
                    scaler_enc,
                )
                offset += count
            plot_distance_kde(dist_groups, title=f"{enc_name} / {label}", ax=axes[row_idx, col_idx])
    fig.suptitle("Exp 5 - Per-Frame vs Clip vs siiRL-Faithful", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp5_encoding_kdes.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_chunked_progress_curves(
        chunked_curves,
        title=f"Exp 5b - Per-Chunk Distance (window={WINDOW_SIZE}, stride={STRIDE})",
        ax=ax,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp5b_chunked_progress.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    plot_pca_explained_variance(x_all, title="Exp 6a - PCA Explained Variance", ax=axes[0])
    plot_dimension_variance(x_all, title="Exp 6b - Per-Dimension Std Dev (top 50)", ax=axes[1])
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp6_dimensionality.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_knn_elbow(
        x_reference,
        k_values=[2, 3, 5, 10],
        title="Exp 7a - k-NN Distances (reference, raw space)",
        ax=axes[0],
    )
    x_ref_scaled = StandardScaler().fit_transform(x_reference)
    plot_knn_elbow(
        x_ref_scaled,
        k_values=[2, 3, 5, 10],
        title="Exp 7b - k-NN Distances (reference, scaled space)",
        ax=axes[1],
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp7_knn_reference.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_knn_elbow_comparison(
        {
            "Demo": x_demo,
            "SFT Success": x_sft_ok,
            "SFT Failed": x_sft_fail,
            "Random": x_random,
            "All References": x_reference,
        },
        k=2,
        title="Exp 7c - k-NN Curves by Group (k=2)",
        ax=ax,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp7_knn_groups.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_progress_curve(
        PROGRESS_LEVELS,
        distances_per_level_siirl,
        title="SRPO Full-Trajectory Distance vs Progress",
        ax=axes[0],
    )
    plot_chunked_progress_curves(
        {"Demo": chunked_curves["Demo"]},
        title="SRPO Chunked (Demo) Distance Over Time",
        ax=axes[1],
    )
    fig.suptitle("Exp 8b - Full-Trajectory vs Chunked Progress Signal", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp8_full_vs_chunked.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_per_frame_evolution(
        per_frame_dists,
        title="Exp 8a - Per-Frame Distance to Success Cluster Over Time",
        ax=ax,
    )
    fig.tight_layout()
    _save_figure(fig, figures_dir / "exp8_per_frame_evolution.png")

    summary_metrics = {
        "sep_auroc_raw": float(sep_table_raw[sep_table_raw["Comparison"] == "Success vs All Failure"]["AUROC"].iloc[0]),
        "sep_auroc_siirl": float(
            sep_table_siirl[sep_table_siirl["Comparison"] == "Success vs All Failure"]["AUROC"].iloc[0]
        ),
        "sep_cohens_d_raw": float(
            sep_table_raw[sep_table_raw["Comparison"] == "Success vs All Failure"]["Cohen's d"].iloc[0]
        ),
        "sep_cohens_d_siirl": float(
            sep_table_siirl[sep_table_siirl["Comparison"] == "Success vs All Failure"]["Cohen's d"].iloc[0]
        ),
        "progress_corr_level_mean": float(corr_siirl),
        "progress_corr_per_traj": float(ptr_corr),
        "progress_corr_per_traj_ci_lo": float(ptr_ci_lo),
        "progress_corr_per_traj_ci_hi": float(ptr_ci_hi),
        "chunked_corr_mean": float(chunk_corr),
        "chunked_corr_ci_lo": float(chunk_ci_lo),
        "chunked_corr_ci_hi": float(chunk_ci_hi),
        "raw_progress_corr_level_mean": float(corr_raw),
        "raw_progress_corr_level_mean_p": float(pval_raw),
        "raw_progress_corr_per_traj": float(ptr_corr_raw),
        "raw_progress_corr_per_traj_p": float(ptr_pval_raw),
        "raw_progress_corr_per_traj_ci_lo": float(ptr_ci_lo_raw),
        "raw_progress_corr_per_traj_ci_hi": float(ptr_ci_hi_raw),
        "siirl_progress_corr_level_mean_p": float(pval_siirl),
        "siirl_progress_corr_per_traj_p": float(ptr_pval),
        "num_raw_centers": int(centers_raw.shape[0]),
        "num_siirl_centers": int(centers_siirl.shape[0]),
        "num_chunk_centers": int(chunk_centers.shape[0]),
        "n_demo": n_demo,
        "n_sft_success": n_ok,
        "n_sft_failed": n_fail,
        "n_random": n_random,
        "variance_dims_90": int(np.searchsorted(cum_var, 0.90) + 1),
        "variance_dims_95": int(np.searchsorted(cum_var, 0.95) + 1),
        "variance_dims_99": int(np.searchsorted(cum_var, 0.99) + 1),
        "top_std_ratio": float(dim_stats["Std"].max() / (dim_stats["Std"].min() + 1e-10)),
    }

    metrics_path = output_root / "metrics" / f"{task_key}.json"
    _save_json(summary_metrics, metrics_path)
    _save_json(summary_metrics, task_output_dir / "summary_metrics.json")
    _write_multi_task_summary(output_root)

    logger.info("Completed %s", task_key)
    logger.info("Artifacts: %s", task_output_dir)
    logger.info("Cache: %s", cache_dir)
    return summary_metrics


def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    suite: LiberoSuite = typer.Option(LiberoSuite.SPATIAL, "--suite", help="LIBERO suite"),
    task_id: int = typer.Option(9, "--task-id", help="Task index within the suite"),
    world_model: WorldModelType = typer.Option(
        WorldModelType.VJEPA2,
        "--world-model",
        help="World model encoder to use for trajectory embeddings",
    ),
    num_demos: int = typer.Option(50, "--num-demos"),
    num_rollouts: int = typer.Option(50, "--num-rollouts"),
    num_envs: int = typer.Option(4, "--num-envs"),
    max_steps: int = typer.Option(300, "--max-steps"),
    subsample_every: int = typer.Option(1, "--subsample-every"),
    dbscan_min_samples: int = typer.Option(2, "--dbscan-min-samples"),
    dbscan_percentile: int = typer.Option(25, "--dbscan-percentile"),
    encoder_batch_size: int = typer.Option(4, "--encoder-batch-size"),
    seed: int = typer.Option(42, "--seed"),
    cache_root: Path = typer.Option(DEFAULT_CACHE_ROOT, "--cache-root", help="Root directory for per-task caches"),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output-root", help="Root directory for saved reports"),
) -> None:
    """Run the full reward-study pipeline for a single LIBERO task."""
    metrics = run_task(
        checkpoint=checkpoint,
        suite=str(suite),
        task_id=task_id,
        num_demos=num_demos,
        num_rollouts=num_rollouts,
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
        subsample_every=subsample_every,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_percentile=dbscan_percentile,
        encoder_batch_size=encoder_batch_size,
        cache_root=cache_root,
        output_root=output_root,
        world_model=world_model,
    )

    print(json.dumps({"task": _task_key(str(suite), task_id), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    typer.run(main)
