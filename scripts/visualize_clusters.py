"""Visualise DBSCAN clustering of trajectory embeddings.

Orchestrates two modules:
  1. ``vla.diagnostics.collect_trajectories`` — collects and caches trajectory
     buffers (demo, SFT success/fail, random).
  2. ``vla.diagnostics.clustering`` — encodes, clusters, and plots the results.

Usage:
    # Default (spatial task 5, vjepa2, 8 envs, 24 rollouts per buffer):
    uv run python scripts/visualize_clusters.py

    # Custom task and suite:
    uv run python scripts/visualize_clusters.py --suite spatial --task-id 3

    # Faster run with fewer rollouts:
    uv run python scripts/visualize_clusters.py --num-rollouts 8 --num-envs 4

    # DINOv2 encoder instead of V-JEPA 2:
    uv run python scripts/visualize_clusters.py --world-model dinov2
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

from vla.constants import LiberoSuite, WorldModelType
from vla.diagnostics.clustering import (
    SOURCE_DEMO,
    SOURCE_FAILED,
    SOURCE_RANDOM,
    SOURCE_SFT_SUCCESS,
    ClusteringConfig,
    fit_dbscan_reference,
    fit_umap,
    get_or_compute_embeddings,
    plot_clustering_figure,
    plot_distance_figure,
    print_composition_table,
)
from vla.diagnostics.collect_trajectories import (
    CollectionConfig,
    collect_demo_trajectories,
    collect_rollouts,
)
from vla.models.world_model import build_world_model
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    suite: LiberoSuite = typer.Option("spatial", "--suite", help="LIBERO suite"),
    task_id: int = typer.Option(5, "--task-id", help="Task index within the suite"),
    world_model: WorldModelType = typer.Option("vjepa2", "--world-model", help="dinov2 or vjepa2"),
    num_demos: int = typer.Option(100, "--num-demos", help="Max demo trajectories to load"),
    num_rollouts: int = typer.Option(24, "--num-rollouts", help="Rollouts per buffer (SFT✓, SFT✗, random)"),
    num_envs: int = typer.Option(8, "--num-envs", help="Parallel LIBERO environments"),
    max_steps: int = typer.Option(300, "--max-steps", help="Max episode length"),
    action_dim: int = typer.Option(7, "--action-dim"),
    state_dim: int = typer.Option(8, "--state-dim"),
    subsample_every: int = typer.Option(5, "--subsample-every"),
    dbscan_eps: float = typer.Option(0.5, "--dbscan-eps"),
    dbscan_min_samples: int = typer.Option(2, "--dbscan-min-samples"),
    dbscan_auto_eps: bool = typer.Option(True, "--dbscan-auto-eps/--no-dbscan-auto-eps"),
    dbscan_percentile: int = typer.Option(25, "--dbscan-percentile"),
    umap_n_neighbors: int = typer.Option(15, "--umap-n-neighbors"),
    umap_min_dist: float = typer.Option(0.1, "--umap-min-dist"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path = typer.Option(Path("notebooks/cache"), "--cache-dir"),
    no_show: bool = typer.Option(False, "--no-show", help="Don't call plt.show()"),
) -> None:
    """Run the full clustering analysis pipeline."""
    seed_everything(seed)
    device = get_device()

    # ── Build configs ──
    col_cfg = CollectionConfig(
        checkpoint=checkpoint,
        libero_suite=suite,
        task_id=task_id,
        action_dim=action_dim,
        state_dim=state_dim,
        num_demos=num_demos,
        num_rollouts=num_rollouts,
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
        cache_dir=cache_dir,
    )
    cls_cfg = ClusteringConfig(
        subsample_every=subsample_every,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        dbscan_auto_eps=dbscan_auto_eps,
        dbscan_percentile=dbscan_percentile,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        seed=seed,
        cache_dir=cache_dir,
    )

    # ── 1. Collect trajectories ──
    logger.info("Step 1: Collecting trajectories")
    demo_trajs = collect_demo_trajectories(col_cfg)
    sft_success, sft_failed, random_failed = collect_rollouts(col_cfg, device)

    logger.info(
        "Trajectories: %d demos, %d SFT✓, %d SFT✗, %d random",
        len(demo_trajs), len(sft_success), len(sft_failed), len(random_failed),
    )

    # ── 2. Compute embeddings ──
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Step 2: Computing trajectory embeddings")
    encoder = build_world_model(model_type=world_model, device=str(device), batch_size=1)

    emb_dir = cache_dir
    emb_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{suite}_task{task_id}"

    demo_embs = get_or_compute_embeddings(demo_trajs, encoder, emb_dir / f"{prefix}_demos_embs.pt", subsample_every)
    sft_ok_embs = get_or_compute_embeddings(sft_success, encoder, emb_dir / f"{prefix}_sft_success_embs.pt", subsample_every)
    sft_fail_embs = get_or_compute_embeddings(sft_failed, encoder, emb_dir / f"{prefix}_sft_failed_embs.pt", subsample_every)
    random_embs = get_or_compute_embeddings(random_failed, encoder, emb_dir / f"{prefix}_random_failed_embs.pt", subsample_every)

    encoder.offload()

    # ── 3. Concatenate + analyse ──
    logger.info("Step 3: DBSCAN + UMAP")
    all_embs = torch.cat([demo_embs, sft_ok_embs, sft_fail_embs, random_embs], dim=0)
    X = all_embs.cpu().numpy()

    sources: list[str] = (
        [SOURCE_DEMO] * len(demo_trajs)
        + [SOURCE_SFT_SUCCESS] * len(sft_success)
        + [SOURCE_FAILED] * len(sft_failed)
        + [SOURCE_RANDOM] * len(random_failed)
    )
    all_trajs = demo_trajs + sft_success + sft_failed + random_failed
    traj_indices = list(range(len(all_trajs)))

    # Cluster on successful trajectories (Demo + SFT✓) to build the reference set
    reference_mask = np.array([s in (SOURCE_DEMO, SOURCE_SFT_SUCCESS) for s in sources])
    labels = fit_dbscan_reference(X, reference_mask, cfg=cls_cfg)
    
    print_composition_table(labels, sources)
    xy = fit_umap(X, cls_cfg)

    # ── 4. Plot ──
    logger.info("Step 4: Plotting")
    save_path = cache_dir / f"{prefix}_clustering.png"
    plot_clustering_figure(
        xy, labels, sources, X, all_trajs, traj_indices,
        title=f"Trajectory Embedding Analysis — {suite} task {task_id}",
        save_path=save_path,
    )

    dist_save_path = cache_dir / f"{prefix}_distances.png"
    plot_distance_figure(
        X, sources, cls_cfg,
        title=f"Distance to Centers — {suite} task {task_id}",
        save_path=dist_save_path,
    )

    if not no_show:
        plt.show()


if __name__ == "__main__":
    typer.run(main)
