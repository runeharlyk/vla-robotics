"""Standalone script for high-throughput trajectory collection.

This script focuses solely on collecting and caching trajectories to avoid 
the overhead of the full visualization pipeline when scaling to many samples.
"""

import logging
from pathlib import Path

import torch
import typer

from vla.diagnostics.collect_trajectories import (
    CollectionConfig,
    collect_demo_trajectories,
    collect_rollouts,
)
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    suite: str = typer.Option("spatial", "--suite"),
    task_id: int = typer.Option(5, "--task-id"),
    num_demos: int = typer.Option(100, "--num-demos"),
    num_rollouts: int = typer.Option(100, "--num-rollouts"),
    num_envs: int = typer.Option(4, "--num-envs"),
    max_steps: int = typer.Option(300, "--max-steps"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path = typer.Option(Path("notebooks/cache"), "--cache-dir"),
) -> None:
    """Collect trajectory buffers and save to cache."""
    seed_everything(seed)
    device = get_device()

    cfg = CollectionConfig(
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

    logger.info("Starting collection for task: %s (ID: %d)", suite, task_id)
    
    # 1. Demos
    demo_trajs = collect_demo_trajectories(cfg)
    logger.info("Demos: %d collected", len(demo_trajs))

    # 2. Rollouts (SFT Success, SFT Failed, Random)
    sft_success, sft_failed, random_trajs = collect_rollouts(cfg, device)
    
    logger.info("Collection complete.")
    logger.info("SFT Success: %d", len(sft_success))
    logger.info("SFT Failed: %d", len(sft_failed))
    logger.info("Random: %d", len(random_trajs))


if __name__ == "__main__":
    typer.run(main)
