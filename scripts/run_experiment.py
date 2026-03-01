"""Run the full SRPO validation experiment matrix.

For each (demo_count, seed) pair this script:
  1. Trains SFT baseline
  2. Runs sparse-RL from the SFT checkpoint
  3. Runs SRPO from the SFT checkpoint (with demo-seeded world-model rewards)

Usage:
    uv run python scripts/run_experiment.py
    uv run python scripts/run_experiment.py --config configs/srpo_pickcube.yaml
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import typer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _run(cmd: list[str]) -> int:
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        log.error(f"Command failed with code {result.returncode}")
    return result.returncode


def main(
    config_path: Path = typer.Option(PROJECT_ROOT / "configs" / "srpo_pickcube.yaml", "--config", "-c", path_type=Path),
    no_wandb: bool = typer.Option(False, "--no-wandb"),
) -> None:
    """Run the full experiment matrix from a YAML config."""
    cfg = yaml.safe_load(config_path.read_text())
    demo_counts: list[int] = cfg["demo_counts"]
    seeds: list[int] = cfg["seeds"]
    data_path = str(PROJECT_ROOT / cfg["data_path"])
    checkpoint = cfg["checkpoint"]
    env_id = cfg["env_id"]
    instruction = cfg["instruction"]
    action_dim = cfg["action_dim"]
    sft_cfg = cfg["sft"]
    srpo_cfg = cfg["srpo"]
    sparse_cfg = cfg["sparse_rl"]

    wandb_flag = "--no-wandb" if no_wandb else "--wandb"

    for n_demos in demo_counts:
        for seed in seeds:
            tag = f"demos{n_demos}_seed{seed}"
            sft_save = str(PROJECT_ROOT / "checkpoints" / "sft" / tag)
            sft_best = str(Path(sft_save) / "best")

            log.info(f"\n{'='*60}\n  SFT  {tag}\n{'='*60}")
            rc = _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_sft.py"),
                    "--data",
                    data_path,
                    "--num-demos",
                    str(n_demos),
                    "--checkpoint",
                    checkpoint,
                    "--lr",
                    str(sft_cfg["lr"]),
                    "--batch-size",
                    str(sft_cfg["batch_size"]),
                    "--micro-batch-size",
                    str(sft_cfg.get("micro_batch_size", 4)),
                    "--epochs",
                    str(sft_cfg["num_epochs"]),
                    "--warmup-steps",
                    str(sft_cfg.get("warmup_steps", 1000)),
                    "--decay-steps",
                    str(sft_cfg.get("decay_steps", 30000)),
                    "--decay-lr",
                    str(sft_cfg.get("decay_lr", 2.5e-6)),
                    "--grad-clip-norm",
                    str(sft_cfg.get("grad_clip_norm", 10.0)),
                    "--eval-every",
                    str(sft_cfg["eval_every"]),
                    "--eval-episodes",
                    str(sft_cfg["eval_episodes"]),
                    "--max-steps",
                    str(sft_cfg["max_steps"]),
                    "--seed",
                    str(seed),
                    "--env",
                    env_id,
                    wandb_flag,
                ]
            )
            if rc != 0:
                log.warning(f"SFT failed for {tag}, skipping RL stages")
                continue

            for mode, rl_cfg in [("sparse_rl", sparse_cfg), ("srpo", srpo_cfg)]:
                log.info(f"\n{'='*60}\n  {mode.upper()}  {tag}\n{'='*60}")
                cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_srpo.py"),
                    "--sft-checkpoint",
                    sft_best,
                    "--checkpoint",
                    checkpoint,
                    "--data",
                    data_path,
                    "--num-demos",
                    str(n_demos),
                    "--action-dim",
                    str(action_dim),
                    "--mode",
                    mode,
                    "--lr",
                    str(rl_cfg["lr"]),
                    "--iterations",
                    str(rl_cfg["num_iterations"]),
                    "--trajs-per-iter",
                    str(rl_cfg["trajectories_per_iter"]),
                    "--ppo-epochs",
                    str(rl_cfg.get("ppo_epochs", 4)),
                    "--clip-epsilon",
                    str(rl_cfg.get("clip_epsilon", 0.2)),
                    "--kl-coeff",
                    str(rl_cfg.get("kl_coeff", 0.01)),
                    "--eval-every",
                    str(rl_cfg["eval_every"]),
                    "--eval-episodes",
                    str(rl_cfg["eval_episodes"]),
                    "--max-steps",
                    str(rl_cfg["max_steps"]),
                    "--seed",
                    str(seed),
                    "--env",
                    env_id,
                    "--instruction",
                    instruction,
                    wandb_flag,
                ]
                if mode == "srpo":
                    cmd.extend(["--world-model", str(srpo_cfg.get("world_model_type", "dinov2"))])
                    cmd.extend(["--subsample-every", str(srpo_cfg.get("subsample_every", 5))])
                    cmd.extend(["--dbscan-eps", str(srpo_cfg.get("dbscan_eps", 0.5))])
                    cmd.extend(["--dbscan-min-samples", str(srpo_cfg.get("dbscan_min_samples", 2))])
                if "gamma" in rl_cfg:
                    cmd.extend(["--gamma", str(rl_cfg["gamma"])])
                if "reward_scale" in rl_cfg:
                    cmd.extend(["--reward-scale", str(rl_cfg["reward_scale"])])
                _run(cmd)

    log.info("\nExperiment matrix complete.")


if __name__ == "__main__":
    typer.run(main)
