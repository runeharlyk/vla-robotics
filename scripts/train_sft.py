"""Supervised fine-tuning (behavior cloning) of SmolVLA on a few-demo dataset.

Usage:
    uv run python scripts/train_sft.py --num-demos 10 --seed 42
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer
import wandb

from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.trainer import SFTConfig, train_sft
from vla.utils import get_device, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "preprocessed" / "pickcube.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(
    data_path: Path = typer.Option(DEFAULT_DATA, "--data", "-d", path_type=Path),
    num_demos: int = typer.Option(10, "--num-demos", "-n"),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    lr: float = typer.Option(1e-4, "--lr"),
    batch_size: int = typer.Option(32, "--batch-size"),
    micro_batch_size: int = typer.Option(4, "--micro-batch-size"),
    num_epochs: int = typer.Option(50, "--epochs"),
    warmup_steps: int = typer.Option(1000, "--warmup-steps"),
    decay_steps: int = typer.Option(30000, "--decay-steps"),
    decay_lr: float = typer.Option(2.5e-6, "--decay-lr"),
    grad_clip_norm: float = typer.Option(10.0, "--grad-clip-norm"),
    eval_every: int = typer.Option(5, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(200, "--max-steps"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option("PegInsertionSide-v1", "--env"),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """Run SFT behavior cloning from few demonstrations."""
    seed_everything(seed)
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = FewDemoDataset(data_path, num_demos=num_demos, seed=seed)
    logging.info(f"Loaded {dataset.num_episodes} episodes ({len(dataset)} timesteps) from {data_path}")

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )

    config = SFTConfig(
        lr=lr,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
        grad_clip_norm=grad_clip_norm,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        save_dir=str(PROJECT_ROOT / "checkpoints" / "sft" / f"demos{num_demos}_seed{seed}"),
        env_id=env_id,
        seed=seed,
    )

    run = None
    if use_wandb:
        run = wandb.init(
            project="srpo-smolvla",
            name=f"sft_demos{num_demos}_seed{seed}",
            config={
                "method": "sft",
                "num_demos": num_demos,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "num_epochs": num_epochs,
                "warmup_steps": warmup_steps,
                "decay_steps": decay_steps,
                "decay_lr": decay_lr,
                "grad_clip_norm": grad_clip_norm,
                "checkpoint": checkpoint,
                "env_id": env_id,
            },
        )

    train_sft(policy, dataset, config, wandb_run=run)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
