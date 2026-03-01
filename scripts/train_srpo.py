"""SRPO / sparse-RL training on top of an SFT-initialised SmolVLA policy.

Usage:
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/demos5_seed42/best --mode srpo
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/demos5_seed42/best --mode sparse_rl
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import wandb

from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import Trajectory
from vla.rl.trainer import SRPOConfig, train_srpo
from vla.utils import get_device, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICK_CUBE_INSTRUCTION = "pick up the red cube and move it to the green goal"
DEFAULT_DATA = PROJECT_ROOT / "data" / "preprocessed" / "pickcube.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(
    sft_checkpoint: Path = typer.Option(..., "--sft-checkpoint", "-s", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path = typer.Option(DEFAULT_DATA, "--data", "-d", path_type=Path),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    action_dim: int = typer.Option(8, "--action-dim"),
    mode: str = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    lr: float = typer.Option(1e-5, "--lr"),
    num_iterations: int = typer.Option(100, "--iterations"),
    trajectories_per_iter: int = typer.Option(16, "--trajs-per-iter"),
    ppo_epochs: int = typer.Option(4, "--ppo-epochs"),
    clip_epsilon: float = typer.Option(0.2, "--clip-epsilon"),
    kl_coeff: float = typer.Option(0.01, "--kl-coeff"),
    gamma: float = typer.Option(0.99, "--gamma"),
    reward_scale: float = typer.Option(1.0, "--reward-scale"),
    eval_every: int = typer.Option(10, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(200, "--max-steps"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option("PickCube-v1", "--env"),
    instruction: str = typer.Option(PICK_CUBE_INSTRUCTION, "--instruction"),
    world_model: str = typer.Option("dinov2", "--world-model", help="dinov2 or vjepa2"),
    subsample_every: int = typer.Option(5, "--subsample-every"),
    dbscan_eps: float = typer.Option(0.5, "--dbscan-eps"),
    dbscan_min_samples: int = typer.Option(2, "--dbscan-min-samples"),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """Run SRPO or sparse-RL training starting from an SFT checkpoint."""
    seed_everything(seed)
    device = get_device()

    dataset = FewDemoDataset(data_path, num_demos=num_demos, seed=seed)
    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )
    policy.load_checkpoint(sft_checkpoint)
    logging.info(f"Loaded SFT checkpoint from {sft_checkpoint}")

    demo_trajectories: list[Trajectory] | None = None
    if mode == "srpo":
        demo_trajectories = dataset.episodes_as_trajectories()
        logging.info(f"Built {len(demo_trajectories)} demo trajectories for reference seeding")

    config = SRPOConfig(
        lr=lr,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        kl_coeff=kl_coeff,
        gamma=gamma,
        reward_scale=reward_scale,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        save_dir=str(PROJECT_ROOT / "checkpoints" / mode / f"seed{seed}"),
        env_id=env_id,
        seed=seed,
        mode=mode,
        world_model_type=world_model,
        subsample_every=subsample_every,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    run = None
    if use_wandb:
        run = wandb.init(
            project="srpo-smolvla",
            name=f"{mode}_seed{seed}",
            config={
                "method": mode,
                "seed": seed,
                "lr": lr,
                "num_iterations": num_iterations,
                "trajectories_per_iter": trajectories_per_iter,
                "ppo_epochs": ppo_epochs,
                "clip_epsilon": clip_epsilon,
                "kl_coeff": kl_coeff,
                "gamma": gamma,
                "reward_scale": reward_scale,
                "checkpoint": checkpoint,
                "sft_checkpoint": str(sft_checkpoint),
                "env_id": env_id,
                "world_model": world_model,
                "num_demos": num_demos,
            },
        )

    train_srpo(policy, config, instruction, demo_trajectories=demo_trajectories, wandb_run=run)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
