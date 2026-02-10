"""Training script for state-based action model.

This script trains an MLP to predict actions from robot proprioceptive state,
bypassing the need for RGB observations.
"""

from pathlib import Path
from typing import Optional

import torch
import typer
from tqdm import tqdm

from diffusion_policy.dataset import create_state_dataloader
from diffusion_policy.state_action_model import StateActionConfig, StateActionModel

app = typer.Typer()


def train_epoch(
    model: StateActionModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    for batch in train_loader:
        state = batch["state"]
        action = batch["action"]

        optimizer.zero_grad()
        losses = model.compute_loss(state, action)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_mse += losses["mse"].item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
    }


def validate(
    model: StateActionModel,
    val_loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            state = batch["state"]
            action = batch["action"]

            losses = model.compute_loss(state, action)

            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
    }


@app.command()
def main(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    epochs: int = typer.Option(100, "--epochs", "-n", help="Number of training epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    max_trajectories: int = typer.Option(100, "--max-traj", help="Max trajectories to load"),
    save_dir: str = typer.Option("checkpoints", "--save-dir", "-o", help="Save directory"),
    hidden_dim: int = typer.Option(256, "--hidden-dim", help="Hidden dimension"),
    num_layers: int = typer.Option(4, "--num-layers", help="Number of MLP layers"),
) -> None:
    """Train state-based action prediction model."""
    print(f"Training state-based action model for {env_id}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    train_loader, val_loader, state_dim, action_dim = create_state_dataloader(
        env_id=env_id,
        batch_size=batch_size,
        max_trajectories=max_trajectories,
        num_workers=0,
    )

    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

    config = StateActionConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model = StateActionModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Training"):
        train_metrics = train_epoch(model, train_loader, optimizer)
        val_metrics = validate(model, val_loader)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"\nEpoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"lr={current_lr:.2e}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_loss": best_val_loss,
            }
            torch.save(checkpoint, save_path / f"state_model_{env_id.replace('-', '_')}_best.pt")

    checkpoint = {
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "val_loss": val_metrics["loss"],
    }
    torch.save(checkpoint, save_path / f"state_model_{env_id.replace('-', '_')}_final.pt")

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {save_path}")


if __name__ == "__main__":
    app()
