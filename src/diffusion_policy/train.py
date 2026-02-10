"""Training script for CLIP action model using behavioral cloning."""

from pathlib import Path
from typing import Optional
import torch
import typer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from diffusion_policy.clip_action_model import create_clip_action_model
from diffusion_policy.dataset import create_dataloader

app = typer.Typer()

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: The CLIP action model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Dictionary of average losses.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        actions = batch["action"].to(device)
        texts = batch["text"]

        optimizer.zero_grad()
        losses = model.compute_loss(images, actions, text=texts)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        total_mse += losses["mse_loss"].item()
        total_l1 += losses["l1_loss"].item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mse_loss": total_mse / num_batches,
        "l1_loss": total_l1 / num_batches,
    }


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model.

    Args:
        model: The CLIP action model.
        val_loader: Validation data loader.
        device: Device to validate on.

    Returns:
        Dictionary of average losses.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            actions = batch["action"].to(device)
            texts = batch["text"]

            losses = model.compute_loss(images, actions, text=texts)

            total_loss += losses["loss"].item()
            total_mse += losses["mse_loss"].item()
            total_l1 += losses["l1_loss"].item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mse_loss": total_mse / num_batches,
        "l1_loss": total_l1 / num_batches,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt",
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        checkpoint_dir: Directory to save checkpoint.
        filename: Checkpoint filename.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Path,
) -> int:
    """Load a training checkpoint.

    Args:
        model: The model to load weights into.
        optimizer: The optimizer to load state into (optional).
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The epoch number from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]


@app.command()
def train(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    task_description: str = typer.Option(
        "insert the peg into the hole", "--task", "-t", help="Task description for CLIP"
    ),
    epochs: int = typer.Option(50, "--epochs", "-n", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    clip_model: str = typer.Option("ViT-B/32", "--clip-model", "-m", help="CLIP model variant"),
    checkpoint_dir: str = typer.Option("checkpoints", "--checkpoint-dir", "-c", help="Checkpoint directory"),
    save_every: int = typer.Option(10, "--save-every", help="Save checkpoint every N epochs"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    num_workers: int = typer.Option(0, "--num-workers", "-w", help="Dataloader workers"),
    max_demos: Optional[int] = typer.Option(None, "--max-demos", help="Max demos to load"),
) -> None:
    """Train the CLIP action model using behavioral cloning."""
    print(f"Training CLIP action model for {env_id}")
    print(f"  Task: {task_description}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloader(
        env_id=env_id,
        batch_size=batch_size,
        num_workers=num_workers,
        task_description=task_description,
        max_demos=max_demos,
    )

    sample_batch = next(iter(train_loader))
    action_dim = sample_batch["action"].shape[-1]
    print(f"Action dimension: {action_dim}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\nCreating model...")
    model = create_clip_action_model(
        action_dim=action_dim,
        clip_model=clip_model,
        freeze_clip=True,
    )
    print(f"Model created on device: {model.device}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_path = Path(checkpoint_dir) / env_id
    start_epoch = 0

    if resume:
        start_epoch = load_checkpoint(model, optimizer, Path(resume))

    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_losses = train_epoch(model, train_loader, optimizer, model.device)
        val_losses = validate(model, val_loader, model.device)

        scheduler.step()

        print(f"  Train Loss: {train_losses['loss']:.4f} (MSE: {train_losses['mse_loss']:.4f})")
        print(f"  Val Loss:   {val_losses['loss']:.4f} (MSE: {val_losses['mse_loss']:.4f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            save_checkpoint(model, optimizer, epoch, val_losses["loss"], checkpoint_path, "best.pt")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_losses["loss"], checkpoint_path, f"epoch_{epoch + 1}.pt")

    save_checkpoint(model, optimizer, epochs - 1, val_losses["loss"], checkpoint_path, "final.pt")
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_path}")


if __name__ == "__main__":
    app()
