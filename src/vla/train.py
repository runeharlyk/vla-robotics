"""
Train VLA models on ManiSkill demonstrations.

Supports training RT-1 and other models on preprocessed demonstration data.

Usage:
    uv run python src/vla/train.py rt1 --env PickCube-v1 --epochs 100
    uv run python src/vla/train.py rt1 --help
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from vla.data import load_dataset

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def discretize_actions(actions: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    actions_clipped = torch.clamp(actions, -1, 1)
    bins = ((actions_clipped + 1) / 2 * (num_bins - 1)).long()
    return bins


def bins_to_continuous(bins: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    return (bins.float() / (num_bins - 1)) * 2 - 1


def create_rt1_model(
    action_dim: int = 8,
    device: str = "cuda",
    model_size: str = "small",
) -> nn.Module:
    """
    Create RT-1 model.
    
    Args:
        action_dim: Dimension of action space
        device: Device to place model on
        model_size: 'tiny', 'small', or 'base'
    
    Returns:
        RT-1 model instance
    """
    from robotic_transformer_pytorch import RT1, MaxViT

    configs = {
        "tiny": {
            "dim_conv_stem": 16,
            "dim": 32,
            "dim_head": 16,
            "depth": (1, 1, 1, 1),
            "rt1_depth": 2,
            "rt1_heads": 2,
            "rt1_dim_head": 16,
        },
        "small": {
            "dim_conv_stem": 32,
            "dim": 48,
            "dim_head": 16,
            "depth": (1, 1, 2, 1),
            "rt1_depth": 4,
            "rt1_heads": 4,
            "rt1_dim_head": 32,
        },
        "base": {
            "dim_conv_stem": 64,
            "dim": 96,
            "dim_head": 32,
            "depth": (2, 2, 5, 2),
            "rt1_depth": 6,
            "rt1_heads": 8,
            "rt1_dim_head": 64,
        },
    }

    cfg = configs.get(model_size, configs["small"])

    vit = MaxViT(
        num_classes=1000,
        dim_conv_stem=cfg["dim_conv_stem"],
        dim=cfg["dim"],
        dim_head=cfg["dim_head"],
        depth=cfg["depth"],
        window_size=8,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    model = RT1(
        vit=vit,
        num_actions=action_dim,
        action_bins=256,
        depth=cfg["rt1_depth"],
        heads=cfg["rt1_heads"],
        dim_head=cfg["rt1_dim_head"],
        cond_drop_prob=0.2,
    ).to(device)

    return model


@app.command()
def rt1(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(3e-5, "--lr", help="Learning rate"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s", help="Path to save model"),
    model_size: str = typer.Option("small", "--model-size", "-m", help="Model size: tiny/small/base"),
    sequence_length: int = typer.Option(1, "--seq-len", help="Number of frames per sample"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use mixed precision training"),
) -> None:
    """Train RT-1 on preprocessed ManiSkill demonstrations."""
    try:
        dataset = load_dataset(env_id, sequence_length=sequence_length)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    action_dim = dataset.action_dim
    instruction = dataset.instruction

    print(f"\nCreating RT-1 model ({model_size})...")
    model = create_rt1_model(
        action_dim=action_dim,
        device=device,
        model_size=model_size,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)

    if save_path is None:
        save_path = str(MODELS_DIR / f"rt1_{env_id.lower().replace('-', '_')}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining RT-1 on {env_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Action dim: {action_dim}")
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        print(f"  Device: {device}")
    print(f"  Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    best_loss = float("inf")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)

            B, T, C, H, W = images.shape
            video = images.permute(0, 2, 1, 3, 4)

            target_bins = discretize_actions(actions[:, 0, :])

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                texts = [instruction] * B
                logits = model(video, texts=texts)

                logits_flat = logits[:, 0, :, :].reshape(-1, 256)
                target_flat = target_bins.reshape(-1)

                loss = criterion(logits_flat, target_flat)

            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({"loss": "nan/inf (skipped)"})
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "config": {
                    "action_dim": action_dim,
                    "model_size": model_size,
                    "env_id": env_id,
                    "instruction": instruction,
                },
            }, save_path)
            print(f"  Saved best model (loss={avg_loss:.4f})")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")


@app.command()
def list_models() -> None:
    """List available model architectures."""
    print("Available models:")
    print("  rt1 - Robotics Transformer 1 (tiny/small/base)")
    print("\nUsage:")
    print("  uv run python src/vla/train.py rt1 --env PickCube-v1 --epochs 100")


if __name__ == "__main__":
    app()
