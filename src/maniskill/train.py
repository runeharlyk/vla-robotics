"""
Train VLA models on ManiSkill demonstrations.

Supports training RT-1 and other models on preprocessed demonstration data.

Usage:
    uv run python src/vla/train.py rt1 --env PickCube-v1 --epochs 200 --seq-len 6 --pretrained
    uv run python src/vla/train.py rt1 --help
"""

from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import typer
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from maniskill.data import load_dataset

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


ACTION_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0], dtype=np.float32)


def compute_action_bounds(dataset, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Compute tight action bounds from dataset with a relative margin."""
    all_actions = dataset.get_all_actions()
    data_min = all_actions.min(dim=0).values.numpy()
    data_max = all_actions.max(dim=0).values.numpy()
    data_range = data_max - data_min
    data_range = np.maximum(data_range, 1e-6)
    action_low = (data_min - margin * data_range).astype(np.float32)
    action_high = (data_max + margin * data_range).astype(np.float32)
    return action_low, action_high


def normalize_actions(
    actions: torch.Tensor, low: np.ndarray = ACTION_LOW, high: np.ndarray = ACTION_HIGH
) -> torch.Tensor:
    low_t = torch.tensor(low, device=actions.device, dtype=actions.dtype)
    high_t = torch.tensor(high, device=actions.device, dtype=actions.dtype)
    return 2.0 * (actions - low_t) / (high_t - low_t) - 1.0


def discretize_actions(actions_norm: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    actions_clipped = torch.clamp(actions_norm, -1, 1)
    bins = ((actions_clipped + 1) / 2 * (num_bins - 1)).long()
    return bins


def bins_to_continuous(bins: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    return (bins.float() / (num_bins - 1)) * 2 - 1


class PretrainedRT1(nn.Module):
    """RT-1 with a frozen pretrained MaxViT backbone from timm."""

    def __init__(
        self,
        model_name: str = "maxvit_tiny_tf_224.in1k",
        num_actions: int = 8,
        action_bins: int = 256,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        num_learned_tokens: int = 8,
    ):
        super().__init__()

        from einops.layers.torch import Rearrange

        from robotic_transformer_pytorch.robotic_transformer_pytorch import (
            LayerNorm,
            TokenLearner,
            Transformer,
            posemb_sincos_1d,
        )

        self.backbone = timm.create_model(model_name, pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        embed_dim = self.backbone.num_features

        self.token_learner = TokenLearner(
            dim=embed_dim,
            ff_mult=2,
            num_output_tokens=num_learned_tokens,
            num_layers=2,
        )
        self.num_learned_tokens = num_learned_tokens

        self.transformer = Transformer(
            dim=embed_dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
        )
        self.transformer_depth = depth

        self.to_logits = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

        self._embed_dim = embed_dim
        self._posemb_sincos_1d = posemb_sincos_1d

    def forward(self, video, texts=None, text_embeds=None, cond_drop_prob=0.0):
        from einops import reduce, repeat

        b, c, f, h, w = video.shape
        device = video.device

        images = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)

        with torch.no_grad():
            tokens = self.backbone.forward_features(images)

        tokens = tokens.reshape(b, f, *tokens.shape[1:])
        learned_tokens = self.token_learner(tokens)
        learned_tokens = learned_tokens.reshape(b, f * self.num_learned_tokens, self._embed_dim)

        attn_mask = torch.ones((f, f), dtype=torch.bool, device=device).triu(1)
        attn_mask = repeat(
            attn_mask,
            "i j -> (i r1) (j r2)",
            r1=self.num_learned_tokens,
            r2=self.num_learned_tokens,
        )

        pos_emb = self._posemb_sincos_1d(f, self._embed_dim, dtype=learned_tokens.dtype, device=device)
        learned_tokens = learned_tokens + repeat(pos_emb, "n d -> (n r) d", r=self.num_learned_tokens)

        attended_tokens = self.transformer(learned_tokens, attn_mask=~attn_mask)
        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=f)
        return self.to_logits(pooled)


def create_rt1_model(
    action_dim: int = 8,
    device: str = "cuda",
    model_size: str = "small",
    pretrained: bool = False,
) -> nn.Module:
    """
    Create RT-1 model.

    Args:
        action_dim: Dimension of action space
        device: Device to place model on
        model_size: 'tiny', 'small', or 'base'
        pretrained: Use pretrained ImageNet MaxViT backbone (frozen)

    Returns:
        RT-1 model instance
    """
    if pretrained:
        model = PretrainedRT1(
            model_name="maxvit_tiny_tf_224.in1k",
            num_actions=action_dim,
            action_bins=256,
            depth=6,
            heads=8,
            dim_head=64,
        ).to(device)
        return model

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
    sequence_length: int = typer.Option(6, "--seq-len", help="Number of frames per sample"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use mixed precision training"),
    pretrained: bool = typer.Option(False, "--pretrained/--no-pretrained", help="Use pretrained ImageNet backbone"),
    wandb_project: str = typer.Option("vla-rt1", "--wandb-project", help="Weights & Biases project name"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name", help="Weights & Biases run name"),
) -> None:
    """Train RT-1 on preprocessed ManiSkill demonstrations."""
    image_size = 224 if pretrained else 256

    try:
        dataset = load_dataset(env_id, sequence_length=sequence_length, image_size=image_size, augment=True)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    action_dim = dataset.action_dim
    instruction = dataset.instruction

    action_low, action_high = compute_action_bounds(dataset)
    print("\nAction bounds (data-derived with margin):")
    print(f"  Low:  {action_low.tolist()}")
    print(f"  High: {action_high.tolist()}")
    bin_resolution = (action_high - action_low) / 255
    print(f"  Bin resolution: {bin_resolution.tolist()}")

    backbone_label = "pretrained" if pretrained else model_size
    print(f"\nCreating RT-1 model ({backbone_label})...")
    model = create_rt1_model(
        action_dim=action_dim,
        device=device,
        model_size=model_size,
        pretrained=pretrained,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_epochs = min(5, epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)

    if save_path is None:
        suffix = "_pretrained" if pretrained else ""
        save_path = str(MODELS_DIR / f"rt1_{env_id.lower().replace('-', '_')}{suffix}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining RT-1 on {env_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Image size: {image_size}")
    print(f"  Pretrained backbone: {pretrained}")
    print(f"  Action dim: {action_dim}")
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        print(f"  Device: {device}")
    print(f"  Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "env_id": env_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "sequence_length": sequence_length,
            "image_size": image_size,
            "model_size": model_size,
            "pretrained": pretrained,
            "action_dim": action_dim,
            "instruction": instruction,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "amp": use_amp,
            "warmup_epochs": warmup_epochs,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)

    best_loss = float("inf")
    model.train()
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)

            B, T, C, H, W = images.shape
            video = images.permute(0, 2, 1, 3, 4)

            target_bins = discretize_actions(normalize_actions(actions[:, 0, :], action_low, action_high))

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

            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

            wandb.log({"train/loss": step_loss, "train/lr": optimizer.param_groups[0]["lr"]}, step=global_step)

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        wandb.log(
            {
                "epoch/avg_loss": avg_loss,
                "epoch/lr": current_lr,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "config": {
                        "action_dim": action_dim,
                        "model_size": model_size,
                        "env_id": env_id,
                        "instruction": instruction,
                        "action_low": action_low.tolist(),
                        "action_high": action_high.tolist(),
                        "pretrained": pretrained,
                        "image_size": image_size,
                        "sequence_length": sequence_length,
                    },
                },
                save_path,
            )
            print(f"  Saved best model (loss={avg_loss:.4f})")
            wandb.log({"epoch/best_loss": avg_loss}, step=global_step)
            artifact = wandb.Artifact(
                f"rt1-{env_id.lower().replace('-', '_')}",
                type="model",
                metadata={"epoch": epoch + 1, "loss": avg_loss},
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

    wandb.log({"final/best_loss": best_loss, "final/epochs_completed": epochs})
    wandb.finish()
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
