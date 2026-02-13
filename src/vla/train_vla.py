"""
Train SmoLVLA on ManiSkill demonstrations.

Usage:
    uv run python src/vla/train_vla.py finetune --env PickCube-v1 --epochs 10 --batch-size 8
"""
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from vla.data import load_dataset_with_split

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"


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


def run_validation(policy, val_loader, preprocess, chunk_size, action_dim, device_obj, use_amp=False):
    """Run one validation pass and return average loss."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_mapped = _build_batch_mapped(batch, chunk_size, action_dim)
            batch_preprocessed = preprocess(batch_mapped)

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    output = policy.forward(batch_preprocessed)
                if isinstance(output, dict):
                    loss = output["loss"]
                elif isinstance(output, (tuple, list)):
                    loss = output[0] if isinstance(output[0], torch.Tensor) and output[0].ndim == 0 else output[0].mean()
                else:
                    loss = output
            except Exception:
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

    policy.train()
    return total_loss / num_batches if num_batches > 0 else float("inf")


def _build_batch_mapped(batch, chunk_size, action_dim):
    """Build the SmolVLA-format batch dict from a raw dataloader batch."""
    imgs = batch["images"]
    B = imgs.shape[0]

    if imgs.shape[-3] == 4:
        imgs = imgs[..., :3, :, :]

    last_frame = imgs[:, -1] if imgs.ndim == 5 else imgs

    batch_mapped = {
        "observation.images.top": last_frame,
    }

    if "state" in batch:
        state = batch["state"]
        if state.shape[-1] > action_dim:
            state = state[..., :action_dim]
        batch_mapped["observation.state"] = state

    actions = batch["actions"]
    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    if actions.shape[1] < chunk_size:
        pad_count = chunk_size - actions.shape[1]
        last_action = actions[:, -1:, :].expand(-1, pad_count, -1)
        actions = torch.cat([actions, last_action], dim=1)
    actions = actions[:, :chunk_size, :]
    batch_mapped["action"] = actions

    if "instruction" in batch:
        inst = batch["instruction"]
        if isinstance(inst, (list, tuple)):
            batch_mapped["task"] = list(inst)
        else:
            batch_mapped["task"] = [inst] * B

    return batch_mapped


@app.command()
def finetune(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s", help="Path to save model"),
    model_id: str = typer.Option("lerobot/smolvla_base", "--model-id", "-m", help="Pretrained model ID"),
    sequence_length: int = typer.Option(4, "--seq-len", help="Action chunk size (number of future actions to predict)"),
    image_size: int = typer.Option(224, "--image-size", help="Image size"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    warmup_steps: int = typer.Option(100, "--warmup-steps", help="Number of warmup steps"),
    wandb_project: str = typer.Option("vla-smolvla", "--wandb-project", help="Weights & Biases project name"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name", help="Weights & Biases run name"),
    freeze_vision: bool = typer.Option(True, "--freeze-vision/--no-freeze-vision", help="Freeze vision encoder"),
    val_split: float = typer.Option(0.1, "--val-split", help="Fraction of episodes for validation"),
    patience: int = typer.Option(5, "--patience", help="Early stopping patience (0 to disable)"),
    amp: bool = typer.Option(False, "--amp/--no-amp", help="Use BF16 mixed precision (faster on Ampere+ GPUs)"),
    num_workers: int = typer.Option(4, "--num-workers", help="Dataloader worker count"),
    compile_model: bool = typer.Option(False, "--compile/--no-compile", help="Use torch.compile for faster training"),
    grad_accum_steps: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps (effective batch = batch_size * grad_accum)"),
    prefetch_factor: int = typer.Option(4, "--prefetch", help="DataLoader prefetch factor per worker"),
) -> None:
    """Finetune SmoLVLA on preprocessed ManiSkill demonstrations."""

    try:
        train_dataset, val_dataset = load_dataset_with_split(
            env_id,
            sequence_length=1,
            action_horizon=sequence_length,
            image_size=image_size,
            augment=True,
            val_ratio=val_split,
        )
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_workers = max(1, num_workers // 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
        prefetch_factor=prefetch_factor if val_workers > 0 else None,
    )

    action_dim = train_dataset.action_dim
    instruction = train_dataset.instruction

    action_low, action_high = compute_action_bounds(train_dataset)
    print("\nAction bounds (data-derived with margin):")
    print(f"  Low:  {action_low.tolist()}")
    print(f"  High: {action_high.tolist()}")

    # Load pretrained SmoLVLA policy
    print(f"\nLoading SmoLVLA model from {model_id}...")
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    policy = SmolVLAPolicy.from_pretrained(model_id)

    # Override config to match our single-camera ManiSkill data
    # This avoids processing 3 identical camera images (3x speedup)
    policy.config.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    policy.config.empty_cameras = 0
    policy.config.chunk_size = sequence_length
    policy.config.n_action_steps = sequence_length

    policy = policy.to(device_obj)
    policy.train()

    # Setup pre/post processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    if freeze_vision and hasattr(policy, 'vision_encoder'):
        print("Freezing vision encoder...")
        for param in policy.vision_encoder.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    chunk_size = policy.config.chunk_size

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    if compile_model:
        print("Compiling model with torch.compile...")
        policy = torch.compile(policy, mode="reduce-overhead")

    effective_batch_size = batch_size * grad_accum_steps
    total_optim_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_optim_steps,
        pct_start=min(warmup_steps / total_optim_steps, 0.3),
    )

    if save_path is None:
        save_path = str(MODELS_DIR / f"smolvla_{env_id.lower().replace('-', '_')}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFinetuning SmoLVLA on {env_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (effective: {effective_batch_size}, accum={grad_accum_steps})")
    print(f"  Learning rate: {lr}")
    print(f"  Action chunk size: {sequence_length}")
    print(f"  Image size: {image_size}")
    print(f"  Action dim: {action_dim}")
    print(f"  Freeze vision: {freeze_vision}")
    print(f"  Val split: {val_split} ({len(val_dataset)} samples)")
    print(f"  Early stopping patience: {patience}")
    print(f"  Mixed precision (AMP): {amp}")
    print(f"  torch.compile: {compile_model}")
    print(f"  Dataloader workers: {num_workers}, prefetch: {prefetch_factor}")
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        print(f"  Device: {device}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "env_id": env_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "sequence_length": sequence_length,
            "image_size": image_size,
            "model_id": model_id,
            "action_dim": action_dim,
            "instruction": instruction,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "freeze_vision": freeze_vision,
            "warmup_steps": warmup_steps,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "val_split": val_split,
            "patience": patience,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "amp": amp,
            "compile": compile_model,
            "num_workers": num_workers,
        },
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    optim_step = 0
    epoch_start = time.time()

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch_mapped = _build_batch_mapped(batch, chunk_size, action_dim)
            batch_preprocessed = preprocess(batch_mapped)

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    output = policy.forward(batch_preprocessed)

                if isinstance(output, dict):
                    loss = output["loss"]
                elif isinstance(output, (tuple, list)):
                    loss = output[0] if isinstance(output[0], torch.Tensor) and output[0].ndim == 0 else output[0].mean()
                else:
                    loss = output
            except Exception as e:
                print(f"\nError during forward pass: {e}")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({"loss": "nan/inf (skipped)"})
                continue

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

            wandb.log({
                "train/loss": step_loss,
                "train/lr": scheduler.get_last_lr()[0],
            }, step=global_step)

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        current_lr = scheduler.get_last_lr()[0]

        val_loss = run_validation(policy, val_loader, preprocess, chunk_size, action_dim, device_obj, use_amp=amp)

        print(f"Epoch {epoch + 1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.0f}s")

        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss": val_loss,
            "epoch/lr": current_lr,
            "epoch/time_seconds": epoch_time,
            "epoch": epoch + 1,
        }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "config": {
                    "model_id": model_id,
                    "action_dim": action_dim,
                    "env_id": env_id,
                    "instruction": instruction,
                    "action_low": action_low.tolist(),
                    "action_high": action_high.tolist(),
                    "image_size": image_size,
                    "sequence_length": sequence_length,
                    "control_mode": train_dataset.metadata.get("control_mode", "pd_joint_pos"),
                },
            }, save_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

            wandb.log({"epoch/best_val_loss": val_loss}, step=global_step)

            artifact = wandb.Artifact(
                f"smolvla-{env_id.lower().replace('-', '_')}",
                type="model",
                metadata={"epoch": epoch + 1, "val_loss": val_loss, "train_loss": avg_train_loss},
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s) (best val_loss={best_val_loss:.4f})")

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    wandb.log({"final/best_val_loss": best_val_loss, "final/epochs_completed": epoch + 1})
    wandb.finish()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    app()