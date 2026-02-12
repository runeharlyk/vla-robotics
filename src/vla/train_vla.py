"""
Train SmoLVLA on ManiSkill demonstrations.

Usage:
    uv run python src/vla/train_vla.py finetune --env PickCube-v1 --epochs 10 --batch-size 8
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import typer
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from vla.data import load_dataset

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


@app.command()
def finetune(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s", help="Path to save model"),
    model_id: str = typer.Option("lerobot/smolvla_base", "--model-id", "-m", help="Pretrained model ID"),
    sequence_length: int = typer.Option(4, "--seq-len", help="Number of frames per sample"),
    image_size: int = typer.Option(224, "--image-size", help="Image size"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    warmup_steps: int = typer.Option(100, "--warmup-steps", help="Number of warmup steps"),
    wandb_project: str = typer.Option("vla-smolvla", "--wandb-project", help="Weights & Biases project name"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name", help="Weights & Biases run name"),
    freeze_vision: bool = typer.Option(True, "--freeze-vision/--no-freeze-vision", help="Freeze vision encoder"),
) -> None:
    """Finetune SmoLVLA on preprocessed ManiSkill demonstrations."""
    
    # Load dataset
    try:
        dataset = load_dataset(
            env_id, 
            sequence_length=sequence_length, 
            image_size=image_size, 
            augment=True
        )
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

    # Freeze vision encoder if requested
    if freeze_vision and hasattr(policy, 'vision_encoder'):
        print("Freezing vision encoder...")
        for param in policy.vision_encoder.parameters():
            param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # Setup save path
    if save_path is None:
        save_path = str(MODELS_DIR / f"smolvla_{env_id.lower().replace('-', '_')}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFinetuning SmoLVLA on {env_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Image size: {image_size}")
    print(f"  Action dim: {action_dim}")
    print(f"  Freeze vision: {freeze_vision}")
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        print(f"  Device: {device}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    # Initialize wandb
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
            "model_id": model_id,
            "action_dim": action_dim,
            "instruction": instruction,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "freeze_vision": freeze_vision,
            "warmup_steps": warmup_steps,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
        },
    )
    wandb.watch(policy, log="gradients", log_freq=100)

    # Training loop
    best_loss = float("inf")
    global_step = 0
    chunk_size = policy.config.chunk_size

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            imgs = batch["images"]
            B = imgs.shape[0]

            # Ensure RGB (drop alpha if RGBA)
            if imgs.shape[-3] == 4:
                imgs = imgs[..., :3, :, :]

            # Use last frame from the sequence: (B, T, C, H, W) -> (B, C, H, W)
            last_frame = imgs[:, -1]

            # Single camera — no need to duplicate across 3 cameras
            batch_mapped = {
                "observation.images.top": last_frame,
            }

            # Map state - truncate to action_dim (model pads to max_state_dim=32 internally)
            if "state" in batch:
                state = batch["state"]
                if state.shape[-1] > action_dim:
                    state = state[..., :action_dim]
                batch_mapped["observation.state"] = state

            # Map actions - truncate/pad to chunk_size
            actions = batch["actions"]
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)
            if actions.shape[1] < chunk_size:
                actions = actions.repeat(1, chunk_size, 1)[:, :chunk_size, :]
            batch_mapped["action"] = actions

            # Task instruction (preprocessor tokenizes this)
            # Must be a list of B strings for proper batched tokenization
            if "instruction" in batch:
                inst = batch["instruction"]
                if isinstance(inst, (list, tuple)):
                    batch_mapped["task"] = list(inst)
                else:
                    batch_mapped["task"] = [inst] * B

            # Debug: print shapes on first batch
            if global_step == 0:
                print("\n[DEBUG] batch_mapped keys and shapes:")
                for k, v in batch_mapped.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: {v.shape} dtype={v.dtype}")
                    else:
                        print(f"  {k}: {type(v).__name__} = {v!r}")

            # Preprocess (tokenize text, normalize, move to device)
            batch_preprocessed = preprocess(batch_mapped)

            if global_step == 0:
                print("\n[DEBUG] batch_preprocessed keys and shapes:")
                for k, v in batch_preprocessed.items():
                    if hasattr(v, "shape"):
                        print(f"  {k}: {v.shape} dtype={v.dtype}")
                    else:
                        print(f"  {k}: {type(v).__name__} = {v!r}")

            optimizer.zero_grad()

            try:
                output = policy.forward(batch_preprocessed)

                # Debug: inspect output type on first batch
                if global_step == 0:
                    print(f"\n[DEBUG] output type: {type(output)}")
                    if isinstance(output, dict):
                        print(f"  keys: {list(output.keys())}")
                    elif isinstance(output, (tuple, list)):
                        print(f"  length: {len(output)}")
                        for i, v in enumerate(output):
                            if hasattr(v, "shape"):
                                print(f"  [{i}]: {v.shape} dtype={v.dtype}")
                            else:
                                print(f"  [{i}]: {type(v).__name__} = {v}")

                # Extract loss from output (may be dict or tuple)
                if isinstance(output, dict):
                    loss = output["loss"]
                elif isinstance(output, (tuple, list)):
                    loss = output[0] if isinstance(output[0], torch.Tensor) and output[0].ndim == 0 else output[0].mean()
                else:
                    loss = output
            except Exception as e:
                if global_step == 0:
                    import traceback
                    traceback.print_exc()
                print(f"\nError during forward pass: {e}")
                print("Skipping batch...")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({"loss": "nan/inf (skipped)"})
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

            wandb.log({
                "train/loss": step_loss,
                "train/lr": scheduler.get_last_lr()[0],
            }, step=global_step)
        
        # Epoch summary
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        wandb.log({
            "epoch/avg_loss": avg_loss,
            "epoch/lr": current_lr,
            "epoch": epoch + 1,
        }, step=global_step)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "config": {
                    "model_id": model_id,
                    "action_dim": action_dim,
                    "env_id": env_id,
                    "instruction": instruction,
                    "action_low": action_low.tolist(),
                    "action_high": action_high.tolist(),
                    "image_size": image_size,
                    "sequence_length": sequence_length,
                },
            }, save_path)
            print(f"  Saved best model (loss={avg_loss:.4f})")
            
            wandb.log({"epoch/best_loss": avg_loss}, step=global_step)
            
            artifact = wandb.Artifact(
                f"smolvla-{env_id.lower().replace('-', '_')}",
                type="model",
                metadata={"epoch": epoch + 1, "loss": avg_loss},
            )
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

    wandb.log({"final/best_loss": best_loss, "final/epochs_completed": epochs})
    wandb.finish()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    app()