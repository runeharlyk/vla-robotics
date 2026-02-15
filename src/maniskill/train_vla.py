"""
Train SmoLVLA on ManiSkill demonstrations.

Usage:
    uv run python src/vla/train_vla.py finetune --env PickCube-v1 --steps 20000 --batch-size 64
"""

import math
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
from maniskill.data import discover_available_envs, load_dataset_with_split, load_multi_dataset_with_split

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"


def compute_action_bounds(datasets, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(datasets, list):
        datasets = [datasets]
    all_actions = torch.cat([ds.get_all_actions() for ds in datasets])
    data_min = all_actions.min(dim=0).values.numpy()
    data_max = all_actions.max(dim=0).values.numpy()
    data_range = data_max - data_min
    data_range = np.maximum(data_range, 1e-6)
    action_low = (data_min - margin * data_range).astype(np.float32)
    action_high = (data_max + margin * data_range).astype(np.float32)
    return action_low, action_high


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, peak_lr, decay_lr, warmup_steps, decay_steps, last_epoch=-1):
        self.peak_lr = peak_lr
        self.decay_lr = decay_lr
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self._warmup_steps:
            lr = self.peak_lr * step / max(1, self._warmup_steps)
        elif step < self._decay_steps:
            progress = (step - self._warmup_steps) / max(1, self._decay_steps - self._warmup_steps)
            lr = self.decay_lr + (self.peak_lr - self.decay_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.decay_lr
        return [lr for _ in self.base_lrs]


def run_validation(policy, val_loader, preprocess, chunk_size, action_dim, device_obj, use_amp=False):
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
                    loss = (
                        output[0] if isinstance(output[0], torch.Tensor) and output[0].ndim == 0 else output[0].mean()
                    )
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
    env_id: str = typer.Option("all", "--env", "-e"),
    steps: int = typer.Option(20000, "--steps"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(1e-4, "--lr"),
    decay_lr: float = typer.Option(2.5e-6, "--decay-lr"),
    warmup_steps: int = typer.Option(1000, "--warmup-steps"),
    decay_steps: int = typer.Option(30000, "--decay-steps"),
    device: str = typer.Option("cuda", "--device", "-d"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s"),
    model_id: str = typer.Option("lerobot/smolvla_base", "--model-id", "-m"),
    sequence_length: int = typer.Option(50, "--seq-len"),
    image_size: int = typer.Option(256, "--image-size"),
    gradient_clip: float = typer.Option(10.0, "--grad-clip"),
    weight_decay: float = typer.Option(1e-10, "--weight-decay"),
    wandb_project: str = typer.Option("vla-smolvla", "--wandb-project"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name"),
    val_split: float = typer.Option(0.1, "--val-split"),
    val_every: int = typer.Option(500, "--val-every"),
    patience: int = typer.Option(0, "--patience"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
    num_workers: int = typer.Option(4, "--num-workers"),
    compile_model: bool = typer.Option(False, "--compile/--no-compile"),
    grad_accum_steps: int = typer.Option(1, "--grad-accum"),
    prefetch_factor: int = typer.Option(4, "--prefetch"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Finetune SmoLVLA on preprocessed ManiSkill demonstrations."""

    if env_id.lower() == "all":
        env_ids = discover_available_envs()
        if not env_ids:
            typer.echo("No preprocessed data found in data directory", err=True)
            raise typer.Exit(1)
        print(f"Discovered {len(env_ids)} environments: {env_ids}")
    else:
        env_ids = [e.strip() for e in env_id.split(",")]

    try:
        if len(env_ids) == 1:
            train_ds, val_ds = load_dataset_with_split(
                env_ids[0],
                sequence_length=1,
                action_horizon=sequence_length,
                image_size=image_size,
                augment=False,
                val_ratio=val_split,
            )
            train_parts = [train_ds]
            val_parts = [val_ds]
            train_dataset = train_ds
            val_dataset = val_ds
        else:
            train_dataset, val_dataset, train_parts, val_parts, env_ids = load_multi_dataset_with_split(
                env_ids,
                sequence_length=1,
                action_horizon=sequence_length,
                image_size=image_size,
                augment=False,
                val_ratio=val_split,
            )
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    action_dim = train_parts[0].action_dim
    for ds in train_parts[1:]:
        if ds.action_dim != action_dim:
            typer.echo(f"Action dim mismatch: {ds.data_path} has {ds.action_dim}, expected {action_dim}", err=True)
            raise typer.Exit(1)

    env_label = env_ids[0] if len(env_ids) == 1 else f"{len(env_ids)}_envs"
    instruction = train_parts[0].instruction if len(train_parts) == 1 else "multi-task"
    control_mode = train_parts[0].metadata.get("control_mode", "pd_joint_pos")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
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

    action_low, action_high = compute_action_bounds(train_parts)
    print("\nAction bounds (data-derived with margin):")
    print(f"  Low:  {action_low.tolist()}")
    print(f"  High: {action_high.tolist()}")

    print(f"\nLoading SmoLVLA model from {model_id}...")
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    policy = SmolVLAPolicy.from_pretrained(model_id)

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

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")

    chunk_size = policy.config.chunk_size

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    scheduler = CosineDecayWithWarmup(
        optimizer,
        peak_lr=lr,
        decay_lr=decay_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )

    if compile_model:
        print("Compiling model with torch.compile...")
        policy = torch.compile(policy, mode="reduce-overhead")

    effective_batch_size = batch_size * grad_accum_steps

    if save_path is None:
        save_path = str(MODELS_DIR / f"smolvla_{env_label.lower().replace('-', '_')}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFinetuning SmoLVLA on {env_label} ({', '.join(env_ids)})")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size} (effective: {effective_batch_size}, accum={grad_accum_steps})")
    print(f"  Learning rate: {lr} -> {decay_lr} (cosine decay)")
    print(f"  Warmup: {warmup_steps}, Decay period: {decay_steps}")
    print(f"  Action chunk size: {sequence_length}")
    print(f"  Image size: {image_size}")
    print(f"  Action dim: {action_dim}")
    print(f"  Val split: {val_split} ({len(val_dataset)} samples)")
    print(f"  Validate every: {val_every} steps")
    print(f"  Early stopping patience: {patience} (0=disabled)")
    print(f"  Mixed precision (AMP): {amp}")
    print(f"  torch.compile: {compile_model}")
    print(f"  Dataloader workers: {num_workers}, prefetch: {prefetch_factor}")
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
    else:
        print(f"  Device: {device}")
    if len(env_ids) == 1:
        print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "env_ids": env_ids,
            "num_envs": len(env_ids),
            "steps": steps,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr": lr,
            "decay_lr": decay_lr,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "sequence_length": sequence_length,
            "image_size": image_size,
            "model_id": model_id,
            "action_dim": action_dim,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "action_low": action_low.tolist(),
            "action_high": action_high.tolist(),
            "val_split": val_split,
            "val_every": val_every,
            "patience": patience,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "amp": amp,
            "compile": compile_model,
            "num_workers": num_workers,
        },
    )

    best_val_loss = float("inf")
    intervals_without_improvement = 0
    optim_step = 0
    fwd_step = 0
    running_loss = 0.0
    running_count = 0
    train_start = time.time()

    data_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training")

    while optim_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        policy.train()
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
            continue

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        fwd_step += 1

        running_loss += loss.item()
        running_count += 1

        if fwd_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optim_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if optim_step % log_every == 0 and running_count > 0:
                avg_loss = running_loss / running_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - train_start
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/elapsed_hours": elapsed / 3600,
                    },
                    step=optim_step,
                )
                running_loss = 0.0
                running_count = 0

            if val_every > 0 and optim_step % val_every == 0:
                val_loss = run_validation(
                    policy, val_loader, preprocess, chunk_size, action_dim, device_obj, use_amp=amp
                )
                elapsed = time.time() - train_start
                print(f"\n[Step {optim_step}/{steps}] Val loss: {val_loss:.4f}, Elapsed: {elapsed / 3600:.1f}h")

                wandb.log(
                    {
                        "val/loss": val_loss,
                    },
                    step=optim_step,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    intervals_without_improvement = 0
                    torch.save(
                        {
                            "model_state_dict": policy.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": optim_step,
                            "val_loss": val_loss,
                            "config": {
                                "model_id": model_id,
                                "action_dim": action_dim,
                                "env_ids": env_ids,
                                "env_id": env_ids[0] if len(env_ids) == 1 else "multi",
                                "instruction": instruction,
                                "action_low": action_low.tolist(),
                                "action_high": action_high.tolist(),
                                "image_size": image_size,
                                "sequence_length": sequence_length,
                                "control_mode": control_mode,
                            },
                        },
                        save_path,
                    )
                    print(f"  Saved best model (val_loss={val_loss:.4f})")

                    wandb.log({"val/best_loss": val_loss}, step=optim_step)

                    artifact = wandb.Artifact(
                        f"smolvla-{env_label.lower().replace('-', '_')}",
                        type="model",
                        metadata={"step": optim_step, "val_loss": val_loss},
                    )
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)
                else:
                    intervals_without_improvement += 1
                    print(
                        f"  No improvement for {intervals_without_improvement} interval(s) (best={best_val_loss:.4f})"
                    )

                    if patience > 0 and intervals_without_improvement >= patience:
                        print(f"\nEarly stopping after {patience} intervals without improvement.")
                        break

    pbar.close()

    if best_val_loss == float("inf"):
        val_loss = run_validation(policy, val_loader, preprocess, chunk_size, action_dim, device_obj, use_amp=amp)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": optim_step,
                "val_loss": val_loss,
                "config": {
                    "model_id": model_id,
                    "action_dim": action_dim,
                    "env_ids": env_ids,
                    "env_id": env_ids[0] if len(env_ids) == 1 else "multi",
                    "instruction": instruction,
                    "action_low": action_low.tolist(),
                    "action_high": action_high.tolist(),
                    "image_size": image_size,
                    "sequence_length": sequence_length,
                    "control_mode": control_mode,
                },
            },
            save_path,
        )
        print(f"  Saved final model (val_loss={val_loss:.4f})")

    wandb.log({"final/best_val_loss": best_val_loss, "final/steps_completed": optim_step})
    wandb.finish()
    elapsed = time.time() - train_start
    print(f"\nTraining complete in {elapsed / 3600:.1f}h! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    app()
