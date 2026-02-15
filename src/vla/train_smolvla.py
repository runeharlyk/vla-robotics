"""
Fine-tune SmolVLA on LIBERO demonstrations.

Uses LeRobot datasets for data loading and our own training loop for full control.

Usage:
    uv run python src/vla/train_smolvla.py --suite spatial --steps 20000
    uv run python src/vla/train_smolvla.py --suite all --steps 50000 --amp
"""

import math
import time
from pathlib import Path
from typing import Optional

import torch
import typer
import wandb
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from vla.data import (
    load_libero_all,
    load_libero_suite,
    make_dataloader,
    split_dataset,
)

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LIBERO_ACTION_DIM = 7
LIBERO_IMAGE_SIZE = 256


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay schedule with linear warmup."""

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


def _build_smolvla_batch(batch: dict, chunk_size: int, action_dim: int) -> dict:
    """Convert our training batch format to SmolVLA's expected input format."""
    imgs = batch["images"]
    B = imgs.shape[0]

    if imgs.shape[-3] == 4:
        imgs = imgs[..., :3, :, :]

    last_frame = imgs[:, -1] if imgs.ndim == 5 else imgs

    batch_mapped = {
        "observation.images.image": last_frame,
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


def _run_validation(policy, val_loader, preprocess, chunk_size, action_dim, use_amp=False):
    """Run validation and return average loss."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_mapped = _build_smolvla_batch(batch, chunk_size, action_dim)
            batch_preprocessed = preprocess(batch_mapped)

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    output = policy.forward(batch_preprocessed)
                if isinstance(output, dict):
                    loss = output["loss"]
                elif isinstance(output, tuple):
                    loss = output[0]
                else:
                    loss = output
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss)
            except Exception:
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

    policy.train()
    return total_loss / num_batches if num_batches > 0 else float("inf")


@app.command()
def finetune(
    suite: str = typer.Option("all", "--suite", "-s", help="LIBERO suite(s) or 'all'"),
    steps: int = typer.Option(20000, "--steps"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(1e-4, "--lr"),
    decay_lr: float = typer.Option(2.5e-6, "--decay-lr"),
    warmup_steps: int = typer.Option(1000, "--warmup-steps"),
    decay_steps: int = typer.Option(30000, "--decay-steps"),
    device: str = typer.Option("cuda", "--device", "-d"),
    save_path: Optional[str] = typer.Option(None, "--save"),
    model_id: str = typer.Option("lerobot/smolvla_base", "--model-id", "-m"),
    chunk_size: int = typer.Option(50, "--chunk-size"),
    image_size: int = typer.Option(LIBERO_IMAGE_SIZE, "--image-size"),
    gradient_clip: float = typer.Option(10.0, "--grad-clip"),
    weight_decay: float = typer.Option(1e-10, "--weight-decay"),
    wandb_project: str = typer.Option("vla-smolvla-libero", "--wandb-project"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name"),
    val_split: float = typer.Option(0.1, "--val-split"),
    val_every: int = typer.Option(500, "--val-every"),
    patience: int = typer.Option(0, "--patience"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
    num_workers: int = typer.Option(4, "--num-workers"),
    grad_accum_steps: int = typer.Option(1, "--grad-accum"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Fine-tune SmolVLA on LIBERO demonstrations."""
    from vla.data import LIBERO_SUITES

    if suite.lower() == "all":
        suite_names = list(LIBERO_SUITES.keys())
    else:
        suite_names = [s.strip().lower() for s in suite.split(",")]

    print(f"Loading LIBERO data for suites: {suite_names}")

    if len(suite_names) == 1:
        full_dataset = load_libero_suite(suite_names[0])
    else:
        full_dataset = load_libero_all(suite_names)

    train_dataset, val_dataset = split_dataset(full_dataset, val_ratio=val_split)
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val samples")

    train_loader = make_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = make_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=max(1, num_workers // 2)
    )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    action_dim = LIBERO_ACTION_DIM

    print(f"\nLoading SmolVLA from {model_id}...")
    policy = SmolVLAPolicy.from_pretrained(model_id)

    policy.config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    policy.config.empty_cameras = 0
    policy.config.chunk_size = chunk_size
    policy.config.n_action_steps = chunk_size

    policy = policy.to(device_obj)
    policy.train()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scheduler = CosineDecayWithWarmup(
        optimizer, peak_lr=lr, decay_lr=decay_lr, warmup_steps=warmup_steps, decay_steps=decay_steps
    )

    effective_batch_size = batch_size * grad_accum_steps
    suite_label = suite_names[0] if len(suite_names) == 1 else f"{len(suite_names)}_suites"

    if save_path is None:
        save_path = str(MODELS_DIR / f"smolvla_libero_{suite_label}.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFine-tuning SmolVLA on LIBERO-{suite_label}")
    print(f"  Steps: {steps}, Batch: {batch_size} (eff: {effective_batch_size})")
    print(f"  LR: {lr} -> {decay_lr}, Warmup: {warmup_steps}, Decay: {decay_steps}")
    print(f"  Chunk size: {chunk_size}, Image: {image_size}, AMP: {amp}")
    print(f"  Save: {save_path}")

    wandb.init(
        project=wandb_project,
        name=wandb_name or f"smolvla-libero-{suite_label}",
        config={
            "suites": suite_names,
            "steps": steps,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "lr": lr,
            "decay_lr": decay_lr,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "chunk_size": chunk_size,
            "image_size": image_size,
            "model_id": model_id,
            "action_dim": action_dim,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "val_split": val_split,
            "amp": amp,
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
        batch_mapped = _build_smolvla_batch(batch, chunk_size, action_dim)
        batch_preprocessed = preprocess(batch_mapped)

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output = policy.forward(batch_preprocessed)
            if isinstance(output, dict):
                loss = output["loss"]
            elif isinstance(output, tuple):
                loss = output[0]
            else:
                loss = output
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)
        except Exception as e:
            print(f"\nForward error: {e}")
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        (loss / grad_accum_steps).backward()
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
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/elapsed_hours": (time.time() - train_start) / 3600,
                    },
                    step=optim_step,
                )
                running_loss = 0.0
                running_count = 0

            if val_every > 0 and optim_step % val_every == 0:
                val_loss = _run_validation(policy, val_loader, preprocess, chunk_size, action_dim, use_amp=amp)
                elapsed = (time.time() - train_start) / 3600
                print(f"\n[Step {optim_step}/{steps}] Val loss: {val_loss:.4f}, Elapsed: {elapsed:.1f}h")
                wandb.log({"val/loss": val_loss}, step=optim_step)

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
                                "suites": suite_names,
                                "image_size": image_size,
                                "chunk_size": chunk_size,
                            },
                        },
                        save_path,
                    )
                    print(f"  Saved best model (val_loss={val_loss:.4f})")
                    wandb.log({"val/best_loss": val_loss}, step=optim_step)
                else:
                    intervals_without_improvement += 1
                    print(f"  No improvement ({intervals_without_improvement}x, best={best_val_loss:.4f})")
                    if patience > 0 and intervals_without_improvement >= patience:
                        print(f"\nEarly stopping after {patience} intervals.")
                        break

    pbar.close()

    if best_val_loss == float("inf"):
        val_loss = _run_validation(policy, val_loader, preprocess, chunk_size, action_dim, use_amp=amp)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "step": optim_step,
                "val_loss": val_loss,
                "config": {
                    "model_id": model_id,
                    "action_dim": action_dim,
                    "suites": suite_names,
                    "image_size": image_size,
                    "chunk_size": chunk_size,
                },
            },
            save_path,
        )

    wandb.log({"final/best_val_loss": best_val_loss, "final/steps_completed": optim_step})
    wandb.finish()
    elapsed = (time.time() - train_start) / 3600
    print(f"\nDone in {elapsed:.1f}h. Best val loss: {best_val_loss:.4f}")
    print(f"Model: {save_path}")


if __name__ == "__main__":
    app()
