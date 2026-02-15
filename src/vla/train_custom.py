"""
Train the custom VLA model on LIBERO demonstrations.

Uses the same training loop pattern as SmolVLA fine-tuning,
with flow matching loss and cosine decay schedule.

Usage:
    uv run python src/vla/train_custom.py --suite spatial --steps 30000
    uv run python src/vla/train_custom.py --suite all --steps 80000 --amp
"""

import math
import time
from pathlib import Path
from typing import Optional

import torch
import typer
import wandb
from tqdm import tqdm

from vla.custom_model import CustomVLA
from vla.data import (
    load_libero_all,
    load_libero_suite,
    make_dataloader,
    split_dataset,
)

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


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


def _prepare_batch(batch: dict, chunk_size: int, action_dim: int) -> dict:
    """Prepare batch for CustomVLA forward pass."""
    images = batch["images"]
    actions = batch["actions"]

    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    if actions.shape[1] < chunk_size:
        pad = actions[:, -1:, :].expand(-1, chunk_size - actions.shape[1], -1)
        actions = torch.cat([actions, pad], dim=1)
    actions = actions[:, :chunk_size, :action_dim]

    instruction = batch.get("instruction", "")
    if isinstance(instruction, str):
        instruction = [instruction] * images.shape[0]

    return {
        "images": images,
        "actions": actions,
        "instruction": instruction,
    }


def _run_validation(model, val_loader, chunk_size, action_dim, device, use_amp=False):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = _prepare_batch(batch, chunk_size, action_dim)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    output = model(batch)
                loss = output["loss"]
            except Exception:
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            n += 1

    model.train()
    return total_loss / n if n > 0 else float("inf")


@app.command()
def train(
    suite: str = typer.Option("all", "--suite", "-s", help="LIBERO suite(s) or 'all'"),
    steps: int = typer.Option(30000, "--steps"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(3e-4, "--lr"),
    decay_lr: float = typer.Option(1e-6, "--decay-lr"),
    warmup_steps: int = typer.Option(1000, "--warmup-steps"),
    decay_steps: int = typer.Option(40000, "--decay-steps"),
    device: str = typer.Option("cuda", "--device", "-d"),
    save_path: Optional[str] = typer.Option(None, "--save"),
    vision_model: str = typer.Option("google/siglip-base-patch16-256", "--vision-model"),
    language_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--language-model"),
    d_model: int = typer.Option(512, "--d-model"),
    chunk_size: int = typer.Option(20, "--chunk-size"),
    n_fusion_layers: int = typer.Option(2, "--n-fusion-layers"),
    n_action_layers: int = typer.Option(4, "--n-action-layers"),
    n_heads: int = typer.Option(8, "--n-heads"),
    freeze_vision: bool = typer.Option(True, "--freeze-vision/--unfreeze-vision"),
    freeze_language: bool = typer.Option(True, "--freeze-language/--unfreeze-language"),
    flow_steps: int = typer.Option(10, "--flow-steps"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    wandb_project: str = typer.Option("vla-custom-libero", "--wandb-project"),
    wandb_name: Optional[str] = typer.Option(None, "--wandb-name"),
    val_split: float = typer.Option(0.1, "--val-split"),
    val_every: int = typer.Option(500, "--val-every"),
    patience: int = typer.Option(0, "--patience"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
    num_workers: int = typer.Option(4, "--num-workers"),
    grad_accum_steps: int = typer.Option(1, "--grad-accum"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Train custom VLA on LIBERO demonstrations."""
    from vla.data import LIBERO_SUITES

    if suite.lower() == "all":
        suite_names = list(LIBERO_SUITES.keys())
    else:
        suite_names = [s.strip().lower() for s in suite.split(",")]

    action_dim = 7
    suite_label = suite_names[0] if len(suite_names) == 1 else f"{len(suite_names)}_suites"

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

    print("\nBuilding Custom VLA...")
    model = CustomVLA(
        vision_model=vision_model,
        language_model=language_model,
        d_model=d_model,
        action_dim=action_dim,
        chunk_size=chunk_size,
        n_fusion_layers=n_fusion_layers,
        n_action_layers=n_action_layers,
        n_heads=n_heads,
        freeze_vision=freeze_vision,
        freeze_language=freeze_language,
        flow_steps=flow_steps,
    )
    model = model.to(device_obj)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    scheduler = CosineDecayWithWarmup(
        optimizer, peak_lr=lr, decay_lr=decay_lr, warmup_steps=warmup_steps, decay_steps=decay_steps
    )

    effective_batch_size = batch_size * grad_accum_steps

    if save_path is None:
        save_path = str(MODELS_DIR / f"custom_vla_libero_{suite_label}.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining Custom VLA on LIBERO-{suite_label}")
    print(f"  Steps: {steps}, Batch: {batch_size} (eff: {effective_batch_size})")
    print(f"  LR: {lr} -> {decay_lr}, Chunk: {chunk_size}, Flow steps: {flow_steps}")
    print(f"  Vision: {vision_model} (frozen={freeze_vision})")
    print(f"  Language: {language_model} (frozen={freeze_language})")
    print(f"  Save: {save_path}")

    wandb.init(
        project=wandb_project,
        name=wandb_name or f"custom-vla-libero-{suite_label}",
        config={
            "suites": suite_names,
            "steps": steps,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            "lr": lr,
            "decay_lr": decay_lr,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "vision_model": vision_model,
            "language_model": language_model,
            "d_model": d_model,
            "chunk_size": chunk_size,
            "n_fusion_layers": n_fusion_layers,
            "n_action_layers": n_action_layers,
            "n_heads": n_heads,
            "freeze_vision": freeze_vision,
            "freeze_language": freeze_language,
            "flow_steps": flow_steps,
            "action_dim": action_dim,
            "trainable_params": trainable_params,
            "total_params": total_params,
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

        model.train()
        batch = _prepare_batch(batch, chunk_size, action_dim)
        batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output = model(batch)
            loss = output["loss"]
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
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
                val_loss = _run_validation(model, val_loader, chunk_size, action_dim, device_obj, use_amp=amp)
                elapsed = (time.time() - train_start) / 3600
                print(f"\n[Step {optim_step}/{steps}] Val loss: {val_loss:.4f}, Elapsed: {elapsed:.1f}h")
                wandb.log({"val/loss": val_loss}, step=optim_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    intervals_without_improvement = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": optim_step,
                            "val_loss": val_loss,
                            "config": {
                                "model_kwargs": model.get_model_kwargs(),
                                "action_dim": action_dim,
                                "suites": suite_names,
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
        val_loss = _run_validation(model, val_loader, chunk_size, action_dim, device_obj, use_amp=amp)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "step": optim_step,
                "val_loss": val_loss,
                "config": {
                    "model_kwargs": model.get_model_kwargs(),
                    "action_dim": action_dim,
                    "suites": suite_names,
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
