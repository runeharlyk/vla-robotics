"""Convert checkpoints between policy.pt and LeRobot formats.

Usage:
    # policy.pt -> LeRobot (config.json + model.safetensors + normalizers)
    uv run python scripts/convert_checkpoint.py to-lerobot --checkpoint-dir checkpoints/best

    # LeRobot -> policy.pt
    uv run python scripts/convert_checkpoint.py to-policy-pt --checkpoint-dir checkpoints/best

    # Both directions (ensure both formats exist)
    uv run python scripts/convert_checkpoint.py ensure-both --checkpoint-dir checkpoints/best
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer
from safetensors.torch import load_file as load_safetensors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Convert between policy.pt and LeRobot checkpoint formats.")

LEROBOT_FILES = [
    "config.json",
    "model.safetensors",
]
NORMALIZER_FILES = [
    "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
]


def _has_policy_pt(path: Path) -> bool:
    return (path / "policy.pt").exists()


def _has_lerobot(path: Path) -> bool:
    return all((path / f).exists() for f in LEROBOT_FILES)


def _policy_pt_to_lerobot(path: Path, base_checkpoint: str) -> None:
    from vla.models.smolvla import SmolVLAPolicy

    data = torch.load(path / "policy.pt", map_location="cpu", weights_only=False)
    action_dim = data.get("action_dim", 7)
    state_dim = data.get("state_dim", 0)
    ckpt = data.get("checkpoint", base_checkpoint)

    policy = SmolVLAPolicy(checkpoint=ckpt, action_dim=action_dim, state_dim=state_dim, device="cpu")
    policy.load_checkpoint(path)
    policy._save_lerobot_format(path)
    logger.info("Converted policy.pt -> LeRobot format in %s", path)


def _lerobot_to_policy_pt(path: Path, base_checkpoint: str) -> None:
    from vla.models.smolvla import SmolVLAPolicy

    with open(path / "config.json") as f:
        config = json.load(f)

    action_shape = config.get("output_features", {}).get("action", {}).get("shape", [7])
    action_dim = action_shape[0]
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape", [0])
    state_dim = state_shape[0] if state_shape else 0

    policy = SmolVLAPolicy(checkpoint=base_checkpoint, action_dim=action_dim, state_dim=state_dim, device="cpu")

    weights = load_safetensors(str(path / "model.safetensors"), device="cpu")
    prefix = "model."
    state_dict = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in weights.items()}
    policy.model.load_state_dict(state_dict)

    for fname in NORMALIZER_FILES:
        fpath = path / fname
        if fpath.exists():
            stats = load_safetensors(str(fpath), device="cpu")
            if "action.mean" in stats:
                am = stats["action.mean"].float()[:action_dim]
                astd = stats["action.std"].float()[:action_dim]
                if am.shape != policy.action_mean.shape:
                    policy.register_buffer("action_mean", torch.zeros_like(am), persistent=True)
                    policy.register_buffer("action_std", torch.ones_like(astd), persistent=True)
                policy.action_mean.copy_(am)
                policy.action_std.copy_(astd)
            if "observation.state.mean" in stats:
                sm = stats["observation.state.mean"].float()[: max(state_dim, 1)]
                sstd = stats["observation.state.std"].float()[: max(state_dim, 1)]
                if sm.shape != policy.state_mean.shape:
                    policy.register_buffer("state_mean", torch.zeros_like(sm), persistent=True)
                    policy.register_buffer("state_std", torch.ones_like(sstd), persistent=True)
                policy.state_mean.copy_(sm)
                policy.state_std.copy_(sstd)
            break

    torch.save(
        {
            "model_state_dict": policy.model.state_dict(),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "checkpoint": base_checkpoint,
            "ckpt_config": config,
            "action_mean": policy.action_mean.detach().cpu(),
            "action_std": policy.action_std.detach().cpu(),
            "state_mean": policy.state_mean.detach().cpu(),
            "state_std": policy.state_std.detach().cpu(),
            "env_metadata": {},
        },
        path / "policy.pt",
    )
    logger.info("Converted LeRobot -> policy.pt format in %s", path)


@app.command()
def to_lerobot(
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", "-d", path_type=Path),
    base_checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--base-checkpoint", "-c"),
) -> None:
    """Convert a policy.pt checkpoint to LeRobot format."""
    if not _has_policy_pt(checkpoint_dir):
        typer.echo(f"No policy.pt found in {checkpoint_dir}", err=True)
        raise typer.Exit(1)
    _policy_pt_to_lerobot(checkpoint_dir, base_checkpoint)


@app.command()
def to_policy_pt(
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", "-d", path_type=Path),
    base_checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--base-checkpoint", "-c"),
) -> None:
    """Convert a LeRobot checkpoint to policy.pt format."""
    if not _has_lerobot(checkpoint_dir):
        typer.echo(f"Missing LeRobot files in {checkpoint_dir}", err=True)
        raise typer.Exit(1)
    _lerobot_to_policy_pt(checkpoint_dir, base_checkpoint)


@app.command()
def ensure_both(
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", "-d", path_type=Path),
    base_checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--base-checkpoint", "-c"),
) -> None:
    """Ensure both formats exist, converting whichever is missing."""
    has_pt = _has_policy_pt(checkpoint_dir)
    has_lr = _has_lerobot(checkpoint_dir)

    if not has_pt and not has_lr:
        typer.echo(f"No checkpoint files found in {checkpoint_dir}", err=True)
        raise typer.Exit(1)

    if has_pt and not has_lr:
        _policy_pt_to_lerobot(checkpoint_dir, base_checkpoint)
    elif has_lr and not has_pt:
        _lerobot_to_policy_pt(checkpoint_dir, base_checkpoint)
    else:
        typer.echo(f"Both formats already exist in {checkpoint_dir}")


if __name__ == "__main__":
    app()
