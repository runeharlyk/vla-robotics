"""Merge an SFT base SmolVLA with an RL-finetuned checkpoint (WiSE-FT).

Implements the Wortsman et al. 2022 weight-space ensemble:

    theta_merged = (1 - alpha) * theta_sft + alpha * theta_rl

and saves the merged policy to a new directory in **both** the internal
``policy.pt`` format and the LeRobot-compatible safetensors format so it
plugs directly into ``scripts/evaluate.py --checkpoint-dir <merged>`` and
``lerobot-eval --policy.path=<merged>`` without further conversion.

Usage:
    uv run python scripts/merge_checkpoints.py \\
        --sft-checkpoint HuggingFaceVLA/smolvla_libero \\
        --rl-checkpoint-dir /work3/.../checkpoints/sparse_rl/.../best \\
        --alpha 0.5 \\
        --output-dir /work3/.../checkpoints/wiseft/spatial_alpha050

For an alpha sweep, run this once per alpha value and feed each output
directory to a separate eval job. ``--alpha 0.0`` short-circuits the
merge and just copies the SFT base; ``--alpha 1.0`` short-circuits to a
straight load of the RL checkpoint.

The diagnostics (alpha, n_merged_keys, n_copied_keys, max_abs_delta)
are written to ``<output_dir>/wise_ft_diagnostics.json`` for
reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)


def _infer_dims_from_rl(rl_checkpoint_dir: Path) -> tuple[int, int]:
    """Read action_dim and state_dim from the RL checkpoint's policy.pt."""
    import torch

    rl_path = rl_checkpoint_dir / "policy.pt"
    if not rl_path.exists():
        raise typer.BadParameter(f"RL checkpoint not found: {rl_path}")
    data = torch.load(rl_path, map_location="cpu", weights_only=False)
    action_dim = int(data.get("action_dim", 7))
    state_dim = int(data.get("state_dim", 8))
    return action_dim, state_dim


@app.command()
def main(
    sft_checkpoint: str = typer.Option(
        "HuggingFaceVLA/smolvla_libero",
        "--sft-checkpoint",
        help="HuggingFace model id or local directory holding the SFT base SmolVLA checkpoint.",
    ),
    rl_checkpoint_dir: Path = typer.Option(
        ...,
        "--rl-checkpoint-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Local directory containing the RL-finetuned policy.pt to interpolate towards.",
    ),
    alpha: float = typer.Option(
        ...,
        "--alpha",
        min=0.0,
        max=1.0,
        help="Interpolation weight. 0.0 = pure SFT, 1.0 = pure RL.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory to write the merged checkpoint to (policy.pt + LeRobot files).",
    ),
    action_dim: int = typer.Option(
        0,
        "--action-dim",
        help="Optional override for SmolVLA action_dim. 0 = infer from RL checkpoint's policy.pt.",
    ),
    state_dim: int = typer.Option(
        0,
        "--state-dim",
        help="Optional override for SmolVLA state_dim. 0 = infer from RL checkpoint's policy.pt.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device used during the merge ('cpu' is safest; merging itself needs no GPU).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Overwrite output_dir if it already contains a checkpoint.",
    ),
) -> None:
    """Linearly interpolate SFT and RL SmolVLA checkpoints and save both formats."""
    from vla.env_metadata import EnvMetadata
    from vla.models.smolvla import SmolVLAPolicy
    from vla.utils.wise_ft import wise_ft_merge_into_policy

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise typer.BadParameter(
            f"Output directory {output_dir} is non-empty. Pass --overwrite to replace its contents."
        )

    inferred_action_dim, inferred_state_dim = _infer_dims_from_rl(rl_checkpoint_dir)
    resolved_action_dim = action_dim or inferred_action_dim
    resolved_state_dim = state_dim or inferred_state_dim
    logger.info(
        "Merging SFT=%s with RL=%s at alpha=%.3f (action_dim=%d, state_dim=%d, device=%s)",
        sft_checkpoint,
        rl_checkpoint_dir,
        alpha,
        resolved_action_dim,
        resolved_state_dim,
        device,
    )

    policy = SmolVLAPolicy(
        checkpoint=sft_checkpoint,
        action_dim=resolved_action_dim,
        state_dim=resolved_state_dim,
        device=device,
    )

    diagnostics = wise_ft_merge_into_policy(policy, rl_checkpoint_dir, alpha)
    logger.info(
        "Merge diagnostics: alpha=%.3f merged_keys=%d copied_keys=%d max_abs_delta=%.4g",
        diagnostics["alpha"],
        diagnostics["n_merged_keys"],
        diagnostics["n_copied_keys"],
        diagnostics["max_abs_delta"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    env_metadata = EnvMetadata(
        simulator="libero",
        suite="spatial",
        env_id="libero_spatial",
    )
    policy.save_checkpoint(output_dir, env_metadata=env_metadata)

    diagnostics_path = output_dir / "wise_ft_diagnostics.json"
    payload = {
        "sft_checkpoint": sft_checkpoint,
        "rl_checkpoint_dir": str(rl_checkpoint_dir),
        "alpha": float(alpha),
        "action_dim": resolved_action_dim,
        "state_dim": resolved_state_dim,
        **diagnostics,
    }
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote merge diagnostics to %s", diagnostics_path)
    logger.info("Saved merged checkpoint to %s (policy.pt + LeRobot format)", output_dir)


if __name__ == "__main__":
    app()
