"""Materialize label-preserving perturbations for an SFT ``.pt`` dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _load_instruction_variants(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return {"*": [str(v) for v in loaded]}
    if isinstance(loaded, dict):
        return {str(k): [str(v) for v in values] for k, values in loaded.items() if isinstance(values, list)}
    raise typer.BadParameter("--instruction-variants must be a JSON list or object")


def _choose_instruction(base: str, variants: dict[str, list[str]], gen: torch.Generator) -> str:
    choices = variants.get(base, []) or variants.get("*", [])
    if not choices:
        return base
    all_choices = [base, *choices]
    idx = int(torch.randint(0, len(all_choices), (1,), generator=gen).item())
    return all_choices[idx]


def _augment_images(
    images: torch.Tensor,
    *,
    brightness: float,
    contrast: float,
    noise_std: float,
    crop_scale: float,
    gen: torch.Generator,
) -> torch.Tensor:
    out = images.float()
    was_uint8_scale = out.max().item() > 2.0 if out.numel() else False
    if was_uint8_scale:
        out = out / 255.0

    if crop_scale < 1.0 and out.shape[-1] > 1 and out.shape[-2] > 1:
        h, w = out.shape[-2:]
        crop_h = max(int(round(h * crop_scale)), 1)
        crop_w = max(int(round(w * crop_scale)), 1)
        top = int(torch.randint(0, h - crop_h + 1, (1,), generator=gen).item())
        left = int(torch.randint(0, w - crop_w + 1, (1,), generator=gen).item())
        cropped = out[..., top : top + crop_h, left : left + crop_w]
        out = torch.nn.functional.interpolate(
            cropped.reshape(-1, 1, crop_h, crop_w),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape_as(out)

    if contrast > 0:
        factor = 1.0 + (torch.rand((), generator=gen).item() * 2.0 - 1.0) * contrast
        mean = out.mean(dim=(-2, -1), keepdim=True)
        out = (out - mean) * factor + mean
    if brightness > 0:
        delta = (torch.rand((), generator=gen).item() * 2.0 - 1.0) * brightness
        out = out + delta
    if noise_std > 0:
        out = out + torch.randn(out.shape, generator=gen, dtype=out.dtype) * noise_std

    out = out.clamp(0.0, 1.0)
    if was_uint8_scale:
        return (out * 255.0).round().to(torch.uint8)
    return out


@app.command()
def main(
    data: Path = typer.Option(..., "--data", exists=True, file_okay=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output"),
    repeats: int = typer.Option(4, "--repeats", min=1),
    brightness: float = typer.Option(0.1, "--brightness", min=0.0),
    contrast: float = typer.Option(0.1, "--contrast", min=0.0),
    noise_std: float = typer.Option(0.01, "--noise-std", min=0.0),
    crop_scale: float = typer.Option(1.0, "--crop-scale", min=0.01, max=1.0),
    instruction_variants: Path | None = typer.Option(
        None,
        "--instruction-variants",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    seed: int = typer.Option(42, "--seed"),
    keep_original: bool = typer.Option(True, "--keep-original/--drop-original"),
) -> None:
    """Write a new ``.pt`` dataset with materialized visual/text perturbations."""
    payload = torch.load(data, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    episodes: list[dict] = payload["episodes"]
    variants = _load_instruction_variants(instruction_variants)

    out_episodes: list[dict] = []
    for ep_idx, episode in enumerate(episodes):
        if keep_original:
            out_episodes.append(dict(episode))
        for repeat_idx in range(repeats):
            gen = torch.Generator().manual_seed(seed + ep_idx * 10_000 + repeat_idx)
            augmented = dict(episode)
            augmented["images"] = _augment_images(
                episode["images"],
                brightness=brightness,
                contrast=contrast,
                noise_std=noise_std,
                crop_scale=crop_scale,
                gen=gen,
            )
            base_instruction = str(episode.get("instruction", metadata.get("instruction", "complete the task")))
            augmented["instruction"] = _choose_instruction(base_instruction, variants, gen)
            augmented["augmentation_repeat"] = repeat_idx
            out_episodes.append(augmented)

    metadata.update(
        {
            "source_dataset": str(data),
            "num_episodes": len(out_episodes),
            "augmentation": {
                "repeats": repeats,
                "keep_original": keep_original,
                "brightness": brightness,
                "contrast": contrast,
                "noise_std": noise_std,
                "crop_scale": crop_scale,
                "instruction_variants": str(instruction_variants or ""),
                "seed": seed,
                "label_preserving_only": True,
            },
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "episodes": out_episodes}, output)
    logger.info("Saved %d episodes to %s", len(out_episodes), output)


if __name__ == "__main__":
    app()
