"""Build a distillation-ready dataset from collected RL rollouts.

Thin wrapper around :class:`FewDemoDataset` and :class:`AugmentedSFTDataset`
that bundles the loader-time perturbations we want for self-distillation
of an RL-tuned SmolVLA: load collected ``.pt`` rollouts produced by
``scripts/collect_success_dataset.py``, then apply the standard color /
contrast / noise / random-crop augmentation block. This is the
single import point ``scripts/distill_from_rollouts.py`` relies on, so
that future tweaks to the distillation perturbation recipe live in
exactly one place.

The actual transforms are implemented by
:class:`vla.data.dataset.AugmentedSFTDataset`; this module only chooses
defaults appropriate for RL-rollout distillation (mild perturbations,
not the aggressive paraphrase-and-warp settings sometimes used for
robustness fine-tuning on demonstrations).

Why these defaults:
    * ``brightness=0.10`` and ``contrast=0.10`` — keep the visual
      identity of the LIBERO scenes intact while exposing the student
      to small camera / lighting drift that the RL teacher already
      sometimes saw in its rollouts.
    * ``noise_std=0.02`` — adds ~2% pixel-domain Gaussian noise (in the
      [0, 1] range). Roughly matches LIBERO's own renderer jitter on
      shadows. Higher values start to wash out colors that are part of
      the task description (e.g. "red cube").
    * ``random_crop_scale=0.92`` — crop to ~92% of the original frame
      and resize back. This shifts the center-of-attention by a few
      pixels without losing the manipulation workspace.
    * ``repeats=2`` — each rollout episode is seen twice per epoch with
      independent augmentation. Cheap variance reduction on the
      ~K=300/task budget we use in practice.

Anything not covered by these label-preserving perturbations (camera
pose, init-state diversity, language paraphrases) must be folded into
the *collection* phase instead, since perturbing those at loader time
would invalidate the executed action targets.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vla.data.dataset import (
    AugmentedSFTDataset,
    ConcatFewDemoDataset,
    FewDemoDataset,
)

DEFAULT_BRIGHTNESS = 0.10
DEFAULT_CONTRAST = 0.10
DEFAULT_NOISE_STD = 0.02
DEFAULT_CROP_SCALE = 0.92
DEFAULT_REPEATS = 2


def build_rollout_distill_dataset(
    rollout_paths: Iterable[str | Path],
    *,
    num_demos: int | None = None,
    seed: int = 42,
    action_chunk_size: int = 50,
    repeats: int = DEFAULT_REPEATS,
    brightness: float = DEFAULT_BRIGHTNESS,
    contrast: float = DEFAULT_CONTRAST,
    noise_std: float = DEFAULT_NOISE_STD,
    random_crop_scale: float = DEFAULT_CROP_SCALE,
    instruction_variants: dict[str, list[str]] | None = None,
    enable_augmentation: bool = True,
) -> AugmentedSFTDataset | FewDemoDataset | ConcatFewDemoDataset:
    """Construct a distillation dataset from one or more collected rollout files.

    Args:
        rollout_paths: Paths to ``.pt`` files written by
            ``scripts/collect_success_dataset.py``.
        num_demos: Optional per-file episode cap (``None`` = use all).
        seed: Subsampling seed (matches train_sft.py / FewDemoDataset).
        action_chunk_size: Action target chunk depth (SmolVLA default 50).
        repeats: Virtual dataset repeats inside ``AugmentedSFTDataset``.
        brightness, contrast, noise_std, random_crop_scale:
            Perturbation strengths. Defaults are tuned for LIBERO RL
            distillation and intentionally mild.
        instruction_variants: Optional task -> paraphrase mapping.
        enable_augmentation: When ``False``, return the bare
            ``FewDemoDataset`` / ``ConcatFewDemoDataset`` without any
            wrapping (useful for an A/B "no-aug" baseline run).

    Returns:
        A dataset object compatible with ``train_sft`` / ``SmolVLAPolicy``
        consumers (``norm_stats``, ``action_dim``, ``state_dim``,
        ``num_episodes``, ``metadata``, ``instruction``, ``control_mode``
        are all preserved through the wrapper).
    """
    paths = [Path(p) for p in rollout_paths]
    if not paths:
        raise ValueError("rollout_paths must contain at least one .pt file")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Rollout dataset not found: {p}")

    if len(paths) == 1:
        base: FewDemoDataset | ConcatFewDemoDataset = FewDemoDataset(
            paths[0],
            num_demos=num_demos,
            seed=seed,
            action_chunk_size=action_chunk_size,
        )
    else:
        base = ConcatFewDemoDataset(
            paths,
            num_demos=num_demos,
            seed=seed,
            action_chunk_size=action_chunk_size,
        )

    if not enable_augmentation:
        return base

    return AugmentedSFTDataset(
        base,
        repeats=repeats,
        brightness=brightness,
        contrast=contrast,
        noise_std=noise_std,
        random_crop_scale=random_crop_scale,
        instruction_variants=instruction_variants,
        seed=seed,
    )


__all__ = [
    "DEFAULT_BRIGHTNESS",
    "DEFAULT_CONTRAST",
    "DEFAULT_CROP_SCALE",
    "DEFAULT_NOISE_STD",
    "DEFAULT_REPEATS",
    "build_rollout_distill_dataset",
]
