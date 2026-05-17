"""Weight-space ensemble of fine-tuned models (WiSE-FT) for SmolVLA.

Implements the linear-interpolation merge from
Wortsman et al. 2022 (`Robust fine-tuning of zero-shot models`,
https://arxiv.org/abs/2109.01903):

    theta_merged = (1 - alpha) * theta_pretrained + alpha * theta_finetuned

Used as a free post-training lever: take an SFT base SmolVLA checkpoint
and an RL-finetuned SmolVLA checkpoint trained from that base, and
evaluate the family of merged policies for various ``alpha``. Often
recovers some of the SFT robustness on tasks the RL run regressed on,
while keeping most of the RL gains on tasks where it improved.

The merge operates on ``policy.model`` only. The normalization buffers
(``action_mean/std``, ``state_mean/std``) are not learnable and are not
updated by RL training in this codebase, so they are kept at the SFT
values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def wise_ft_merge_into_policy(
    policy,
    rl_checkpoint_dir: str | Path,
    alpha: float,
) -> dict[str, float]:
    """Apply a WiSE-FT merge in-place to ``policy.model``.

    Args:
        policy: A freshly constructed :class:`SmolVLAPolicy` whose
            ``model`` currently holds the SFT base weights (i.e. nothing
            else has called ``load_checkpoint`` on it yet).
        rl_checkpoint_dir: Directory containing the RL ``policy.pt``.
        alpha: Interpolation weight in ``[0, 1]``.
            ``alpha=0.0`` keeps pure SFT (no-op); ``alpha=1.0`` is
            equivalent to ``policy.load_checkpoint(rl_checkpoint_dir)``.

    Returns:
        A small diagnostics dict suitable for logging:
        ``{"alpha": ..., "n_merged_keys": ..., "n_copied_keys": ...,
        "max_abs_delta": ...}``.

    Raises:
        ValueError: if ``alpha`` is outside ``[0, 1]`` or if the SFT
            and RL state dicts have mismatched keys/shapes.
        FileNotFoundError: if ``rl_checkpoint_dir/policy.pt`` is missing.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"wise_ft_alpha must be in [0, 1], got {alpha!r}")

    rl_dir = Path(rl_checkpoint_dir)
    rl_path = rl_dir / "policy.pt"
    if not rl_path.exists():
        raise FileNotFoundError(f"RL checkpoint not found at {rl_path}")

    sft_sd: dict[str, torch.Tensor] = {
        k: v.detach().to("cpu", copy=True) for k, v in policy.model.state_dict().items()
    }

    rl_data = torch.load(rl_path, map_location="cpu", weights_only=False)
    rl_sd: dict[str, torch.Tensor] = rl_data["model_state_dict"]

    sft_keys = set(sft_sd.keys())
    rl_keys = set(rl_sd.keys())
    missing = sft_keys - rl_keys
    extra = rl_keys - sft_keys
    if missing or extra:
        raise ValueError(
            "WiSE-FT merge cannot proceed: state dicts have different keys. "
            f"missing-from-RL={sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            f"extra-in-RL={sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )

    n_merged = 0
    n_copied = 0
    max_abs_delta = 0.0
    merged: dict[str, torch.Tensor] = {}
    for k, sft_v in sft_sd.items():
        rl_v = rl_sd[k]
        if sft_v.shape != rl_v.shape:
            raise ValueError(f"WiSE-FT merge shape mismatch for {k}: sft={sft_v.shape} rl={rl_v.shape}")

        if sft_v.dtype.is_floating_point:
            sft_f = sft_v.to(dtype=torch.float32)
            rl_f = rl_v.to(dtype=torch.float32)
            merged_f = (1.0 - alpha) * sft_f + alpha * rl_f
            delta = (merged_f - sft_f).abs().max().item()
            if delta > max_abs_delta:
                max_abs_delta = delta
            merged[k] = merged_f.to(dtype=sft_v.dtype)
            n_merged += 1
        else:
            merged[k] = sft_v.clone()
            n_copied += 1

    target_dtype = next(policy.model.parameters()).dtype
    target_device = next(policy.model.parameters()).device
    merged_for_load = {
        k: (v.to(device=target_device, dtype=target_dtype) if v.dtype.is_floating_point else v.to(target_device))
        for k, v in merged.items()
    }
    policy.model.load_state_dict(merged_for_load, strict=True)

    logger.info(
        "WiSE-FT merge applied: alpha=%.3f, merged %d float tensors, copied %d non-float tensors, "
        "max |theta_merged - theta_sft| = %.4g",
        alpha,
        n_merged,
        n_copied,
        max_abs_delta,
    )
    return {
        "alpha": float(alpha),
        "n_merged_keys": n_merged,
        "n_copied_keys": n_copied,
        "max_abs_delta": float(max_abs_delta),
    }


__all__ = ["wise_ft_merge_into_policy"]
