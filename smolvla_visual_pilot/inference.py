"""Step-wise model inference for clean and noisy observations.

Reuses :class:`~vla.models.smolvla.SmolVLAPolicy` and follows the same
closed-loop pattern as ``smolvla_language_pilot/language_class.py``:
one ``predict_action`` call per timestep.
"""

from __future__ import annotations

import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.utils.device import get_device
from vla.utils.seed import seed_everything

try:
    from .noise import NoiseConfig, apply_noise
except ImportError:
    from noise import NoiseConfig, apply_noise


# ---------------------------------------------------------------------------
# Policy bundle (mirrors language_class.load_policy_bundle)
# ---------------------------------------------------------------------------

def load_policy_bundle(
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    device: str = "cuda",
) -> dict:
    """Load SmolVLA policy and return a reusable bundle dict.

    Returns
    -------
    dict
        Keys: ``policy``, ``device``, ``model_dtype``.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda, but CUDA is not available. "
            "Use --device cpu as fallback."
        )

    device_obj = get_device(device)
    if device == "cuda" and device_obj.type != "cuda":
        raise RuntimeError(
            f"Requested CUDA but resolved device is '{device_obj}'. "
            "Use --device cpu or fix your CUDA environment."
        )

    policy = SmolVLAPolicy(checkpoint, action_dim=7, device=str(device_obj))
    policy.eval()

    return {
        "policy": policy,
        "device": device_obj,
        "model_dtype": policy.dtype,
    }


# ---------------------------------------------------------------------------
# Trajectory replay
# ---------------------------------------------------------------------------

def run_trajectory(
    images: torch.Tensor,
    states: torch.Tensor,
    instruction: str,
    policy_bundle: dict,
    noise_config: NoiseConfig | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Replay a trajectory through the policy, one timestep at a time.

    Parameters
    ----------
    images : torch.Tensor
        ``(T, C, H, W)`` or ``(T, V, C, H, W)`` observations in [0, 1].
    states : torch.Tensor
        ``(T, state_dim)`` proprioceptive states.
    instruction : str
        Language instruction for the task.
    policy_bundle : dict
        From :func:`load_policy_bundle`.
    noise_config : NoiseConfig, optional
        If provided, applies this noise to each frame before inference.
        If ``None``, runs on clean observations.
    seed : int
        Reproducibility seed.

    Returns
    -------
    torch.Tensor
        ``(T, action_dim)`` predicted actions (denormalized).
    """
    seed_everything(seed)

    policy = policy_bundle["policy"]

    if hasattr(policy, "reset"):
        policy.reset()

    actions: list[torch.Tensor] = []

    with torch.inference_mode():
        for img, state in zip(images, states):
            # -- optionally corrupt the observation --
            if noise_config is not None:
                if img.ndim == 3:
                    img = apply_noise(img, noise_config)
                elif img.ndim == 4:
                    img = torch.stack([apply_noise(view, noise_config) for view in img], dim=0)
                else:
                    raise ValueError(
                        f"Expected image shape (C,H,W) or (V,C,H,W), got {tuple(img.shape)}"
                    )

            action = policy.predict_action(img, instruction, state)
            actions.append(action.cpu())

    return torch.stack(actions)
