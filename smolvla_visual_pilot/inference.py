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
def _apply_noise_to_batch(
    images: torch.Tensor,
    noise_config: NoiseConfig,
) -> torch.Tensor:
    """Apply one noise config to a timestep batch of observations.

    Parameters
    ----------
    images : torch.Tensor
        ``(B,C,H,W)`` or ``(B,V,C,H,W)``.
    """
    if images.ndim == 4:
        return torch.stack([apply_noise(frame, noise_config) for frame in images], dim=0)
    if images.ndim == 5:
        return torch.stack(
            [torch.stack([apply_noise(view, noise_config) for view in frame], dim=0) for frame in images],
            dim=0,
        )
    raise ValueError(f"Expected image batch shape (B,C,H,W) or (B,V,C,H,W), got {tuple(images.shape)}")


def run_trajectory(
    images: torch.Tensor,
    states: torch.Tensor,
    instruction: str,
    policy_bundle: dict,
    noise_config: NoiseConfig | None = None,
    seed: int = 0,
    timestep_batch_size: int = 1,
) -> torch.Tensor:
    """Replay a trajectory through the policy.

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
    timestep_batch_size : int
        Number of timesteps to evaluate per model forward pass.
        Use ``1`` to preserve strictly step-wise behavior.

    Returns
    -------
    torch.Tensor
        ``(T, action_dim)`` predicted actions (denormalized).
    """
    seed_everything(seed)

    if timestep_batch_size < 1:
        raise ValueError("timestep_batch_size must be >= 1")

    policy = policy_bundle["policy"]

    if hasattr(policy, "reset"):
        policy.reset()

    actions: list[torch.Tensor] = []

    with torch.inference_mode():
        num_steps = images.shape[0]
        for start in range(0, num_steps, timestep_batch_size):
            end = min(start + timestep_batch_size, num_steps)

            img_batch = images[start:end]
            state_batch = states[start:end]

            if noise_config is not None:
                img_batch = _apply_noise_to_batch(img_batch, noise_config)

            action_batch = policy.predict_action_batch(img_batch, instruction, state_batch)
            actions.append(action_batch.cpu())

    return torch.cat(actions, dim=0)


def run_trajectories_for_noises(
    images: torch.Tensor,
    states: torch.Tensor,
    instruction: str,
    policy_bundle: dict,
    noise_configs: list[NoiseConfig],
    seed: int = 0,
    noise_batch_size: int = 0,
    timestep_batch_size: int = 1,
) -> torch.Tensor:
    """Replay one trajectory for multiple noise variants with batched inference.

    Returns
    -------
    torch.Tensor
        ``(N, T, action_dim)`` where ``N=len(noise_configs)``.
    """
    if not noise_configs:
        raise ValueError("noise_configs must not be empty")

    if timestep_batch_size < 1:
        raise ValueError("timestep_batch_size must be >= 1")

    n_noises = len(noise_configs)
    if noise_batch_size <= 0:
        noise_batch_size = n_noises
    noise_batch_size = min(noise_batch_size, n_noises)

    # Exact legacy ordering fallback.
    if noise_batch_size == 1:
        by_noise = [
            run_trajectory(
                images,
                states,
                instruction,
                policy_bundle,
                noise_config=nc,
                seed=seed,
                timestep_batch_size=timestep_batch_size,
            )
            for nc in noise_configs
        ]
        return torch.stack(by_noise, dim=0)

    seed_everything(seed)

    policy = policy_bundle["policy"]
    if hasattr(policy, "reset"):
        policy.reset()

    actions_per_noise: list[list[torch.Tensor]] = [[] for _ in range(n_noises)]
    num_steps = images.shape[0]

    with torch.inference_mode():
        for t_start in range(0, num_steps, timestep_batch_size):
            t_end = min(t_start + timestep_batch_size, num_steps)
            img_chunk = images[t_start:t_end]
            state_chunk = states[t_start:t_end]
            if state_chunk.ndim == 1:
                state_chunk = state_chunk.unsqueeze(-1)
            chunk_steps = state_chunk.shape[0]

            for n_start in range(0, n_noises, noise_batch_size):
                n_end = min(n_start + noise_batch_size, n_noises)
                cfg_batch = noise_configs[n_start:n_end]
                batch_noises = len(cfg_batch)

                noisy_chunks = [_apply_noise_to_batch(img_chunk, nc) for nc in cfg_batch]
                stacked = torch.stack(noisy_chunks, dim=0)

                flat_images = stacked.reshape(batch_noises * chunk_steps, *stacked.shape[2:])
                flat_states = (
                    state_chunk.unsqueeze(0)
                    .expand(batch_noises, -1, -1)
                    .reshape(batch_noises * chunk_steps, state_chunk.shape[-1])
                )

                pred_flat = policy.predict_action_batch(flat_images, instruction, flat_states).cpu()
                pred = pred_flat.reshape(batch_noises, chunk_steps, -1)

                for local_idx in range(batch_noises):
                    actions_per_noise[n_start + local_idx].append(pred[local_idx])

    return torch.stack([torch.cat(parts, dim=0) for parts in actions_per_noise], dim=0)
