"""Self-implemented visual noise corruptions matching Libero+ sensor noise.

Implements the five Libero+ sensor noise types at a given severity level.
These are standard image corruption functions based on the ``imagecorruptions``
library conventions.  We implement them directly using NumPy / SciPy / OpenCV
so there is **no dependency on the Libero+ package** for noise injection.

All functions accept and return ``(C, H, W)`` float32 tensors in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    from scipy.ndimage import zoom as scipy_zoom
except ImportError:
    scipy_zoom = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Noise config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseConfig:
    """Describes one noise perturbation to apply."""

    noise_type: str
    severity: int  # 1–5, matching imagecorruptions convention

    def __str__(self) -> str:
        return f"{self.noise_type}_s{self.severity}"


def get_noise_configs(
    noise_types: list[str],
    severity: int = 3,
) -> list[NoiseConfig]:
    """Build a list of :class:`NoiseConfig` — one per noise type at the given severity."""
    return [NoiseConfig(noise_type=nt, severity=severity) for nt in noise_types]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] → (H, W, C) uint8."""
    arr = img.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))       # H, W, C
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _from_hwc_uint8(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 → (C, H, W) float [0,1]."""
    arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def _clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 1)


# ---------------------------------------------------------------------------
# Individual corruption functions
# ---------------------------------------------------------------------------

# Severity parameters follow the imagecorruptions defaults used by Libero+.

def _motion_blur(img_uint8: np.ndarray, severity: int) -> np.ndarray:
    """Apply directional motion blur."""
    if cv2 is None:
        raise ImportError("opencv-python is required for motion_blur")

    kernel_sizes = [5, 9, 13, 17, 21]
    k = kernel_sizes[severity - 1]

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= k

    return cv2.filter2D(img_uint8, -1, kernel)


def _gaussian_blur(img_uint8: np.ndarray, severity: int) -> np.ndarray:
    """Apply Gaussian blur."""
    if cv2 is None:
        raise ImportError("opencv-python is required for gaussian_blur")

    sigmas = [1, 2, 3, 4, 5]
    sigma = sigmas[severity - 1]
    ksize = int(2 * round(2 * sigma) + 1)

    return cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)


def _zoom_blur(img_uint8: np.ndarray, severity: int) -> np.ndarray:
    """Simulate zoom blur by averaging zoomed-in copies."""
    zoom_factors = [
        [1.02, 1.04],
        [1.04, 1.08],
        [1.06, 1.12, 1.18],
        [1.08, 1.16, 1.24],
        [1.10, 1.20, 1.30, 1.40],
    ]
    factors = zoom_factors[severity - 1]

    h, w, c = img_uint8.shape
    img_f = img_uint8.astype(np.float32)
    accum = img_f.copy()

    for factor in factors:
        # Centre-crop a zoomed version back to original size
        zh, zw = int(h * factor), int(w * factor)
        if cv2 is not None:
            zoomed = cv2.resize(img_uint8, (zw, zh), interpolation=cv2.INTER_LINEAR)
        elif scipy_zoom is not None:
            zoomed = scipy_zoom(img_uint8, (factor, factor, 1), order=1).astype(np.uint8)
        else:
            # Fallback: no zoom, just accumulate original
            accum += img_f
            continue

        # Centre crop
        y0 = (zh - h) // 2
        x0 = (zw - w) // 2
        cropped = zoomed[y0 : y0 + h, x0 : x0 + w]

        # Handle potential off-by-one from rounding
        if cropped.shape[0] != h or cropped.shape[1] != w:
            cropped = cv2.resize(cropped, (w, h)) if cv2 is not None else cropped[:h, :w]

        accum += cropped.astype(np.float32)

    result = accum / (1 + len(factors))
    return np.clip(result, 0, 255).astype(np.uint8)


def _fog(img_uint8: np.ndarray, severity: int) -> np.ndarray:
    """Overlay a foggy haze."""
    strengths = [0.3, 0.45, 0.6, 0.75, 0.9]
    strength = strengths[severity - 1]

    h, w, c = img_uint8.shape
    img_f = img_uint8.astype(np.float32) / 255.0

    # Create a smooth fog pattern using low-frequency noise
    rng = np.random.RandomState(42)  # deterministic fog
    fog_map = rng.rand(h // 8 + 1, w // 8 + 1).astype(np.float32)

    if cv2 is not None:
        fog_map = cv2.resize(fog_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Simple repeat upscale as fallback
        fog_map = np.repeat(np.repeat(fog_map, 8, axis=0), 8, axis=1)[:h, :w]

    fog_map = fog_map[..., np.newaxis]  # (H, W, 1)

    # Blend with white fog
    result = img_f * (1 - strength * fog_map) + strength * fog_map
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _glass_blur(img_uint8: np.ndarray, severity: int) -> np.ndarray:
    """Simulate glass-like distortion via local pixel shuffling + blur.

    Produces bit-for-bit identical output to the original triple-nested loop.
    Speed-up comes from generating all random offsets in one vectorised numpy
    call (same RNG sequence) and pre-computing swap indices, so the hot loop
    only does integer array look-ups instead of calling rng.randint() per pixel.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for glass_blur")

    sigmas = [0.5, 0.7, 0.9, 1.1, 1.5]
    deltas = [1, 1, 2, 2, 3]
    iters_list = [1, 2, 2, 2, 2]

    sigma = sigmas[severity - 1]
    delta = deltas[severity - 1]
    n_iters = iters_list[severity - 1]

    h, w, c = img_uint8.shape
    result = cv2.GaussianBlur(img_uint8, (0, 0), sigma).astype(np.float32)

    rng = np.random.RandomState(42)

    # --- Pre-compute everything outside the hot loop ---

    # Pixel coordinates visited per iteration (same order as original nested loop)
    i_range = np.arange(h - delta, delta, -1)   # shape (n_rows,)
    j_range = np.arange(w - delta, delta, -1)   # shape (n_cols,)
    ii, jj = np.meshgrid(i_range, j_range, indexing="ij")  # (n_rows, n_cols)
    ii_flat = np.tile(ii.ravel(), n_iters)       # (total,)
    jj_flat = np.tile(jj.ravel(), n_iters)       # (total,)
    total = len(ii_flat)

    # Generate ALL random offsets in one call — produces the identical sequence
    # as calling rng.randint() twice per pixel in the original loop order:
    #   all_offsets[0]  = di for pixel 0,  all_offsets[1]  = dj for pixel 0, ...
    all_offsets = rng.randint(-delta, delta + 1, size=2 * total)
    di_all = all_offsets[0::2]   # shape (total,)
    dj_all = all_offsets[1::2]   # shape (total,)

    # Clamped neighbour indices (vectorised)
    ni_all = np.clip(ii_flat + di_all, 0, h - 1)
    nj_all = np.clip(jj_flat + dj_all, 0, w - 1)

    # Pre-compute flat (1-D) indices for faster array access
    src = (ii_flat * w + jj_flat).astype(np.intp)
    dst = (ni_all * w + nj_all).astype(np.intp)

    # Flat view — mutations here are reflected in `result`
    flat = result.reshape(-1, c)

    # Sequential swaps — order matches original; bit-identical output guaranteed.
    # The remaining Python loop is fast because every per-iteration cost is a
    # plain integer array look-up, with no function calls.
    for k in range(total):
        s, d = src[k], dst[k]
        tmp = flat[s].copy()
        flat[s] = flat[d]
        flat[d] = tmp

    result = cv2.GaussianBlur(result.astype(np.uint8), (0, 0), sigma)
    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_CORRUPTION_FNS = {
    "motion_blur": _motion_blur,
    "gaussian_blur": _gaussian_blur,
    "zoom_blur": _zoom_blur,
    "fog": _fog,
    "glass_blur": _glass_blur,
}


def apply_noise(
    image: torch.Tensor,
    noise_config: NoiseConfig,
) -> torch.Tensor:
    """Apply a visual corruption to a single image tensor.

    Parameters
    ----------
    image : torch.Tensor
        ``(C, H, W)`` float32 tensor in [0, 1].
    noise_config : NoiseConfig
        Which corruption and severity to apply.

    Returns
    -------
    torch.Tensor
        Corrupted ``(C, H, W)`` float32 tensor in [0, 1].
    """
    fn = _CORRUPTION_FNS.get(noise_config.noise_type)
    if fn is None:
        raise ValueError(
            f"Unknown noise type: {noise_config.noise_type!r}. "
            f"Available: {list(_CORRUPTION_FNS)}"
        )

    img_uint8 = _to_hwc_uint8(image)
    corrupted = fn(img_uint8, noise_config.severity)
    return _from_hwc_uint8(corrupted)
