"""Frozen world-model encoder for SRPO trajectory embedding.

Provides a unified interface for encoding trajectory frames into latent
embeddings used by the SRPO world-progress reward model.  Two backends are
supported:

* **DINOv2** (default) – ``facebook/dinov2-large`` (ViT-L, 1024-dim).
  Well-tested, available out-of-the-box via ``transformers``.
* **V-JEPA 2** – ``facebook/vjepa2-vitg-384`` (ViT-G, 1536-dim).  The
  world model used in the SRPO paper.  Falls back to DINOv2 if the
  checkpoint cannot be loaded.

Both models are kept **frozen** (no gradient) and run in fp16 for
efficiency.  Frame subsampling is applied to reduce compute when encoding
long trajectories.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

DINOV2_MODEL_ID = "facebook/dinov2-large"
VJEPA2_MODEL_ID = "facebook/vjepa2-vitg-384"


class WorldModelEncoder(ABC):
    """Abstract interface for a frozen trajectory encoder."""

    @abstractmethod
    def embed_dim(self) -> int: ...

    @abstractmethod
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into embeddings.

        Args:
            images: ``(B, 3, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            ``(B, D)`` embedding tensor.
        """
        ...

    @torch.no_grad()
    def encode_trajectory(
        self,
        images: torch.Tensor,
        subsample_every: int = 5,
    ) -> torch.Tensor:
        """Encode a trajectory into a single embedding via mean-pooled frames.

        Args:
            images: ``(T, 3, H, W)`` float tensor (full trajectory).
            subsample_every: Take every *k*-th frame.

        Returns:
            ``(D,)`` trajectory embedding.
        """
        indices = list(range(0, images.shape[0], subsample_every))
        frames = images[indices]
        if frames.ndim == 5:
            t, v, c, h, w = frames.shape
            frames = frames.reshape(t * v, c, h, w)
        frame_embs = self.encode_frames(frames)
        return frame_embs.mean(dim=0)

    @torch.no_grad()
    def encode_trajectories(
        self,
        trajectories_images: list[torch.Tensor],
        subsample_every: int = 5,
    ) -> torch.Tensor:
        """Encode multiple trajectories into trajectory-level embeddings.

        Args:
            trajectories_images: List of ``(T_i, 3, H, W)`` frame tensors.
            subsample_every: Take every *k*-th frame per trajectory.

        Returns:
            ``(N, D)`` tensor of trajectory embeddings.
        """
        embs = [self.encode_trajectory(imgs, subsample_every) for imgs in trajectories_images]
        return torch.stack(embs, dim=0)


class DINOv2Encoder(WorldModelEncoder):
    """Frozen DINOv2 ViT-L encoder (1024-dim embeddings).

    Args:
        model_id: HuggingFace model id.
        device: Torch device.
        dtype: Model precision.
        batch_size: Max batch size for frame encoding (to limit VRAM).
    """

    def __init__(
        self,
        model_id: str = DINOV2_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size

        logger.info("Loading DINOv2 encoder: %s", model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._embed_dim = self.model.config.hidden_size
        logger.info("DINOv2 loaded – embed_dim=%d", self._embed_dim)

    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        all_embs = []
        for start in range(0, B, self.batch_size):
            batch = images[start : start + self.batch_size]
            batch_pil = [self._tensor_to_pil_format(img) for img in batch]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0]
            all_embs.append(cls_emb.float())
        return torch.cat(all_embs, dim=0)

    @staticmethod
    def _tensor_to_pil_format(img: torch.Tensor) -> torch.Tensor:
        """Ensure image is float in [0,1] with shape (C,H,W)."""
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        return img.float()


class VJEPA2Encoder(WorldModelEncoder):
    """Frozen V-JEPA 2 ViT-G/384 encoder.

    Falls back to :class:`DINOv2Encoder` if the V-JEPA 2 checkpoint
    cannot be loaded (e.g. if the model is not available on HuggingFace).

    Args:
        model_id: HuggingFace model id for V-JEPA 2.
        device: Torch device.
        dtype: Model precision.
        batch_size: Max batch per encode call.
    """

    def __init__(
        self,
        model_id: str = VJEPA2_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self._fallback: DINOv2Encoder | None = None

        try:
            logger.info("Attempting to load V-JEPA 2: %s", model_id)
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._embed_dim = self.model.config.hidden_size
            logger.info("V-JEPA 2 loaded – embed_dim=%d", self._embed_dim)
        except Exception as e:
            logger.warning("Failed to load V-JEPA 2 (%s), falling back to DINOv2.", e)
            self._fallback = DINOv2Encoder(device=device, dtype=dtype, batch_size=batch_size)
            self._embed_dim = self._fallback.embed_dim()

    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        if self._fallback is not None:
            return self._fallback.encode_frames(images)

        B = images.shape[0]
        all_embs = []
        for start in range(0, B, self.batch_size):
            batch = images[start : start + self.batch_size]
            batch_pil = [DINOv2Encoder._tensor_to_pil_format(img) for img in batch]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0]
            all_embs.append(cls_emb.float())
        return torch.cat(all_embs, dim=0)


def build_world_model(
    model_type: str = "dinov2",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,
) -> WorldModelEncoder:
    """Factory for world model encoders.

    Args:
        model_type: ``"dinov2"`` or ``"vjepa2"``.
        device: Torch device string.
        dtype: Model precision.
        batch_size: Max encoding batch size.

    Returns:
        A frozen :class:`WorldModelEncoder`.
    """
    if model_type == "vjepa2":
        return VJEPA2Encoder(device=device, dtype=dtype, batch_size=batch_size)
    return DINOv2Encoder(device=device, dtype=dtype, batch_size=batch_size)
