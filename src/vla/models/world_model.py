"""Frozen world-model encoder for SRPO trajectory embedding.

Provides a unified interface for encoding trajectory frames into latent
embeddings used by the SRPO world-progress reward model.  Two backends are
supported:

* **DINOv2** (default) - ``facebook/dinov2-large`` (ViT-L, 1024-dim).
  Well-tested, available out-of-the-box via ``transformers``.
* **V-JEPA 2** - ``facebook/vjepa2-vitg-fpc64-384-ssv2`` (ViT-G, 1536-dim).
  The world model used in the SRPO paper.  Loaded via ``transformers``
  native V-JEPA 2 support.  Falls back to loading
  ``Sylvest/vjepa2-vit-g`` raw checkpoint via ``timm``, and ultimately
  to DINOv2 if all attempts fail.

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
VJEPA2_MODEL_ID = "facebook/vjepa2-vitg-fpc64-384-ssv2"
VJEPA2_SRPO_RAW_ID = "Sylvest/vjepa2-vit-g"


class WorldModelEncoder(ABC):
    """Abstract interface for a frozen trajectory encoder."""

    @abstractmethod
    def embed_dim(self) -> int: ...

    def offload(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()

    def reload(self, device: torch.device | str) -> None:
        if hasattr(self, "model") and self.model is not None:
            self.model.to(device)

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

        Batches frames from all trajectories into large GPU batches for
        maximum throughput.  Falls back to sequential encoding only when
        subclasses override :meth:`encode_trajectory` with special logic
        (e.g. V-JEPA 2 video-clip mode).

        Args:
            trajectories_images: List of ``(T_i, [V,] 3, H, W)`` frame tensors.
            subsample_every: Take every *k*-th frame per trajectory.

        Returns:
            ``(N, D)`` tensor of trajectory embeddings.
        """
        # Subsample and flatten across all trajectories
        all_frames: list[torch.Tensor] = []
        traj_sizes: list[int] = []  # frames per trajectory after subsampling
        for imgs in trajectories_images:
            indices = list(range(0, imgs.shape[0], subsample_every))
            frames = imgs[indices]
            if frames.ndim == 5:  # (T, V, C, H, W) -> (T*V, C, H, W)
                t, v, c, h, w = frames.shape
                frames = frames.reshape(t * v, c, h, w)
            all_frames.append(frames)
            traj_sizes.append(frames.shape[0])

        mega_batch = torch.cat(all_frames, dim=0)  # (total_frames, C, H, W)
        all_embs = self.encode_frames(mega_batch)   # (total_frames, D)

        # Mean-pool per trajectory
        results = []
        offset = 0
        for sz in traj_sizes:
            results.append(all_embs[offset: offset + sz].mean(dim=0))
            offset += sz
        return torch.stack(results, dim=0)


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
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype).to(self.device)
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
            batch = images[start: start + self.batch_size]
            batch_pil = [self._tensor_to_pil_format(img) for img in batch]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=self.dtype)
                      for k, v in inputs.items()}
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
    """Frozen V-JEPA 2 ViT-G encoder for SRPO trajectory embedding.

    Loading strategy (tried in order):

    1. ``AutoModel.from_pretrained`` with the official Facebook HF
       checkpoint (``facebook/vjepa2-vitg-fpc64-384-ssv2``).  Works
       out-of-the-box with ``transformers >= 4.52``.
    2. Raw ``.pt`` checkpoint from ``Sylvest/vjepa2-vit-g`` loaded into a
       ``timm`` ViT-G model.  This is the checkpoint format used by the
       original SRPO codebase (siiRL).
    3. Falls back to :class:`DINOv2Encoder` if all attempts fail.

    When loaded via ``transformers``, :meth:`encode_trajectory` passes the
    subsampled frames as a single video clip ``(B, C, T, H, W)`` so V-JEPA
    2 can leverage temporal context.

    Args:
        model_id: HuggingFace model id for V-JEPA 2.
        device: Torch device.
        dtype: Model precision.
        batch_size: Max clips / frame-batches per forward pass.
    """

    _WEIGHT_EXTENSIONS = (".safetensors", ".pt", ".pth", ".bin")

    def __init__(
        self,
        model_id: str = VJEPA2_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self._fallback: DINOv2Encoder | None = None
        self._backend: str = "none"

        if self._try_automodel(model_id):
            return
        if self._try_raw_checkpoint(model_id):
            return
        if model_id != VJEPA2_SRPO_RAW_ID and self._try_raw_checkpoint(VJEPA2_SRPO_RAW_ID):
            return

        logger.warning(
            "All V-JEPA 2 loading attempts failed. Falling back to DINOv2.")
        self._fallback = DINOv2Encoder(
            device=device, dtype=dtype, batch_size=32)
        self._embed_dim = self._fallback.embed_dim()
        self._backend = "dinov2_fallback"

    def _try_automodel(self, model_id: str) -> bool:
        try:
            logger.info("Trying AutoModel for V-JEPA 2: %s", model_id)
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._embed_dim = self.model.config.hidden_size
            self._backend = "transformers"
            logger.info(
                "V-JEPA 2 loaded via AutoModel – embed_dim=%d", self._embed_dim)
            return True
        except Exception as e:
            logger.warning("AutoModel failed for %s: %s", model_id, e)
            return False

    def _try_raw_checkpoint(self, model_id: str) -> bool:
        try:
            logger.info(
                "Trying raw checkpoint loading for %s via timm…", model_id)
            self._load_raw_checkpoint(model_id)
            logger.info("V-JEPA 2 loaded via timm – embed_dim=%d",
                        self._embed_dim)
            return True
        except Exception as e:
            logger.warning(
                "Raw checkpoint loading failed for %s: %s", model_id, e)
            return False

    def _load_raw_checkpoint(self, model_id: str) -> None:
        from huggingface_hub import hf_hub_download, list_repo_files
        import timm

        files = list_repo_files(model_id)
        weight_files = [f for f in files if any(
            f.endswith(ext) for ext in self._WEIGHT_EXTENSIONS)]
        if not weight_files:
            raise FileNotFoundError(
                f"No weight files ({self._WEIGHT_EXTENSIONS}) in {model_id}. Repo contains: {files}"
            )

        logger.info("Found weight files in %s: %s", model_id, weight_files)
        path = hf_hub_download(model_id, weight_files[0])

        if weight_files[0].endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path)
        else:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict):
                for key in ("model", "state_dict", "encoder", "target_encoder"):
                    if key in raw:
                        raw = raw[key]
                        break
            state_dict = raw if isinstance(raw, dict) else raw.state_dict()

        embed_dim = self._infer_embed_dim(state_dict)
        timm_name = self._pick_timm_model(embed_dim)
        logger.info("Creating timm model %s (embed_dim=%d)",
                    timm_name, embed_dim)
        model = timm.create_model(timm_name, pretrained=False, num_classes=0)

        msg = model.load_state_dict(state_dict, strict=False)
        if msg.unexpected_keys:
            logger.info("Unexpected keys (ignored, first 5): %s",
                        msg.unexpected_keys[:5])
        if msg.missing_keys:
            logger.info("Missing keys (first 5): %s", msg.missing_keys[:5])

        self.model = model.to(self.device, dtype=self.dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._embed_dim = embed_dim
        self._backend = "timm"

    @staticmethod
    def _infer_embed_dim(state_dict: dict) -> int:
        for key in ("cls_token", "pos_embed", "patch_embed.proj.bias"):
            if key in state_dict:
                return state_dict[key].shape[-1]
        for key, val in state_dict.items():
            if "norm" in key and "weight" in key and val.ndim == 1:
                return val.shape[0]
        raise ValueError("Cannot infer embed_dim from state dict keys")

    @staticmethod
    def _pick_timm_model(embed_dim: int) -> str:
        if embed_dim == 1536:
            return "vit_giant_patch14_dinov2.lvd142m"
        if embed_dim == 1408:
            return "vit_giant_patch14_clip_224.laion2b"
        return "vit_giant_patch14_dinov2.lvd142m"

    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        if self._fallback is not None:
            return self._fallback.encode_frames(images)

        B = images.shape[0]
        all_embs = []
        for start in range(0, B, self.batch_size):
            batch = images[start: start + self.batch_size]
            batch = batch.float()
            if batch.max() > 1.0:
                batch = batch / 255.0

            if self._backend == "timm":
                batch = batch.to(self.device, dtype=self.dtype)
                out = self.model.forward_features(batch)
                emb = out[:, 0] if out.ndim == 3 else out
            else:
                batch = batch.to(self.device, dtype=self.dtype)
                batch_video = batch.unsqueeze(1)
                outputs = self.model(pixel_values_videos=batch_video)
                hs = outputs.last_hidden_state
                emb = hs[:, 0] if hs.ndim == 3 else hs.mean(dim=1)

            all_embs.append(emb.float())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_trajectory(
        self,
        images: torch.Tensor,
        subsample_every: int = 5,
    ) -> torch.Tensor:
        """Encode a full trajectory, leveraging temporal context when possible.

        With the ``transformers`` backend the subsampled frames are passed
        as a single video clip ``(1, C, T, H, W)`` so V-JEPA 2 sees
        temporal structure.  With the ``timm`` backend (raw checkpoint)
        frames are encoded independently and mean-pooled.

        Args:
            images: ``(T, 3, H, W)`` or ``(T, V, 3, H, W)`` float tensor.
            subsample_every: Take every *k*-th frame.

        Returns:
            ``(D,)`` trajectory embedding.
        """
        if self._fallback is not None:
            return self._fallback.encode_trajectory(images, subsample_every)

        indices = list(range(0, images.shape[0], subsample_every))
        frames = images[indices]
        if frames.ndim == 5:
            t, v, c, h, w = frames.shape
            frames = frames.reshape(t * v, c, h, w)

        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0

        if self._backend == "transformers":
            clip = frames.unsqueeze(0)
            clip = clip.to(self.device, dtype=self.dtype)
            outputs = self.model(pixel_values_videos=clip)
            hs = outputs.last_hidden_state
            return hs[0, 0].float() if hs.ndim == 3 else hs[0].mean(dim=0).float()

        frame_embs = self.encode_frames(frames)
        return frame_embs.mean(dim=0)

    @torch.no_grad()
    def encode_trajectories(
        self,
        trajectories_images: list[torch.Tensor],
        subsample_every: int = 5,
    ) -> torch.Tensor:
        """Encode multiple trajectories with batched video-clip inference.

        For the ``transformers`` backend, packs multiple trajectory clips
        into a single ``(N, C, T, H, W)`` batch (preserving temporal
        context within each clip).  For ``timm`` / fallback, delegates to
        the base class which flattens all frames into one mega-batch.

        Args:
            trajectories_images: List of ``(T_i, [V,] 3, H, W)`` tensors.
            subsample_every: Take every *k*-th frame per trajectory.

        Returns:
            ``(N, D)`` tensor of trajectory embeddings.
        """
        if self._fallback is not None:
            return self._fallback.encode_trajectories(trajectories_images, subsample_every)

        if self._backend != "transformers":
            # timm backend: use the base-class mega-batch strategy
            return super().encode_trajectories(trajectories_images, subsample_every)

        # ── transformers backend: batch video clips ──────────────────────
        # Subsample each trajectory and normalise to (T_sub*V, C, H, W)
        clips: list[torch.Tensor] = []
        for imgs in trajectories_images:
            indices = list(range(0, imgs.shape[0], subsample_every))
            frames = imgs[indices]
            if frames.ndim == 5:  # (T, V, C, H, W)
                t, v, c, h, w = frames.shape
                frames = frames.reshape(t * v, c, h, w)
            frames = frames.float()
            if frames.max() > 1.0:
                frames = frames / 255.0
            clips.append(frames)  # (F_i, C, H, W)

        # Pad to the same temporal length so we can stack into one tensor
        max_frames = max(c.shape[0] for c in clips)
        padded: list[torch.Tensor] = []
        lengths: list[int] = []
        for c in clips:
            lengths.append(c.shape[0])
            if c.shape[0] < max_frames:
                pad = c[-1:].expand(max_frames - c.shape[0], -1, -1, -1)
                c = torch.cat([c, pad], dim=0)
            padded.append(c)

        # Stack: (N, F, C, H, W) – this is the pixel_values_videos format
        batch_clips = torch.stack(padded, dim=0).to(
            self.device, dtype=self.dtype)
        N = batch_clips.shape[0]

        # Process in sub-batches of self.batch_size clips
        all_embs: list[torch.Tensor] = []
        for start in range(0, N, self.batch_size):
            sub = batch_clips[start: start + self.batch_size]
            outputs = self.model(pixel_values_videos=sub)
            hs = outputs.last_hidden_state
            if hs.ndim == 3:
                emb = hs[:, 0]  # (sub_N, D)
            else:
                emb = hs.mean(dim=1)  # (sub_N, D)
            all_embs.append(emb.float())

        return torch.cat(all_embs, dim=0)  # (N, D)


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
