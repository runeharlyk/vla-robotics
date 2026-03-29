"""Frozen world-model encoder for SRPO trajectory embedding.

Provides a unified interface for encoding trajectory frames into latent
embeddings used by the SRPO world-progress reward model.  Two backends are
supported:

* **DINOv2** - ``facebook/dinov2-large`` (ViT-L, 1024-dim).
* **V-JEPA 2** - ``facebook/vjepa2-vitg-fpc64-384-ssv2`` (ViT-G, 1536-dim).
  The world model used in the SRPO paper.  Loaded via ``transformers``
  native V-JEPA 2 support (``transformers >= 4.52``), with a fallback to
  loading ``Sylvest/vjepa2-vit-g`` as a raw checkpoint via ``timm``.

Both models are kept **frozen** (no gradient) and run in fp16 for
efficiency.  Frame subsampling is applied to reduce compute when encoding
long trajectories.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from transformers import AutoImageProcessor, AutoModel

from vla.utils.tensor import to_float01

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
        all_embs = self.encode_frames(mega_batch)  # (total_frames, D)

        # Mean-pool per trajectory
        results = []
        offset = 0
        for sz in traj_sizes:
            results.append(all_embs[offset : offset + sz].mean(dim=0))
            offset += sz
        return torch.stack(results, dim=0)


class DINOv2Encoder(WorldModelEncoder):
    """Frozen DINOv2 ViT-L encoder (1024-dim embeddings)."""

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
        logger.info("DINOv2 loaded - embed_dim=%d", self._embed_dim)

    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        all_embs = []
        for start in range(0, B, self.batch_size):
            batch = images[start : start + self.batch_size]
            batch = to_float01(batch)
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0]
            all_embs.append(cls_emb.float())
        return torch.cat(all_embs, dim=0)


class VJEPA2Encoder(WorldModelEncoder):
    """Frozen V-JEPA 2 ViT-G encoder for SRPO trajectory embedding."""

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
        self._backend: str = "none"

        if self._try_automodel(model_id):
            return
        if self._try_raw_checkpoint(model_id):
            return
        if model_id != VJEPA2_SRPO_RAW_ID and self._try_raw_checkpoint(VJEPA2_SRPO_RAW_ID):
            return

        raise RuntimeError(f"All V-JEPA 2 loading attempts failed for {model_id!r}")

    def _try_automodel(self, model_id: str) -> bool:
        try:
            logger.info("Trying AutoModel for V-JEPA 2: %s", model_id)
            self.model = AutoModel.from_pretrained(
                model_id, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._embed_dim = self.model.config.hidden_size
            self._backend = "transformers"
            logger.info("V-JEPA 2 loaded via AutoModel - embed_dim=%d", self._embed_dim)
            return True
        except Exception as e:
            logger.warning("AutoModel failed for %s: %s", model_id, e)
            return False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """siiRL Public Spec: Resize short side to 438px, then Center-Crop 384x384.
        
        Reference: 
        - img_size = 384
        - short_side = int(256 / 224 * img_size) = 438
        - ImageNet Normalization
        """
        from torchvision.transforms.functional import center_crop, resize

        if x.shape[-2:] != (384, 384):
            h, w = x.shape[-2:]
            short = min(h, w)
            # Alignment: siiRL uses 256/224 scaling factor for the short side
            img_size = 384
            scale = (256 / 224) * img_size / short
            new_size = [int(h * scale), int(w * scale)]
            x = resize(x, new_size, antialias=True)
            x = center_crop(x, [img_size, img_size])

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        all_embs = []
        for start in range(0, B, self.batch_size):
            batch = to_float01(images[start : start + self.batch_size], auto_scale=True)
            batch = self._normalize(batch.to(self.device, dtype=self.dtype))

            if self._backend == "timm":
                out = self.model.forward_features(batch)
                emb = out[:, 0] if out.ndim == 3 else out
            else:
                # Video backend expects (B, C, T, H, W) -> T=1 for single frames
                batch_video = batch.unsqueeze(1)
                outputs = self.model(pixel_values_videos=batch_video)
                hs = outputs.last_hidden_state
                # Average pooling over temporal/spatial tokens if needed
                emb = hs[:, 0] if hs.ndim == 3 else hs.mean(dim=1)

            all_embs.append(emb.float())
        return torch.cat(all_embs, dim=0)

    def _prepare_clip(self, images: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
        """Subsample/pad images to ``target_frames`` on CPU in fp16, normalized."""
        frames = images
        if frames.ndim == 5:
            t, v, c, h, w = frames.shape
            frames = frames.reshape(t * v, c, h, w)
        frames = to_float01(frames, auto_scale=True)
        frames = self._normalize(frames.to(dtype=self.dtype))

        T = frames.shape[0]
        if target_frames < T:
            idx = torch.linspace(0, T - 1, target_frames).long()
            frames = frames[idx]
        elif target_frames > T:
            pad = frames[-1:].expand(target_frames - T, -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)
        return frames

    @torch.no_grad()
    def encode_trajectory(
        self,
        images: torch.Tensor,
        subsample_every: int = 1,
    ) -> torch.Tensor:
        if subsample_every > 1:
            indices = list(range(0, images.shape[0], subsample_every))
            images = images[indices]

        if self._backend == "transformers":
            frames = self._prepare_clip(images)
            clip = frames.unsqueeze(0).to(self.device)
            outputs = self.model(pixel_values_videos=clip)
            hs = outputs.last_hidden_state
            emb = hs[0, 0].float() if hs.ndim == 3 else hs[0].mean(dim=0).float()
            del clip, outputs, hs
            torch.cuda.empty_cache()
            return emb

        frames = to_float01(images, auto_scale=True)
        frames = self._normalize(frames.to(dtype=self.dtype))
        frame_embs = self.encode_frames(frames)
        return frame_embs.mean(dim=0)

    @torch.no_grad()
    def encode_trajectories(
        self,
        trajectories_images: list[torch.Tensor],
        subsample_every: int = 1,
    ) -> torch.Tensor:
        if self._backend != "transformers":
            return super().encode_trajectories(trajectories_images, subsample_every)

        all_embs: list[torch.Tensor] = []
        for imgs in trajectories_images:
            if subsample_every > 1:
                indices = list(range(0, imgs.shape[0], subsample_every))
                imgs = imgs[indices]
            clip_cpu = self._prepare_clip(imgs)
            clip = clip_cpu.unsqueeze(0).to(self.device)
            outputs = self.model(pixel_values_videos=clip)
            hs = outputs.last_hidden_state
            emb = hs[0, 0].float() if hs.ndim == 3 else hs[0].mean(dim=0).float()
            all_embs.append(emb.cpu())
            del clip, outputs, hs, emb
            torch.cuda.empty_cache()

        return torch.stack(all_embs, dim=0).to(self.device)

    # ── Fallback Loading ──────────────────────────────────────────────

    def _try_raw_checkpoint(self, model_id: str) -> bool:
        try:
            logger.info("Trying raw checkpoint loading for %s via timm…", model_id)
            import timm
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(model_id)
            weight_files = [f for f in files if any(f.endswith(ext) for ext in self._WEIGHT_EXTENSIONS)]
            if not weight_files: return False
            path = hf_hub_download(model_id, weight_files[0])

            if weight_files[0].endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(path)
            else:
                raw = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    for key in ("model", "state_dict", "encoder"):
                        if key in raw: raw = raw[key]; break
                state_dict = raw if isinstance(raw, dict) else raw.state_dict()

            embed_dim = self._infer_embed_dim(state_dict)
            timm_name = self._pick_timm_model(embed_dim)
            model = timm.create_model(timm_name, pretrained=False, num_classes=0)
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(self.device, dtype=self.dtype).eval()
            for p in self.model.parameters(): p.requires_grad_(False)
            self._embed_dim = embed_dim
            self._backend = "timm"
            logger.info("V-JEPA 2 loaded via timm - embed_dim=%d", self._embed_dim)
            return True
        except Exception as e:
            logger.warning("Raw checkpoint failed for %s: %s", model_id, e)
            return False

    @staticmethod
    def _infer_embed_dim(state_dict: dict) -> int:
        for key in ("cls_token", "pos_embed", "patch_embed.proj.bias"):
            if key in state_dict: return state_dict[key].shape[-1]
        for key, val in state_dict.items():
            if "norm" in key and "weight" in key and val.ndim == 1: return val.shape[0]
        raise ValueError("Cannot infer embed_dim")

    @staticmethod
    def _pick_timm_model(embed_dim: int) -> str:
        if embed_dim == 1536: return "vit_giant_patch14_dinov2.lvd142m"
        if embed_dim == 1408: return "vit_giant_patch14_clip_224.laion2b"
        return "vit_giant_patch14_dinov2.lvd142m"

    def embed_dim(self) -> int:
        return self._embed_dim


def build_world_model(
    model_type: str = "dinov2",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,
) -> WorldModelEncoder:
    """Factory for world model encoders."""
    if model_type == "vjepa2":
        return VJEPA2Encoder(device=device, dtype=dtype, batch_size=batch_size)
    return DINOv2Encoder(device=device, dtype=dtype, batch_size=batch_size)
