"""SmolVLA policy wrapper for ManiSkill manipulation tasks.

Handles checkpoint loading, normalization, tokenization, and the
user-facing predict / train API.  The core flow-matching architecture
(:class:`~vla.models.vla_flow_matching.VLAFlowMatching`) lives in its
own module so that the network definition stays separate from the
higher-level policy logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor

from vla.env_metadata import EnvMetadata
from vla.models.vla_flow_matching import VLAFlowMatching
from vla.utils.tensor import to_float01

DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
logger = logging.getLogger(__name__)


def _resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: float = -1) -> torch.Tensor:
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    rh, rw = int(cur_height / ratio), int(cur_width / ratio)
    resized = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    ph, pw = max(0, height - rh), max(0, width - rw)
    return F.pad(resized, (pw, 0, ph, 0), value=pad_value)


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    cur = shape[-1]
    shape[-1] = new_dim
    out = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    out[..., :cur] = vector
    return out


class SmolVLAPolicy(nn.Module):
    """Thin wrapper that loads a lerobot SmolVLA checkpoint.

    Downloads the checkpoint config to discover the VLM backbone,
    reconstructs the full VLAFlowMatching architecture, then loads
    the safetensors weights.

    Args:
        checkpoint: HuggingFace model id or local path to a lerobot SmolVLA checkpoint.
        action_dim: Dimensionality of the robot action space (actual, before padding).
        state_dim: Dimensionality of the robot proprioceptive state (actual).
        device: Torch device string.
        dtype: Model precision (default ``torch.bfloat16``).
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        action_dim: int = 8,
        state_dim: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.eval_zero_sample = False

        ckpt_config = self._load_ckpt_config(checkpoint)
        self.ckpt_config = ckpt_config

        self.vlm_model_name: str = ckpt_config["vlm_model_name"]
        self.max_state_dim: int = ckpt_config.get("max_state_dim", 32)
        self.max_action_dim: int = ckpt_config.get("max_action_dim", 32)
        self.chunk_size: int = ckpt_config.get("chunk_size", 50)
        input_features = ckpt_config.get("input_features", {})
        self.num_image_inputs = max(
            1,
            sum(1 for spec in input_features.values() if isinstance(spec, dict) and spec.get("type") == "VISUAL"),
        )
        self._warned_camera_fallback = False

        self.processor = AutoProcessor.from_pretrained(self.vlm_model_name)

        self.model = VLAFlowMatching(ckpt_config)

        weights_path = self._resolve_weights(checkpoint)
        raw_sd = load_safetensors(weights_path, device="cpu")
        prefix = "model."
        state_dict = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in raw_sd.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            import logging

            logging.warning("Unexpected keys when loading SmolVLA checkpoint: %s", unexpected[:10])

        self.model.to(device=self.device, dtype=self.dtype)
        self._gradient_checkpointing = False

        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)
        self.register_buffer("state_mean", torch.zeros(max(state_dim, 1)), persistent=True)
        self.register_buffer("state_std", torch.ones(max(state_dim, 1)), persistent=True)

        self._load_checkpoint_norm_stats(checkpoint, action_dim, max(state_dim, 1))

    def _load_checkpoint_norm_stats(self, checkpoint: str, action_dim: int, state_dim: int) -> None:
        """Load normalization stats from the HuggingFace checkpoint's processor files.

        These are the stats the model was actually trained with, stored in
        ``policy_postprocessor_step_1_unnormalizer_processor.safetensors``.
        Falls back silently to identity normalization if the files are absent
        (e.g. for local-only checkpoints).
        """
        files_to_try = [
            "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
        ]
        stats: dict[str, torch.Tensor] = {}
        for fname in files_to_try:
            try:
                if Path(checkpoint).is_dir():
                    fpath = Path(checkpoint) / fname
                    if not fpath.exists():
                        continue
                    fpath = str(fpath)
                else:
                    fpath = hf_hub_download(checkpoint, fname)
                stats = load_safetensors(fpath, device="cpu")
                break
            except Exception:
                continue

        if not stats:
            return

        if "action.mean" in stats and "action.std" in stats:
            am = stats["action.mean"].float()[:action_dim]
            astd = stats["action.std"].float()[:action_dim]
            self.action_mean.copy_(am)
            self.action_std.copy_(astd)
        if "observation.state.mean" in stats and "observation.state.std" in stats:
            sm = stats["observation.state.mean"].float()[:state_dim]
            sstd = stats["observation.state.std"].float()[:state_dim]
            self.state_mean.copy_(sm)
            self.state_std.copy_(sstd)
        logger.info(
            "Loaded normalization stats from checkpoint: action_mean=%s action_std=%s",
            self.action_mean.tolist(),
            self.action_std.tolist(),
        )

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        self._gradient_checkpointing = enable
        self.model.vlm_with_expert.enable_gradient_checkpointing(enable)

    def set_normalization(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        """Set mean/std normalization statistics for actions and state.

        Args:
            action_mean: ``(action_dim,)`` tensor.
            action_std: ``(action_dim,)`` tensor.
            state_mean: ``(state_dim,)`` tensor.
            state_std: ``(state_dim,)`` tensor.
        """
        self.action_mean.copy_(action_mean.to(self.device))
        self.action_std.copy_(action_std.to(self.device))
        self.state_mean.copy_(state_mean.to(self.device))
        self.state_std.copy_(state_std.to(self.device))

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(action.device, dtype=action.dtype)
        std = self.action_std.to(action.device, dtype=action.dtype).clamp(min=1e-8)
        return (action - mean) / std

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(action.device, dtype=action.dtype)
        std = self.action_std.to(action.device, dtype=action.dtype).clamp(min=1e-8)
        return action * std + mean

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_dim == 0:
            return state
        sdim = min(state.shape[-1], self.state_mean.shape[0])
        mean = self.state_mean[:sdim].to(state.device, dtype=state.dtype)
        std = self.state_std[:sdim].to(state.device, dtype=state.dtype).clamp(min=1e-8)
        out = state.clone()
        out[..., :sdim] = (state[..., :sdim] - mean) / std
        return out

    def _prepare_state_input(self, state: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """Prepare a ``(B, max_state_dim)`` state tensor from raw observation state.

        Truncates the input to ``self.state_dim`` dimensions so that extra
        state components present during evaluation (e.g. object poses that
        ManiSkill appends to the flat state vector) are discarded.  This
        ensures the model sees exactly the same state structure it was
        trained on.
        """
        if state is None or self.state_dim == 0:
            return torch.zeros(batch_size, self.max_state_dim, device=self.device, dtype=self.dtype)
        s = state.to(self.device, dtype=self.dtype)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        sdim = min(s.shape[-1], self.state_dim)
        s_trunc = s[..., :sdim]
        s_norm = self._normalize_state(s_trunc)
        return _pad_vector(s_norm, self.max_state_dim)

    @staticmethod
    def _load_ckpt_config(checkpoint: str) -> dict[str, Any]:
        if Path(checkpoint).is_dir():
            cfg_path = Path(checkpoint) / "config.json"
        else:
            cfg_path = hf_hub_download(checkpoint, "config.json")
        with open(cfg_path) as f:
            return json.load(f)

    @staticmethod
    def _resolve_weights(checkpoint: str) -> str:
        if Path(checkpoint).is_dir():
            return str(Path(checkpoint) / "model.safetensors")
        return hf_hub_download(checkpoint, "model.safetensors")

    def _tokenize(self, instruction: str | list[str], batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        tok_max = self.ckpt_config.get("tokenizer_max_length", 48)
        if isinstance(instruction, str):
            text = instruction if instruction.endswith("\n") else instruction + "\n"
            encoded = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=tok_max,
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoded["input_ids"].to(self.device).expand(batch_size, -1)
            masks = encoded["attention_mask"].to(self.device).bool().expand(batch_size, -1)
        else:
            texts = [t if t.endswith("\n") else t + "\n" for t in instruction]
            encoded = self.processor.tokenizer(
                texts,
                padding="max_length",
                max_length=tok_max,
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoded["input_ids"].to(self.device)
            masks = encoded["attention_mask"].to(self.device).bool()
        return tokens, masks

    def _prepare_images(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected image tensor with 4 or 5 dims, got shape {tuple(images.shape)}")

        bsize, n_views, _, _, _ = images.shape
        if n_views < self.num_image_inputs:
            if not self._warned_camera_fallback:
                logger.warning(
                    "Only %d camera view(s) provided, but checkpoint expects %d. "
                    "Falling back by duplicating the last camera view.",
                    n_views,
                    self.num_image_inputs,
                )
                self._warned_camera_fallback = True
            pad = images[:, -1:].expand(-1, self.num_image_inputs - n_views, -1, -1, -1)
            images = torch.cat([images, pad], dim=1)
        elif n_views > self.num_image_inputs:
            images = images[:, : self.num_image_inputs]

        resize = self.ckpt_config.get("resize_imgs_with_padding")
        img_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        for i in range(self.num_image_inputs):
            img = images[:, i]
            if resize is not None:
                img = _resize_with_pad(img, *resize, pad_value=0)
            img = img * 2.0 - 1.0
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
            img_list.append(img)
            mask_list.append(mask)
        return img_list, mask_list

    def _prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        return _pad_vector(state, self.max_state_dim)

    def _prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        return _pad_vector(action, self.max_action_dim)

    @staticmethod
    def _to_float01(img: torch.Tensor) -> torch.Tensor:
        return to_float01(img, auto_scale=True)

    def predict_action(self, image: torch.Tensor, instruction: str, state: torch.Tensor | None = None) -> torch.Tensor:
        """Predict a single action from an image observation.

        Args:
            image: (C, H, W) tensor (uint8 or float in [0, 1]).
            instruction: Language instruction.
            state: Optional (state_dim,) proprioceptive state tensor.

        Returns:
            (action_dim,) float tensor (denormalized to environment scale).
        """
        self.eval()
        with torch.no_grad():
            img = self._to_float01(image)
            if img.ndim not in (3, 4):
                raise ValueError(f"Expected image shape (C,H,W) or (N,C,H,W), got {tuple(img.shape)}")
            imgs = img.unsqueeze(0).to(self.device, dtype=self.dtype)
            img_list, mask_list = self._prepare_images(imgs)
            tokens, tmasks = self._tokenize(instruction, batch_size=1)
            s = self._prepare_state_input(state, batch_size=1)
            noise = None
            if self.eval_zero_sample:
                noise = torch.zeros((1, self.chunk_size, self.max_action_dim), device=self.device, dtype=self.dtype)
            actions = self.model.sample_actions(img_list, mask_list, tokens, tmasks, s, noise=noise)
            raw = actions[0, 0, : self.action_dim].float()
            return self._denormalize_action(raw)

    def predict_action_batch(
        self, images: torch.Tensor, instruction: str, states: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict actions for a batch of images.

        Args:
            images: (B, C, H, W) tensor (uint8 or float in [0, 1]).
            instruction: Shared language instruction.
            states: Optional (B, state_dim) proprioceptive state tensor.

        Returns:
            (B, action_dim) float tensor (denormalized).
        """
        self.eval()
        with torch.no_grad():
            if images.ndim not in (4, 5):
                raise ValueError(f"Expected image batch shape (B,C,H,W) or (B,N,C,H,W), got {tuple(images.shape)}")
            imgs = self._to_float01(images).to(self.device, dtype=self.dtype)
            img_list, mask_list = self._prepare_images(imgs)
            tokens, tmasks = self._tokenize(instruction, batch_size=imgs.shape[0])
            s = self._prepare_state_input(states, batch_size=imgs.shape[0])
            noise = None
            if self.eval_zero_sample:
                noise = torch.zeros(
                    (imgs.shape[0], self.chunk_size, self.max_action_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
            actions = self.model.sample_actions(img_list, mask_list, tokens, tmasks, s, noise=noise)
            raw = actions[:, 0, : self.action_dim].float()
            return self._denormalize_action(raw)

    def forward(
        self,
        images: torch.Tensor,
        instruction: str | list[str],
        target_action_chunks: torch.Tensor,
        target_action_mask: torch.Tensor,
        states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute flow-matching MSE loss for behavior cloning.

        Args:
            images: (B, C, H, W) image batch in [0, 1].
            instruction: Shared language instruction string, or per-sample
                list of length B for multi-task training.
            target_actions: (B, action_dim) ground-truth actions (unnormalized).
            states: Optional (B, state_dim) proprioceptive state.

        Returns:
            Dict with ``loss``.
        """
        self.train()
        imgs = self._to_float01(images).to(self.device, dtype=self.dtype)
        img_list, mask_list = self._prepare_images(imgs)
        tokens, tmasks = self._tokenize(instruction, batch_size=imgs.shape[0])
        s = self._prepare_state_input(states, batch_size=imgs.shape[0])

        chunks = target_action_chunks.to(self.device, dtype=self.dtype)
        chunks = self._normalize_action(chunks)
        chunks = _pad_vector(chunks, self.max_action_dim)

        losses = self.model.forward(img_list, mask_list, tokens, tmasks, s, chunks)
        losses = losses[:, :, : self.action_dim]

        valid = target_action_mask.unsqueeze(-1).to(losses.dtype)
        loss = (losses * valid).sum() / valid.sum().clamp(min=1.0)

        return {"loss": loss}

    def _build_rl_action_targets(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = actions.shape[0]
        actions = self._normalize_action(actions)
        actions = self._prepare_action(actions)

        chunks = torch.zeros(
            T,
            self.chunk_size,
            self.max_action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        mask = torch.zeros(
            T,
            self.chunk_size,
            device=actions.device,
            dtype=torch.bool,
        )

        for t in range(T):
            end = min(T, t + self.chunk_size)
            n = end - t
            chunks[t, :n] = actions[t:end]
            mask[t, :n] = True

        mask[:, 1:] = False
        return chunks, mask

    def _build_action_chunks(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = actions.shape[0]
        actions = self._normalize_action(actions)
        actions = self._prepare_action(actions)

        chunks = torch.zeros(
            T,
            self.chunk_size,
            self.max_action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        mask = torch.zeros(
            T,
            self.chunk_size,
            device=actions.device,
            dtype=torch.bool,
        )

        for t in range(T):
            end = min(T, t + self.chunk_size)
            n = end - t
            chunks[t, :n] = actions[t:end]
            mask[t, :n] = True

        return chunks, mask

    def compute_fm_loss_batched(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor | None,
        instruction: str,
        fixed_noise: torch.Tensor,
        fixed_time: torch.Tensor,
        batch_size: int = 32,
        full_chunk_target: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute per-timestep flow-matching loss in mini-batches.

        Packs ``batch_size`` timesteps into a single forward pass, giving
        a near-linear speedup over a naive per-step loop.  Used by the
        SRPO trainer for advantage-weighted and PPO updates.

        Args:
            images: ``(T, [V,] C, H, W)`` observation images.
            actions: ``(T, action_dim)`` actions (unnormalized).
            states: ``(T, state_dim)`` or ``None``.
            instruction: Language instruction (shared across all steps).
            fixed_noise: ``(T, chunk_size, max_action_dim)`` pre-sampled noise.
            fixed_time: ``(T,)`` pre-sampled time values.
            batch_size: Number of timesteps per forward pass.

        Returns:
            ``(T,)`` per-step mean FM loss.
        """
        T = images.shape[0]
        all_losses: list[torch.Tensor] = []
        use_amp = self.device.type == "cuda"

        actions = actions.to(self.device, dtype=self.dtype)
        if full_chunk_target:
            action_chunks, action_mask = self._build_action_chunks(actions)
        else:
            action_chunks, action_mask = self._build_rl_action_targets(actions)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            B = end - start

            imgs = self._to_float01(images[start:end]).to(self.device, dtype=self.dtype)
            img_list, mask_list = self._prepare_images(imgs)
            tokens, tmasks = self._tokenize(instruction, batch_size=B)

            state_raw = states[start:end] if states is not None else None
            state = self._prepare_state_input(state_raw, batch_size=B)

            target_chunks = action_chunks[start:end]
            target_mask = action_mask[start:end]

            noise = fixed_noise[start:end].to(self.device, dtype=self.dtype)
            time = fixed_time[start:end].to(self.device, dtype=self.dtype)

            with torch.autocast("cuda", enabled=use_amp):
                losses = self.model.forward(
                    img_list,
                    mask_list,
                    tokens,
                    tmasks,
                    state,
                    target_chunks,
                    noise=noise,
                    time=time,
                )

            losses = losses.float()[:, :, : self.action_dim]
            valid = target_mask.unsqueeze(-1).float()
            per_pos = (losses * valid).mean(dim=2)

            if reduction == "sum":
                per_step = per_pos.sum(dim=1)
            elif reduction == "mean":
                n_valid_positions = target_mask.sum(dim=1).clamp(min=1.0)
                per_step = per_pos.sum(dim=1) / n_valid_positions
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

            all_losses.append(per_step)

        return torch.cat(all_losses)

    def get_embedding(self, image: torch.Tensor, instruction: str, state: torch.Tensor | None = None) -> torch.Tensor:
        """Return the VLM backbone embedding for a single observation (for Tier B SRPO).

        Args:
            image: (C, H, W) tensor in [0, 1].
            instruction: Language instruction.
            state: Optional (state_dim,) proprioceptive state tensor.

        Returns:
            (hidden_dim,) float tensor.
        """
        self.eval()
        with torch.no_grad():
            img = self._to_float01(image)
            imgs = img.unsqueeze(0).to(self.device, dtype=self.dtype)
            img_list, mask_list = self._prepare_images(imgs)
            tokens, tmasks = self._tokenize(instruction, batch_size=1)
            s = self._prepare_state_input(state, batch_size=1)
            embs, _, _ = self.model.embed_prefix(img_list, mask_list, tokens, tmasks, s)
            return embs[0, -1].float()

    def save_checkpoint(
        self,
        path: str | Path,
        env_metadata: EnvMetadata | None = None,
        **extra_metadata: Any,
    ) -> None:
        """Save full model weights, normalization statistics, and optional metadata.

        Args:
            path: Directory to write ``policy.pt`` into.
            env_metadata: Typed environment metadata.  If ``None`` but keyword
                arguments are provided, an :class:`EnvMetadata` is constructed
                from them for backward compatibility.
        """
        if env_metadata is None and extra_metadata:
            env_metadata = EnvMetadata.from_dict(extra_metadata)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "checkpoint": self.checkpoint,
                "ckpt_config": self.ckpt_config,
                "action_mean": self.action_mean.detach().cpu(),
                "action_std": self.action_std.detach().cpu(),
                "state_mean": self.state_mean.detach().cpu(),
                "state_std": self.state_std.detach().cpu(),
                "env_metadata": env_metadata.to_dict() if env_metadata else {},
            },
            path / "policy.pt",
        )

    def load_checkpoint(self, path: str | Path) -> EnvMetadata:
        """Load model weights and normalization statistics from a previously saved checkpoint.

        Returns:
            :class:`EnvMetadata` parsed from the stored metadata dict.
        """
        path = Path(path)
        # weights_only=False: policy.pt contains metadata dicts alongside the model state dict
        data = torch.load(path / "policy.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        if "state_dim" in data:
            self.state_dim = data["state_dim"]
        if "action_dim" in data:
            self.action_dim = data["action_dim"]
        if "action_mean" in data:
            am = data["action_mean"].to(self.device)
            astd = data["action_std"].to(self.device)
            if am.shape != self.action_mean.shape:
                self.register_buffer("action_mean", torch.zeros_like(am), persistent=True)
                self.register_buffer("action_std", torch.ones_like(astd), persistent=True)
            self.action_mean.copy_(am)
            self.action_std.copy_(astd)
        if "state_mean" in data:
            sm = data["state_mean"].to(self.device)
            sstd = data["state_std"].to(self.device)
            if sm.shape != self.state_mean.shape:
                self.register_buffer("state_mean", torch.zeros_like(sm), persistent=True)
                self.register_buffer("state_std", torch.ones_like(sstd), persistent=True)
            self.state_mean.copy_(sm)
            self.state_std.copy_(sstd)
        return EnvMetadata.from_dict(data.get("env_metadata", {}))
