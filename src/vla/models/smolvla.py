"""SmolVLA policy wrapper for ManiSkill manipulation tasks.

Reconstructs the lerobot SmolVLA (VLAFlowMatching) architecture locally so
we can load checkpoints such as ``HuggingFaceVLA/smolvla_libero`` without
pulling in the full lerobot package (which has an incompatible gymnasium
version).  The VLM backbone + processor are loaded from the original model
id stored in the checkpoint config (e.g. ``HuggingFaceTB/SmolVLM2-500M-Instruct``).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor

from vla.models.vendor.smolvlm_with_expert import SmolVLMWithExpertModel

DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
logger = logging.getLogger(__name__)


def _get_safe_dtype(dtype: torch.dtype, device_type: str) -> torch.dtype:
    if dtype == torch.float64 and device_type == "mps":
        return torch.float32
    return dtype


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> torch.Tensor:
    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


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


def _pad_tensor(tensor: torch.Tensor, max_len: int, pad_value: float = 0) -> torch.Tensor:
    b, d = tensor.shape[:2]
    padded = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded[:, :d] = tensor
    return padded


class VLAFlowMatching(nn.Module):
    """Standalone reimplementation of the lerobot VLAFlowMatching model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config
        self.chunk_size: int = config["chunk_size"]
        self.max_state_dim: int = config["max_state_dim"]
        self.max_action_dim: int = config["max_action_dim"]
        self.num_steps: int = config["num_steps"]
        self.min_period: float = config.get("min_period", 4e-3)
        self.max_period: float = config.get("max_period", 4.0)
        self.use_cache: bool = config.get("use_cache", True)
        self.resize_imgs_with_padding: list[int] | None = config.get("resize_imgs_with_padding")
        self.add_image_special_tokens: bool = config.get("add_image_special_tokens", False)
        self.prefix_length: int = config.get("prefix_length", 0)

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config["vlm_model_name"],
            freeze_vision_encoder=config.get("freeze_vision_encoder", True),
            train_expert_only=config.get("train_expert_only", True),
            load_vlm_weights=config.get("load_vlm_weights", True),
            attention_mode=config.get("attention_mode", "cross_attn"),
            num_expert_layers=config.get("num_expert_layers", -1),
            num_vlm_layers=config.get("num_vlm_layers", 0),
            self_attn_every_n_layers=config.get("self_attn_every_n_layers", 2),
            expert_width_multiplier=config.get("expert_width_multiplier", 0.5),
        )

        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        self.state_proj = nn.Linear(self.max_state_dim, vlm_hidden)
        self.action_in_proj = nn.Linear(self.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, self.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long)
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)

    def _sample_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def _sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(1.5, 1.0)
        return (beta_dist.sample((bsize,)) * 0.999 + 0.001).to(device=device, dtype=torch.float32)

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []
        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                start_tok = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                start_mask = torch.ones_like(start_tok[:, :, 0], dtype=torch.bool)
                att_masks += [0] * start_mask.shape[-1]
                embs.append(start_tok)
                pad_masks.append(start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            dim = img_emb.shape[-1]
            img_emb = img_emb * (dim**0.5)
            bsize, n_img = img_emb.shape[:2]
            img_mask_exp = img_mask[:, None].expand(bsize, n_img)
            embs.append(img_emb)
            pad_masks.append(img_mask_exp)
            att_masks += [0] * n_img

            if self.add_image_special_tokens:
                end_tok = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                end_mask = torch.ones_like(end_tok[:, :, 0], dtype=torch.bool)
                embs.append(end_tok)
                pad_masks.append(end_mask)
                att_masks += [0] * end_mask.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        s_len = state_emb.shape[1]
        pad_masks.append(torch.ones(bsize, s_len, dtype=torch.bool, device=device))
        att_masks += [1] * s_len

        embs_t = torch.cat(embs, dim=1)
        pad_t = torch.cat(pad_masks, dim=1)
        att_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_t.device)[None, :]

        seq_len = pad_t.shape[1]
        if self.prefix_length > 0 and seq_len < self.prefix_length:
            embs_t = _pad_tensor(embs_t, self.prefix_length)
            pad_t = _pad_tensor(pad_t, self.prefix_length)
            att_t = _pad_tensor(att_t, self.prefix_length)

        att_t = att_t.expand(bsize, -1)
        return embs_t, pad_t, att_t

    def embed_suffix(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = _create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size, self.min_period, self.max_period, device
        ).to(dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        at_emb = F.silu(self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2)))
        at_emb = self.action_time_mlp_out(at_emb)

        at_mask = torch.ones(bsize, at_emb.shape[1], dtype=torch.bool, device=device)
        att = [1] * self.chunk_size
        att_t = torch.tensor(att, dtype=at_emb.dtype, device=device)[None, :].expand(bsize, -1)
        return at_emb, at_mask, att_t

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        act_dtype = actions.dtype
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device).to(act_dtype)
        else:
            noise = noise.to(act_dtype)
        if time is None:
            time = self._sample_time(actions.shape[0], actions.device).to(act_dtype)
        else:
            time = time.to(act_dtype)

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        pre_embs, pre_pad, pre_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        suf_embs, suf_pad, suf_att = self.embed_suffix(x_t, time)

        pad = torch.cat([pre_pad, suf_pad], dim=1)
        att = torch.cat([pre_att, suf_att], dim=1)
        att_2d = _make_att_2d_masks(pad, att)
        pos_ids = torch.cumsum(pad, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[pre_embs, suf_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.chunk_size :]
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    def sample_actions(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsize = state.shape[0]
        device = state.device
        mdtype = state.dtype
        if noise is None:
            noise = self._sample_noise((bsize, self.chunk_size, self.max_action_dim), device).to(mdtype)

        pre_embs, pre_pad, pre_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        pre_att_2d = _make_att_2d_masks(pre_pad, pre_att)
        pre_pos = torch.cumsum(pre_pad, dim=1) - 1

        _, past_kv = self.vlm_with_expert.forward(
            attention_mask=pre_att_2d,
            position_ids=pre_pos,
            past_key_values=None,
            inputs_embeds=[pre_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.num_steps
        x_t = noise
        for step in range(self.num_steps):
            t_val = 1.0 + step * dt
            t_tensor = torch.tensor(t_val, dtype=mdtype, device=device).expand(bsize)
            v_t = self._denoise_step(x_t, pre_pad, past_kv, t_tensor)
            x_t = x_t + dt * v_t
        return x_t

    def _denoise_step(
        self,
        x_t: torch.Tensor,
        prefix_pad: torch.Tensor,
        past_kv: dict,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suf_embs, suf_pad, suf_att = self.embed_suffix(x_t, timestep)
        suf_len = suf_pad.shape[1]
        bsize = prefix_pad.shape[0]
        pre_len = prefix_pad.shape[1]
        pre_2d = prefix_pad[:, None, :].expand(bsize, suf_len, pre_len)
        suf_2d = _make_att_2d_masks(suf_pad, suf_att)
        full_att = torch.cat([pre_2d, suf_2d], dim=2)
        offsets = prefix_pad.sum(dim=-1)[:, None]
        pos_ids = offsets + torch.cumsum(suf_pad, dim=1) - 1

        out, _ = self.vlm_with_expert.forward(
            attention_mask=full_att,
            position_ids=pos_ids,
            past_key_values=past_kv,
            inputs_embeds=[None, suf_embs],
            use_cache=self.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = out[1][:, -self.chunk_size :]
        return self.action_out_proj(suffix_out)


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

        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)
        self.register_buffer("state_mean", torch.zeros(max(state_dim, 1)), persistent=True)
        self.register_buffer("state_std", torch.ones(max(state_dim, 1)), persistent=True)

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
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        if img.max() > 1.5:
            return img.float() / 255.0
        return img.float()

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
            actions = self.model.sample_actions(img_list, mask_list, tokens, tmasks, s)
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
            actions = self.model.sample_actions(img_list, mask_list, tokens, tmasks, s)
            raw = actions[:, 0, : self.action_dim].float()
            return self._denormalize_action(raw)

    def forward(
        self,
        images: torch.Tensor,
        instruction: str | list[str],
        target_actions: torch.Tensor,
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
        normalized_actions = self._normalize_action(target_actions.to(self.device, dtype=self.dtype))
        actions_padded = self._prepare_action(normalized_actions)
        actions_padded = actions_padded.unsqueeze(1).expand(-1, self.chunk_size, -1)

        losses = self.model.forward(img_list, mask_list, tokens, tmasks, s, actions_padded)
        loss = losses[:, :, : self.max_action_dim].mean()
        return {"loss": loss}

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

    def save_checkpoint(self, path: str | Path, **extra_metadata: Any) -> None:
        """Save full model weights, normalization statistics, and optional metadata.

        Any additional keyword arguments are persisted under an ``"env_metadata"``
        key so that downstream evaluate / visualize scripts can auto-configure
        ``env_id``, ``instruction``, ``control_mode``, etc. without requiring
        the user to re-specify them.
        """
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
                "env_metadata": extra_metadata,
            },
            path / "policy.pt",
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load model weights and normalization statistics from a previously saved checkpoint.
        """
        path = Path(path)
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
        return data.get("env_metadata", {})
