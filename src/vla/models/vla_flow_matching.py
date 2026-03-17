"""Standalone VLAFlowMatching model - the core flow-matching architecture.

This module contains the neural network itself (embedding, denoising,
sampling).  The higher-level policy wrapper that handles normalization,
tokenization, checkpointing, and the user-facing API lives in
:mod:`vla.models.smolvla`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.models.vendor.smolvlm_with_expert import SmolVLMWithExpertModel


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
        time = self._sample_time(actions.shape[0], actions.device).to(act_dtype) if time is None else time.to(act_dtype)

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

        loss_type = self.cfg.get("fm_loss_type", "epsilon")

        if loss_type == "epsilon":
            eps_pred = x_t.float() + (1.0 - t.float()) * v_t.float()
            return F.mse_loss(eps_pred, noise.float(), reduction="none")

        return F.mse_loss(v_t.float(), u_t.float(), reduction="none")

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
