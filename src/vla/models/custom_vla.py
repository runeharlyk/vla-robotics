"""
Custom VLA architecture for LIBERO.

Components:
    - Vision encoder: SigLIP (frozen or fine-tuned)
    - Language encoder: Sentence-level encoder (frozen)
    - Fusion: Cross-attention conditioning visual features on language
    - Action head: Flow matching transformer that generates action chunks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention to fuse vision and language features.

    Visual tokens attend to language tokens, producing language-conditioned
    visual representations.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of cross-attention layers
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                        "norm1": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                    }
                )
            )

    def forward(self, vision_tokens: torch.Tensor, lang_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse vision tokens with language tokens via cross-attention.

        Args:
            vision_tokens: (B, N_v, D)
            lang_tokens: (B, N_l, D)

        Returns:
            Language-conditioned vision tokens (B, N_v, D)
        """
        x = vision_tokens
        for layer in self.layers:
            residual = x
            x = layer["norm1"](x)
            x, _ = layer["cross_attn"](query=x, key=lang_tokens, value=lang_tokens)
            x = residual + x
            residual = x
            x = layer["norm2"](x)
            x = layer["ffn"](x)
            x = residual + x
        return x


class FlowMatchingActionHead(nn.Module):
    """Conditional flow matching transformer for action generation.

    Generates action chunks by learning a vector field that transports
    noise to action distributions, conditioned on vision-language features.

    Args:
        d_model: Hidden dimension
        action_dim: Dimension of action space
        chunk_size: Number of actions to predict per step
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int = 7,
        chunk_size: int = 20,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.action_proj = nn.Linear(action_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                        "norm1": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                        "norm3": nn.LayerNorm(d_model),
                    }
                )
            )

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Args:
            noisy_actions: (B, chunk_size, action_dim) noisy action samples
            t: (B, 1) or (B,) diffusion time in [0, 1]
            condition: (B, N_cond, d_model) conditioning features from fusion

        Returns:
            Predicted velocity (B, chunk_size, action_dim)
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        x = self.action_proj(noisy_actions) + self.pos_embed[:, : noisy_actions.shape[1], :]
        time_emb = self.time_mlp(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + time_emb

        for layer in self.layers:
            residual = x
            x = layer["norm1"](x)
            x, _ = layer["self_attn"](x, x, x)
            x = residual + x

            residual = x
            x = layer["norm2"](x)
            x, _ = layer["cross_attn"](query=x, key=condition, value=condition)
            x = residual + x

            residual = x
            x = layer["norm3"](x)
            x = layer["ffn"](x)
            x = residual + x

        x = self.final_norm(x)
        return self.out_proj(x)


class CustomVLA(nn.Module):
    """Custom Vision-Language-Action model for LIBERO.

    Architecture:
        Image -> [Vision Encoder] -> vision tokens
        Instruction -> [Language Encoder] -> language tokens
        (vision tokens, language tokens) -> [Cross-Attention Fusion] -> conditioned tokens
        conditioned tokens -> [Flow Matching Action Head] -> action chunk

    Args:
        vision_model: SigLIP or DINOv2 model name
        language_model: Sentence encoder model name
        d_model: Internal hidden dimension for fusion and action head
        action_dim: Action space dimensionality
        chunk_size: Number of actions to predict
        n_fusion_layers: Cross-attention layers
        n_action_layers: Flow matching transformer layers
        n_heads: Attention heads
        freeze_vision: Whether to freeze the vision encoder
        freeze_language: Whether to freeze the language encoder
        dropout: Dropout rate
        flow_steps: Number of ODE steps during inference
    """

    def __init__(
        self,
        vision_model: str = "google/siglip-base-patch16-256",
        language_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        d_model: int = 512,
        action_dim: int = 7,
        chunk_size: int = 20,
        n_fusion_layers: int = 2,
        n_action_layers: int = 4,
        n_heads: int = 8,
        freeze_vision: bool = True,
        freeze_language: bool = True,
        dropout: float = 0.0,
        flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.flow_steps = flow_steps
        self.d_model = d_model
        self._vision_model_name = vision_model
        self._language_model_name = language_model
        self._freeze_vision = freeze_vision
        self._freeze_language = freeze_language

        self._init_vision_encoder(vision_model, freeze_vision)
        self._init_language_encoder(language_model, freeze_language)

        self.fusion = CrossAttentionFusion(d_model, n_heads=n_heads, n_layers=n_fusion_layers, dropout=dropout)
        self.action_head = FlowMatchingActionHead(
            d_model=d_model,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_layers=n_action_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def _init_vision_encoder(self, model_name: str, freeze: bool) -> None:
        from transformers import AutoModel, AutoProcessor

        self.vision_encoder = AutoModel.from_pretrained(model_name)
        self.vision_processor = AutoProcessor.from_pretrained(model_name)
        vision_dim = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_dim, self.d_model)

        if freeze:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def _init_language_encoder(self, model_name: str, freeze: bool) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.language_encoder = AutoModel.from_pretrained(model_name)
        self.language_tokenizer = AutoTokenizer.from_pretrained(model_name)
        lang_dim = self.language_encoder.config.hidden_size
        self.language_proj = nn.Linear(lang_dim, self.d_model)

        if freeze:
            for p in self.language_encoder.parameters():
                p.requires_grad = False

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to vision tokens.

        Args:
            images: (B, C, H, W) normalized images

        Returns:
            (B, N_patches, d_model) vision tokens
        """
        if self._freeze_vision:
            with torch.no_grad():
                outputs = self.vision_encoder(pixel_values=images)
        else:
            outputs = self.vision_encoder(pixel_values=images)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs[0]

        return self.vision_proj(tokens)

    def encode_language(self, instructions: list[str]) -> torch.Tensor:
        """Encode text instructions to language tokens.

        Args:
            instructions: List of B instruction strings

        Returns:
            (B, N_tokens, d_model) language tokens
        """
        device = next(self.language_proj.parameters()).device
        tokenized = self.language_tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        if self._freeze_language:
            with torch.no_grad():
                outputs = self.language_encoder(**tokenized)
        else:
            outputs = self.language_encoder(**tokenized)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            tokens = outputs[0]

        return self.language_proj(tokens)

    def compute_flow_loss(
        self,
        condition: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the conditional flow matching loss.

        Samples a random time t ~ U(0,1) and a noise sample, then
        constructs the interpolant x_t = (1-t)*noise + t*target.
        The loss is MSE between the predicted velocity and the
        ground-truth velocity (target - noise).

        Args:
            condition: (B, N_cond, d_model) conditioning tokens
            target_actions: (B, chunk_size, action_dim) ground truth actions

        Returns:
            Scalar flow matching loss
        """
        B = target_actions.shape[0]
        device = target_actions.device

        t = torch.rand(B, device=device)
        noise = torch.randn_like(target_actions)

        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * noise + t_expand * target_actions
        velocity_target = target_actions - noise

        velocity_pred = self.action_head(x_t, t, condition)
        return F.mse_loss(velocity_pred, velocity_target)

    @torch.no_grad()
    def sample_actions(self, condition: torch.Tensor) -> torch.Tensor:
        """Generate action chunk via Euler ODE integration.

        Args:
            condition: (B, N_cond, d_model) conditioning tokens

        Returns:
            (B, chunk_size, action_dim) sampled actions
        """
        B = condition.shape[0]
        device = condition.device

        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt = 1.0 / self.flow_steps

        for i in range(self.flow_steps):
            t = torch.full((B,), i * dt, device=device)
            velocity = self.action_head(x, t, condition)
            x = x + velocity * dt

        return x

    def forward(self, batch: dict) -> dict:
        """Training forward pass.

        Args:
            batch: Dict with keys:
                - "images": (B, C, H, W) or (B, T, C, H, W)
                - "actions": (B, chunk_size, action_dim)
                - "instruction": list of B strings
                - (optional) "state": (B, state_dim)

        Returns:
            Dict with "loss" key
        """
        images = batch["images"]
        if images.ndim == 5:
            images = images[:, -1]
        actions = batch["actions"]
        instructions = batch["instruction"]
        if isinstance(instructions, str):
            instructions = [instructions] * images.shape[0]

        vision_tokens = self.encode_vision(images)
        lang_tokens = self.encode_language(instructions)
        condition = self.fusion(vision_tokens, lang_tokens)
        loss = self.compute_flow_loss(condition, actions)

        return {"loss": loss}

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Inference: generate actions for a single observation.

        Args:
            batch: Same keys as forward()

        Returns:
            (B, action_dim) first action from the generated chunk
        """
        images = batch["images"]
        if images.ndim == 5:
            images = images[:, -1]
        instructions = batch.get("instruction", batch.get("task", [""]))
        if isinstance(instructions, str):
            instructions = [instructions] * images.shape[0]

        vision_tokens = self.encode_vision(images)
        lang_tokens = self.encode_language(instructions)
        condition = self.fusion(vision_tokens, lang_tokens)
        action_chunk = self.sample_actions(condition)

        return action_chunk[:, 0, :]

    def reset(self) -> None:
        pass

    def get_model_kwargs(self) -> dict:
        """Return constructor kwargs for checkpointing."""
        return {
            "vision_model": self._vision_model_name,
            "language_model": self._language_model_name,
            "d_model": self.d_model,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "n_fusion_layers": len(self.fusion.layers),
            "n_action_layers": len(self.action_head.layers),
            "n_heads": self.fusion.layers[0]["cross_attn"].num_heads,
            "freeze_vision": self._freeze_vision,
            "freeze_language": self._freeze_language,
            "flow_steps": self.flow_steps,
        }
