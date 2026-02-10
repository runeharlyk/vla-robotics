"""CLIP-based action model for robot manipulation.

This module adapts the CLIP model to predict robot actions conditioned on
visual observations and optional text instructions.
"""

from dataclasses import dataclass
from typing import Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPActionConfig:
    """Configuration for CLIP action model.

    Args:
        clip_model: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14").
        action_dim: Dimension of the action space.
        hidden_dim: Hidden dimension for action MLP.
        num_action_layers: Number of layers in action head.
        use_text_conditioning: Whether to condition on text instructions.
        dropout: Dropout probability.
        freeze_clip: Whether to freeze CLIP encoder weights.
    """

    clip_model: str = "ViT-B/32"
    action_dim: int = 7
    hidden_dim: int = 512
    num_action_layers: int = 3
    use_text_conditioning: bool = True
    dropout: float = 0.1
    freeze_clip: bool = True


class ActionHead(nn.Module):
    """MLP head for predicting continuous robot actions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize the action head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output action dimension.
            num_layers: Number of MLP layers.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through action head.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Actions of shape (batch_size, output_dim).
        """
        return self.mlp(x)


class CLIPActionModel(nn.Module):
    """CLIP-based model for predicting robot actions.

    This model uses CLIP's visual encoder to extract features from observations
    and optionally conditions on text instructions to predict robot actions.
    """

    def __init__(self, config: CLIPActionConfig):
        """Initialize the CLIP action model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=self.device)

        if config.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        clip_dim = self.clip_model.visual.output_dim
        action_input_dim = clip_dim * 2 if config.use_text_conditioning else clip_dim

        self.action_head = ActionHead(
            input_dim=action_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.action_dim,
            num_layers=config.num_action_layers,
            dropout=config.dropout,
        ).to(self.device)

        self.action_scale = nn.Parameter(torch.ones(config.action_dim, device=self.device))
        self.action_bias = nn.Parameter(torch.zeros(config.action_dim, device=self.device))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP visual encoder.

        Args:
            images: Input images of shape (batch_size, 3, H, W).

        Returns:
            Image features of shape (batch_size, clip_dim).
        """
        with torch.set_grad_enabled(not self.config.freeze_clip):
            features = self.clip_model.encode_image(images)
        return features.float()

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode text instructions using CLIP text encoder.

        Args:
            text: List of text instructions.

        Returns:
            Text features of shape (batch_size, clip_dim).
        """
        tokens = clip.tokenize(text, truncate=True).to(self.device)
        with torch.set_grad_enabled(not self.config.freeze_clip):
            features = self.clip_model.encode_text(tokens)
        return features.float()

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Forward pass to predict actions.

        Args:
            images: Input images of shape (batch_size, 3, H, W).
            text: Optional list of text instructions for conditioning.

        Returns:
            Predicted actions of shape (batch_size, action_dim).
        """
        image_features = self.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        if self.config.use_text_conditioning and text is not None:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            combined_features = torch.cat([image_features, text_features], dim=-1)
        else:
            if self.config.use_text_conditioning:
                zeros = torch.zeros_like(image_features)
                combined_features = torch.cat([image_features, zeros], dim=-1)
            else:
                combined_features = image_features

        raw_actions = self.action_head(combined_features)
        actions = raw_actions * self.action_scale + self.action_bias

        return actions

    def compute_loss(
        self,
        images: torch.Tensor,
        target_actions: torch.Tensor,
        text: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            images: Input images of shape (batch_size, 3, H, W).
            target_actions: Ground truth actions of shape (batch_size, action_dim).
            text: Optional list of text instructions.

        Returns:
            Dictionary containing loss values.
        """
        predicted_actions = self.forward(images, text)
        mse_loss = F.mse_loss(predicted_actions, target_actions)
        l1_loss = F.l1_loss(predicted_actions, target_actions)

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
        }


class CLIPActionModelWithHistory(CLIPActionModel):
    """CLIP action model with observation history support.

    This variant processes multiple past observations to capture temporal
    information for better action prediction.
    """

    def __init__(self, config: CLIPActionConfig, history_length: int = 4):
        """Initialize the model with history support.

        Args:
            config: Model configuration.
            history_length: Number of past observations to consider.
        """
        super().__init__(config)
        self.history_length = history_length

        clip_dim = self.clip_model.visual.output_dim
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=2,
        ).to(self.device)

        self.temporal_proj = nn.Linear(clip_dim, clip_dim).to(self.device)

    def encode_image_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of images.

        Args:
            images: Input images of shape (batch_size, history_length, 3, H, W).

        Returns:
            Aggregated features of shape (batch_size, clip_dim).
        """
        batch_size, seq_len, c, h, w = images.shape
        images_flat = images.view(batch_size * seq_len, c, h, w)

        with torch.set_grad_enabled(not self.config.freeze_clip):
            features_flat = self.clip_model.encode_image(images_flat)

        features = features_flat.float().view(batch_size, seq_len, -1)
        temporal_features = self.temporal_encoder(features)
        aggregated = self.temporal_proj(temporal_features[:, -1, :])

        return aggregated

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Forward pass with observation history.

        Args:
            images: Input images of shape (batch_size, history_length, 3, H, W)
                   or (batch_size, 3, H, W) for single frame.
            text: Optional list of text instructions.

        Returns:
            Predicted actions of shape (batch_size, action_dim).
        """
        if images.dim() == 4:
            image_features = self.encode_image(images)
        else:
            image_features = self.encode_image_sequence(images)

        image_features = F.normalize(image_features, dim=-1)

        if self.config.use_text_conditioning and text is not None:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            combined_features = torch.cat([image_features, text_features], dim=-1)
        else:
            if self.config.use_text_conditioning:
                zeros = torch.zeros_like(image_features)
                combined_features = torch.cat([image_features, zeros], dim=-1)
            else:
                combined_features = image_features

        raw_actions = self.action_head(combined_features)
        actions = raw_actions * self.action_scale + self.action_bias

        return actions


def create_clip_action_model(
    action_dim: int = 7,
    clip_model: str = "ViT-B/32",
    use_history: bool = False,
    history_length: int = 4,
    freeze_clip: bool = True,
    **kwargs,
) -> CLIPActionModel:
    """Factory function to create a CLIP action model.

    Args:
        action_dim: Dimension of the action space.
        clip_model: CLIP model variant.
        use_history: Whether to use observation history.
        history_length: Number of past observations (if use_history=True).
        freeze_clip: Whether to freeze CLIP encoder weights.
        **kwargs: Additional config arguments.

    Returns:
        Configured CLIP action model.
    """
    config = CLIPActionConfig(
        clip_model=clip_model,
        action_dim=action_dim,
        freeze_clip=freeze_clip,
        **kwargs,
    )

    if use_history:
        return CLIPActionModelWithHistory(config, history_length=history_length)
    return CLIPActionModel(config)
