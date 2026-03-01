"""SmolVLA policy wrapper for ManiSkill manipulation tasks.

Loads a pretrained SmolVLA checkpoint from HuggingFace and provides a
unified interface for action prediction and supervised fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"


class SmolVLAPolicy(nn.Module):
    """Thin wrapper around a HuggingFace SmolVLA checkpoint.

    Args:
        checkpoint: HuggingFace model id or local path.
        action_dim: Dimensionality of the robot action space.
        device: Torch device string.
        dtype: Model precision (default ``torch.bfloat16``).
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        action_dim: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.checkpoint = checkpoint

        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)

        self.action_head = nn.Linear(self.model.config.text_config.hidden_size, action_dim).to(
            device=self.device, dtype=torch.float32
        )

    def _encode(self, image: torch.Tensor, instruction: str) -> torch.Tensor:
        """Run the VLA backbone and return the last hidden state (pooled).

        Args:
            image: (C, H, W) uint8 or float tensor.
            instruction: Language instruction string.

        Returns:
            (hidden_dim,) float tensor on ``self.device``.
        """
        if image.dtype == torch.uint8:
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = (image.permute(1, 2, 0).cpu().clamp(0, 255)).to(torch.uint8).numpy()
        pil_image = Image.fromarray(image_np)
        inputs = self.processor(text=instruction, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[:, -1, :]
        return pooled.squeeze(0).float()

    def predict_action(self, image: torch.Tensor, instruction: str) -> torch.Tensor:
        """Predict a single action from an image observation.

        Args:
            image: (C, H, W) tensor.
            instruction: Language instruction.

        Returns:
            (action_dim,) float tensor.
        """
        self.eval()
        with torch.no_grad():
            embedding = self._encode(image, instruction)
            action = self.action_head(embedding)
        return action.detach()

    def predict_action_batch(self, images: torch.Tensor, instruction: str) -> torch.Tensor:
        """Predict actions for a batch of images.

        Args:
            images: (B, C, H, W) tensor.
            instruction: Shared language instruction.

        Returns:
            (B, action_dim) float tensor.
        """
        self.eval()
        actions = []
        with torch.no_grad():
            for i in range(images.shape[0]):
                actions.append(self.predict_action(images[i], instruction))
        return torch.stack(actions, dim=0)

    def forward(
        self, images: torch.Tensor, instruction: str, target_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute MSE loss for behavior cloning.

        Args:
            images: (B, C, H, W) image batch.
            instruction: Shared language instruction.
            target_actions: (B, action_dim) ground-truth actions.

        Returns:
            Dict with ``loss`` and ``predicted_actions``.
        """
        self.train()
        embeddings = []
        for i in range(images.shape[0]):
            emb = self._encode(images[i], instruction)
            embeddings.append(emb)
        embeddings_t = torch.stack(embeddings, dim=0)
        predicted = self.action_head(embeddings_t)
        target = target_actions.to(device=self.device, dtype=torch.float32)
        loss = nn.functional.mse_loss(predicted, target)
        return {"loss": loss, "predicted_actions": predicted.detach()}

    def get_embedding(self, image: torch.Tensor, instruction: str) -> torch.Tensor:
        """Return the backbone embedding for a single observation (for Tier B SRPO).

        Args:
            image: (C, H, W) tensor.
            instruction: Language instruction.

        Returns:
            (hidden_dim,) float tensor.
        """
        return self._encode(image, instruction).detach()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save action head weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_head": self.action_head.state_dict(),
                "action_dim": self.action_dim,
                "checkpoint": self.checkpoint,
            },
            path / "policy.pt",
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load action head weights."""
        path = Path(path)
        data = torch.load(path / "policy.pt", map_location=self.device, weights_only=False)
        self.action_head.load_state_dict(data["action_head"])
