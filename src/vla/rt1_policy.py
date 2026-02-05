"""
RT-1 (Robotics Transformer) model wrapper.

RT-1 is a transformer-based model for robotic control from Google Research.
This wrapper uses the lucidrains PyTorch implementation.

Installation:
    pip install robotic-transformer-pytorch

Usage:
    1. Train with: uv run python src/vla/train_rt1.py train --env PickCube-v1
    2. Evaluate with: uv run python src/vla/train_rt1.py evaluate --model models/rt1_pickcube_v1.pt
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class RT1Policy:
    """
    Wrapper for RT-1 (Robotics Transformer) model.
    
    Uses the lucidrains/robotic-transformer-pytorch implementation.
    Supports loading checkpoints from train_rt1.py training script.
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to run model on
        action_dim: Dimension of action space (auto-detected from checkpoint if available)
        model_size: Model size: tiny/small/base (auto-detected from checkpoint if available)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        action_dim: int = 8,
        model_size: str = "small",
    ):
        self.model_path = model_path
        self.device = device
        self.action_dim = action_dim
        self.model_size = model_size
        self.model = None
        self.instruction = "pick up the cube"

    def load(self) -> None:
        try:
            from robotic_transformer_pytorch import RT1, MaxViT
        except ImportError:
            print("robotic-transformer-pytorch not installed. Install with:")
            print("  pip install robotic-transformer-pytorch")
            raise ImportError("robotic_transformer_pytorch package not found")

        import torch

        if self.model_path and Path(self.model_path).exists():
            print(f"Loading checkpoint from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            if "config" in checkpoint:
                config = checkpoint["config"]
                self.action_dim = config.get("action_dim", self.action_dim)
                self.model_size = config.get("model_size", self.model_size)
                self.instruction = config.get("instruction", self.instruction)

        configs = {
            "tiny": {
                "dim_conv_stem": 16, "dim": 32, "dim_head": 16, "depth": (1, 1, 1, 1),
                "rt1_depth": 2, "rt1_heads": 2, "rt1_dim_head": 16,
            },
            "small": {
                "dim_conv_stem": 32, "dim": 48, "dim_head": 16, "depth": (1, 1, 2, 1),
                "rt1_depth": 4, "rt1_heads": 4, "rt1_dim_head": 32,
            },
            "base": {
                "dim_conv_stem": 64, "dim": 96, "dim_head": 32, "depth": (2, 2, 5, 2),
                "rt1_depth": 6, "rt1_heads": 8, "rt1_dim_head": 64,
            },
        }
        cfg = configs.get(self.model_size, configs["small"])

        print(f"Creating RT-1 model ({self.model_size}, action_dim={self.action_dim})...")

        vit = MaxViT(
            num_classes=1000,
            dim_conv_stem=cfg["dim_conv_stem"],
            dim=cfg["dim"],
            dim_head=cfg["dim_head"],
            depth=cfg["depth"],
            window_size=8,
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
        )

        self.model = RT1(
            vit=vit,
            num_actions=self.action_dim,
            action_bins=256,
            depth=cfg["rt1_depth"],
            heads=cfg["rt1_heads"],
            dim_head=cfg["rt1_dim_head"],
            cond_drop_prob=0.2,
        ).to(self.device)

        if self.model_path and Path(self.model_path).exists():
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print("Weights loaded successfully!")
        else:
            print("WARNING: No weights loaded - model will output random actions!")

        self.model.eval()

    def predict_action(
        self,
        image: np.ndarray | Image.Image,
        instruction: Optional[str] = None,
        unnorm_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict robot action from image and language instruction.
        
        Args:
            image: RGB image (H, W, 3) as numpy array or PIL Image
            instruction: Natural language task instruction (uses loaded instruction if None)
            unnorm_key: Not used
        
        Returns:
            Action array of shape (action_dim,)
        """
        import torch

        if self.model is None:
            self.load()

        if instruction is None:
            instruction = self.instruction

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        img_pil = Image.fromarray(image, mode="RGB")
        img_resized = img_pil.resize((256, 256))
        image = np.array(img_resized)

        img_tensor = torch.from_numpy(image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        video = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        video = video.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            logits = self.model(video, texts=[instruction])
            action_bins = logits.argmax(dim=-1)[0, 0]

        action_continuous = (action_bins.float() / 255 * 2 - 1).cpu().numpy()

        return action_continuous[: self.action_dim].astype(np.float32)
