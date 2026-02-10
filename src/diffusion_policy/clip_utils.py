"""Utility functions for loading and using CLIP models."""

import clip
import torch
from PIL import Image


def load_clip_model(model_name: str = "ViT-B/32") -> tuple:
    """Load a CLIP model and its preprocessing function.

    Args:
        model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14").

    Returns:
        Tuple of (model, preprocess) where model is the CLIP model and
        preprocess is the image preprocessing function.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def get_available_models() -> list[str]:
    """Get list of available CLIP model variants.

    Returns:
        List of available model names.
    """
    return clip.available_models()


def preprocess_image(image: Image.Image, preprocess) -> torch.Tensor:
    """Preprocess a PIL image for CLIP.

    Args:
        image: PIL Image to preprocess.
        preprocess: CLIP preprocessing function.

    Returns:
        Preprocessed image tensor.
    """
    return preprocess(image).unsqueeze(0)


def encode_text_batch(model, texts: list[str], device: str = None) -> torch.Tensor:
    """Encode a batch of text strings using CLIP.

    Args:
        model: CLIP model.
        texts: List of text strings to encode.
        device: Device to use (defaults to model's device).

    Returns:
        Text embeddings of shape (batch_size, embed_dim).
    """
    if device is None:
        device = next(model.parameters()).device
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features
