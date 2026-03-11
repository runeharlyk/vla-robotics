"""Serialization helpers."""

from __future__ import annotations

from typing import Any


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert an object tree to JSON-serializable types.

    Handles PyTorch/numpy scalar tensors (via ``.item()``), nested dicts,
    lists, and tuples.  Useful for writing metrics dicts to JSONL files or
    sending them to logging backends (wandb, etc.).
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj
