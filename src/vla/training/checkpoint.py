"""Shared checkpoint management utilities."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def save_best_checkpoint(
    current_value: float,
    best_value: float,
    save_fn: Any,
    tag: str = "",
) -> float:
    """Compare ``current_value`` against ``best_value`` and save if improved.

    Args:
        current_value: The metric to compare (e.g. success rate).
        best_value: Previous best.
        save_fn: Zero-arg callable invoked when a new best is found.
        tag: Label used in the log message.

    Returns:
        Updated best value.
    """
    if current_value > best_value:
        save_fn()
        logger.info("New best %scheckpoint: %.2f%%", f"{tag} " if tag else "", current_value * 100)
        return current_value
    return best_value
