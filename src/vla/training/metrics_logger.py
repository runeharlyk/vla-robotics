"""Unified metrics logging for training loops.

Wraps both W&B remote logging and local JSONL persistence behind a
single :meth:`MetricsLogger.log` call, eliminating duplication across
SFT and SRPO training loops.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vla.utils.serialization import to_json_serializable

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Unified metrics sink for W&B and local JSONL persistence.

    Args:
        jsonl_path: Optional path to a ``.jsonl`` file for local persistence.
            Created (along with parent dirs) on first :meth:`log` call.
        wandb_run: Optional wandb run object (anything with a ``.log()`` method).
    """

    def __init__(
        self,
        jsonl_path: Path | str | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self._wandb_run = wandb_run
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def has_wandb(self) -> bool:
        return self._wandb_run is not None

    @property
    def has_jsonl(self) -> bool:
        return self._jsonl_path is not None

    def log(self, data: dict[str, Any]) -> None:
        """Log a metrics dictionary to all configured sinks."""
        if self._wandb_run is not None:
            self._wandb_run.log(data)

        if self._jsonl_path is not None:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(to_json_serializable(data)) + "\n")
