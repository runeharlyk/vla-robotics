"""Centralized configuration for the visual noise robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# The five Libero+ sensor noise types.
NOISE_TYPES: list[str] = [
    "motion_blur",
    "gaussian_blur",
    "zoom_blur",
    "fog",
    "glass_blur",
]

# Fixed severity level (user chose L3).
NOISE_SEVERITY: int = 3
DEFAULT_EVAL_CAMERAS: list[str] = [
    "observation.images.front",
    "observation.images.wrist",
]


@dataclass
class EvalConfig:
    """All knobs for one evaluation run."""

    # -- Model --
    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    device: str = "cuda"
    seed: int = 0

    # -- Data --
    rollout_path: str = ""  # path to the combined Libero+ h5 file (TBD)
    cameras: list[str] = field(default_factory=lambda: list(DEFAULT_EVAL_CAMERAS))

    # -- Noise --
    noise_types: list[str] = field(default_factory=lambda: list(NOISE_TYPES))
    noise_severity: int = NOISE_SEVERITY

    # -- Iteration caps (None = process all) --
    max_tasks: int | None = None    # cap on number of tasks (up to 15)
    max_demos: int | None = None    # cap on rollouts per task (up to 50)

    # -- Output --
    output_dir: str = "smolvla_visual_pilot/outputs"

    def resolve_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
