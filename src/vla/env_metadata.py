"""Typed metadata stored alongside policy checkpoints.

Replaces the raw ``dict`` that ``save_checkpoint`` / ``load_checkpoint``
previously used under the ``"env_metadata"`` key.  Having a dataclass makes
the schema discoverable, prevents typos in key strings, and provides
sensible defaults so callers no longer need chains of ``.get("key", fallback)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EnvMetadata:
    """Environment configuration persisted inside policy checkpoints.

    Downstream scripts (evaluate, visualize, RL training) read these fields
    to auto-configure the simulator without requiring the user to re-specify
    every CLI flag.
    """

    env_id: str = "PickCube-v1"
    instruction: str = "complete the manipulation task"
    control_mode: str = "pd_joint_delta_pos"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EnvMetadata:
        """Construct from a plain dict, ignoring unknown keys."""
        known = {f.name for f in field()} if False else set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})
