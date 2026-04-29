"""Hydra wrapper for grouped SRPO experiment configs.

This script intentionally delegates to ``scripts/train_srpo.py`` after
composing a Hydra config.  The existing Typer entrypoint remains the
single implementation of training, while Hydra provides cleaner
experiment files and override syntax.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SRPO = REPO_ROOT / "scripts" / "train_srpo.py"

PAIRED_BOOL_FLAGS = {
    "wandb",
    "failure-rewards",
    "standard-scaler",
    "rollout.gradient-checkpointing",
    "fpo.full-chunk-target",
    "fpo.positive-adv-only",
    "fpo.use-ref-policy-kl",
    "rollout.eval-zero-sample",
    "kl.adaptive",
    "replay.include-demos",
    "sampling.dynamic",
}

KEY_ALIASES = {
    "wandb": "wandb",
    "failure_rewards": "failure-rewards",
    "standard_scaler": "standard-scaler",
}
IGNORED_KEYS = {"hydra", "metadata"}


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(next_prefix, child)
        return
    yield prefix, value


def _key_to_flag(key: str) -> str:
    if key in KEY_ALIASES:
        return KEY_ALIASES[key]
    return ".".join(part.replace("_", "-") for part in key.split("."))


def config_to_train_srpo_args(cfg: DictConfig) -> list[str]:
    """Convert a Hydra config tree to ``scripts/train_srpo.py`` CLI args."""
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict config, got {type(data)!r}")

    args: list[str] = []
    for key, value in _flatten("", data):
        if key in IGNORED_KEYS or key.split(".", 1)[0] in IGNORED_KEYS or value is None:
            continue

        flag = _key_to_flag(key)
        option = f"--{flag}"

        if isinstance(value, bool):
            if value:
                args.append(option)
            elif flag in PAIRED_BOOL_FLAGS:
                args.append(f"--no-{flag}")
            continue

        args.extend([option, str(value)])

    return args


@hydra.main(version_base=None, config_path="../configs/train_srpo", config_name="base")
def main(cfg: DictConfig) -> None:
    args = config_to_train_srpo_args(cfg)
    cmd = [sys.executable, str(TRAIN_SRPO), *args]
    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
