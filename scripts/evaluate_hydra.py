"""Hydra wrapper for evaluation configs.

Like ``scripts/train_srpo_hydra.py``, this keeps the existing Typer
evaluation script as the implementation and uses Hydra only for composing
repeatable eval recipes.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = REPO_ROOT / "scripts" / "evaluate.py"

PAIRED_BOOL_FLAGS = {"wandb"}
IGNORED_KEYS = {"hydra", "metadata", "evaluations"}


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(next_prefix, child)
        return
    yield prefix, value


def _key_to_flag(key: str) -> str:
    return ".".join(part.replace("_", "-") for part in key.split("."))


def config_to_evaluate_args(cfg: dict[str, Any] | DictConfig) -> list[str]:
    """Convert one composed eval config to ``scripts/evaluate.py`` CLI args."""
    data = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
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


def expand_eval_configs(cfg: DictConfig) -> list[dict[str, Any]]:
    """Return one concrete config per eval run.

    If ``evaluations`` is present, each item is merged over the common
    top-level config.  This is useful for protocols that compare SFT and
    several RL checkpoints under identical eval settings.
    """
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict config, got {type(data)!r}")

    evaluations = data.get("evaluations")
    if not evaluations:
        return [data]

    base = {k: deepcopy(v) for k, v in data.items() if k != "evaluations"}
    expanded: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise TypeError(f"Expected evaluation entry to be dict, got {type(evaluation)!r}")
        item = deepcopy(base)
        item.update(evaluation)
        expanded.append(item)
    return expanded


@hydra.main(version_base=None, config_path="../configs/evaluate", config_name="base")
def main(cfg: DictConfig) -> None:
    for eval_cfg in expand_eval_configs(cfg):
        args = config_to_evaluate_args(eval_cfg)
        cmd = [sys.executable, str(EVALUATE), *args]
        print("Launching:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
