from __future__ import annotations

import os
from pathlib import Path

from vla.constants import PROJECT_ROOT


def repo_local_runtime_env(
    *,
    project_root: Path | None = None,
    platform_name: str | None = None,
) -> dict[str, str]:
    root = (project_root or PROJECT_ROOT).resolve()
    system = (platform_name or os.name).lower()
    if system not in {"nt", "windows"}:
        return {}

    tmp_dir = root / ".tmp"
    return {
        "HF_HOME": str((root / ".hf-cache").resolve()),
        "WANDB_DIR": str((root / ".wandb").resolve()),
        "TEMP": str(tmp_dir.resolve()),
        "TMP": str(tmp_dir.resolve()),
    }


def apply_repo_local_runtime_env(
    *,
    project_root: Path | None = None,
    platform_name: str | None = None,
) -> dict[str, str]:
    env = repo_local_runtime_env(project_root=project_root, platform_name=platform_name)
    for value in env.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


__all__ = ["apply_repo_local_runtime_env", "repo_local_runtime_env"]
