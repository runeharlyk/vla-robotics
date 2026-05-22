from __future__ import annotations

import os
import shutil
from pathlib import Path

from vla.utils.runtime_env import apply_repo_local_runtime_env, repo_local_runtime_env


def _fresh_runtime_env_root(case_name: str) -> Path:
    root = Path("tests/.tmp") / case_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_repo_local_runtime_env_is_empty_off_windows() -> None:
    assert repo_local_runtime_env(project_root=Path("repo"), platform_name="posix") == {}


def test_apply_repo_local_runtime_env_bootstraps_repo_local_dirs(monkeypatch) -> None:
    root = _fresh_runtime_env_root("runtime_env_case1")
    try:
        for key in ("HF_HOME", "WANDB_DIR", "TEMP", "TMP"):
            monkeypatch.delenv(key, raising=False)

        env = apply_repo_local_runtime_env(project_root=root, platform_name="Windows")

        assert Path(env["HF_HOME"]).exists()
        assert Path(env["WANDB_DIR"]).exists()
        assert Path(env["TEMP"]).exists()
        assert Path(env["TMP"]).exists()
        assert os.environ["HF_HOME"] == env["HF_HOME"]
        assert os.environ["WANDB_DIR"] == env["WANDB_DIR"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_apply_repo_local_runtime_env_preserves_explicit_env(monkeypatch) -> None:
    root = _fresh_runtime_env_root("runtime_env_case2")
    try:
        explicit_hf = str((root / "external-hf").resolve())
        monkeypatch.setenv("HF_HOME", explicit_hf)
        monkeypatch.delenv("WANDB_DIR", raising=False)
        monkeypatch.delenv("TEMP", raising=False)
        monkeypatch.delenv("TMP", raising=False)

        env = apply_repo_local_runtime_env(project_root=root, platform_name="Windows")

        assert os.environ["HF_HOME"] == explicit_hf
        assert env["HF_HOME"] != explicit_hf
        assert os.environ["WANDB_DIR"] == env["WANDB_DIR"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
