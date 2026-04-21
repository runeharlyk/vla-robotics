from __future__ import annotations

import shutil
from pathlib import Path

from vla.envs.robocasa_runtime import configure_robocasa_runtime, probe_robocasa_runtime


def _fresh_case_dir(case_name: str) -> Path:
    root = Path("tests/.tmp") / case_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_source_tree(root: Path) -> Path:
    source_root = root / ".robocasa-src"
    package_root = source_root / "robocasa"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "macros.py").write_text("# upstream macros\n", encoding="utf-8")

    for rel in (
        "models/assets/textures",
        "models/assets/fixtures",
        "models/assets/objects/objaverse",
    ):
        (package_root / rel).mkdir(parents=True, exist_ok=True)

    return source_root


def test_configure_robocasa_runtime_bootstraps_workspace(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("robocasa_runtime_case1")
    try:
        source_root = _make_source_tree(tmp_path)
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setattr("vla.envs.robocasa_runtime._module_version", lambda name: {
            "numpy": "2.2.5",
            "mujoco": "3.3.1",
            "robosuite": "1.5.2",
        }.get(name, ""))
        monkeypatch.setattr("vla.envs.robocasa_runtime._has_composite_controller_api", lambda: True)
        monkeypatch.setattr("vla.envs.robocasa_runtime._ensure_windows_mujoco_dll", lambda: None)
        monkeypatch.setattr("vla.envs.robocasa_runtime._patch_windows_robosuite_macros_module", lambda: None)
        monkeypatch.setattr("vla.envs.robocasa_runtime._ensure_windows_robosuite_macros_private", lambda: None)
        monkeypatch.setattr("vla.envs.robocasa_runtime._patch_robosuite", lambda: None)

        info = configure_robocasa_runtime(platform_name="Windows", source_dir=source_root)

        assert info["mujoco_gl"] == "wgl"
        assert info["install_mode"] == "source"
        assert info["workspace_ready"] is True
        assert info["ready"] is True
        assert (source_root / "robocasa" / "macros_private.py").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_probe_robocasa_runtime_reports_version_conflicts(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("robocasa_runtime_case2")
    try:
        source_root = _make_source_tree(tmp_path)
        (source_root / "robocasa" / "macros_private.py").write_text("# private macros\n", encoding="utf-8")

        monkeypatch.setattr("vla.envs.robocasa_runtime._module_version", lambda name: {
            "numpy": "2.2.6",
            "mujoco": "3.3.1",
            "robosuite": "1.4.0",
        }.get(name, ""))
        monkeypatch.setattr("vla.envs.robocasa_runtime._has_composite_controller_api", lambda: False)

        info = probe_robocasa_runtime(platform_name="Windows", source_dir=source_root)

        assert info["workspace_ready"] is True
        assert info["ready"] is False
        assert info["compatibility"]["numpy_ok"] is False
        assert info["compatibility"]["robosuite_ok"] is False
        assert info["compatibility"]["composite_controller_api"] is False
        assert info["compatibility"]["issues"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
