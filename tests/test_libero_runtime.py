from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from vla.envs.libero_runtime import (
    configure_libero_runtime,
    probe_libero_runtime,
)


def _fake_libero_spec(package_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        submodule_search_locations=[str(package_root)],
        origin=str(package_root / "__init__.py"),
    )


def _fake_package_spec(package_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        submodule_search_locations=[str(package_root)],
        origin=str(package_root / "__init__.py"),
    )


def _fresh_case_dir(case_name: str) -> Path:
    root = Path("tests/.tmp") / case_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_configure_libero_runtime_bootstraps_windows_config(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("libero_runtime_case1")
    try:
        package_root = tmp_path / "site-packages" / "libero"
        benchmark_root = package_root / "libero"
        for rel in ("bddl_files", "init_files", "assets"):
            (benchmark_root / rel).mkdir(parents=True, exist_ok=True)

        config_dir = tmp_path / "cfg"
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
        monkeypatch.setattr(
            "vla.envs.libero_runtime.importlib.util.find_spec",
            lambda name: _fake_libero_spec(package_root),
        )
        monkeypatch.setattr("vla.envs.libero_runtime._patch_robosuite", lambda: None)

        info = configure_libero_runtime(platform_name="Windows", config_dir=config_dir)

        assert info["mujoco_gl"] == "wgl"
        config_file = config_dir / "config.yaml"
        assert config_file.exists()

        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert config["benchmark_root"] == str(benchmark_root.resolve())
        assert config["bddl_files"] == str((benchmark_root / "bddl_files").resolve())
        assert config["init_states"] == str((benchmark_root / "init_files").resolve())
        assert config["assets"] == str((benchmark_root / "assets").resolve())
        assert Path(config["datasets"]).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_configure_libero_runtime_preserves_explicit_mujoco_gl(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("libero_runtime_case2")
    try:
        package_root = tmp_path / "site-packages" / "libero"
        benchmark_root = package_root / "libero"
        benchmark_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("MUJOCO_GL", "glfw")
        monkeypatch.setattr(
            "vla.envs.libero_runtime.importlib.util.find_spec",
            lambda name: _fake_libero_spec(package_root),
        )
        monkeypatch.setattr("vla.envs.libero_runtime._patch_robosuite", lambda: None)

        info = configure_libero_runtime(platform_name="Windows", config_dir=tmp_path / "cfg")

        assert info["mujoco_gl"] == "glfw"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_probe_libero_runtime_reports_missing_install(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("libero_runtime_case3")
    try:
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "cfg"))
        monkeypatch.setattr("vla.envs.libero_runtime.importlib.util.find_spec", lambda name: None)

        info = probe_libero_runtime(platform_name="Windows")

        assert info["ready"] is False
        assert info["benchmark_root"] is None
        assert info["mujoco_gl"] == "wgl"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_windows_default_config_dir_is_repo_local(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.setattr("vla.envs.libero_runtime.platform.system", lambda: "Windows")

    info = probe_libero_runtime()

    assert info["config_file"] == (Path.cwd() / ".libero" / "config.yaml").resolve()


def test_configure_libero_runtime_stages_windows_mujoco_dll(monkeypatch) -> None:
    tmp_path = _fresh_case_dir("libero_runtime_case4")
    try:
        libero_root = tmp_path / "site-packages" / "libero"
        benchmark_root = libero_root / "libero"
        for rel in (
            "bddl_files",
            "init_files",
            "assets/articulated_objects",
            "assets/scenes",
            "assets/stable_hope_objects",
            "assets/stable_scanned_objects",
            "assets/textures",
            "assets/turbosquid_objects",
        ):
            (benchmark_root / rel).mkdir(parents=True, exist_ok=True)

        mujoco_root = tmp_path / "site-packages" / "mujoco"
        mujoco_root.mkdir(parents=True, exist_ok=True)
        source_dll = mujoco_root / "mujoco.dll"
        source_dll.write_bytes(b"fake-mujoco-dll")

        robosuite_root = tmp_path / "site-packages" / "robosuite"
        robosuite_utils = robosuite_root / "utils"
        robosuite_utils.mkdir(parents=True, exist_ok=True)
        (robosuite_root / "macros.py").write_text(
            'MUJOCO_GPU_RENDERING = True\nFILE_LOGGING_LEVEL = "DEBUG"\n',
            encoding="utf-8",
        )

        def _find_spec(name: str):
            mapping = {
                "libero": _fake_package_spec(libero_root),
                "mujoco": _fake_package_spec(mujoco_root),
                "robosuite": _fake_package_spec(robosuite_root),
            }
            return mapping.get(name)

        monkeypatch.setattr("vla.envs.libero_runtime.importlib.util.find_spec", _find_spec)
        monkeypatch.setattr("vla.envs.libero_runtime._patch_robosuite", lambda: None)

        info = configure_libero_runtime(platform_name="Windows", config_dir=tmp_path / "cfg")

        assert info["ready"] is True
        staged_dll = robosuite_utils / "mujoco.dll"
        assert staged_dll.read_bytes() == source_dll.read_bytes()
        macros_private = robosuite_utils.parent / "macros_private.py"
        assert macros_private.exists()
        assert "MUJOCO_GPU_RENDERING = False" in macros_private.read_text(encoding="utf-8")
        assert "FILE_LOGGING_LEVEL = None" in macros_private.read_text(encoding="utf-8")
        macros_module = robosuite_root / "macros.py"
        macros_text = macros_module.read_text(encoding="utf-8")
        assert "MUJOCO_GPU_RENDERING = False" in macros_text
        assert "FILE_LOGGING_LEVEL = None" in macros_text
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
