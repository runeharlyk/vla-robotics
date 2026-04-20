from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROBOSUITE_PATCHED = False
_DLL_DIR_HANDLES: list[Any] = []
_LIBERO_ASSET_SUBDIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def _normalize_platform_name(platform_name: str | None = None) -> str:
    return (platform_name or platform.system()).strip().lower()


def _resolve_libero_config_dir(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()

    configured = os.environ.get("LIBERO_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    if _normalize_platform_name() == "windows":
        return (Path(__file__).resolve().parents[3] / ".libero").resolve()

    return (Path.home() / ".libero").resolve()


def _discover_libero_benchmark_root() -> Path | None:
    importlib.invalidate_caches()
    with contextlib.suppress(Exception):
        package_root = _resolve_package_root("libero")
        if package_root is None:
            return None

        benchmark_root = package_root / "libero"
        if benchmark_root.exists():
            return benchmark_root.resolve()
    return None


def _resolve_package_root(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None

    search_locations = list(spec.submodule_search_locations or [])
    if search_locations:
        return Path(search_locations[0]).resolve()
    if spec.origin is not None:
        return Path(spec.origin).parent.resolve()
    return None


def _remember_dll_directory(path: Path) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    with contextlib.suppress(OSError):
        _DLL_DIR_HANDLES.append(add_dll_directory(str(path)))


def _ensure_windows_mujoco_dll() -> Path | None:
    mujoco_root = _resolve_package_root("mujoco")
    robosuite_root = _resolve_package_root("robosuite")
    if mujoco_root is None or robosuite_root is None:
        return None

    source = mujoco_root / "mujoco.dll"
    target_dir = robosuite_root / "utils"
    target = target_dir / "mujoco.dll"
    if not source.exists() or not target_dir.exists():
        return None

    _remember_dll_directory(mujoco_root)
    _remember_dll_directory(target_dir)

    source_size = source.stat().st_size
    target_size = target.stat().st_size if target.exists() else None
    if target_size != source_size:
        shutil.copy2(source, target)

    return target


def _ensure_windows_robosuite_macros_private() -> Path | None:
    robosuite_root = _resolve_package_root("robosuite")
    if robosuite_root is None:
        return None

    target = robosuite_root / "macros_private.py"
    content = (
        "from robosuite.macros import *\n"
        "\n"
        "# Windows uses WGL rather than robosuite's default EGL path.\n"
        "MUJOCO_GPU_RENDERING = False\n"
        "FILE_LOGGING_LEVEL = None\n"
    )

    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")

    return target


def _patch_windows_robosuite_macros_module() -> Path | None:
    robosuite_root = _resolve_package_root("robosuite")
    if robosuite_root is None:
        return None

    target = robosuite_root / "macros.py"
    if not target.exists():
        return None

    content = target.read_text(encoding="utf-8")
    patched = content.replace("MUJOCO_GPU_RENDERING = True", "MUJOCO_GPU_RENDERING = False")
    patched = patched.replace('FILE_LOGGING_LEVEL = "DEBUG"', "FILE_LOGGING_LEVEL = None")

    if patched != content:
        target.write_text(patched, encoding="utf-8")

    return target


def _default_libero_paths(
    benchmark_root: Path,
    config_dir: Path,
    datasets_dir: str | Path | None = None,
) -> dict[str, str]:
    dataset_root = (
        Path(datasets_dir).expanduser().resolve()
        if datasets_dir is not None
        else Path(os.environ.get("LIBERO_DATASETS_PATH", config_dir / "datasets")).expanduser().resolve()
    )
    dataset_root.mkdir(parents=True, exist_ok=True)

    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str(dataset_root),
        "assets": str((benchmark_root / "assets").resolve()),
    }


def ensure_libero_config(
    *,
    config_dir: str | Path | None = None,
    datasets_dir: str | Path | None = None,
) -> Path | None:
    benchmark_root = _discover_libero_benchmark_root()
    if benchmark_root is None:
        return None

    config_root = _resolve_libero_config_dir(config_dir)
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)

    config_file = config_root / "config.yaml"
    desired = _default_libero_paths(benchmark_root, config_root, datasets_dir)

    current: dict[str, Any] = {}
    if config_file.exists():
        with contextlib.suppress(Exception):
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded

    merged = {
        "benchmark_root": desired["benchmark_root"],
        "bddl_files": desired["bddl_files"],
        "init_states": desired["init_states"],
        "datasets": current.get("datasets") or desired["datasets"],
        "assets": desired["assets"],
    }

    if merged != current:
        config_file.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

    return config_file


def _safe_close_env(env: Any) -> None:
    try:
        env.close()
    except BaseException:
        logger.debug("Ignoring LIBERO env close failure during shutdown", exc_info=True)


def _patch_robosuite() -> None:
    global _ROBOSUITE_PATCHED
    if _ROBOSUITE_PATCHED:
        return

    try:
        from robosuite.renderers.context.egl_context import EGLGLContext
        from robosuite.utils.binding_utils import MjRenderContext
    except Exception:
        return

    orig_mj_del = getattr(MjRenderContext, "__del__", None)
    if callable(orig_mj_del):

        def _safe_mj_del(self: object) -> None:
            if not hasattr(self, "con"):
                return
            with contextlib.suppress(Exception):
                orig_mj_del(self)

        MjRenderContext.__del__ = _safe_mj_del

    orig_egl_free = getattr(EGLGLContext, "free", None)
    if callable(orig_egl_free):

        def _safe_egl_free(self: object) -> None:
            with contextlib.suppress(Exception):
                orig_egl_free(self)

        EGLGLContext.free = _safe_egl_free

    orig_egl_del = getattr(EGLGLContext, "__del__", None)
    if callable(orig_egl_del):

        def _safe_egl_del(self: object) -> None:
            with contextlib.suppress(Exception):
                orig_egl_del(self)

        EGLGLContext.__del__ = _safe_egl_del

    _ROBOSUITE_PATCHED = True


def probe_libero_runtime(platform_name: str | None = None) -> dict[str, Any]:
    system = _normalize_platform_name(platform_name)
    benchmark_root = _discover_libero_benchmark_root()
    assets_path = benchmark_root / "assets" if benchmark_root is not None else None
    assets_ready = assets_path is not None and all((assets_path / subdir).exists() for subdir in _LIBERO_ASSET_SUBDIRS)
    config_dir = _resolve_libero_config_dir()
    config_file = config_dir / "config.yaml"
    default_mujoco_gl = "wgl" if system == "windows" else os.environ.get("MUJOCO_GL", "")

    return {
        "platform": system,
        "benchmark_root": benchmark_root,
        "assets_path": assets_path,
        "assets_ready": assets_ready,
        "config_file": config_file,
        "ready": benchmark_root is not None and assets_ready,
        "mujoco_gl": os.environ.get("MUJOCO_GL", default_mujoco_gl),
    }


def configure_libero_runtime(
    *,
    platform_name: str | None = None,
    config_dir: str | Path | None = None,
    datasets_dir: str | Path | None = None,
) -> dict[str, Any]:
    system = _normalize_platform_name(platform_name)
    if system == "windows":
        os.environ.setdefault("MUJOCO_GL", "wgl")
        _ensure_windows_mujoco_dll()
        _patch_windows_robosuite_macros_module()
        _ensure_windows_robosuite_macros_private()

    config_file = ensure_libero_config(config_dir=config_dir, datasets_dir=datasets_dir)
    _patch_robosuite()

    info = probe_libero_runtime(system)
    if config_file is not None:
        info["config_file"] = config_file
    return info


__all__ = [
    "_patch_robosuite",
    "_safe_close_env",
    "configure_libero_runtime",
    "ensure_libero_config",
    "probe_libero_runtime",
]
