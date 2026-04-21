from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from vla.constants import PROJECT_ROOT
from vla.envs.libero_runtime import (
    _ensure_windows_mujoco_dll,
    _ensure_windows_robosuite_macros_private,
    _patch_robosuite,
    _patch_windows_robosuite_macros_module,
)

ROBOCASA_REQUIRED_NUMPY = "2.2.5"
ROBOCASA_REQUIRED_MUJOCO = "3.3.1"
ROBOCASA_MIN_ROBOSUITE = (1, 5, 2)
ROBOCASA_ASSET_SUBDIRS = (
    "models/assets/textures",
    "models/assets/generative_textures",
    "models/assets/fixtures",
    "models/assets/objects",
    "models/assets/objects/objaverse",
)


def _normalize_platform_name(platform_name: str | None = None) -> str:
    return (platform_name or platform.system()).strip().lower()


def _resolve_robocasa_source_dir(source_dir: str | Path | None = None) -> Path:
    if source_dir is not None:
        return Path(source_dir).expanduser().resolve()

    configured = os.environ.get("ROBOCASA_SOURCE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    return (PROJECT_ROOT / ".robocasa-src").resolve()


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


def _resolve_robocasa_package_root(
    source_dir: str | Path | None = None,
) -> tuple[Path | None, str]:
    source_root = _resolve_robocasa_source_dir(source_dir)
    package_root = source_root / "robocasa"
    if package_root.exists():
        return package_root.resolve(), "source"

    importlib.invalidate_caches()
    installed_root = _resolve_package_root("robocasa")
    if installed_root is not None:
        return installed_root, "site-packages"

    return None, "missing"


def _assets_present(package_root: Path) -> bool:
    return all((package_root / rel).exists() for rel in ROBOCASA_ASSET_SUBDIRS)


def ensure_robocasa_macros(package_root: str | Path) -> Path | None:
    root = Path(package_root).expanduser().resolve()
    macros_path = root / "macros.py"
    macros_private_path = root / "macros_private.py"
    if not macros_path.exists():
        return None

    if not macros_private_path.exists():
        macros_private_path.write_text(macros_path.read_text(encoding="utf-8"), encoding="utf-8")

    return macros_private_path


def _module_version(package_name: str) -> str:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(package_name)

    with contextlib.suppress(Exception):
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "")
        if isinstance(version, str):
            return version

    return ""


def _parse_version(version: str) -> tuple[int, ...]:
    numbers = [int(piece) for piece in re.findall(r"\d+", version)]
    return tuple(numbers)


def _version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    if not version:
        return False
    parsed = _parse_version(version)
    if not parsed:
        return False
    if len(parsed) < len(minimum):
        parsed = parsed + (0,) * (len(minimum) - len(parsed))
    return parsed >= minimum


def _has_composite_controller_api() -> bool:
    try:
        controllers_mod = importlib.import_module("robosuite.controllers")
    except Exception:
        return False
    return hasattr(controllers_mod, "load_composite_controller_config")


def _compatibility_status() -> dict[str, Any]:
    numpy_version = _module_version("numpy")
    mujoco_version = _module_version("mujoco")
    robosuite_version = _module_version("robosuite")
    composite_controller_api = _has_composite_controller_api()

    numpy_ok = numpy_version == ROBOCASA_REQUIRED_NUMPY
    mujoco_ok = mujoco_version == ROBOCASA_REQUIRED_MUJOCO
    robosuite_ok = _version_at_least(robosuite_version, ROBOCASA_MIN_ROBOSUITE)

    issues: list[str] = []
    if not numpy_ok:
        issues.append(f"numpy=={ROBOCASA_REQUIRED_NUMPY} required (found {numpy_version or 'missing'})")
    if not mujoco_ok:
        issues.append(f"mujoco=={ROBOCASA_REQUIRED_MUJOCO} required (found {mujoco_version or 'missing'})")
    if not robosuite_ok:
        issues.append(
            "robosuite>="
            f"{'.'.join(str(part) for part in ROBOCASA_MIN_ROBOSUITE)} required "
            f"(found {robosuite_version or 'missing'})"
        )
    if not composite_controller_api:
        issues.append("robosuite composite-controller API is missing in the active Python environment")

    return {
        "numpy_version": numpy_version,
        "mujoco_version": mujoco_version,
        "robosuite_version": robosuite_version,
        "numpy_ok": numpy_ok,
        "mujoco_ok": mujoco_ok,
        "robosuite_ok": robosuite_ok,
        "composite_controller_api": composite_controller_api,
        "overall": numpy_ok and mujoco_ok and robosuite_ok and composite_controller_api,
        "issues": issues,
    }


def probe_robocasa_runtime(
    platform_name: str | None = None,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    system = _normalize_platform_name(platform_name)
    source_root = _resolve_robocasa_source_dir(source_dir)
    package_root, install_mode = _resolve_robocasa_package_root(source_root)
    assets_path = package_root / "models" / "assets" if package_root is not None else None
    assets_ready = package_root is not None and _assets_present(package_root)
    macros_file = package_root / "macros_private.py" if package_root is not None else None
    macros_ready = bool(macros_file is not None and macros_file.exists())
    compatibility = _compatibility_status()
    workspace_ready = package_root is not None and assets_ready and macros_ready
    default_mujoco_gl = "wgl" if system == "windows" else os.environ.get("MUJOCO_GL", "")

    return {
        "platform": system,
        "source_root": source_root,
        "package_root": package_root,
        "install_mode": install_mode,
        "assets_path": assets_path,
        "assets_ready": assets_ready,
        "macros_file": macros_file,
        "macros_ready": macros_ready,
        "workspace_ready": workspace_ready,
        "compatibility": compatibility,
        "ready": workspace_ready and compatibility["overall"],
        "mujoco_gl": os.environ.get("MUJOCO_GL", default_mujoco_gl),
    }


def configure_robocasa_runtime(
    *,
    platform_name: str | None = None,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    system = _normalize_platform_name(platform_name)
    source_root = _resolve_robocasa_source_dir(source_dir)
    package_root = source_root / "robocasa"
    if package_root.exists():
        os.environ.setdefault("ROBOCASA_SOURCE_PATH", str(source_root))
        source_parent = str(source_root)
        if source_parent not in sys.path:
            sys.path.insert(0, source_parent)

    if system == "windows":
        os.environ.setdefault("MUJOCO_GL", "wgl")
        _ensure_windows_mujoco_dll()
        _patch_windows_robosuite_macros_module()
        _ensure_windows_robosuite_macros_private()

    resolved_package_root, _ = _resolve_robocasa_package_root(source_root)
    if resolved_package_root is not None:
        ensure_robocasa_macros(resolved_package_root)

    _patch_robosuite()
    return probe_robocasa_runtime(platform_name=system, source_dir=source_root)


__all__ = [
    "ROBOCASA_ASSET_SUBDIRS",
    "ROBOCASA_MIN_ROBOSUITE",
    "ROBOCASA_REQUIRED_MUJOCO",
    "ROBOCASA_REQUIRED_NUMPY",
    "configure_robocasa_runtime",
    "ensure_robocasa_macros",
    "probe_robocasa_runtime",
]
