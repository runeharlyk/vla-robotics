from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def _load_setup_libero_module():
    module_path = Path("scripts/setup_libero.py").resolve()
    spec = importlib.util.spec_from_file_location("setup_libero_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fresh_case_dir(case_name: str) -> Path:
    root = Path("tests/.tmp") / case_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_resolve_source_tree_root_accepts_repo_root() -> None:
    module = _load_setup_libero_module()
    tmp_path = _fresh_case_dir("setup_libero_case1")
    try:
        source_root = tmp_path / "LIBERO-plus"
        (source_root / "libero").mkdir(parents=True, exist_ok=True)
        (source_root / "libero" / "__init__.py").write_text("", encoding="utf-8")

        assert module._resolve_source_tree_root(source_root) == source_root.resolve()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_resolve_source_tree_root_accepts_inner_package_dir() -> None:
    module = _load_setup_libero_module()
    tmp_path = _fresh_case_dir("setup_libero_case2")
    try:
        source_root = tmp_path / "LIBERO-plus"
        package_dir = source_root / "libero"
        package_dir.mkdir(parents=True, exist_ok=True)
        (source_root / "setup.py").write_text("", encoding="utf-8")
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

        assert module._resolve_source_tree_root(package_dir) == source_root.resolve()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_install_libero_from_source_tree_preserves_nested_layout(monkeypatch) -> None:
    module = _load_setup_libero_module()
    tmp_path = _fresh_case_dir("setup_libero_case3")
    try:
        source_root = tmp_path / "LIBERO-plus"
        site_packages = tmp_path / "site-packages"
        package_dir = source_root / "libero"
        package_dir.mkdir(parents=True, exist_ok=True)
        (source_root / "setup.py").write_text("", encoding="utf-8")
        (package_dir / "__init__.py").write_text("# inner package\n", encoding="utf-8")
        (package_dir / "benchmark").mkdir(parents=True, exist_ok=True)
        (package_dir / "benchmark" / "__init__.py").write_text("", encoding="utf-8")

        monkeypatch.setattr(module, "_site_packages_dir", lambda: site_packages.resolve())

        installed_root = module._install_libero_from_source_tree(source_root)

        assert installed_root == (site_packages / "libero").resolve()
        assert not (installed_root / "__init__.py").exists()
        assert (installed_root / "libero" / "__init__.py").exists()
        assert (installed_root / "libero" / "benchmark" / "__init__.py").exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
