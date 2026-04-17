from __future__ import annotations

import json
import os
import shutil
import sysconfig
import tarfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import typer

from vla.envs.libero_runtime import configure_libero_runtime, probe_libero_runtime

LIBERO_ASSETS_REPO_ID = "jadechoghari/libero-assets"
LIBERO_ASSET_SUBDIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def _download_libero_archive(version: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"libero-{version}.tar.gz"
    if archive_path.exists():
        return archive_path

    metadata_url = f"https://pypi.org/pypi/libero/{version}/json"
    with urlopen(metadata_url) as response:
        metadata = json.load(response)

    sdist = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    urlretrieve(sdist["url"], archive_path)
    return archive_path


def _site_packages_dir() -> Path:
    return Path(sysconfig.get_paths()["purelib"]).resolve()


def _remove_existing_libero_install(site_packages: Path) -> None:
    shutil.rmtree(site_packages / "libero", ignore_errors=True)

    for pattern in ("libero-*.dist-info", "__editable__.libero-*.pth", "__editable___libero_*_finder.py"):
        for path in site_packages.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def _install_libero_package_tree(archive_path: Path, site_packages: Path) -> Path:
    package_root = site_packages / "libero"
    package_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        package_prefix = next(
            member.name for member in members if member.isdir() and member.name.endswith("/libero")
        ).rstrip("/")
        package_prefix = f"{package_prefix}/"

        for member in members:
            if not member.name.startswith(package_prefix):
                continue
            relative_path = Path(member.name[len(package_prefix) :])
            if not relative_path.parts:
                continue
            if any(part == ".." for part in relative_path.parts):
                raise RuntimeError(f"Refusing to extract unsafe path from archive: {member.name}")

            destination = package_root / relative_path
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Failed to extract archive member: {member.name}")
            with source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)

    return package_root


def _install_libero_from_archive(archive_path: Path) -> Path:
    site_packages = _site_packages_dir()
    _remove_existing_libero_install(site_packages)
    return _install_libero_package_tree(archive_path, site_packages)


def _assets_present(assets_dir: Path) -> bool:
    return all((assets_dir / subdir).exists() for subdir in LIBERO_ASSET_SUBDIRS)


def _install_libero_assets(package_root: Path) -> Path:
    assets_dir = package_root / "libero" / "assets"
    if _assets_present(assets_dir):
        return assets_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download LIBERO assets.") from exc

    hf_home = Path(os.environ.get("HF_HOME", Path(".hf-cache").resolve()))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    assets_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        snapshot_download(
            repo_id=LIBERO_ASSETS_REPO_ID,
            repo_type="model",
            cache_dir=str(hf_home / "hub"),
        )
    )

    for candidate in (snapshot_root, snapshot_root / "assets"):
        if not _assets_present(candidate):
            continue
        for child in candidate.iterdir():
            destination = assets_dir / child.name
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
            elif child.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination)
        break

    if not _assets_present(assets_dir):
        raise RuntimeError(
            f"Downloaded LIBERO assets snapshot at {snapshot_root}, but required asset folders were not found."
        )

    return assets_dir


def main(
    config_dir: str | None = typer.Option(None, "--config-dir"),
    datasets_dir: str | None = typer.Option(None, "--datasets-dir"),
    install: bool = typer.Option(
        False,
        "--install",
        help="Download LIBERO source and install it into the active venv.",
    ),
    libero_version: str = typer.Option("0.1.1", "--libero-version"),
    source_dir: str | None = typer.Option(None, "--source-dir", help="Use an existing local LIBERO source tree."),
) -> None:
    probe = probe_libero_runtime()
    if install and not probe["ready"]:
        if source_dir is not None:
            source_root = Path(source_dir).expanduser().resolve()
            if not (source_root / "libero").exists():
                raise RuntimeError(f"Expected a 'libero' package under {source_root}")
            print(f"Installing LIBERO package tree from {source_root}")
            _remove_existing_libero_install(_site_packages_dir())
            installed_root = _site_packages_dir() / "libero"
            shutil.copytree(source_root / "libero", installed_root, dirs_exist_ok=True)
        else:
            archive_path = _download_libero_archive(libero_version, Path(".libero-src-cache"))
            installed_root = _install_libero_from_archive(archive_path)
            print(f"Installed LIBERO package tree to {installed_root}")

        assets_dir = _install_libero_assets(installed_root)
        print(f"Installed LIBERO assets to {assets_dir}")

    info = configure_libero_runtime(config_dir=config_dir, datasets_dir=datasets_dir)

    print("LIBERO runtime configuration")
    print(f"  Platform: {info['platform']}")
    print(f"  MUJOCO_GL: {info['mujoco_gl'] or '<unset>'}")
    print(f"  Benchmark root: {info['benchmark_root'] or '<libero not installed>'}")
    print(f"  Assets path: {info['assets_path'] or '<assets missing>'}")
    print(f"  Config file: {info['config_file'] or '<not created>'}")

    config_file = info.get("config_file")
    if config_file and info.get("benchmark_root"):
        print("\nRuntime is ready. You can now run:")
        print(
            "  uv run python -m vla visualize --checkpoint HuggingFaceVLA/smolvla_libero "
            "--simulator libero --suite spatial"
        )
        print(f"  Videos will be written under {Path('videos').resolve()}")
    else:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
