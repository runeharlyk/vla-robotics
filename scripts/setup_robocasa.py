from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import typer
from huggingface_hub import hf_hub_download

from vla.envs.robocasa_runtime import (
    ROBOCASA_ASSET_SUBDIRS,
    configure_robocasa_runtime,
    ensure_robocasa_macros,
    probe_robocasa_runtime,
)

ROBOCASA_ASSET_DOWNLOADS = {
    "textures": (
        "robocasa/robocasa-assets",
        "textures.zip",
        "models/assets/textures",
    ),
    "generative_textures": (
        "robocasa/robocasa-assets",
        "generative_textures.zip",
        "models/assets/generative_textures",
    ),
    "fixtures_lightwheel": (
        "nvidia/PhysicalAI-Kitchen-Assets",
        "fixtures_lightwheel.zip",
        "models/assets/fixtures",
    ),
    "objaverse": (
        "robocasa/robocasa-assets",
        "objaverse.zip",
        "models/assets/objects/objaverse",
    ),
    "objects_lightwheel": (
        "nvidia/PhysicalAI-Kitchen-Assets",
        "objects_lightwheel.zip",
        "models/assets/objects/lightwheel",
    ),
}


def _download_robocasa_archive(git_ref: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"robocasa-{git_ref}.tar.gz"
    if archive_path.exists():
        return archive_path

    archive_url = f"https://github.com/robocasa/robocasa/archive/refs/tags/{git_ref}.tar.gz"
    urlretrieve(archive_url, archive_path)
    return archive_path


def _install_repo_tree(archive_path: Path, source_root: Path) -> Path:
    shutil.rmtree(source_root, ignore_errors=True)
    source_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        root_prefix = next(member.name.split("/", 1)[0] for member in members if "/" in member.name)
        prefix = f"{root_prefix}/"

        for member in members:
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix) :]
            if not relative_name:
                continue

            destination = source_root / relative_name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract archive member: {member.name}")
            with extracted, destination.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)

    return source_root / "robocasa"


def _assets_present(package_root: Path) -> bool:
    return all((package_root / rel).exists() for rel in ROBOCASA_ASSET_SUBDIRS)


def _download_and_extract_hf_zip(repo_id: str, filename: str, target_dir: Path, cache_dir: Path) -> None:
    from zipfile import ZipFile

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / filename
    if not archive_path.exists():
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision="main",
        )
        shutil.copyfile(downloaded, archive_path)

    with ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(path=target_dir.parent)


def _install_assets(package_root: Path, cache_dir: Path) -> None:
    if _assets_present(package_root):
        return

    for asset_name, (repo_id, filename, rel_dir) in ROBOCASA_ASSET_DOWNLOADS.items():
        target_dir = package_root / rel_dir
        if target_dir.exists():
            continue
        print(f"Downloading RoboCasa asset pack: {asset_name}")
        _download_and_extract_hf_zip(repo_id, filename, target_dir, cache_dir)

    if not _assets_present(package_root):
        raise RuntimeError(
            "RoboCasa assets are still incomplete after download. "
            "Expected textures, fixtures, and objaverse folders under models/assets."
        )


def _patch_local_assets(source_root: Path) -> None:
    script_path = Path("scripts/fix_robocasa_gentex.py").resolve()
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path.cwd(),
        check=True,
    )


def main(
    source_dir: str | None = typer.Option(None, "--source-dir", help="Repo-local RoboCasa source checkout path."),
    install: bool = typer.Option(
        False,
        "--install",
        help="Download the official RoboCasa source archive and the required kitchen assets.",
    ),
    robocasa_ref: str = typer.Option("v1.0", "--robocasa-ref", help="Official RoboCasa git tag to download."),
) -> None:
    probe = probe_robocasa_runtime(source_dir=source_dir)
    source_root = probe["source_root"]

    if install:
        if not probe["workspace_ready"]:
            archive_path = _download_robocasa_archive(robocasa_ref, Path(".robocasa-src-cache"))
            package_root = _install_repo_tree(archive_path, source_root)
            ensure_robocasa_macros(package_root)
            _install_assets(package_root, Path(".robocasa-asset-cache"))
        _patch_local_assets(source_root)

    info = configure_robocasa_runtime(source_dir=source_root)
    compatibility = info["compatibility"]

    print("RoboCasa runtime configuration")
    print(f"  Platform: {info['platform']}")
    print(f"  MUJOCO_GL: {info['mujoco_gl'] or '<unset>'}")
    print(f"  Source root: {info['source_root']}")
    print(f"  Package root: {info['package_root'] or '<missing>'}")
    print(f"  Install mode: {info['install_mode']}")
    print(f"  Assets path: {info['assets_path'] or '<missing>'}")
    print(f"  Macros file: {info['macros_file'] or '<missing>'}")
    print(f"  Workspace ready: {info['workspace_ready']}")
    print("")
    print("Active Python compatibility")
    print(f"  numpy: {compatibility['numpy_version'] or '<missing>'} (need 2.2.5)")
    print(f"  mujoco: {compatibility['mujoco_version'] or '<missing>'} (need 3.3.1)")
    print(f"  robosuite: {compatibility['robosuite_version'] or '<missing>'} (need >= 1.5.2)")
    print(f"  composite controller API: {compatibility['composite_controller_api']}")

    if info["ready"]:
        print("\nWorkspace files and the active Python stack are both compatible.")
        print("You can now use this source tree from a dedicated RoboCasa environment.")
        return

    if info["workspace_ready"] and not compatibility["overall"]:
        print("\nWorkspace files are ready, but the current Python environment is not RoboCasa-compatible.")
        print("This repo's shared LIBERO / ManiSkill environment uses different versions than RoboCasa 1.0 expects.")
        print("Recommended next step:")
        print(f"  Create a separate Python 3.11 environment and point ROBOCASA_SOURCE_PATH at {info['source_root']}")
        for issue in compatibility["issues"]:
            print(f"  - {issue}")
        raise typer.Exit(code=0)

    print("\nRoboCasa source/assets are not ready in this workspace yet.")
    print("Run this script with --install to download the official source tree and kitchen assets.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
