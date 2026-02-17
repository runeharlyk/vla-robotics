"""
Download ManiSkill demonstration data into data/raw.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --skill PickCube-v1
"""

import subprocess
import sys
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SKILL = "PickCube-v1"


def main(
    skill: str = typer.Option(DEFAULT_SKILL, "--skill", "-s", help="ManiSkill env ID (e.g. PickCube-v1)"),
    output_dir: Path = typer.Option(RAW_DIR, "--output-dir", "-o", path_type=Path, help="Directory to save demos"),
) -> None:
    """Download ManiSkill demonstrations for the given skill to data/raw."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mani_skill.utils.download_demo",
        skill,
        "-o",
        str(output_dir),
    ]
    typer.echo(f"Downloading {skill} to {output_dir} ...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    typer.echo(f"Done. Raw data in {output_dir}")


if __name__ == "__main__":
    typer.run(main)
