"""
Preprocess only Panda-based ManiSkill demos into HDF5 files.

Uses the updated preprocess_data.py pipeline with:
- instruction mapping
- JPEG compression
- multi-arm support
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
PREPROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_data.py"


PANDA_ENVS = {
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PushCube-v1",
    "PushT-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
    "PokeCube-v1",
    "RollBall-v1",
    "StackPyramid-v1",
    "PlugCharger-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
}


def main(
    max_episodes: int | None = None,
    image_size: int = 256,
    jpeg_quality: int = 95,
):
    if not RAW_DIR.exists():
        print("data/raw does not exist. Download demos first.")
        return

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Only preprocess Panda environments that actually exist locally
    envs = sorted(
        d.name
        for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name in PANDA_ENVS
    )

    if not envs:
        print("No Panda-based environments found in data/raw")
        return

    print(f"Found {len(envs)} Panda environments")

    for env_id in envs:
        out_name = env_id.replace("-v1", "").replace("-", "_").lower() + ".h5"
        out_path = PREPROCESSED_DIR / out_name

        print(f"\n=== Preprocessing {env_id} ===")

        if out_path.exists():
            print("Already preprocessed, skipping")
            continue

        cmd = [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--skill", env_id,
            "--image-size", str(image_size),
            "--jpeg-quality", str(jpeg_quality),
        ]

        if max_episodes is not None:
            cmd += ["--max-episodes", str(max_episodes)]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"❌ Preprocessing failed for {env_id}")
        else:
            print(f"✅ Finished {env_id}")


if __name__ == "__main__":
    main()
