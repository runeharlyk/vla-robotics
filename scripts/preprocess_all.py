"""
Preprocess only Panda-based ManiSkill demos.

Avoids downloading assets for other robots (ANYmal, humanoid, etc.).
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

PREPROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_data.py"

# ✅ Known Panda demo environments
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


def main():
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    envs = sorted(
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name in PANDA_ENVS
    )

    for env_id in envs:
        out_name = env_id.replace("-v1", "").replace("-", "_").lower() + ".pt"
        out_path = PREPROCESSED_DIR / out_name

        print(f"\n=== Preprocessing {env_id} ===")

        if out_path.exists():
            print("Already preprocessed, skipping")
            continue

        cmd = [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--skill",
            env_id,
        ]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"Preprocessing failed for {env_id}")


if __name__ == "__main__":
    main()
