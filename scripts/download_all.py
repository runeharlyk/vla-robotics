"""
Download ManiSkill demonstrations for all environments listed in skills.txt.

- Skips envs with no demos
- Skips envs that are already downloaded
- Writes a log file
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_LIST = PROJECT_ROOT / "skills.txt"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "download_all.log"

DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_data.py"


def main():
    if not ENV_LIST.exists():
        raise FileNotFoundError(f"{ENV_LIST} not found")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    envs = [e.strip() for e in ENV_LIST.read_text().splitlines() if e.strip()]

    with LOG_FILE.open("w") as log:
        for env_id in envs:
            env_dir = RAW_DIR / env_id

            header = f"\n=== {env_id} ===\n"
            print(header, end="")
            log.write(header)

            # ✅ Already downloaded → treat as success
            if env_dir.exists():
                msg = f"Already downloaded, skipping\n"
                print(msg, end="")
                log.write(msg)
                continue

            cmd = [
                sys.executable,
                str(DOWNLOAD_SCRIPT),
                "--skill",
                env_id,
            ]

            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=log,
            )

            if result.returncode != 0:
                msg = f"No demos available or download failed\n"
                print(msg, end="")
                log.write(msg)


if __name__ == "__main__":
    main()
