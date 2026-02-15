"""
Download LIBERO demonstration data via LeRobot from HuggingFace.

Usage:
    uv run python scripts/download_libero.py --suite spatial
    uv run python scripts/download_libero.py --suite all
"""

import typer

SUITES = {
    "spatial": "lerobot/libero_spatial_image",
    "object": "lerobot/libero_object_image",
    "goal": "lerobot/libero_goal_image",
    "long": "lerobot/libero_10_image",
}


def main(
    suite: str = typer.Option("all", "--suite", "-s", help="Suite to download: spatial, object, goal, long, or all"),
) -> None:
    """Download LIBERO dataset(s) from HuggingFace via LeRobot."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if suite.lower() == "all":
        suites_to_download = list(SUITES.keys())
    else:
        suites_to_download = [s.strip().lower() for s in suite.split(",")]

    for suite_name in suites_to_download:
        repo_id = SUITES.get(suite_name, suite_name)
        print(f"\nDownloading {suite_name} ({repo_id})...")
        ds = LeRobotDataset(repo_id)
        print(f"  Loaded {len(ds)} samples from {repo_id}")
        print(f"  Features: {list(ds.meta.features.keys()) if hasattr(ds, 'meta') else 'N/A'}")

    print("\nAll requested suites downloaded and cached.")


if __name__ == "__main__":
    typer.run(main)
