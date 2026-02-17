import typer

from vla.constants import LIBERO_SUITES, resolve_suites


def main(
    suite: str = typer.Option("all", "--suite", "-s"),
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suites_to_download = resolve_suites(suite)

    for suite_name in suites_to_download:
        repo_id = LIBERO_SUITES.get(suite_name, suite_name)
        print(f"\nDownloading {suite_name} ({repo_id})...")
        ds = LeRobotDataset(repo_id)
        print(f"  Loaded {len(ds)} samples from {repo_id}")
        print(f"  Features: {list(ds.meta.features.keys()) if hasattr(ds, 'meta') else 'N/A'}")

    print("\nAll requested suites downloaded and cached.")


if __name__ == "__main__":
    typer.run(main)
