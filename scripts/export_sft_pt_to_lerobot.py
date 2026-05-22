"""Convert this repo's SFT ``.pt`` dataset format into a LeRobot dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _normalise_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"Expected episode images shaped (T,C,H,W) or (T,V,C,H,W), got {tuple(images.shape)}")
    if images.dtype != torch.uint8:
        max_val = images.max().item() if images.numel() else 0.0
        images = images.clamp(0, 255 if max_val > 2.0 else 1.0)
        if max_val <= 2.0:
            images = images * 255.0
        images = images.to(torch.uint8)
    return images


@app.command()
def main(
    data: Path = typer.Option(..., "--data", exists=True, file_okay=True, dir_okay=False, readable=True),
    repo_id: str = typer.Option(..., "--repo-id", help="LeRobot/HF repo id, e.g. user/libero-spatial-rl-success."),
    root: Path | None = typer.Option(None, "--root", help="Local LeRobot dataset root. Defaults to LeRobot cache."),
    fps: int = typer.Option(10, "--fps", min=1),
    robot_type: str = typer.Option("libero_panda", "--robot-type"),
    push: bool = typer.Option(False, "--push/--no-push"),
    private: bool = typer.Option(False, "--private/--public"),
    use_videos: bool = typer.Option(False, "--videos/--no-videos"),
) -> None:
    """Create a LeRobot dataset from a collected SFT ``.pt`` file."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    payload = torch.load(data, map_location="cpu", weights_only=False)
    episodes: list[dict] = payload["episodes"]
    metadata: dict = payload.get("metadata", {})
    if not episodes:
        raise typer.BadParameter("Input dataset has no episodes")

    first_images = _normalise_images(episodes[0]["images"])
    _, num_cameras, channels, height, width = first_images.shape
    action_dim = int(metadata.get("action_dim", episodes[0]["actions"].shape[-1]))
    state_dim = int(metadata.get("state_dim", episodes[0]["states"].shape[-1]))

    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {"dtype": "float32", "shape": (action_dim,), "names": [f"action_{i}" for i in range(action_dim)]},
    }
    for camera_idx in range(num_cameras):
        key = "observation.images.image" if camera_idx == 0 else f"observation.images.image{camera_idx + 1}"
        features[key] = {
            "dtype": "image" if not use_videos else "video",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    for episode_idx, episode in enumerate(episodes):
        images = _normalise_images(episode["images"])
        states = episode["states"].float()
        actions = episode["actions"].float()
        instruction = str(episode.get("instruction", metadata.get("instruction", "complete the task")))
        T = int(actions.shape[0])
        for t in range(T):
            frame: dict = {
                "observation.state": states[t],
                "action": actions[t],
                "task": instruction,
            }
            for camera_idx in range(num_cameras):
                key = "observation.images.image" if camera_idx == 0 else f"observation.images.image{camera_idx + 1}"
                frame[key] = images[t, camera_idx]
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
        logger.info("Saved LeRobot episode %d/%d", episode_idx + 1, len(episodes))

    logger.info("Created LeRobot dataset at %s", dataset.root)
    if push:
        dataset.push_to_hub(private=private)
        logger.info("Pushed dataset to %s", repo_id)


if __name__ == "__main__":
    app()
