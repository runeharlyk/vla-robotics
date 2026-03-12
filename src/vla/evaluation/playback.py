"""Demo playback - replay recorded demonstrations as videos.

Two modes:
- ``replay``: Step through the simulator with recorded actions and capture live video.
- ``render``: Stitch recorded images from the dataset into a video (no env needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import typer
from tqdm import tqdm


@dataclass
class DemoEpisode:
    actions: np.ndarray
    images: list[np.ndarray] = field(default_factory=list)
    instruction: str = ""
    task_index: int = 0


def _extract_libero_episodes(suite: str) -> list[DemoEpisode]:
    from vla.data.libero import load_libero_suite

    ds = load_libero_suite(suite)
    lerobot_ds = ds.lerobot_dataset

    ep_index = lerobot_ds.episode_data_index
    num_episodes = len(ep_index["from"])
    episodes: list[DemoEpisode] = []

    for ep in tqdm(range(num_episodes), desc=f"Loading {suite} episodes"):
        start = ep_index["from"][ep].item()
        end = ep_index["to"][ep].item()

        actions_list: list[np.ndarray] = []
        frame_list: list[np.ndarray] = []
        instruction = ""
        task_idx = 0

        for i in range(start, end):
            sample = lerobot_ds[i]

            action = sample["action"]
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            actions_list.append(action)

            img = sample.get("observation.images.image")
            if img is not None:
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                frame_list.append(img)

            if not instruction:
                if "task" in sample:
                    instruction = str(sample["task"])
                else:
                    ti = sample.get("task_index", 0)
                    if isinstance(ti, torch.Tensor):
                        ti = ti.item()
                    task_idx = int(ti)
                    instruction = ds._task_map.get(task_idx, "")

        episodes.append(
            DemoEpisode(
                actions=np.stack(actions_list),
                images=frame_list,
                instruction=instruction,
                task_index=task_idx,
            )
        )

    return episodes


def _extract_maniskill_episodes(data_path: str, instruction: str = "") -> list[DemoEpisode]:
    from vla.data.maniskill import load_maniskill_dataset

    ds = load_maniskill_dataset(data_path, instruction=instruction)
    episodes: list[DemoEpisode] = []

    for i in tqdm(range(len(ds)), desc="Loading ManiSkill episodes"):
        sample = ds._samples[i]

        actions = sample["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        frame_list: list[np.ndarray] = []
        if "images" in sample:
            imgs = sample["images"]
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.numpy()
            for t in range(imgs.shape[0]):
                img = imgs[t]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                frame_list.append(img)

        ep_instr = sample.get("instruction", instruction)
        episodes.append(
            DemoEpisode(
                actions=actions,
                images=frame_list,
                instruction=ep_instr,
                task_index=0,
            )
        )

    return episodes


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {path} ({len(frames)} frames, {len(frames) / fps:.1f}s)")


def _render_episodes(
    episodes: list[DemoEpisode],
    ep_indices: list[int],
    out_dir: Path,
    fps: int,
) -> None:
    for idx in ep_indices:
        ep = episodes[idx]
        if not ep.images:
            print(f"  Episode {idx}: no recorded images, skipping render")
            continue
        vid_name = f"ep{idx:04d}_render.mp4"
        _save_video(ep.images, out_dir / vid_name, fps=fps)


def _replay_episodes(
    episodes: list[DemoEpisode],
    ep_indices: list[int],
    simulator: str,
    suite: str | None,
    env_id: str | None,
    out_dir: Path,
    fps: int,
    seed: int,
) -> None:
    from vla.evaluation.evaluate import _make_factory

    for idx in ep_indices:
        ep = episodes[idx]
        T = ep.actions.shape[0]

        if simulator == "libero":
            env_factory = _make_factory(simulator, suite=suite)
            env = env_factory(ep.task_index)
        else:
            env_factory = _make_factory(simulator, env_id=env_id)
            env = env_factory(0)

        task_desc = env.task_description
        print(f"  Episode {idx}: replaying {T} actions in '{task_desc}'")

        obs_raw, _info = env.reset(seed=seed + idx)
        frames: list[np.ndarray] = [env.get_frame(obs_raw)]
        success = False

        for t in tqdm(range(T), desc=f"    Ep {idx}", unit="step", leave=False):
            action = ep.actions[t]
            obs_raw, _reward, terminated, truncated, info = env.step(action)
            frames.append(env.get_frame(obs_raw))

            if env.is_success(info):
                success = True
                break
            if terminated or truncated:
                break

        env.close()

        tag = "success" if success else "done"
        vid_name = f"ep{idx:04d}_replay_{tag}.mp4"
        _save_video(frames, out_dir / vid_name, fps=fps)


def playback(
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("long", "--suite", "-s"),
    env_id: str | None = typer.Option(None, "--env-id"),
    data_path: str | None = typer.Option(None, "--data-path"),
    mode: str = typer.Option("replay", "--mode", help="'replay' (actions in env) or 'render' (image stitch)"),
    episodes: str = typer.Option("0", "--episodes", "-n", help="Comma-separated episode indices"),
    output_dir: str = typer.Option("videos/playback", "--output-dir", "-o"),
    fps: int = typer.Option(30, "--fps"),
    seed: int = typer.Option(0, "--seed"),
    instruction: str = typer.Option("", "--instruction"),
) -> None:
    ep_indices = [int(e.strip()) for e in episodes.split(",")]

    print(f"\nPlayback mode={mode}, simulator={simulator}, suite={suite}")
    print(f"Episodes: {ep_indices}")

    if simulator == "libero":
        all_episodes = _extract_libero_episodes(suite)
    elif simulator == "maniskill":
        if data_path is None:
            raise typer.BadParameter("--data-path is required for ManiSkill playback")
        all_episodes = _extract_maniskill_episodes(data_path, instruction=instruction)
    else:
        raise typer.BadParameter(f"Unknown simulator: {simulator}")

    max_idx = len(all_episodes) - 1
    for idx in ep_indices:
        if idx > max_idx:
            raise typer.BadParameter(
                f"Episode {idx} out of range (dataset has {len(all_episodes)} episodes, 0..{max_idx})"
            )

    out = Path(output_dir) / simulator / suite
    out.mkdir(parents=True, exist_ok=True)

    if mode == "render":
        _render_episodes(all_episodes, ep_indices, out, fps)
    elif mode == "replay":
        _replay_episodes(all_episodes, ep_indices, simulator, suite, env_id, out, fps, seed)
    else:
        raise typer.BadParameter(f"Unknown mode: {mode!r}. Use 'replay' or 'render'.")

    print(f"\nDone. Videos saved to {out}/")


if __name__ == "__main__":
    typer.run(playback)
