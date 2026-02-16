"""
Record videos of a VLA policy solving LIBERO tasks.

Usage:
    uv run python src/vla/visualize.py --checkpoint HuggingFaceVLA/smolvla_libero --suite long
    uv run python src/vla/visualize.py -c models/smolvla_libero_long.pt -s long --tasks 0,2,5
"""

from pathlib import Path
from typing import Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SUITE_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


def _save_video(frames: list, path: Path, fps: int = 30) -> None:
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {path} ({len(frames)} frames, {len(frames) / fps:.1f}s)")


def _collect_frame(obs_raw: dict) -> "np.ndarray":
    import numpy as np

    if "pixels" not in obs_raw or not isinstance(obs_raw["pixels"], dict):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    cams = list(obs_raw["pixels"].values())
    flipped = [np.flip(c, axis=(0, 1)).copy() for c in cams]

    if len(flipped) == 1:
        return flipped[0]
    return np.concatenate(flipped, axis=1)


def main(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Local .pt path or HF model id"),
    suite: str = typer.Option("long", "--suite", "-s", help="LIBERO suite: spatial, object, goal, long"),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Episodes to record per task"),
    device: str = typer.Option("cuda", "--device", "-d"),
    output_dir: str = typer.Option("videos", "--output-dir", "-o", help="Output directory for videos"),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated task ids (default: all)"),
    fps: int = typer.Option(30, "--fps", help="Video frame rate"),
    seed: int = typer.Option(0, "--seed", help="Starting seed for episodes"),
) -> None:
    """Record videos of SmolVLA solving LIBERO tasks."""
    import numpy as np
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from vla.evaluate import _obs_to_batch

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
        print(f"Loading SmolVLA from local checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        config = ckpt["config"]
        model_id = config["model_id"]
        action_dim = config.get("action_dim", 7)
        image_size = config.get("image_size", 256)
        chunk_size = config.get("chunk_size", 50)

        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.config.input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
        }
        policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        }
        policy.config.empty_cameras = 0
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = chunk_size
        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Loading SmolVLA from HuggingFace: {checkpoint}")
        model_id = checkpoint
        policy = SmolVLAPolicy.from_pretrained(checkpoint)

    policy = policy.to(device_obj)
    policy.eval()

    state_feature = policy.config.input_features.get("observation.state")
    state_dim = state_feature.shape[0] if state_feature else 8

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    libero_suite_name = SUITE_MAP.get(suite.lower(), f"libero_{suite}")
    benchmark_suite = _get_suite(libero_suite_name)
    num_tasks = len(benchmark_suite.tasks)

    task_ids = list(range(num_tasks))
    if tasks is not None:
        task_ids = [int(t.strip()) for t in tasks.split(",")]

    out = Path(output_dir) / suite
    out.mkdir(parents=True, exist_ok=True)

    total_videos = len(task_ids) * episodes
    print(f"\nRecording {total_videos} videos -> {out}/")
    print(f"Suite: {suite} ({libero_suite_name}), Tasks: {task_ids}, Episodes/task: {episodes}\n")

    from tqdm import tqdm

    task_bar = tqdm(task_ids, desc="Tasks", unit="task", position=0)
    for task_id in task_bar:
        env = LiberoEnv(
            task_suite=benchmark_suite,
            task_id=task_id,
            task_suite_name=libero_suite_name,
            obs_type="pixels_agent_pos",
        )

        task_desc = env.task_description
        max_steps = env._max_episode_steps
        task_bar.set_description(f"Task {task_id}: {task_desc}")

        ep_bar = tqdm(range(episodes), desc="  Episodes", unit="ep", position=1, leave=False)
        for ep in ep_bar:
            obs_raw, info = env.reset(seed=seed + ep)
            policy.reset()
            frames: list = [_collect_frame(obs_raw)]
            success = False

            step_bar = tqdm(range(max_steps), desc="    Steps", unit="step", position=2, leave=False)
            for _step in step_bar:
                batch = _obs_to_batch(obs_raw, task_desc, state_dim)
                batch = preprocessor(batch)

                with torch.no_grad():
                    action = policy.select_action(batch)

                action = postprocessor(action)
                action_np = action.to("cpu").numpy()
                if action_np.ndim == 2:
                    action_np = action_np[0]

                obs_raw, reward, terminated, truncated, info = env.step(action_np)
                frames.append(_collect_frame(obs_raw))

                if info.get("is_success", False):
                    success = True
                    step_bar.set_postfix(result="success")
                    break
                if terminated or truncated:
                    step_bar.set_postfix(result="done")
                    break
            step_bar.close()

            tag = "success" if success else "fail"
            vid_name = f"task{task_id:02d}_ep{ep:02d}_{tag}.mp4"
            _save_video(frames, out / vid_name, fps=fps)
            ep_bar.set_postfix(last=tag)
        ep_bar.close()

        env.close()

    task_bar.close()
    print(f"\nDone. Videos saved to {out}/")


if __name__ == "__main__":
    typer.run(main)
