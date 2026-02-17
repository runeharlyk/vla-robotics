"""
Unified task visualizer for ManiSkill and LIBERO benchmarks.

Supports random-action exploration and policy rollouts (SmolVLA).

Usage:
    uv run python scripts/visualize.py maniskill --env PickCube-v1
    uv run python scripts/visualize.py libero --suite long --task 0
    uv run python scripts/visualize.py smolvla -c HuggingFaceVLA/smolvla_libero -s long
    uv run python scripts/visualize.py list
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"


def _has_gui() -> bool:
    """Check whether OpenCV highgui can open a window."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    """Write a list of RGB frames to an mp4 file."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    typer.echo(f"Saved {path} ({len(frames)} frames, {len(frames) / fps:.1f}s)")


def _display_frames(frames: list[np.ndarray], title: str, speed: float = 1.0) -> None:
    """Show frames in an OpenCV window."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 640, 480)
    for frame in frames:
        cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(int(33 / speed)) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def _output_frames(
    frames: list[np.ndarray],
    default_path: Path,
    title: str,
    output: Optional[str],
    save: bool,
    speed: float,
) -> None:
    """Save or display frames, falling back to save when GUI is unavailable."""
    if output:
        _save_video(frames, Path(output))
    elif save or not _has_gui():
        if not save:
            typer.echo("No GUI available — saving video instead.")
        _save_video(frames, default_path)
    else:
        _display_frames(frames, title, speed)


def _get_maniskill_frame(env) -> np.ndarray:
    """Extract an RGB numpy array from a ManiSkill environment render."""
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def _get_libero_frame(obs_raw: dict) -> np.ndarray:
    """Extract an RGB numpy array from a LIBERO observation."""
    if "pixels" not in obs_raw or not isinstance(obs_raw["pixels"], dict):
        return np.zeros((256, 256, 3), dtype=np.uint8)
    cams = list(obs_raw["pixels"].values())
    flipped = [np.flip(c, axis=(0, 1)).copy() for c in cams]
    if len(flipped) == 1:
        return flipped[0]
    return np.concatenate(flipped, axis=1)


LIBERO_SUITE_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


@app.command()
def maniskill(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="ManiSkill environment id"),
    steps: int = typer.Option(200, "--steps", "-s", help="Number of steps to run"),
    seed: int = typer.Option(0, "--seed", help="Random seed"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier"),
    save: bool = typer.Option(False, "--save", help="Force saving video to file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output video path"),
) -> None:
    """Visualize a ManiSkill task with random actions."""
    import gymnasium as gym
    import mani_skill  # noqa: F401

    typer.echo(f"Creating ManiSkill environment: {env_id}")
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend="cpu",
    )

    env.reset(seed=seed)
    frames: list[np.ndarray] = [_get_maniskill_frame(env)]

    typer.echo(f"Running {steps} random steps …")
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(_get_maniskill_frame(env))

        done = terminated.any() if hasattr(terminated, "any") else terminated
        if done:
            env.reset()

    env.close()

    _output_frames(
        frames,
        default_path=VIDEOS_DIR / f"maniskill_{env_id}.mp4",
        title=f"ManiSkill — {env_id}",
        output=output,
        save=save,
        speed=speed,
    )
    typer.echo("Done.")


@app.command()
def libero(
    suite: str = typer.Option("long", "--suite", "-s", help="LIBERO suite: spatial, object, goal, long"),
    task: int = typer.Option(0, "--task", "-t", help="Task index within the suite"),
    steps: int = typer.Option(300, "--steps", "-n", help="Max steps per episode"),
    seed: int = typer.Option(0, "--seed", help="Random seed"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier"),
    save: bool = typer.Option(False, "--save", help="Force saving video to file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output video path"),
) -> None:
    """Visualize a LIBERO task with random actions."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    libero_suite_name = LIBERO_SUITE_MAP.get(suite.lower(), f"libero_{suite}")
    benchmark_suite = _get_suite(libero_suite_name)
    num_tasks = len(benchmark_suite.tasks)

    if task < 0 or task >= num_tasks:
        typer.echo(f"Task index {task} out of range (0–{num_tasks - 1})", err=True)
        raise typer.Exit(1)

    env = LiberoEnv(
        task_suite=benchmark_suite,
        task_id=task,
        task_suite_name=libero_suite_name,
        obs_type="pixels_agent_pos",
    )
    task_desc = env.task_description
    max_steps = min(steps, env._max_episode_steps)

    typer.echo(f"Suite: {suite} ({libero_suite_name})")
    typer.echo(f"Task {task}/{num_tasks - 1}: {task_desc}")
    typer.echo(f"Running {max_steps} random steps …")

    obs_raw, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = [_get_libero_frame(obs_raw)]

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs_raw, reward, terminated, truncated, info = env.step(action)
        frames.append(_get_libero_frame(obs_raw))

        if info.get("is_success", False) or terminated or truncated:
            break

    env.close()

    _output_frames(
        frames,
        default_path=VIDEOS_DIR / f"libero_{suite}_task{task:02d}.mp4",
        title=f"LIBERO — {suite} task {task}: {task_desc}",
        output=output,
        save=save,
        speed=speed,
    )
    typer.echo("Done.")


def _load_smolvla(checkpoint: str, device: torch.device, image_key: str = "observation.images.image"):
    """Load a SmolVLA policy from a local .pt or HuggingFace id."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config_dict: dict = {}
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
        typer.echo(f"Loading SmolVLA from local checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        config_dict = ckpt["config"]
        model_id = config_dict["model_id"]
        action_dim = config_dict.get("action_dim", 7)
        image_size = config_dict.get("image_size", 256)
        chunk_size = config_dict.get("chunk_size", 50)

        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.config.input_features = {
            image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
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
        typer.echo(f"Loading SmolVLA from HuggingFace: {checkpoint}")
        model_id = checkpoint
        policy = SmolVLAPolicy.from_pretrained(checkpoint)

    policy = policy.to(device)
    policy.eval()
    return policy, model_id, config_dict


@app.command()
def smolvla(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Local .pt path or HF model id"),
    suite: str = typer.Option("long", "--suite", "-s", help="LIBERO suite: spatial, object, goal, long"),
    tasks: Optional[str] = typer.Option(None, "--tasks", "-t", help="Comma-separated task ids (default: all)"),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Episodes per task"),
    device: str = typer.Option("cuda", "--device", "-d"),
    seed: int = typer.Option(0, "--seed", help="Starting seed"),
    fps: int = typer.Option(30, "--fps", help="Video frame rate"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
) -> None:
    """Visualize SmolVLA policy solving LIBERO tasks."""
    import torch
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.policies.factory import make_pre_post_processors
    from tqdm import tqdm

    from vla.evaluate import _obs_to_batch

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    policy, model_id, _ = _load_smolvla(checkpoint, device_obj, image_key="observation.images.image")

    state_feature = policy.config.input_features.get("observation.state")
    state_dim = state_feature.shape[0] if state_feature else 8

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    libero_suite_name = LIBERO_SUITE_MAP.get(suite.lower(), f"libero_{suite}")
    benchmark_suite = _get_suite(libero_suite_name)
    num_tasks = len(benchmark_suite.tasks)

    task_ids = list(range(num_tasks))
    if tasks is not None:
        task_ids = [int(t.strip()) for t in tasks.split(",")]

    out = Path(output_dir) if output_dir else VIDEOS_DIR / f"smolvla_{suite}"
    out.mkdir(parents=True, exist_ok=True)

    total_videos = len(task_ids) * episodes
    typer.echo(f"\nRecording {total_videos} videos -> {out}/")
    typer.echo(f"Suite: {suite} ({libero_suite_name}), Tasks: {task_ids}, Episodes/task: {episodes}\n")

    successes = 0
    total = 0

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
            obs_raw, _ = env.reset(seed=seed + ep)
            policy.reset()
            frames: list[np.ndarray] = [_get_libero_frame(obs_raw)]
            success = False

            for _ in range(max_steps):
                batch = _obs_to_batch(obs_raw, task_desc, state_dim)
                batch = preprocessor(batch)

                with torch.no_grad():
                    action = policy.select_action(batch)

                action = postprocessor(action)
                action_np = action.to("cpu").numpy()
                if action_np.ndim == 2:
                    action_np = action_np[0]

                obs_raw, reward, terminated, truncated, info = env.step(action_np)
                frames.append(_get_libero_frame(obs_raw))

                if info.get("is_success", False):
                    success = True
                    break
                if terminated or truncated:
                    break

            total += 1
            if success:
                successes += 1

            tag = "success" if success else "fail"
            vid_name = f"task{task_id:02d}_ep{ep:02d}_{tag}.mp4"
            _save_video(frames, out / vid_name, fps=fps)
            ep_bar.set_postfix(last=tag)
        ep_bar.close()
        env.close()

    task_bar.close()
    typer.echo(f"\nDone. {successes}/{total} successful. Videos saved to {out}/")


def _maniskill_obs_to_batch(
    env,
    obs,
    instruction: str,
    action_dim: int,
    image_size: int,
    image_key: str,
) -> dict:
    """Build a SmolVLA batch dict from a ManiSkill env + observation."""
    import torch
    from PIL import Image as PILImage

    frame = _get_maniskill_frame(env)
    rgb = np.array(PILImage.fromarray(frame).resize((image_size, image_size)))
    img = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0)

    if isinstance(obs, dict) and "agent" in obs:
        agent = obs["agent"]
        qpos = agent["qpos"]
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        state_np = qpos.flatten().astype(np.float32)
    elif isinstance(obs, dict) and "state" in obs:
        s = obs["state"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        state_np = s.flatten().astype(np.float32)
    elif hasattr(obs, "cpu"):
        state_np = obs.cpu().numpy().flatten().astype(np.float32)
    else:
        state_np = np.array(obs, dtype=np.float32).flatten()

    if len(state_np) > action_dim:
        state_np = state_np[:action_dim]
    elif len(state_np) < action_dim:
        state_np = np.pad(state_np, (0, action_dim - len(state_np)))

    state = torch.from_numpy(state_np).unsqueeze(0)

    return {
        image_key: img,
        "observation.state": state,
        "task": [instruction],
    }


@app.command(name="smolvla-maniskill")
def smolvla_maniskill(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Local .pt checkpoint path"),
    env_id: Optional[str] = typer.Option(None, "--env", "-e", help="Override env id (default: from checkpoint)"),
    instruction: Optional[str] = typer.Option(None, "--instruction", "-i", help="Override task instruction"),
    steps: int = typer.Option(200, "--steps", "-s", help="Max steps per episode"),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Number of episodes"),
    device: str = typer.Option("cuda", "--device", "-d"),
    seed: int = typer.Option(0, "--seed", help="Starting seed"),
    fps: int = typer.Option(30, "--fps", help="Video frame rate"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output video path or directory"),
) -> None:
    """Visualize SmolVLA policy on a ManiSkill environment."""
    import gymnasium as gym
    import mani_skill  # noqa: F401
    import torch
    from lerobot.policies.factory import make_pre_post_processors

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    image_key = "observation.images.top"
    policy, model_id, ckpt_config = _load_smolvla(checkpoint, device_obj, image_key=image_key)

    resolved_env = env_id or ckpt_config.get("env_id", ckpt_config.get("env_ids", ["PickCube-v1"])[0])
    resolved_instruction = instruction or ckpt_config.get("instruction", "pick up the cube")
    action_dim = ckpt_config.get("action_dim", 8)
    image_size = ckpt_config.get("image_size", 256)

    state_feature = policy.config.input_features.get("observation.state")
    state_dim = state_feature.shape[0] if state_feature else action_dim

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    typer.echo(f"Environment: {resolved_env}")
    typer.echo(f"Instruction: '{resolved_instruction}'")
    typer.echo(f"Action dim: {action_dim}, Image size: {image_size}")

    env = gym.make(
        resolved_env,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend="cpu",
    )

    out_dir = Path(output) if output else VIDEOS_DIR / f"smolvla_{resolved_env}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset()
        frames: list[np.ndarray] = [_get_maniskill_frame(env)]

        typer.echo(f"\nEpisode {ep + 1}/{episodes} …")
        for step in range(steps):
            batch = _maniskill_obs_to_batch(env, obs, resolved_instruction, state_dim, image_size, image_key)
            batch = preprocessor(batch)

            with torch.no_grad():
                action = policy.select_action(batch)

            action = postprocessor(action)
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            frames.append(_get_maniskill_frame(env))

            success = info.get("success", False)
            if hasattr(success, "item"):
                success = success.item()
            done = terminated.any() if hasattr(terminated, "any") else terminated

            if success or done:
                break

        tag = "success" if success else "fail"
        vid_path = out_dir / f"ep{ep:02d}_{tag}.mp4"
        _save_video(frames, vid_path, fps=fps)
        typer.echo(f"  {tag} after {step + 1} steps")

    env.close()
    typer.echo(f"\nDone. Videos saved to {out_dir}/")


@app.command(name="list")
def list_tasks(
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Filter: maniskill or libero (default: both)"
    ),
) -> None:
    """List available tasks for each benchmark."""
    show_maniskill = benchmark is None or benchmark.lower() == "maniskill"
    show_libero = benchmark is None or benchmark.lower() == "libero"

    if show_maniskill:
        _list_maniskill_tasks()
    if show_libero:
        _list_libero_tasks()


def _list_maniskill_tasks() -> None:
    try:
        import gymnasium as gym
        import mani_skill  # noqa: F401
    except ImportError:
        typer.echo("\n[ManiSkill] not installed – skipping")
        return

    envs = sorted(
        e
        for e in gym.registry
        if "v1" in e
        and any(
            x in e
            for x in [
                "PickCube",
                "PushCube",
                "StackCube",
                "PegInsertion",
                "PlugCharger",
                "OpenCabinet",
                "PushChairs",
                "Drawer",
                "LiftPeg",
                "TurnFaucet",
            ]
        )
    )
    typer.echo(f"\n{'=' * 50}")
    typer.echo("ManiSkill environments")
    typer.echo(f"{'=' * 50}")
    for e in envs:
        typer.echo(f"  {e}")
    typer.echo(f"  Total: {len(envs)}")
    typer.echo(f"  Run:   uv run python scripts/visualize.py maniskill -e <ENV_ID>")


def _list_libero_tasks() -> None:
    try:
        from lerobot.envs.libero import _get_suite
    except ImportError:
        typer.echo("\n[LIBERO] not installed – install with: uv pip install -e '.[sim]'")
        return

    typer.echo(f"\n{'=' * 50}")
    typer.echo("LIBERO tasks")
    typer.echo(f"{'=' * 50}")

    for short_name, libero_name in LIBERO_SUITE_MAP.items():
        try:
            suite = _get_suite(libero_name)
            typer.echo(f"\n  Suite: {short_name} ({libero_name}) — {len(suite.tasks)} tasks")
            for idx, task in enumerate(suite.tasks):
                name = task.name if hasattr(task, "name") else str(task)
                typer.echo(f"    [{idx:2d}] {name}")
        except Exception as exc:
            typer.echo(f"\n  Suite: {short_name} — unavailable ({exc})")

    typer.echo(f"\n  Run:   uv run python scripts/visualize.py libero -s <SUITE> -t <TASK_ID>")


if __name__ == "__main__":
    app()
