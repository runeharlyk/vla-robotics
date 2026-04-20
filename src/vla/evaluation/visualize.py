from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import typer


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


class _PreviewWindow:
    def __init__(self, enabled: bool, fps: int, scale: float = 1.0, title: str = "VLA Rollout Preview") -> None:
        self.enabled = enabled
        self.fps = max(fps, 1)
        self.scale = max(scale, 0.1)
        self.title = title
        self._available = enabled

    def show(self, frame: np.ndarray, status_lines: tuple[str, ...] = ()) -> bool:
        if not self._available:
            return False

        try:
            import cv2
        except Exception as exc:
            print(f"Preview disabled: failed to import OpenCV GUI support ({exc}).")
            self._available = False
            return False

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.scale != 1.0:
            width = max(int(round(frame_bgr.shape[1] * self.scale)), 1)
            height = max(int(round(frame_bgr.shape[0] * self.scale)), 1)
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

        if status_lines:
            y = 24
            for line in status_lines:
                cv2.putText(
                    frame_bgr,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 26

        try:
            cv2.imshow(self.title, frame_bgr)
            key = cv2.waitKey(max(int(round(1000 / self.fps)), 1)) & 0xFF
            if key in (27, ord("q")):
                return True
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except Exception as exc:
            print(f"Preview disabled: failed to open/update preview window ({exc}).")
            self._available = False
            return False

    def close(self) -> None:
        if not self.enabled:
            return
        with contextlib.suppress(Exception):
            import cv2

            cv2.destroyWindow(self.title)


def _trajectory_to_frames(trajectory) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for step_images in trajectory.images:
        if step_images.ndim == 3:
            step_images = step_images.unsqueeze(0)

        cams: list[np.ndarray] = []
        for cam in step_images:
            frame = cam.detach().cpu().permute(1, 2, 0).numpy()
            if frame.dtype in (np.float32, np.float64):
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8, copy=False)
            cams.append(frame)

        frames.append(cams[0] if len(cams) == 1 else np.concatenate(cams, axis=1))
    return frames


def _select_parallel_trajectory(trajectories: list) -> tuple[int, object]:
    for idx, traj in enumerate(trajectories):
        if traj.success:
            return idx, traj
    return 0, trajectories[0]


def _parallel_video_name(task_id: int, episode: int, attempt_idx: int, success: bool) -> str:
    tag = "success" if success else "fail"
    return f"task{task_id:02d}_ep{episode:02d}_try{attempt_idx:02d}_{tag}.mp4"


def _preview_frames(
    preview: _PreviewWindow,
    frames: list[np.ndarray],
    status_lines: tuple[str, ...],
) -> bool:
    return any(preview.show(frame, status_lines=status_lines) for frame in frames)


def _batched_select_action(policy, images, instruction: str, states, *, use_amp: bool):
    import torch

    if images.ndim == 4:
        images = images.unsqueeze(1)

    batch = {}
    for idx in range(images.shape[1]):
        key = "observation.images.image" if idx == 0 else f"observation.images.image{idx + 1}"
        batch[key] = images[:, idx]

    if states is not None:
        batch["observation.state"] = states
    batch["task"] = [instruction] * images.shape[0]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        return policy.select_action(batch)


def main(
    model: str = typer.Option("smolvla", "--model", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("long", "--suite", "-s"),
    env_id: str | None = typer.Option(None, "--env-id"),
    episodes: int = typer.Option(1, "--episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    output_dir: str = typer.Option("videos", "--output-dir", "-o"),
    tasks: str | None = typer.Option(None, "--tasks", "-t"),
    fps: int = typer.Option(30, "--fps"),
    seed: int = typer.Option(0, "--seed"),
    num_envs: int = typer.Option(
        1,
        "--num-envs",
        min=1,
        help=(
            "For LIBERO, run this many rollout attempts in parallel per requested "
            "episode and save the first successful attempt, if any."
        ),
    ),
    save_all_parallel_videos: bool = typer.Option(
        False,
        "--save-all-parallel-videos/--save-selected-parallel-video",
        help="When --num-envs > 1 on LIBERO, save every parallel attempt instead of only the selected one.",
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Open a live preview window while recording."),
    preview_scale: float = typer.Option(2.0, "--preview-scale", help="Scale factor for the live preview window."),
) -> None:
    import torch
    from tqdm import tqdm

    from vla.evaluation.evaluate import _make_factory
    from vla.models import load_policy

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    use_amp = device_obj.type == "cuda"

    loaded = load_policy(model, checkpoint, device)
    policy = loaded.policy
    policy.eval()

    preprocessor = loaded.preprocessor
    postprocessor = loaded.postprocessor

    env_factory = _make_factory(simulator, suite=suite, env_id=env_id, state_dim=loaded.state_dim)

    if simulator.lower() != "libero" and num_envs != 1:
        raise typer.BadParameter("--num-envs is only supported for LIBERO visualize right now")

    task_ids = list(range(env_factory.num_tasks))
    if tasks is not None:
        task_ids = [int(t.strip()) for t in tasks.split(",")]

    out = Path(output_dir) / model / env_factory.suite_name
    out.mkdir(parents=True, exist_ok=True)
    preview = _PreviewWindow(enabled=show, fps=fps, scale=preview_scale)
    abort_requested = False

    videos_per_episode = num_envs if simulator.lower() == "libero" and num_envs > 1 and save_all_parallel_videos else 1
    total_videos = len(task_ids) * episodes * videos_per_episode
    print(f"\nRecording {total_videos} videos -> {out}/")
    print(f"Model: {model}, Simulator: {simulator}, Suite/Task: {env_factory.suite_name}")
    print(f"Tasks: {task_ids}, Episodes/task: {episodes}\n")

    task_bar = tqdm(task_ids, desc="Tasks", unit="task", position=0)
    for task_id in task_bar:
        if simulator.lower() == "libero" and num_envs > 1:
            from vla.rl.libero_rollout import LiberoRollout

            env = env_factory(task_id)
            task_desc = env.task_description
            max_steps = env.max_episode_steps
            env.close()
            task_bar.set_description(f"Task {task_id}: {task_desc}")

            ep_bar = tqdm(range(episodes), desc="  Episodes", unit="ep", position=1, leave=False)
            for ep in ep_bar:
                rollout = LiberoRollout(
                    suite_name=env_factory.suite_name,
                    task_id=task_id,
                    num_envs=num_envs,
                    max_steps=max_steps,
                    state_dim=loaded.state_dim,
                )
                try:
                    trajectories = rollout.collect_batch(
                        policy_fn=None,
                        instruction=task_desc,
                        num_trajectories=num_envs,
                        seed=seed + ep * num_envs,
                        policy_batch_fn=lambda imgs, instr, states: _batched_select_action(
                            policy,
                            imgs.to(device_obj, non_blocking=True),
                            instr,
                            states.to(device_obj, non_blocking=True),
                            use_amp=use_amp,
                        ),
                    )
                finally:
                    rollout.close()

                selected_idx, selected = _select_parallel_trajectory(trajectories)
                success_count = sum(1 for traj in trajectories if traj.success)
                frames = _trajectory_to_frames(selected)
                success = bool(selected.success)

                if show and _preview_frames(
                    preview,
                    frames,
                    status_lines=(
                        f"Task {task_id} | Episode {ep} | Parallel attempts {success_count}/{len(trajectories)}",
                        task_desc,
                        f"Showing selected attempt {selected_idx}",
                    ),
                ):
                    abort_requested = True

                if save_all_parallel_videos:
                    for attempt_idx, trajectory in enumerate(trajectories):
                        attempt_frames = _trajectory_to_frames(trajectory)
                        _save_video(
                            attempt_frames,
                            out / _parallel_video_name(task_id, ep, attempt_idx, bool(trajectory.success)),
                            fps=fps,
                        )
                else:
                    tag = "success" if success else "fail"
                    vid_name = f"task{task_id:02d}_ep{ep:02d}_{tag}.mp4"
                    _save_video(frames, out / vid_name, fps=fps)
                print(
                    "  Parallel LIBERO attempts: "
                    f"{success_count}/{len(trajectories)} success; saved attempt {selected_idx}"
                )
                selected_tag = "success" if success else "fail"
                ep_bar.set_postfix(last=f"{selected_tag} ({success_count}/{len(trajectories)})")
                if abort_requested:
                    break
            ep_bar.close()

            if abort_requested:
                break
            continue

        env = env_factory(task_id)

        task_desc = env.task_description
        max_steps = env.max_episode_steps
        task_bar.set_description(f"Task {task_id}: {task_desc}")

        ep_bar = tqdm(range(episodes), desc="  Episodes", unit="ep", position=1, leave=False)
        for ep in ep_bar:
            obs_raw, info = env.reset(seed=seed + ep)
            policy.reset()
            frames: list[np.ndarray] = [env.get_frame(obs_raw)]
            success = False
            if preview.show(
                frames[-1],
                status_lines=(
                    f"Task {task_id} | Episode {ep}",
                    task_desc,
                    "Press q or Esc to stop preview/run",
                ),
            ):
                abort_requested = True

            step_bar = tqdm(range(max_steps), desc="    Steps", unit="step", position=2, leave=False)
            for _step in step_bar:
                if abort_requested:
                    step_bar.set_postfix(result="stopped")
                    break

                batch = env.obs_to_batch(obs_raw, device=device_obj)
                batch = preprocessor(batch)

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    action = policy.select_action(batch)

                action = postprocessor(action)
                action_np = action.to("cpu").numpy()
                if action_np.ndim == 2:
                    action_np = action_np[0]

                obs_raw, reward, terminated, truncated, info = env.step(action_np)
                frames.append(env.get_frame(obs_raw))
                if preview.show(
                    frames[-1],
                    status_lines=(
                        f"Task {task_id} | Episode {ep} | Step {_step + 1}/{max_steps}",
                        task_desc,
                        "Press q or Esc to stop preview/run",
                    ),
                ):
                    abort_requested = True
                    step_bar.set_postfix(result="stopped")
                    break

                if env.is_success(info):
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
            if abort_requested:
                break
        ep_bar.close()

        env.close()
        if abort_requested:
            break

    task_bar.close()
    preview.close()
    if abort_requested:
        print(f"\nStopped early. Partial videos saved to {out}/")
        raise typer.Exit(0)
    print(f"\nDone. Videos saved to {out}/")


if __name__ == "__main__":
    typer.run(main)
