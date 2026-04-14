from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer

from vla.evaluation.runtime import (
    EvalConfig,
    make_runtime_env_factory,
    predict_action_from_batch,
    resolve_eval_runtime,
)
from vla.utils import seed_everything


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


def main(
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
) -> None:
    from tqdm import tqdm

    seed_everything(seed)
    runtime = resolve_eval_runtime(
        EvalConfig(
            checkpoint=checkpoint,
            simulator=simulator,
            suite=suite,
            env_id=env_id,
            device=device,
        )
    )
    runtime.policy.eval()
    env_factory = make_runtime_env_factory(runtime)

    task_ids = list(range(env_factory.num_tasks))
    if tasks is not None:
        task_ids = [int(t.strip()) for t in tasks.split(",")]

    out = Path(output_dir) / runtime.checkpoint_tag / env_factory.suite_name
    out.mkdir(parents=True, exist_ok=True)

    total_videos = len(task_ids) * episodes
    print(f"\nRecording {total_videos} videos -> {out}/")
    print(f"Checkpoint: {runtime.checkpoint_tag}, Simulator: {simulator}, Suite/Task: {env_factory.suite_name}")
    print(f"Tasks: {task_ids}, Episodes/task: {episodes}\n")

    task_bar = tqdm(task_ids, desc="Tasks", unit="task", position=0)
    for task_id in task_bar:
        env = env_factory(task_id)

        task_desc = env.task_description
        max_steps = env.max_episode_steps
        task_bar.set_description(f"Task {task_id}: {task_desc}")

        ep_bar = tqdm(range(episodes), desc="  Episodes", unit="ep", position=1, leave=False)
        for ep in ep_bar:
            obs_raw, info = env.reset(seed=seed + ep)
            frames: list[np.ndarray] = [env.get_frame(obs_raw)]
            success = False

            step_bar = tqdm(range(max_steps), desc="    Steps", unit="step", position=2, leave=False)
            for _step in step_bar:
                batch = env.obs_to_batch(obs_raw, device=runtime.device)

                with torch.no_grad():
                    action = predict_action_from_batch(runtime.policy, batch, runtime.instruction)

                action_np = action.to("cpu").numpy()

                obs_raw, reward, terminated, truncated, info = env.step(action_np)
                frames.append(env.get_frame(obs_raw))

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
        ep_bar.close()

        env.close()

    task_bar.close()
    print(f"\nDone. Videos saved to {out}/")


if __name__ == "__main__":
    typer.run(main)
