"""
LIBERO evaluation harness.

Wraps LeRobot's LIBERO environment and SmolVLA preprocessor pipeline
to evaluate fine-tuned models with the same processing as training.

Usage:
    uv run python src/vla/evaluate.py smolvla --checkpoint models/smolvla_libero_long.pt --suite long
    uv run python src/vla/evaluate.py custom --checkpoint models/custom_vla_libero.pt --suite all
"""

from pathlib import Path
from typing import Optional
from models import SmollVLA
import wandb

import torch
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tqdm import tqdm

import typer

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SUITE_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


def _resolve_suites(suite: str) -> list[str]:
    if suite.lower() == "all":
        return list(SUITE_MAP.keys())
    return [s.strip().lower() for s in suite.split(",")]


def _obs_to_batch(raw_obs: dict, task_description: str, state_dim: int = 8) -> dict:
    import einops
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep

    batch: dict = {}

    if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
        for cam_key, img_np in raw_obs["pixels"].items():
            img = torch.from_numpy(img_np)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.float() / 255.0
            img = torch.flip(img, dims=[2, 3])
            batch[f"observation.images.{cam_key}"] = img

    if "robot_state" in raw_obs:
        rs = raw_obs["robot_state"]
        eef_pos = torch.from_numpy(rs["eef"]["pos"]).float().unsqueeze(0)
        eef_quat = torch.from_numpy(rs["eef"]["quat"]).float().unsqueeze(0)
        gripper_qpos = torch.from_numpy(rs["gripper"]["qpos"]).float().unsqueeze(0)

        proc = LiberoProcessorStep()
        eef_axisangle = proc._quat2axisangle(eef_quat)

        state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)
        if state_dim < state.shape[-1]:
            state = state[..., :state_dim]
        batch["observation.state"] = state

    batch["task"] = [task_description]

    return batch


@app.command()
def evaluate(
    model: str = typer.Option("smolvla","--model","-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Local .pt path or HF model id"),
    suite: str = typer.Option("all", "--suite", "-s", help="LIBERO suite(s): spatial,object,goal,long or all"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n", help="Episodes per task"),
    device: str = typer.Option("cuda", "--device", "-d"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Not used for sequential eval"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project"),
) -> None:
    """Evaluate model on LIBERO."""

    suites = _resolve_suites(suite)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    if model == "smolvla" :
        policy,model_id,action_dim = smolvla(checkpoint,device)
    else: 
        print(f"No model named {model}")
        exit()

    policy.eval()

    state_feature = policy.config.input_features.get("observation.state")
    state_dim = state_feature.shape[0] if state_feature else 8

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    print(f"  Action dim: {action_dim}, State dim: {state_dim}, Chunk: {policy.config.chunk_size}")
    print(f"  Device: {device_obj}")

    results = _run_libero_eval(
        policy,
        preprocessor,
        postprocessor,
        suites,
        num_episodes,
        state_dim,
        device_obj,
    )
    _print_results(results, "SmolVLA")

    if wandb_project:
        _log_wandb(results, wandb_project, "smolvla", checkpoint)


def _run_libero_eval(
    policy,
    preprocessor,
    postprocessor,
    suites: list[str],
    num_episodes: int,
    state_dim: int,
    device: "torch.device",
) -> dict[str, dict[str, float]]:

    all_results: dict[str, dict[str, float]] = {}

    for suite_name in suites:
        libero_suite_name = SUITE_MAP.get(suite_name, f"libero_{suite_name}")
        print(f"\n{'=' * 60}")
        print(f"Evaluating on LIBERO-{suite_name.capitalize()} ({libero_suite_name})")
        print(f"{'=' * 60}")

        benchmark_suite = _get_suite(libero_suite_name)
        num_tasks = len(benchmark_suite.tasks)
        task_results: dict[str, float] = {}

        task_bar = tqdm(range(num_tasks), desc="Tasks", unit="task", position=0)
        for task_id in task_bar:
            env = LiberoEnv(
                task_suite=benchmark_suite,
                task_id=task_id,
                task_suite_name=libero_suite_name,
                obs_type="pixels_agent_pos",
            )

            task_name = env.task
            task_desc = env.task_description
            max_steps = env._max_episode_steps
            successes = 0

            task_bar.set_description(f"Task {task_id}: {task_desc}")

            ep_bar = tqdm(range(num_episodes), desc="  Episodes", unit="ep", position=1, leave=False)
            for ep in ep_bar:
                obs_raw, info = env.reset(seed=ep)
                policy.reset()
                episode_success = False

                step_bar = tqdm(range(max_steps), desc="    Steps", unit="step", position=2, leave=False)
                for _step in step_bar:
                    batch = _obs_to_batch(obs_raw, task_desc, state_dim)

                    if ep == 0 and _step == 0 and task_id == 0:
                        tqdm.write("\n    [DEBUG] First observation diagnostics:")
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                tqdm.write(
                                    f"      {k}: shape={v.shape}, dtype={v.dtype}, "
                                    f"min={v.min().item():.4f}, max={v.max().item():.4f}, "
                                    f"mean={v.mean().item():.4f}"
                                )
                            else:
                                tqdm.write(f"      {k}: {v}")

                    batch = preprocessor(batch)

                    if ep == 0 and _step == 0 and task_id == 0:
                        tqdm.write("    [DEBUG] After preprocessor:")
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                tqdm.write(
                                    f"      {k}: shape={v.shape}, dtype={v.dtype}, "
                                    f"min={v.min().item():.4f}, max={v.max().item():.4f}"
                                )

                    with torch.no_grad():
                        action = policy.select_action(batch)

                    if ep == 0 and _step == 0 and task_id == 0:
                        tqdm.write(
                            f"    [DEBUG] Raw model action: shape={action.shape}, "
                            f"min={action.min().item():.4f}, max={action.max().item():.4f}, "
                            f"mean={action.mean().item():.4f}"
                        )

                    action = postprocessor(action)

                    if ep == 0 and _step == 0 and task_id == 0:
                        tqdm.write(
                            f"    [DEBUG] After postprocessor: shape={action.shape}, "
                            f"min={action.min().item():.4f}, max={action.max().item():.4f}, "
                            f"mean={action.mean().item():.4f}"
                        )

                    action_np = action.to("cpu").numpy()
                    if action_np.ndim == 2:
                        action_np = action_np[0]

                    if ep == 0 and _step < 3 and task_id == 0:
                        tqdm.write(f"    [DEBUG] Step {_step} action sent to env: {action_np}")

                    obs_raw, reward, terminated, truncated, info = env.step(action_np)

                    if info.get("is_success", False):
                        episode_success = True
                        step_bar.set_postfix(result="success")
                        break
                    if terminated or truncated:
                        step_bar.set_postfix(result="done")
                        break
                step_bar.close()

                if episode_success:
                    successes += 1
                ep_bar.set_postfix(sr=f"{successes}/{ep + 1}")
            ep_bar.close()

            env.close()
            sr = successes / num_episodes
            task_results[task_name] = sr
            task_bar.set_postfix(last_sr=f"{sr * 100:.0f}%")
            tqdm.write(f"  Task {task_id}: {sr * 100:.1f}% ({successes}/{num_episodes}) - {task_desc}")
        task_bar.close()

        all_results[suite_name] = task_results

    return all_results


def _print_results(results: dict[str, dict[str, float]], model_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Results: {model_name}")
    print(f"{'=' * 60}")

    suite_avgs = {}
    for suite_name, task_results in results.items():
        print(f"\n  LIBERO-{suite_name.capitalize()}:")
        for task_name, success_rate in task_results.items():
            print(f"    {task_name}: {success_rate * 100:.1f}%")
        avg = sum(task_results.values()) / max(len(task_results), 1)
        suite_avgs[suite_name] = avg
        print(f"    Average: {avg * 100:.1f}%")

    overall = sum(suite_avgs.values()) / max(len(suite_avgs), 1)
    print(f"\n  Overall Average: {overall * 100:.1f}%")

    print("\n  | Suite | Success Rate |")
    print("  |-------|-------------|")
    for suite_name, avg in suite_avgs.items():
        print(f"  | {suite_name.capitalize():12s} | {avg * 100:10.1f}% |")
    print(f"  | {'Overall':12s} | {overall * 100:10.1f}% |")


def _log_wandb(
    results: dict[str, dict[str, float]],
    project: str,
    model_name: str,
    checkpoint: str,
) -> None:

    wandb.init(project=project, name=f"eval-{model_name}", config={"checkpoint": checkpoint})

    flat = {}
    suite_avgs = {}
    for suite_name, task_results in results.items():
        for task_name, sr in task_results.items():
            flat[f"eval/{suite_name}/{task_name}"] = sr
        avg = sum(task_results.values()) / max(len(task_results), 1)
        flat[f"eval/{suite_name}/average"] = avg
        suite_avgs[suite_name] = avg

    flat["eval/overall_average"] = sum(suite_avgs.values()) / max(len(suite_avgs), 1)
    wandb.log(flat)
    wandb.finish()


if __name__ == "__main__":
    app()
