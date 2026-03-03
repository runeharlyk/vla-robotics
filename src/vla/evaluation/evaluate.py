import einops
import torch
import typer
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.env_processor import LiberoProcessorStep
from tqdm import tqdm

import wandb
from vla.constants import SUITE_MAP, resolve_suites
from vla.models.octo import octo
from vla.models.smolvla import smolvla

app = typer.Typer(no_args_is_help=True)

_LIBERO_PROC = LiberoProcessorStep()


def _obs_to_batch(raw_obs: dict, task_description: str, state_dim: int = 8, device: torch.device | None = None) -> dict:
    batch: dict = {}

    if "pixels" in raw_obs and isinstance(raw_obs["pixels"], dict):
        for cam_key, img_np in raw_obs["pixels"].items():
            img = torch.from_numpy(img_np)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.float() / 255.0
            img = torch.flip(img, dims=[2, 3])
            if device is not None:
                img = img.to(device, non_blocking=True)
            batch[f"observation.images.{cam_key}"] = img

    if "robot_state" in raw_obs:
        rs = raw_obs["robot_state"]
        eef_pos = torch.from_numpy(rs["eef"]["pos"]).float().unsqueeze(0)
        eef_quat = torch.from_numpy(rs["eef"]["quat"]).float().unsqueeze(0)
        gripper_qpos = torch.from_numpy(rs["gripper"]["qpos"]).float().unsqueeze(0)

        eef_axisangle = _LIBERO_PROC._quat2axisangle(eef_quat)

        state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)
        if state_dim < state.shape[-1]:
            state = state[..., :state_dim]
        if device is not None:
            state = state.to(device, non_blocking=True)
        batch["observation.state"] = state

    batch["task"] = [task_description]

    return batch


@app.command()
def evaluate(
    model: str = typer.Option("smolvla", "--model", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    suite: str = typer.Option("all", "--suite", "-s"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    compile_model: bool = typer.Option(False, "--compile/--no-compile"),
) -> None:
    suites = resolve_suites(suite)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if model == "smolvla":
        policy, model_id, action_dim = smolvla(checkpoint, device)
    if model == "octo":
        policy, model_id, action_dim = octo(checkpoint, device)
    else:
        print(f"Unknown model: {model}")
        raise typer.Exit(1)

    policy.eval()

    if compile_model and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile ...")
        policy = torch.compile(policy, mode="default")

    state_feature = policy.config.input_features.get("observation.state")
    state_dim = state_feature.shape[0] if state_feature else 8

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    print(f"  Action dim: {action_dim}, State dim: {state_dim}, Chunk: {policy.config.chunk_size}")
    print(f"  Device: {device_obj}")

    if wandb_project:
        wandb.init(project=wandb_project, name=f"eval-{model}", config={"checkpoint": checkpoint})

    try:
        results = _run_libero_eval(
            policy,
            preprocessor,
            postprocessor,
            suites,
            num_episodes,
            state_dim,
            device_obj,
        )
        _print_results(results, model)
    finally:
        if wandb_project and wandb.run is not None:
            wandb.finish()


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

        use_amp = device.type == "cuda"

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

                for _step in range(max_steps):
                    batch = _obs_to_batch(obs_raw, task_desc, state_dim, device=device)
                    batch = preprocessor(batch)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        action = policy.select_action(batch)

                    action = postprocessor(action)
                    action_np = action.to("cpu").numpy()
                    if action_np.ndim == 2:
                        action_np = action_np[0]

                    obs_raw, reward, terminated, truncated, info = env.step(action_np)

                    if info.get("is_success", False):
                        episode_success = True
                        break
                    if terminated or truncated:
                        break

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

        if wandb.run is not None:
            suite_metrics = {}
            for task_name, sr in task_results.items():
                suite_metrics[f"eval/{suite_name}/{task_name}"] = sr
            avg = sum(task_results.values()) / max(len(task_results), 1)
            suite_metrics[f"eval/{suite_name}/average"] = avg
            suite_avgs = {s: sum(r.values()) / max(len(r), 1) for s, r in all_results.items()}
            suite_metrics["eval/overall_average"] = sum(suite_avgs.values()) / max(len(suite_avgs), 1)
            wandb.log(suite_metrics)

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


if __name__ == "__main__":
    app()
