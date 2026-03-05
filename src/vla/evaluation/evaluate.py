from __future__ import annotations

import torch
import typer
from tqdm import tqdm

import wandb
from vla.envs import SimEnvFactory, make_env_factory
from vla.models import load_policy

app = typer.Typer(no_args_is_help=True)


@app.command()
def evaluate(
    model: str = typer.Option("smolvla", "--model", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("all", "--suite", "-s"),
    env_id: str = typer.Option(None, "--env-id"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
    compile_model: bool = typer.Option(False, "--compile/--no-compile"),
) -> None:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loaded = load_policy(model, checkpoint, device)
    policy = loaded.policy
    policy.eval()

    if compile_model and hasattr(torch, "compile"):
        policy = torch.compile(policy, mode="default")

    preprocessor = loaded.preprocessor
    postprocessor = loaded.postprocessor

    if simulator.lower() == "libero" and suite.lower() == "all":
        from vla.constants import resolve_suites
        libero_suites = [s for s in resolve_suites("all") if s != "long"]
        factories = [
            _make_factory(simulator, suite=s, env_id=env_id, state_dim=loaded.state_dim)
            for s in libero_suites
        ]
        suite_label = "all (object, spatial, goal)"
    else:
        factories = [_make_factory(
            simulator, suite=suite, env_id=env_id, state_dim=loaded.state_dim)]
        suite_label = factories[0].suite_name

    print(
        f"  Model: {model}, Action dim: {loaded.action_dim}, State dim: {loaded.state_dim}")
    print(f"  Simulator: {simulator}, Suite/Task: {suite_label}")
    print(f"  Device: {device_obj}")

    if wandb_project:
        wandb.init(project=wandb_project,
                   name=f"eval-{model}", config={"checkpoint": checkpoint})

    try:
        results = {}
        for env_factory in factories:
            suite_results = _run_eval(
                policy,
                preprocessor,
                postprocessor,
                env_factory,
                num_episodes,
                device_obj,
            )
            results.update(suite_results)
        _print_results(results, model)
    finally:
        if wandb_project and wandb.run is not None:
            wandb.finish()


def _make_factory(
    simulator: str,
    suite: str | None = None,
    env_id: str | None = None,
    state_dim: int = 8,
) -> SimEnvFactory:
    sim = simulator.lower()
    kwargs: dict = {}

    if sim == "libero":
        kwargs["suite"] = suite or "all"
        kwargs["state_dim"] = state_dim
    elif sim == "maniskill":
        if not env_id:
            raise typer.BadParameter(
                "--env-id is required for ManiSkill evaluation")
        from vla.constants import MANISKILL_TASKS

        task_meta = MANISKILL_TASKS.get(env_id, {})
        kwargs["env_id"] = env_id
        kwargs["instruction"] = task_meta.get("instruction", env_id)
        kwargs["max_episode_steps"] = task_meta.get("max_episode_steps")

    return make_env_factory(sim, **kwargs)


def _run_eval(
    policy,
    preprocessor,
    postprocessor,
    env_factory: SimEnvFactory,
    num_episodes: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:

    all_results: dict[str, dict[str, float]] = {}
    use_amp = device.type == "cuda"

    task_bar = tqdm(range(env_factory.num_tasks),
                    desc="Tasks", unit="task", position=0)
    for task_id in task_bar:
        env = env_factory(task_id)

        task_desc = env.task_description
        max_steps = env.max_episode_steps
        task_bar.set_description(f"Task {task_id}: {task_desc}")

        successes = 0
        ep_bar = tqdm(range(num_episodes), desc="  Episodes",
                      unit="ep", position=1, leave=False)
        for ep in ep_bar:
            obs_raw, info = env.reset(seed=ep)
            policy.reset()
            episode_success = False

            for _step in range(max_steps):
                batch = env.obs_to_batch(obs_raw, device=device)
                batch = preprocessor(batch)

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    action = policy.select_action(batch)

                action = postprocessor(action)
                action_np = action.to("cpu").numpy()
                if action_np.ndim == 2:
                    action_np = action_np[0]

                obs_raw, reward, terminated, truncated, info = env.step(
                    action_np)

                if env.is_success(info):
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
        task_results = all_results.setdefault(env_factory.suite_name, {})
        task_results[task_desc] = sr
        task_bar.set_postfix(last_sr=f"{sr * 100:.0f}%")
        tqdm.write(
            f"  Task {task_id}: {sr * 100:.1f}% ({successes}/{num_episodes}) - {task_desc}")
    task_bar.close()

    if wandb.run is not None:
        for suite_name, task_results in all_results.items():
            metrics = {}
            for task_name, sr in task_results.items():
                metrics[f"eval/{suite_name}/{task_name}"] = sr
            avg = sum(task_results.values()) / max(len(task_results), 1)
            metrics[f"eval/{suite_name}/average"] = avg
            wandb.log(metrics)

    return all_results


def _print_results(results: dict[str, dict[str, float]], model_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Results: {model_name}")
    print(f"{'=' * 60}")

    suite_avgs = {}
    for suite_name, task_results in results.items():
        print(f"\n  {suite_name}:")
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
        print(f"  | {suite_name:12s} | {avg * 100:10.1f}% |")
    print(f"  | {'Overall':12s} | {overall * 100:10.1f}% |")


if __name__ == "__main__":
    app()
