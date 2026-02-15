"""
LIBERO evaluation harness.

Wraps LeRobot's built-in LIBERO environment and eval utilities to provide
a simple CLI that works with both SmolVLA and our custom model.

Usage:
    uv run python src/vla/evaluate.py smolvla --checkpoint HuggingFaceVLA/smolvla_libero --suite spatial
    uv run python src/vla/evaluate.py custom --checkpoint models/custom_vla_libero.pt --suite all
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LIBERO_SUITES = ["spatial", "object", "goal", "long"]


def _resolve_suites(suite: str) -> list[str]:
    if suite.lower() == "all":
        return LIBERO_SUITES
    return [s.strip().lower() for s in suite.split(",")]


@app.command()
def smolvla(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="HF model id or local .pt path"),
    suite: str = typer.Option("all", "--suite", "-s", help="LIBERO suite(s): spatial,object,goal,long or all"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n", help="Episodes per task"),
    device: str = typer.Option("cuda", "--device", "-d"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Parallel eval envs"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project"),
) -> None:
    """Evaluate SmolVLA on LIBERO using LeRobot evaluation."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    suites = _resolve_suites(suite)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
        print(f"Loading SmolVLA from local checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        config = ckpt["config"]
        policy = SmolVLAPolicy.from_pretrained(config["model_id"])
        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Loading SmolVLA from HuggingFace: {checkpoint}")
        policy = SmolVLAPolicy.from_pretrained(checkpoint)

    policy = policy.to(device_obj)
    policy.eval()

    results = _run_libero_eval(policy, suites, num_episodes, batch_size)
    _print_results(results, "SmolVLA")

    if wandb_project:
        _log_wandb(results, wandb_project, "smolvla", checkpoint)


@app.command()
def custom(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to custom model .pt"),
    suite: str = typer.Option("all", "--suite", "-s"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    batch_size: int = typer.Option(10, "--batch-size", "-b"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project"),
) -> None:
    """Evaluate custom VLA model on LIBERO."""
    import torch

    from vla.custom_model import CustomVLA
    from vla.policy_wrapper import PolicyWrapper

    suites = _resolve_suites(suite)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading custom model from: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    config = ckpt["config"]

    model = CustomVLA(**config["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device_obj)
    model.eval()

    policy = PolicyWrapper(model, device=device_obj)

    results = _run_libero_eval(policy, suites, num_episodes, batch_size)
    _print_results(results, "CustomVLA")

    if wandb_project:
        _log_wandb(results, wandb_project, "custom", checkpoint)


def _run_libero_eval(
    policy,
    suites: list[str],
    num_episodes: int = 20,
    batch_size: int = 10,
) -> dict[str, dict[str, float]]:
    """Run LIBERO evaluation across suites using LeRobot's LiberoEnv.

    Args:
        policy: A policy with select_action(batch) -> Tensor
        suites: List of suite names to evaluate
        num_episodes: Episodes per task
        batch_size: Number of parallel environments

    Returns:
        Dict mapping suite name to {task_name: success_rate}
    """
    import numpy as np
    import torch
    from lerobot.envs.libero import LiberoEnv

    all_results: dict[str, dict[str, float]] = {}

    for suite_name in suites:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on LIBERO-{suite_name.capitalize()}")
        print(f"{'=' * 60}")

        env = LiberoEnv(
            task_suite_name=f"libero_{suite_name}",
            num=batch_size,
            obs_type="pixels_agent_pos",
        )

        task_results: dict[str, float] = {}
        num_tasks = env.num_tasks if hasattr(env, "num_tasks") else 10

        for task_id in range(num_tasks):
            task_name = f"task_{task_id}"
            successes = 0

            for ep_start in range(0, num_episodes, batch_size):
                ep_batch = min(batch_size, num_episodes - ep_start)
                obs, info = env.reset(task_id=task_id, seed=ep_start)
                policy.reset()
                done = np.zeros(ep_batch, dtype=bool)

                for _step in range(env.max_episode_steps if hasattr(env, "max_episode_steps") else 400):
                    if done.all():
                        break

                    with torch.no_grad():
                        action = policy.select_action(obs)

                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = done | terminated | truncated

                    if "is_success" in info:
                        batch_success = np.array(info["is_success"], dtype=bool)
                        done = done | batch_success

                if "is_success" in info:
                    successes += int(np.sum(info.get("is_success", np.zeros(ep_batch))))

            sr = successes / num_episodes
            task_results[task_name] = sr
            print(f"  {task_name}: {sr * 100:.1f}%")

        env.close()
        all_results[suite_name] = task_results

    return all_results


def _print_results(results: dict[str, dict[str, float]], model_name: str) -> None:
    """Print evaluation results table."""
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
    """Log evaluation results to Weights & Biases."""
    import wandb

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
