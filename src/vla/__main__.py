import importlib

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Vision-Language-Action robotics CLI."""


@app.command()
def train(
    ctx: typer.Context,
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("all", "--suite", "-s"),
    data_path: str | None = typer.Option(None, "--data-path"),
    instruction: str = typer.Option("", "--instruction"),
    action_dim: int = typer.Option(7, "--action-dim"),
    steps: int = typer.Option(30000, "--steps"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(3e-4, "--lr"),
    device: str = typer.Option("cuda", "--device", "-d"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
) -> None:
    """SFT training of custom VLA on demonstration data."""
    mod = importlib.import_module("vla.training.sft")
    mod.train(
        simulator=simulator,
        suite=suite,
        data_path=data_path,
        instruction=instruction,
        action_dim=action_dim,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        amp=amp,
    )


@app.command()
def train_rl(
    ctx: typer.Context,
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("long", "--suite", "-s"),
    env_id: str | None = typer.Option(None, "--env-id"),
    num_iterations: int = typer.Option(100, "--num-iterations"),
    trajectories_per_iter: int = typer.Option(16, "--trajectories-per-iter"),
    lr: float = typer.Option(1e-5, "--lr"),
    max_steps: int = typer.Option(300, "--max-steps"),
    device: str = typer.Option("cuda", "--device", "-d"),
) -> None:
    """RL fine-tuning of a pre-trained policy via online rollouts."""
    mod = importlib.import_module("vla.training.rl")
    mod.train_rl(
        checkpoint=checkpoint,
        simulator=simulator,
        suite=suite,
        env_id=env_id,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        lr=lr,
        max_steps=max_steps,
        device=device,
    )


@app.command()
def evaluate(
    ctx: typer.Context,
    model: str = typer.Option("smolvla", "--model", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("all", "--suite", "-s"),
    env_id: str | None = typer.Option(None, "--env-id"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    wandb_project: str | None = typer.Option(None, "--wandb-project"),
) -> None:
    """Evaluate VLA models on a simulator."""
    mod = importlib.import_module("vla.evaluation.evaluate")
    mod.evaluate(
        model=model,
        checkpoint=checkpoint,
        simulator=simulator,
        suite=suite,
        env_id=env_id,
        num_episodes=num_episodes,
        device=device,
        wandb_project=wandb_project,
    )


@app.command()
def visualize(
    ctx: typer.Context,
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
) -> None:
    """Record policy rollout videos on a simulator."""
    mod = importlib.import_module("vla.evaluation.visualize")
    mod.main(
        model=model,
        checkpoint=checkpoint,
        simulator=simulator,
        suite=suite,
        env_id=env_id,
        episodes=episodes,
        device=device,
        output_dir=output_dir,
        tasks=tasks,
        fps=fps,
        seed=seed,
    )


@app.command()
def playback(
    ctx: typer.Context,
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
    """Playback recorded demonstrations as videos."""
    mod = importlib.import_module("vla.evaluation.playback")
    mod.playback(
        simulator=simulator,
        suite=suite,
        env_id=env_id,
        data_path=data_path,
        mode=mode,
        episodes=episodes,
        output_dir=output_dir,
        fps=fps,
        seed=seed,
        instruction=instruction,
    )


if __name__ == "__main__":
    app()
