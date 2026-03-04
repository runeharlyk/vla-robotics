import importlib

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Vision-Language-Action robotics CLI — SmolVLA training & evaluation."""


@app.command()
def train(
    ctx: typer.Context,
) -> None:
    """SFT training. Use ``scripts/train_sft.py`` for full control."""
    typer.echo(
        "For SmolVLA SFT training, run:\n"
        "  uv run python scripts/train_sft.py --help\n\n"
        "Example:\n"
        "  uv run python scripts/train_sft.py --data data/preprocessed/peginsertionside.pt"
    )
    raise typer.Exit(0)


@app.command()
def train_rl(
    ctx: typer.Context,
) -> None:
    """RL fine-tuning (SRPO / sparse-RL). Use ``scripts/train_srpo.py`` for full control."""
    typer.echo(
        "For SRPO / sparse-RL training, run:\n"
        "  uv run python scripts/train_srpo.py --help\n\n"
        "Example:\n"
        "  uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/.../best"
    )
    raise typer.Exit(0)


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
    """Evaluate SmolVLA on a simulator."""
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
