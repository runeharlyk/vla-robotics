import importlib

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Vision-Language-Action robotics CLI."""


@app.command()
def train(
    ctx: typer.Context,
    suite: str = typer.Option("all", "--suite", "-s"),
    steps: int = typer.Option(30000, "--steps"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(3e-4, "--lr"),
    device: str = typer.Option("cuda", "--device", "-d"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
) -> None:
    """Train custom VLA on LIBERO demonstrations."""
    mod = importlib.import_module("vla.training.train_custom")
    mod.train(
        suite=suite,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        device=device,
        amp=amp,
    )


@app.command()
def evaluate(
    ctx: typer.Context,
    model: str = typer.Option("smolvla", "--model", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    suite: str = typer.Option("all", "--suite", "-s"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
    wandb_project: str = typer.Option(None, "--wandb-project"),
) -> None:
    """Evaluate VLA models on LIBERO."""
    mod = importlib.import_module("vla.evaluation.evaluate")
    mod.evaluate(
        model=model,
        checkpoint=checkpoint,
        suite=suite,
        num_episodes=num_episodes,
        device=device,
        wandb_project=wandb_project,
    )


@app.command()
def visualize(
    ctx: typer.Context,
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    suite: str = typer.Option("long", "--suite", "-s"),
    episodes: int = typer.Option(1, "--episodes", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
) -> None:
    """Record SmolVLA rollout videos on LIBERO."""
    mod = importlib.import_module("vla.evaluation.visualize")
    mod.main(
        checkpoint=checkpoint,
        suite=suite,
        episodes=episodes,
        device=device,
    )


if __name__ == "__main__":
    app()
