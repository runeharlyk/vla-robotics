import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "diffusion_policy"
PYTHON_VERSION = "3.11"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

@task
def visualize(ctx: Context, env: str = "PickCube-v1", shader: str = "default") -> None:
    """Run ManiSkill visualizer."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py visualize --env {env} --shader {shader}", echo=True, pty=not WINDOWS)

@task
def list_envs(ctx: Context) -> None:
    """List available ManiSkill environments."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py list-envs", echo=True, pty=not WINDOWS)

@task
def record(ctx: Context, env: str = "PickCube-v1", output: str = "output.mp4") -> None:
    """Record a video of the environment."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py record --env {env} --output {output}", echo=True, pty=not WINDOWS)
