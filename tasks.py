import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "diffusion_policy"
PYTHON_VERSION = "3.11"

# Project commands
@task
def download_data(ctx: Context, skill: str = "PickCube-v1") -> None:
    """Download ManiSkill demonstrations to data/raw."""
    ctx.run(f"uv run python scripts/download_data.py --skill {skill}", echo=True, pty=not WINDOWS)


@task
def preprocess_data(ctx: Context, skill: str = "PickCube-v1", max_episodes: str = "") -> None:
    """Preprocess raw demos in data/raw to VLA .pt files in data/preprocessed."""
    cmd = f"uv run python scripts/preprocess_data.py --skill {skill}"
    if max_episodes:
        cmd += f" --max-episodes {max_episodes}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

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
def replay(ctx: Context, env: str = "PegInsertionSide-v1", episode: int = 0, speed: float = 1.0, use_actions: bool = False) -> None:
    """Replay a ManiSkill demonstration."""
    mode_flag = "--use-actions" if use_actions else "--use-env-states"
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py replay --env {env} --episode {episode} --speed {speed} {mode_flag}", echo=True, pty=not WINDOWS)

@task
def list_demos(ctx: Context, env: str = "") -> None:
    """List available ManiSkill demonstrations."""
    env_flag = f"--env {env}" if env else ""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py list-demos {env_flag}", echo=True, pty=not WINDOWS)

@task
def list_envs(ctx: Context) -> None:
    """List available ManiSkill environments."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py list-envs", echo=True, pty=not WINDOWS)
