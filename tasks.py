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

@task
def train_rt1(
    ctx: Context,
    env: str = "PickCube-v1",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_size: str = "small",
    device: str = "cuda",
    amp: bool = True,
) -> None:
    """Train RT-1 on preprocessed ManiSkill demonstrations."""
    amp_flag = "--amp" if amp else "--no-amp"
    cmd = (
        f"uv run python src/vla/train_rt1.py train "
        f"--env {env} --epochs {epochs} --batch-size {batch_size} "
        f"--lr {lr} --model-size {model_size} --device {device} {amp_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def eval_rt1(
    ctx: Context,
    env: str = "PickCube-v1",
    model: str = "",
    num_episodes: int = 20,
    device: str = "cuda",
    render: bool = False,
) -> None:
    """Evaluate trained RT-1 model on ManiSkill environment."""
    if not model:
        model = f"models/rt1_{env.lower().replace('-', '_')}.pt"
    render_flag = "--render" if render else "--no-render"
    cmd = (
        f"uv run python src/vla/train_rt1.py evaluate "
        f"--env {env} --model {model} --num-episodes {num_episodes} "
        f"--device {device} {render_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)
