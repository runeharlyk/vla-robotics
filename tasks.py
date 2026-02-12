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
def replay(ctx: Context, env: str = "PickCube-v1", episode: int = 0, speed: float = 1.0, loop: bool = False) -> None:
    """Replay a ManiSkill demonstration."""
    loop_flag = "--loop" if loop else ""
    ctx.run(f"uv run python src/vla/visualizer.py replay --env {env} --episode {episode} --speed {speed} {loop_flag}", echo=True, pty=not WINDOWS)

@task
def list_demos(ctx: Context, env: str = "") -> None:
    """List available ManiSkill demonstrations."""
    env_flag = f"--env {env}" if env else ""
    ctx.run(f"uv run python src/vla/visualizer.py list-demos {env_flag}", echo=True, pty=not WINDOWS)

@task
def list_envs(ctx: Context) -> None:
    """List all available ManiSkill environments."""
    ctx.run("uv run python src/vla/visualizer.py list-envs", echo=True, pty=not WINDOWS)


@task
def visualize_policy(ctx: Context, model: str = "", env: str = "PickCube-v1", speed: float = 1.0, loop: bool = False, device: str = "cuda") -> None:
    """Visualize trained policy running in environment."""
    if not model:
        model = f"models/rt1_{env.lower().replace('-', '_')}.pt"
    loop_flag = "--loop" if loop else ""
    ctx.run(f"uv run python src/vla/visualizer.py policy --model {model} --env {env} --speed {speed} --device {device} {loop_flag}", echo=True, pty=not WINDOWS)

@task
def test_env(ctx: Context, env: str = "PickCube-v1", steps: int = 100, speed: float = 1.0) -> None:
    """Test environment rendering with random actions."""
    ctx.run(f"uv run python src/vla/visualizer.py test-env --env {env} --steps {steps} --speed {speed}", echo=True, pty=not WINDOWS)


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
        f"uv run python src/vla/train.py rt1 "
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
    save_video: bool = False,
) -> None:
    """Evaluate trained RT-1 model on ManiSkill environment."""
    if not model:
        model = f"models/rt1_{env.lower().replace('-', '_')}.pt"
    video_flag = "--save-video" if save_video else ""
    cmd = (
        f"uv run python src/vla/evaluate.py rt1 "
        f"--env {env} --model {model} --num-episodes {num_episodes} "
        f"--device {device} {video_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def finetune(
    ctx: Context,
    env: str = "PickCube-v1",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-5,
    model_id: str = "lerobot/smolvla_base",
    seq_len: int = 4,
    device: str = "cuda",
    freeze_vision: bool = True,
    wandb_project: str = "vla-smolvla",
    val_split: float = 0.1,
    patience: int = 5,
    amp: bool = False,
    num_workers: int = 4,
) -> None:
    """Finetune SmoLVLA on preprocessed ManiSkill demonstrations."""
    freeze_flag = "--freeze-vision" if freeze_vision else "--no-freeze-vision"
    amp_flag = "--amp" if amp else "--no-amp"
    cmd = (
        f"uv run python src/vla/train_vla.py "
        f"--env {env} --epochs {epochs} --batch-size {batch_size} "
        f"--lr {lr} --model-id {model_id} --seq-len {seq_len} "
        f"--device {device} {freeze_flag} --wandb-project {wandb_project} "
        f"--val-split {val_split} --patience {patience} "
        f"{amp_flag} --num-workers {num_workers}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)