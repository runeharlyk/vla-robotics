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
def train_clip(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    task: str = "insert the peg into the hole",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    clip_model: str = "ViT-B/32",
) -> None:
    """Train CLIP action model using behavioral cloning."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/train.py "
        f'--env {env} --task "{task}" --epochs {epochs} --batch-size {batch_size} --lr {lr} '
        f"--clip-model {clip_model}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate_clip(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    checkpoint: str = "checkpoints/PegInsertionSide-v1/best.pt",
    task: str = "insert the peg into the hole",
    num_episodes: int = 10,
) -> None:
    """Evaluate trained CLIP action model in simulation."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/evaluate.py evaluate "
        f'--env {env} --checkpoint {checkpoint} --task "{task}" --num-episodes {num_episodes}'
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def visualize_trained(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    checkpoint: str = "checkpoints/PegInsertionSide-v1/best.pt",
    task: str = "insert the peg into the hole",
    max_steps: int = 1000,
) -> None:
    """Visualize trained model in ManiSkill simulation."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/evaluate.py visualize-trained "
        f'--env {env} --checkpoint {checkpoint} --task "{task}" --max-steps {max_steps}'
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def download_demos(ctx: Context, env: str = "PegInsertionSide-v1") -> None:
    """Download demonstration data for an environment."""
    ctx.run(f'uv run python -m mani_skill.utils.download_demo "{env}"', echo=True, pty=not WINDOWS)


@task
def convert_demos(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    source: str = "motionplanning",
    max_trajectories: int = 100,
) -> None:
    """Convert demos to include RGB observations."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/convert_demos.py convert "
        f"--env {env} --source {source} --max {max_trajectories}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def generate_dataset(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    source: str = "motionplanning",
    max_trajectories: int = 100,
) -> None:
    """Generate RGB dataset by replaying demos with viewer."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/generate_dataset.py generate "
        f"--env {env} --source {source} --max {max_trajectories}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def check_demos(ctx: Context, env: str = "PegInsertionSide-v1") -> None:
    """Check available demo files."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/generate_dataset.py check --env {env}", echo=True, pty=not WINDOWS)


@task
def train_state(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_traj: int = 100,
) -> None:
    """Train state-based action model (no RGB needed)."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/train_state.py "
        f"--env {env} --epochs {epochs} --batch-size {batch_size} --lr {lr} --max-traj {max_traj}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate_state(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    checkpoint: str = "",
    num_episodes: int = 10,
    max_steps: int = 200,
    vis: bool = False,
) -> None:
    """Evaluate state-based action model in simulation."""
    if not checkpoint:
        checkpoint = f"checkpoints/state_model_{env.replace('-', '_')}_best.pt"
    vis_flag = "--vis" if vis else ""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/evaluate_state.py "
        f"--env {env} --checkpoint {checkpoint} --num-episodes {num_episodes} --max-steps {max_steps} {vis_flag}"
    )
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
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
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
def visualize(ctx: Context, env: str = "PegInsertionSide-v1", shader: str = "default") -> None:
    """Run ManiSkill visualizer."""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/visualizer.py visualize --env {env} --shader {shader}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def list_envs(ctx: Context) -> None:
    """List available ManiSkill environments."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/visualizer.py list-envs", echo=True, pty=not WINDOWS)


@task
def record(ctx: Context, env: str = "PegInsertionSide-v1", output: str = "output.mp4") -> None:
    """Record a video of the environment."""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/visualizer.py record --env {env} --output {output}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def clip_visualize(
    ctx: Context,
    env: str = "PegInsertionSide-v1",
    task: str = "pick up the apple and place it in the bowl",
    clip_model: str = "ViT-B/32",
    max_steps: int = 1000,
) -> None:
    """Run CLIP action model in ManiSkill visualizer (untrained model)."""
    cmd = (
        f"uv run python src/{PROJECT_NAME}/visualizer.py clip-policy "
        f'--env {env} --task "{task}" --clip-model {clip_model} --max-steps {max_steps}'
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def clip_demo(
    ctx: Context,
    action_dim: int = 7,
    clip_model: str = "ViT-B/32",
    use_history: bool = False,
    history_length: int = 4,
) -> None:
    """Run CLIP action model demo."""
    cmd = f"uv run python src/{PROJECT_NAME}/clip_demo.py demo --action-dim {action_dim} --clip-model {clip_model}"
    if use_history:
        cmd += f" --use-history --history-length {history_length}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def clip_info(ctx: Context) -> None:
    """Show available CLIP models and configuration."""
    ctx.run(f"uv run python src/{PROJECT_NAME}/clip_demo.py info", echo=True, pty=not WINDOWS)
