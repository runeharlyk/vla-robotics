import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "vla"
PYTHON_VERSION = "3.11"


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


@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def download_libero(ctx: Context, suite: str = "all") -> None:
    """Download LIBERO datasets from HuggingFace via LeRobot."""
    ctx.run(f"uv run python scripts/download_libero.py --suite {suite}", echo=True, pty=not WINDOWS)


@task
def finetune_smolvla(
    ctx: Context,
    suite: str = "all",
    steps: int = 20000,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda",
    amp: bool = False,
) -> None:
    """Fine-tune SmolVLA on LIBERO demonstrations."""
    amp_flag = "--amp" if amp else "--no-amp"
    cmd = (
        f"uv run python src/vla/train_smolvla.py "
        f"--suite {suite} --steps {steps} --batch-size {batch_size} "
        f"--lr {lr} --device {device} {amp_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_custom(
    ctx: Context,
    suite: str = "all",
    steps: int = 30000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    amp: bool = False,
) -> None:
    """Train custom VLA on LIBERO demonstrations."""
    amp_flag = "--amp" if amp else "--no-amp"
    cmd = (
        f"uv run python src/vla/train_custom.py "
        f"--suite {suite} --steps {steps} --batch-size {batch_size} "
        f"--lr {lr} --device {device} {amp_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def eval_smolvla(
    ctx: Context,
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    suite: str = "all",
    num_episodes: int = 20,
    device: str = "cuda",
) -> None:
    """Evaluate SmolVLA on LIBERO."""
    cmd = (
        f"uv run python src/vla/evaluate.py smolvla "
        f"--checkpoint {checkpoint} --suite {suite} "
        f"--num-episodes {num_episodes} --device {device}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def eval_custom(
    ctx: Context,
    checkpoint: str = "",
    suite: str = "all",
    num_episodes: int = 20,
    device: str = "cuda",
) -> None:
    """Evaluate custom VLA on LIBERO."""
    if not checkpoint:
        checkpoint = "models/custom_vla_libero_4_suites.pt"
    cmd = (
        f"uv run python src/vla/evaluate.py custom "
        f"--checkpoint {checkpoint} --suite {suite} "
        f"--num-episodes {num_episodes} --device {device}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def download_maniskill(ctx: Context, skill: str = "PickCube-v1") -> None:
    """Download ManiSkill demonstrations to data/raw."""
    ctx.run(f"uv run python scripts/download_data.py --skill {skill}", echo=True, pty=not WINDOWS)


@task
def preprocess_maniskill(ctx: Context, skill: str = "PickCube-v1", max_episodes: str = "") -> None:
    """Preprocess raw ManiSkill demos to HDF5 in data/preprocessed."""
    cmd = f"uv run python scripts/preprocess_data.py --skill {skill}"
    if max_episodes:
        cmd += f" --max-episodes {max_episodes}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


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
        f"uv run python src/maniskill/train.py rt1 "
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
        f"uv run python src/maniskill/evaluate.py rt1 "
        f"--env {env} --model {model} --num-episodes {num_episodes} "
        f"--device {device} {video_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def replay(ctx: Context, env: str = "PickCube-v1", episode: int = 0, speed: float = 1.0, loop: bool = False) -> None:
    """Replay a ManiSkill demonstration."""
    loop_flag = "--loop" if loop else ""
    ctx.run(
        f"uv run python src/maniskill/visualizer.py replay --env {env} --episode {episode} --speed {speed} {loop_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def list_demos(ctx: Context, env: str = "") -> None:
    """List available ManiSkill demonstrations."""
    env_flag = f"--env {env}" if env else ""
    ctx.run(f"uv run python src/maniskill/visualizer.py list-demos {env_flag}", echo=True, pty=not WINDOWS)


@task
def list_envs(ctx: Context) -> None:
    """List all available ManiSkill environments."""
    ctx.run("uv run python src/maniskill/visualizer.py list-envs", echo=True, pty=not WINDOWS)


@task
def visualize_maniskill(
    ctx: Context,
    env: str = "PickCube-v1",
    steps: int = 200,
    seed: int = 0,
    save: bool = False,
) -> None:
    """Visualize a ManiSkill task with random actions."""
    save_flag = "--save" if save else "--no-save"
    ctx.run(
        f"uv run python scripts/visualize.py maniskill --env {env} --steps {steps} --seed {seed} {save_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize_libero(
    ctx: Context,
    suite: str = "long",
    task_id: int = 0,
    steps: int = 300,
    seed: int = 0,
    save: bool = False,
) -> None:
    """Visualize a LIBERO task with random actions."""
    save_flag = "--save" if save else "--no-save"
    ctx.run(
        f"uv run python scripts/visualize.py libero --suite {suite} --task {task_id} --steps {steps} --seed {seed} {save_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize_smolvla(
    ctx: Context,
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    suite: str = "long",
    tasks: str = "",
    episodes: int = 1,
    device: str = "cuda",
) -> None:
    """Visualize SmolVLA policy on LIBERO tasks."""
    tasks_flag = f"--tasks {tasks}" if tasks else ""
    ctx.run(
        f"uv run python scripts/visualize.py smolvla "
        f"--checkpoint {checkpoint} --suite {suite} {tasks_flag} "
        f"--episodes {episodes} --device {device}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize_smolvla_maniskill(
    ctx: Context,
    checkpoint: str = "",
    env: str = "",
    episodes: int = 1,
    steps: int = 200,
    device: str = "cuda",
) -> None:
    """Visualize SmolVLA policy on a ManiSkill environment."""
    if not checkpoint:
        checkpoint = "models/smolvla_pickcube_v1.pt"
    env_flag = f"--env {env}" if env else ""
    ctx.run(
        f"uv run python scripts/visualize.py smolvla-maniskill "
        f"--checkpoint {checkpoint} {env_flag} "
        f"--episodes {episodes} --steps {steps} --device {device}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def list_tasks(ctx: Context, benchmark: str = "") -> None:
    """List available tasks for ManiSkill and/or LIBERO."""
    bench_flag = f"--benchmark {benchmark}" if benchmark else ""
    ctx.run(f"uv run python scripts/visualize.py list {bench_flag}", echo=True, pty=not WINDOWS)