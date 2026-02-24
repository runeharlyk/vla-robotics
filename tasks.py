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
    config: str = "",
    suite: str = "all",
    steps: int = 20000,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    if config:
        cmd = f"uv run lerobot_train --config_path {config}"
    else:
        cmd = (
            f"uv run lerobot_train "
            f"--dataset.repo_id=lerobot/libero_10_image "
            f"--policy.type=smolvla "
            f"--steps={steps} "
            f"--batch_size={batch_size} "
            f"--device={device}"
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
        f"uv run python -m vla train "
        f"--suite {suite} --steps {steps} --batch-size {batch_size} "
        f"--lr {lr} --device {device} {amp_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate(
    ctx: Context,
    model: str = "smolvla",
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    suite: str = "all",
    num_episodes: int = 20,
    device: str = "cuda",
) -> None:
    cmd = (
        f"uv run python -m vla evaluate "
        f"--model {model} --checkpoint {checkpoint} --suite {suite} "
        f"--num-episodes {num_episodes} --device {device}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def visualize_libero(
    ctx: Context,
    suite: str = "long",
    task_id: int = 0,
    steps: int = 300,
    seed: int = 0,
    save: bool = False,
) -> None:
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
    tasks_flag = f"--tasks {tasks}" if tasks else ""
    ctx.run(
        f"uv run python scripts/visualize.py smolvla "
        f"--checkpoint {checkpoint} --suite {suite} {tasks_flag} "
        f"--episodes {episodes} --device {device}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def list_tasks(ctx: Context, benchmark: str = "") -> None:
    bench_flag = f"--benchmark {benchmark}" if benchmark else ""
    ctx.run(f"uv run python scripts/visualize.py list {bench_flag}", echo=True, pty=not WINDOWS)

@task
def export_images(
    ctx: Context,
    dataset: str = "libero",
    suite: str = "",
    max_episodes: int = 0,
) -> None:
    """Export PNG images from LIBERO / LIBERO-Pro datasets to data/images/."""
    dataset_flag = f"--dataset {dataset}"
    suite_flag = f"--suites {suite}" if suite else ""
    max_flag = f"--max-episodes {max_episodes}" if max_episodes > 0 else ""
    cmd = f"uv run python -m vla.data.image_export {dataset_flag} {suite_flag} {max_flag}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)
