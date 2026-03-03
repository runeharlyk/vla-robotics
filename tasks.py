import os
import sys
from pathlib import Path

import yaml
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "vla"
PYTHON_VERSION = "3.11"

JOBS_DIR = Path(__file__).parent / "jobs"
PROFILES_PATH = JOBS_DIR / "_profiles.yaml"
ACTIONS_PATH = JOBS_DIR / "_actions.yaml"

LSF_HEADER = """\
#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -W {walltime}
#BSUB -n {cores}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem={mem}]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/{job_name}/%J.out
# -------------------------------------------------
"""


@task
def lint(ctx: Context) -> None:
    """Run ruff linter."""
    ctx.run("uv run ruff check src/ tests/", echo=True, pty=not WINDOWS)


@task
def format(ctx: Context, check: bool = False) -> None:
    """Run ruff formatter."""
    flag = "--check" if check else ""
    ctx.run(f"uv run ruff format {flag} src/ tests/", echo=True, pty=not WINDOWS)


@task
def type_check(ctx: Context) -> None:
    """Run mypy type checking."""
    ctx.run("uv run mypy src/", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docs(ctx: Context, serve: bool = False) -> None:
    """Build (or serve) mkdocs documentation."""
    cmd = "uv run mkdocs serve" if serve else "uv run mkdocs build --strict"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


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
    simulator: str = "libero",
    suite: str = "all",
    steps: int = 30000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    amp: bool = False,
) -> None:
    """SFT training of custom VLA on demonstration data."""
    amp_flag = "--amp" if amp else "--no-amp"
    cmd = (
        f"uv run python -m vla train "
        f"--simulator {simulator} --suite {suite} --steps {steps} --batch-size {batch_size} "
        f"--lr {lr} --device {device} {amp_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate(
    ctx: Context,
    model: str = "smolvla",
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    simulator: str = "libero",
    suite: str = "all",
    env_id: str = "",
    num_episodes: int = 20,
    device: str = "cuda",
) -> None:
    env_flag = f"--env-id {env_id}" if env_id else ""
    cmd = (
        f"uv run python -m vla evaluate "
        f"--model {model} --checkpoint {checkpoint} --simulator {simulator} "
        f"--suite {suite} {env_flag} --num-episodes {num_episodes} --device {device}"
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


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_job_script(config_path: str, gpu: str) -> tuple[str, str]:
    profiles = _load_yaml(PROFILES_PATH)
    actions = _load_yaml(ACTIONS_PATH)
    config = _load_yaml(Path(config_path))

    if gpu not in profiles:
        available = ", ".join(profiles)
        print(f"Unknown GPU profile '{gpu}'. Available: {available}", file=sys.stderr)
        raise SystemExit(1)

    action = config.get("action")
    if not action:
        print(f"Config '{config_path}' is missing the 'action' field.", file=sys.stderr)
        print(f"  Available actions: {', '.join(actions)}", file=sys.stderr)
        raise SystemExit(1)

    if action not in actions:
        available = ", ".join(actions)
        print(f"Unknown action '{action}' in config. Available: {available}", file=sys.stderr)
        raise SystemExit(1)

    profile = profiles[gpu]
    template = actions[action]

    config_stem = Path(config_path).stem
    job_name = f"{config_stem}_{gpu}"

    header = LSF_HEADER.format(job_name=job_name, **profile)
    env_source = '. "$LSB_SUBCWD/jobs/_env.sh"\n'

    placeholders = {**config, "job_name": job_name}
    body = template.format_map(placeholders)

    script = header + env_source + "\n" + body
    return job_name, script


@task
def create_job(
    ctx: Context,
    config: str = "",
    gpu: str = "a100",
) -> None:
    """Generate a job script from an experiment config and save it to jobs/."""
    if not config:
        print("Usage: invoke create-job --config <path> [--gpu <profile>]")
        print(f"  GPU profiles: {', '.join(_load_yaml(PROFILES_PATH))}")
        raise SystemExit(1)

    job_name, script = _generate_job_script(config, gpu)
    script_path = JOBS_DIR / f"{job_name}.sh"

    script_path.write_text(script, newline="\n")
    print(f"Saved: {script_path}\n")
    print("--- generated script ---")
    print(script)
    print("--- end ---")
    print(f"\nSubmit with:  bsub < {script_path}")
