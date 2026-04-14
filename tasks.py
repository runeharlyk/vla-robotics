import os
import sys
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task

WINDOWS = os.name == "nt"

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
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff linter."""
    flag = "--fix" if fix else ""
    ctx.run(f"uv run ruff check {flag} src/ tests/", echo=True, pty=not WINDOWS)


@task
def format(ctx: Context, check: bool = False) -> None:
    """Run ruff formatter."""
    flag = "--check" if check else ""
    ctx.run(f"uv run ruff format {flag} src/ tests/", echo=True, pty=not WINDOWS)


@task
def type_check(ctx: Context) -> None:
    """Run pyright type checking."""
    ctx.run("uv run pyright", echo=True, pty=not WINDOWS)


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
def visualize(
    ctx: Context,
    model: str = "smolvla",
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    simulator: str = "libero",
    suite: str = "long",
    env_id: str = "",
    tasks: str = "",
    episodes: int = 1,
    device: str = "cuda",
    output_dir: str = "videos",
    fps: int = 30,
    seed: int = 0,
) -> None:
    """Record policy rollout videos via `vla visualize`."""
    env_flag = f"--env-id {env_id}" if env_id else ""
    tasks_flag = f"--tasks {tasks}" if tasks else ""
    cmd = (
        f"uv run python -m vla visualize "
        f"--model {model} --checkpoint {checkpoint} --simulator {simulator} "
        f"--suite {suite} {env_flag} {tasks_flag} --episodes {episodes} "
        f"--device {device} --output-dir {output_dir} --fps {fps} --seed {seed}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def playback_demos(
    ctx: Context,
    simulator: str = "libero",
    suite: str = "long",
    env_id: str = "",
    data_path: str = "",
    mode: str = "replay",
    episodes: str = "0",
    output_dir: str = "videos/playback",
    fps: int = 30,
    seed: int = 0,
    instruction: str = "",
) -> None:
    """Playback recorded demonstrations via `vla playback`."""
    env_flag = f"--env-id {env_id}" if env_id else ""
    data_flag = f"--data-path {data_path}" if data_path else ""
    instr_flag = f'--instruction "{instruction}"' if instruction else ""
    cmd = (
        f"uv run python -m vla playback "
        f"--simulator {simulator} --suite {suite} {env_flag} {data_flag} "
        f"--mode {mode} --episodes {episodes} --output-dir {output_dir} "
        f"--fps {fps} --seed {seed} {instr_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


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
    import yaml  # lazy import - not available in the bare `uvx invoke` env

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
    env_source = ". jobs/_env.sh\n"

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


@task
def preprocess(
    ctx: Context,
    skill: str = "PegInsertionSide-v1",
    raw_dir: str = "data/raw",
    out_dir: str = "data/preprocessed",
    num_cameras: int = 2,
    image_size: int = 256,
    max_traj: int = 0,
) -> None:
    """Preprocess ManiSkill raw demos into a VLA-ready .pt file."""
    max_flag = f"--max-traj {max_traj}" if max_traj > 0 else ""
    cmd = (
        f"uv run python scripts/preprocess_data.py "
        f"--skill {skill} --raw-dir {raw_dir} --out-dir {out_dir} "
        f"--num-cameras {num_cameras} --image-size {image_size} {max_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_sft(
    ctx: Context,
    num_demos: int = 10,
    seed: int = 42,
    data: str = "",
    libero_suite: str = "",
    simulator: str = "",
    eval_suite: str = "all",
    resume: str = "",
    no_wandb: bool = False,
) -> None:
    """Run SmolVLA SFT (behavior cloning) training.

    Data from --data (.pt files) or --libero-suite (HF direct).
    """
    wandb_flag = "--no-wandb" if no_wandb else "--wandb"
    data_flag = f"--data {data}" if data else ""
    libero_flag = f"--libero-suite {libero_suite}" if libero_suite else ""
    sim_flag = f"--simulator {simulator}" if simulator else ""
    resume_flag = f"--resume {resume}" if resume else ""
    ctx.run(
        f"uv run python scripts/train_sft.py {data_flag} {libero_flag} "
        f"--num-demos {num_demos} --seed {seed} "
        f"{sim_flag} --eval-suite {eval_suite} {resume_flag} {wandb_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train_srpo(
    ctx: Context,
    sft_checkpoint: str = "checkpoints/sft/demos5_seed42/best",
    mode: str = "srpo",
    num_demos: int = 5,
    seed: int = 42,
    world_model: str = "dinov2",
    no_wandb: bool = False,
) -> None:
    """Run SRPO or sparse-RL training from an SFT checkpoint."""
    wandb_flag = "--no-wandb" if no_wandb else "--wandb"
    ctx.run(
        f"uv run python scripts/train_srpo.py --sft-checkpoint {sft_checkpoint} --mode {mode} "
        f"--num-demos {num_demos} --seed {seed} --world-model {world_model} {wandb_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def run_experiment(ctx: Context, config: str = "configs/srpo_pickcube.yaml", no_wandb: bool = False) -> None:
    """Run the full SRPO validation experiment matrix."""
    wandb_flag = "--no-wandb" if no_wandb else ""
    ctx.run(
        f"uv run python scripts/run_experiment.py --config {config} {wandb_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def evaluate_policy(
    ctx: Context,
    checkpoint_dir: str = "checkpoints/sft/demos10_seed42/best",
    num_episodes: int = 100,
    env: str = "",
    simulator: str = "maniskill",
    suite: str = "all",
) -> None:
    """Evaluate a saved policy checkpoint (env_id auto-read from checkpoint metadata)."""
    env_flag = f"--env {env}" if env else ""
    cmd = (
        f"uv run python scripts/evaluate.py "
        f"--checkpoint-dir {checkpoint_dir} --num-episodes {num_episodes} "
        f"--simulator {simulator} --suite {suite} {env_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)
