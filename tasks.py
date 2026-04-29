import os
import shlex
import sys
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task

from vla.utils.experiment_jobs import (
    GENERATED_JOBS_DIR,
    PROFILES_PATH,
    as_str_set,
    format_percent,
    generate_hydra_job_script,
    generate_train_eval_job_script,
    hydra_experiment_dir,
    load_train_experiment,
    load_training_records,
    load_yaml,
    matching_training_records,
    print_submit_validation,
    read_experiment_summary,
    validate_hydra_submit,
    validate_train_eval_submit,
)

WINDOWS = os.name == "nt"

PROJECT_ROOT = Path(__file__).parent.resolve()


def _task_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if WINDOWS:
        tmp_dir = (PROJECT_ROOT / ".tmp").resolve()
        env.update(
            {
                "UV_CACHE_DIR": str((PROJECT_ROOT / ".uv-cache").resolve()),
                "HF_HOME": str((PROJECT_ROOT / ".hf-cache").resolve()),
                "WANDB_DIR": str((PROJECT_ROOT / ".wandb").resolve()),
                "TEMP": str(tmp_dir),
                "TMP": str(tmp_dir),
            }
        )
        for value in env.values():
            Path(value).mkdir(parents=True, exist_ok=True)
    return env


def _run(ctx: Context, cmd: str) -> None:
    ctx.run(cmd, echo=True, pty=not WINDOWS, env=_task_env())


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff linter."""
    flag = "--fix" if fix else ""
    _run(ctx, f"uv run ruff check {flag} src/ tests/")


@task
def format(ctx: Context, check: bool = False) -> None:
    """Run ruff formatter."""
    flag = "--check" if check else ""
    _run(ctx, f"uv run ruff format {flag} src/ tests/")


@task
def type_check(ctx: Context) -> None:
    """Run pyright type checking."""
    _run(ctx, "uv run pyright")


@task
def test(ctx: Context) -> None:
    """Run tests."""
    _run(ctx, "uv run coverage run -m pytest tests/")
    _run(ctx, "uv run coverage report -m -i")


@task
def docs(ctx: Context, serve: bool = False) -> None:
    """Build (or serve) mkdocs documentation."""
    cmd = "uv run mkdocs serve" if serve else "uv run mkdocs build --strict"
    _run(ctx, cmd)


@task
def download_libero(ctx: Context, suite: str = "all") -> None:
    """Download LIBERO datasets from HuggingFace via LeRobot."""
    _run(ctx, f"uv run python scripts/download_libero.py --suite {suite}")


@task
def setup_libero(
    ctx: Context,
    config_dir: str = "",
    datasets_dir: str = "",
    install: bool = False,
    source_dir: str = "",
    variant: str = "libero",
) -> None:
    """Create a non-interactive LIBERO config and print runtime details."""
    config_flag = f' --config-dir "{config_dir}"' if config_dir else ""
    datasets_flag = f' --datasets-dir "{datasets_dir}"' if datasets_dir else ""
    install_flag = " --install" if install else ""
    source_flag = f' --source-dir "{source_dir}" --variant {variant}' if source_dir else ""
    _run(ctx, f"uv run python scripts/setup_libero.py{config_flag}{datasets_flag}{install_flag}{source_flag}")


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
    _run(ctx, cmd)


@task
def visualize(
    ctx: Context,
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
    num_envs: int = 1,
    save_all_parallel_videos: bool = False,
    show: bool = False,
) -> None:
    """Record policy rollout videos via `vla visualize`."""
    env_flag = f"--env-id {env_id}" if env_id else ""
    tasks_flag = f"--tasks {tasks}" if tasks else ""
    show_flag = "--show" if show else ""
    num_envs_flag = f"--num-envs {num_envs}" if num_envs != 1 else ""
    save_all_flag = "--save-all-parallel-videos" if save_all_parallel_videos else ""
    cmd = (
        f"uv run python -m vla visualize "
        f"--checkpoint {checkpoint} --simulator {simulator} "
        f"--suite {suite} {env_flag} {tasks_flag} --episodes {episodes} "
        f"--device {device} --output-dir {output_dir} --fps {fps} --seed {seed} "
        f"{num_envs_flag} {save_all_flag} {show_flag}"
    )
    _run(ctx, cmd)


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
    _run(ctx, cmd)


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
    _run(ctx, cmd)


def _write_or_submit_hydra_job(ctx: Context, kind: str, experiment: str, profile: str, submit: bool) -> None:
    if not experiment:
        print(f"Usage: invoke submit-{kind} --experiment <name> [--profile <profile>] [--submit]")
        print(f"  GPU profiles: {', '.join(load_yaml(PROFILES_PATH))}")
        raise SystemExit(1)

    validation = validate_hydra_submit(kind, experiment, profile, submit=submit)
    print_submit_validation(validation)
    if validation.errors:
        raise SystemExit(1)

    generated = generate_hydra_job_script(kind, experiment, profile)
    GENERATED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = GENERATED_JOBS_DIR / f"{generated.name}.sh"
    script_path.write_text(generated.script, newline="\n")
    print(f"Saved: {script_path}")

    if submit:
        if WINDOWS:
            print("Refusing to call bsub from Windows. Copy or submit the generated script on HPC.", file=sys.stderr)
            raise SystemExit(1)
        ctx.run(f"bsub < {shlex.quote(str(script_path))}", echo=True, pty=True)
    else:
        print(f"Submit with: bsub < {script_path}")


@task
def submit_train(
    ctx: Context,
    experiment: str = "",
    profile: str = "l40s-16",
    submit: bool = False,
) -> None:
    """Create or submit a Hydra train job from configs/train_srpo/experiment."""
    _write_or_submit_hydra_job(ctx, "train", experiment, profile, submit)


@task
def submit_eval(
    ctx: Context,
    experiment: str = "",
    profile: str = "l40s-16",
    checkpoint: str = "best",
    submit: bool = False,
) -> None:
    """Create or submit an eval job for a train experiment checkpoint."""
    if not experiment:
        print("Usage: invoke submit-eval --experiment <train-config> [--checkpoint best|last|best-rollout]")
        print(f"  GPU profiles: {', '.join(load_yaml(PROFILES_PATH))}")
        raise SystemExit(1)

    train_experiment_path = hydra_experiment_dir("train") / f"{experiment}.yaml"
    if train_experiment_path.exists():
        validation = validate_train_eval_submit(experiment, profile, checkpoint, submit=submit)
        print_submit_validation(validation)
        if validation.errors:
            raise SystemExit(1)

        generated = generate_train_eval_job_script(experiment, profile, checkpoint)
        GENERATED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        script_path = GENERATED_JOBS_DIR / f"{generated.name}.sh"
        script_path.write_text(generated.script, newline="\n")
        print(f"Saved: {script_path}")

        if submit:
            if WINDOWS:
                print(
                    "Refusing to call bsub from Windows. Copy or submit the generated script on HPC.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            ctx.run(f"bsub < {shlex.quote(str(script_path))}", echo=True, pty=True)
        else:
            print(f"Submit with: bsub < {script_path}")
        return

    # Backward-compatible path for explicit eval protocol configs.
    _write_or_submit_hydra_job(ctx, "eval", experiment, profile, submit)


@task
def list_experiments(ctx: Context, kind: str = "train") -> None:
    """List Hydra train/eval experiment configs and their metadata."""
    del ctx
    experiment_dir = hydra_experiment_dir(kind)
    rows = [read_experiment_summary(path) for path in sorted(experiment_dir.glob("*.yaml"))]
    if not rows:
        print(f"No {kind} experiments found in {experiment_dir}")
        return

    print(f"{kind} experiments in {experiment_dir}:")
    for row in rows:
        suffix = []
        if row["wandb_name"]:
            suffix.append(f"wandb={row['wandb_name']}")
        if row["source_job"]:
            suffix.append(f"source={row['source_job']}")
        suffix_text = f" ({', '.join(suffix)})" if suffix else ""
        label = f" - {row['label']}" if row["label"] else ""
        print(f"  {row['name']}{label}{suffix_text}")
        if row["notes"]:
            print(f"    {row['notes']}")


@task
def list_unrun_experiments(ctx: Context, fuzzy: bool = True) -> None:
    """List train experiment configs with no matching local training/W&B record."""
    del ctx
    experiment_dir = hydra_experiment_dir("train")
    records = load_training_records()
    missing: list[dict[str, str]] = []
    matched_count = 0

    for path in sorted(experiment_dir.glob("*.yaml")):
        config = load_yaml(path)
        matches = matching_training_records(path.stem, config, records, fuzzy=fuzzy)
        if matches:
            matched_count += 1
            continue
        summary = read_experiment_summary(path)
        missing.append(summary)

    print(f"Train experiment configs: {matched_count + len(missing)}")
    print(f"Matched local training records: {matched_count}")
    print(f"Without matched local training/W&B record: {len(missing)}")

    if not missing:
        return

    print("\nUnmatched train experiments:")
    for row in missing:
        suffix = []
        if row["wandb_name"]:
            suffix.append(f"wandb={row['wandb_name']}")
        if row["source_job"]:
            suffix.append(f"source={row['source_job']}")
        suffix_text = f" ({', '.join(suffix)})" if suffix else ""
        label = f" - {row['label']}" if row["label"] else ""
        print(f"  {row['name']}{label}{suffix_text}")
        if row["notes"]:
            print(f"    {row['notes']}")


@task
def list_training_runs(ctx: Context, experiment: str = "", limit: int = 20, fuzzy: bool = True) -> None:
    """List local training result records matching a train experiment config."""
    del ctx
    if not experiment:
        print("Usage: invoke list-training-runs --experiment <train_config_name> [--limit 20] [--no-fuzzy]")
        raise SystemExit(1)

    experiment_path, config = load_train_experiment(experiment)
    metadata = config.get("metadata") or {}
    wandb_name = str(config.get("wandb_name") or "")
    job_ids = as_str_set(metadata.get("training_job_ids") or metadata.get("training_job_id"))

    matches = matching_training_records(experiment, config, load_training_records(), fuzzy=fuzzy)

    print(f"Experiment: {experiment}")
    print(f"Config: {experiment_path}")
    if metadata.get("label"):
        print(f"Label: {metadata['label']}")
    if wandb_name:
        print(f"Configured wandb_name: {wandb_name}")
    if job_ids:
        print(f"Configured training_job_ids: {', '.join(sorted(job_ids))}")
    if metadata.get("notes"):
        print(f"Notes: {metadata['notes']}")

    if not matches:
        print("\nNo local training records matched this config.")
        print("Tip: fetch/sync WandB first, or add metadata.training_job_ids to the config.")
        return

    print(f"\nMatched training records: {len(matches)}")
    for reason, record in matches[: max(limit, 0)]:
        print(f"- {record.get('wandb_run_name') or Path(str(record.get('_record_path'))).stem}")
        print(f"  match: {reason}")
        if record.get("lsf_job_id"):
            print(f"  job: {record['lsf_job_id']}")
        if record.get("wandb_url"):
            print(f"  wandb: {record['wandb_url']}")
        if record.get("save_dir"):
            print(f"  save_dir: {record['save_dir']}")
        print(
            "  eval: "
            f"best={format_percent(record.get('best_eval_metric_value'))}"
            f" @ {record.get('best_eval_iteration') or '-'}, "
            f"final={format_percent(record.get('final_eval_metric_value'))}"
            f" @ {record.get('final_eval_iteration') or '-'}"
        )
        if record.get("metrics_jsonl"):
            print(f"  curves: {record['metrics_jsonl']}")
        if record.get("git_commit"):
            print(f"  commit: {record['git_commit']}")

    if len(matches) > limit:
        print(f"\nShowing {limit} of {len(matches)} matches.")


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
    _run(
        ctx,
        f"uv run python scripts/train_sft.py {data_flag} {libero_flag} "
        f"--num-demos {num_demos} --seed {seed} "
        f"{sim_flag} --eval-suite {eval_suite} {resume_flag} {wandb_flag}",
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
    _run(
        ctx,
        f"uv run python scripts/train_srpo.py --sft-checkpoint {sft_checkpoint} --mode {mode} "
        f"--num-demos {num_demos} --seed {seed} --world-model {world_model} {wandb_flag}",
    )


@task
def run_experiment(ctx: Context, config: str = "configs/srpo_pickcube.yaml", no_wandb: bool = False) -> None:
    """Run the full SRPO validation experiment matrix."""
    wandb_flag = "--no-wandb" if no_wandb else ""
    _run(ctx, f"uv run python scripts/run_experiment.py --config {config} {wandb_flag}")


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
    _run(ctx, cmd)
