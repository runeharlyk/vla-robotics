from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from typer.models import OptionInfo

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = PROJECT_ROOT / "jobs"
PROFILES_PATH = JOBS_DIR / "_profiles.yaml"
GENERATED_JOBS_DIR = JOBS_DIR / "generated"

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


@dataclass(frozen=True)
class GeneratedJob:
    name: str
    script: str


@dataclass(frozen=True)
class TrainEvalTarget:
    experiment: str
    checkpoint_kind: str
    checkpoint_dir: Path
    policy_path: Path
    checkpoint: str
    simulator: str
    suite: str
    num_episodes: int
    max_steps: int | None
    seed: int
    num_envs: int
    fixed_noise_seed: int
    wandb_name: str
    training_record: dict[str, Any]
    match_reason: str


@dataclass(frozen=True)
class SubmitValidation:
    summary: dict[str, Any]
    errors: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def hydra_experiment_dir(kind: str) -> Path:
    if kind == "train":
        return PROJECT_ROOT / "configs" / "train_srpo" / "experiment"
    if kind == "eval":
        return PROJECT_ROOT / "configs" / "evaluate" / "experiment"
    print("kind must be 'train' or 'eval'", file=sys.stderr)
    raise SystemExit(1)


def hydra_config_root(kind: str) -> Path:
    if kind == "train":
        return PROJECT_ROOT / "configs" / "train_srpo"
    if kind == "eval":
        return PROJECT_ROOT / "configs" / "evaluate"
    print("kind must be 'train' or 'eval'", file=sys.stderr)
    raise SystemExit(1)


def hydra_script(kind: str) -> str:
    return "scripts/train_srpo_hydra.py" if kind == "train" else "scripts/evaluate_hydra.py"


def target_script(kind: str) -> str:
    return "scripts/train_srpo.py" if kind == "train" else "scripts/evaluate.py"


def generated_job_script(
    *,
    job_name: str,
    profile_name: str,
    command: str,
    kind: str,
    experiment: str,
    config_dir: str,
    entrypoint: str,
) -> GeneratedJob:
    profiles = load_yaml(PROFILES_PATH)
    if profile_name not in profiles:
        available = ", ".join(profiles)
        print(f"Unknown GPU profile '{profile_name}'. Available: {available}", file=sys.stderr)
        raise SystemExit(1)

    header = LSF_HEADER.format(job_name=job_name, **profiles[profile_name])
    body = f"""\
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\\n/work3/s234814/libero\\nY\\n" \\
  | uv run --no-sync python -c "import libero.libero; print('Libero configured')"

echo "Kind: {kind}"
echo "Experiment: {experiment}"
echo "Profile: {profile_name}"
echo "Git commit: $(git rev-parse HEAD)"
git status --short || true
git diff HEAD -- {config_dir} {entrypoint} scripts/train_srpo.py scripts/evaluate.py src/vla/rl || true

{command}
"""
    return GeneratedJob(name=job_name, script=header + body)


def compose_hydra_experiment(kind: str, experiment: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_root = hydra_config_root(kind)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        return compose(config_name="base", overrides=[f"experiment={experiment}"])


def hydra_args_for_submit(kind: str, experiment: str) -> list[str]:
    cfg = compose_hydra_experiment(kind, experiment)
    if kind == "train":
        from scripts.train_srpo_hydra import config_to_train_srpo_args

        return config_to_train_srpo_args(cfg)

    from scripts.evaluate_hydra import config_to_evaluate_args, expand_eval_configs

    args: list[str] = []
    for eval_cfg in expand_eval_configs(cfg):
        args.extend(config_to_evaluate_args(eval_cfg))
    return args


def split_param_decl(decl: str) -> list[str]:
    return [part for part in decl.split("/") if part.startswith("-")]


def is_bool_annotation(annotation: Any) -> bool:
    return annotation is bool or bool(get_origin(annotation) is None and annotation == "bool")


def enum_type(annotation: Any) -> type[Enum] | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, Enum):
            return arg
    return None


def cli_options(script: str) -> dict[str, tuple[str, Any, bool]]:
    module_name = script.removesuffix(".py").replace("/", ".")
    module = __import__(module_name, fromlist=["main"])
    main = getattr(module, "main", None)
    if main is None:
        return {}

    import inspect

    options: dict[str, tuple[str, Any, bool]] = {}
    type_hints = get_type_hints(main)
    for name, param in inspect.signature(main).parameters.items():
        default = param.default
        if not isinstance(default, OptionInfo):
            continue
        annotation = type_hints.get(name, param.annotation)
        requires_value = not is_bool_annotation(annotation)
        for decl in default.param_decls:
            for flag in split_param_decl(decl):
                options[flag] = (name, annotation, requires_value)
    return options


def validate_cli_args(script: str, args: list[str]) -> list[str]:
    options = cli_options(script)
    errors: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            break
        if not token.startswith("-"):
            i += 1
            continue

        flag, sep, inline_value = token.partition("=")
        if flag not in options:
            errors.append(f"{script} unknown option {flag!r}")
            i += 1
            continue

        _, annotation, requires_value = options[flag]
        value: str | None = inline_value if sep else None
        if requires_value and value is None:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                errors.append(f"{script} option {flag!r} is missing a value")
                i += 1
                continue
            value = args[i + 1]
            i += 1

        enum_cls = enum_type(annotation)
        if enum_cls is not None and value is not None and not value.startswith("$"):
            choices = {str(member.value) for member in enum_cls}
            if value not in choices:
                errors.append(
                    f"{script} option {flag!r} has invalid value {value!r}; "
                    f"expected one of {sorted(choices)}"
                )

        i += 1
    return errors


def sanitize_job_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_gb(value: str) -> int | None:
    match = re.fullmatch(r"(\d+)\s*GB", str(value).strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def cli_path(value: Path | str) -> str:
    return str(value).replace("\\", "/")


def profile_summary(profile_name: str, profile: dict[str, Any]) -> str:
    cores = int(profile["cores"])
    mem_per_core = str(profile["mem"])
    total_gb = parse_gb(mem_per_core)
    total = f"{cores * total_gb}GB RAM" if total_gb is not None else f"{cores} x {mem_per_core} RAM"
    return (
        f"running on {profile_name}: queue={profile['queue']}, "
        f"cores={cores}, {total}, walltime={profile['walltime']}, gpu=1"
    )


def hpc_setup_issues(*, submit: bool) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []

    if not (JOBS_DIR / "_env.sh").exists():
        missing.append("jobs/_env.sh")
    for command in ("uv", "git"):
        if shutil.which(command) is None:
            missing.append(f"command:{command}")
    if submit and shutil.which("bsub") is None:
        missing.append("command:bsub")

    if os.name != "nt" and shutil.which("nvidia-smi") is None:
        warnings.append("command:nvidia-smi")
    for env_name in ("VLA_WORK3", "HF_HOME", "WANDB_DIR", "UV_PROJECT_ENVIRONMENT"):
        if not os.environ.get(env_name):
            warnings.append(f"env:{env_name} (source jobs/_env.sh to set)")

    return missing, warnings


def validate_hydra_submit(kind: str, experiment: str, profile_name: str, *, submit: bool) -> SubmitValidation:
    errors: list[str] = []
    profiles = load_yaml(PROFILES_PATH)
    profile = profiles.get(profile_name)
    if profile is None:
        errors.append(f"unknown profile {profile_name!r}; available: {', '.join(profiles)}")
        profile = {}

    experiment_path = hydra_experiment_dir(kind) / f"{experiment}.yaml"
    if not experiment_path.exists():
        errors.append(f"unknown {kind} experiment {experiment!r}; expected {experiment_path}")

    args: list[str] = []
    eval_targets: list[dict[str, str]] = []
    if experiment_path.exists():
        try:
            args = hydra_args_for_submit(kind, experiment)
            if kind == "eval":
                eval_targets = eval_target_summaries(experiment)
        except Exception as exc:
            errors.append(f"Hydra compose failed for {experiment!r}: {exc}")
        else:
            errors.extend(validate_cli_args(target_script(kind), args))

    missing, warnings = hpc_setup_issues(submit=submit)
    if submit:
        errors.extend(f"missing HPC prerequisite: {item}" for item in missing)

    return SubmitValidation(
        summary={
            "kind": kind,
            "experiment": experiment,
            "profile_name": profile_name,
            "profile": profile,
            "experiment_path": experiment_path,
            "args": args,
            "eval_targets": eval_targets,
            "hpc_missing": missing,
            "hpc_warnings": warnings,
        },
        errors=errors,
    )


def print_submit_validation(validation: SubmitValidation) -> None:
    summary = validation.summary
    profile = summary["profile"]
    setup_line = "HPC setup correct" if not summary["hpc_missing"] else "Missing HPC setup"
    print(setup_line)
    for item in summary["hpc_missing"]:
        print(f"  missing: {item}")
    for item in summary["hpc_warnings"]:
        print(f"  warning: {item}")

    print("\nJob summary:")
    if profile:
        print(f"  {profile_summary(summary['profile_name'], profile)}")
    print(f"  kind: {summary['kind']}")
    print(f"  experiment: {summary['experiment']}")
    print(f"  config: {summary['experiment_path']}")
    print(f"  entrypoint: {hydra_script(summary['kind'])}")
    command = summary.get("command") or (
        f"uv run --no-sync python {hydra_script(summary['kind'])} experiment={summary['experiment']}"
    )
    print(f"  command: {command}")
    if summary["eval_targets"]:
        print("\nEval targets:")
        for index, target in enumerate(summary["eval_targets"], start=1):
            source = target["checkpoint_dir"] or target["checkpoint"]
            details = []
            if target["wandb_name"]:
                details.append(f"wandb={target['wandb_name']}")
            if target["training_job_id"]:
                details.append(f"job={target['training_job_id']}")
            detail_text = f" ({', '.join(details)})" if details else ""
            print(f"  {index}. {target['label']}: {source}{detail_text}")
        if len(summary["eval_targets"]) > 1:
            print(f"  note: this config launches {len(summary['eval_targets'])} separate eval runs.")

    if validation.errors:
        print("\nValidation failed:")
        for error in validation.errors:
            print(f"  - {error}")
    else:
        print("\nValidation passed.")


def generate_hydra_job_script(kind: str, experiment: str, profile_name: str) -> GeneratedJob:
    profiles = load_yaml(PROFILES_PATH)
    if profile_name not in profiles:
        available = ", ".join(profiles)
        print(f"Unknown GPU profile '{profile_name}'. Available: {available}", file=sys.stderr)
        raise SystemExit(1)

    if kind not in {"train", "eval"}:
        print(f"Unknown Hydra job kind '{kind}'. Expected train or eval.", file=sys.stderr)
        raise SystemExit(1)

    script_name = Path(hydra_script(kind)).name
    config_dir = "configs/train_srpo" if kind == "train" else "configs/evaluate"
    job_name = sanitize_job_part(f"{kind}_{experiment}_{profile_name}")
    experiment_arg = shlex.quote(f"experiment={experiment}")
    command = f"uv run --no-sync python scripts/{script_name} {experiment_arg}"
    return generated_job_script(
        job_name=job_name,
        profile_name=profile_name,
        command=command,
        kind=kind,
        experiment=experiment,
        config_dir=config_dir,
        entrypoint=f"scripts/{script_name}",
    )


def eval_target_summaries(experiment: str) -> list[dict[str, str]]:
    from scripts.evaluate_hydra import expand_eval_configs

    cfg = compose_hydra_experiment("eval", experiment)
    targets: list[dict[str, str]] = []
    for index, eval_cfg in enumerate(expand_eval_configs(cfg), start=1):
        metadata = eval_cfg.get("metadata") or {}
        label = metadata.get("label") or eval_cfg.get("wandb_name") or f"eval-{index}"
        targets.append(
            {
                "label": str(label),
                "checkpoint": str(eval_cfg.get("checkpoint") or ""),
                "checkpoint_dir": str(eval_cfg.get("checkpoint_dir") or ""),
                "wandb_name": str(eval_cfg.get("wandb_name") or ""),
                "training_job_id": str(metadata.get("training_job_id") or ""),
            }
        )
    return targets


def read_experiment_summary(path: Path) -> dict[str, str]:
    data = load_yaml(path)
    metadata = data.get("metadata") or {}
    return {
        "name": path.stem,
        "label": str(metadata.get("label") or ""),
        "wandb_name": str(data.get("wandb_name") or ""),
        "source_job": str(metadata.get("source_job") or ""),
        "notes": str(metadata.get("notes") or ""),
    }


def load_train_experiment(experiment: str) -> tuple[Path, dict[str, Any]]:
    experiment_path = hydra_experiment_dir("train") / f"{experiment}.yaml"
    if not experiment_path.exists():
        print(f"Unknown train experiment '{experiment}'.", file=sys.stderr)
        print("Available experiments:", file=sys.stderr)
        for path in sorted(hydra_experiment_dir("train").glob("*.yaml")):
            print(f"  {path.stem}", file=sys.stderr)
        raise SystemExit(1)
    return experiment_path, load_yaml(experiment_path)


def compose_train_experiment_dict(experiment: str) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = compose_hydra_experiment("train", experiment)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected train config dict, got {type(data)!r}")
    return data


def load_training_records(results_dir: Path | None = None) -> list[dict[str, Any]]:
    root = results_dir or PROJECT_ROOT / "results"
    records: list[dict[str, Any]] = []
    for path in sorted((root / "training").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record["_record_path"] = str(path)
        records.append(record)
    return records


def as_str_set(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, int)):
        return {str(values)}
    if isinstance(values, list):
        return {str(value) for value in values}
    return set()


def run_matches_train_experiment(record: dict, experiment: str, config: dict, *, fuzzy: bool) -> tuple[bool, str]:
    metadata = config.get("metadata") or {}
    wandb_name = str(config.get("wandb_name") or "")
    labels = {experiment, str(metadata.get("label") or "")}
    job_ids = as_str_set(metadata.get("training_job_ids") or metadata.get("training_job_id"))

    run_name = str(record.get("wandb_run_name") or "")
    lsf_job_id = str(record.get("lsf_job_id") or "")

    if wandb_name and (run_name == wandb_name or run_name.startswith(f"{wandb_name}_") or wandb_name in run_name):
        return True, "wandb_name"
    if lsf_job_id and lsf_job_id in job_ids:
        return True, "training_job_id"

    if fuzzy:
        haystack = " ".join(
            [
                run_name,
                str(record.get("save_dir") or ""),
                str(record.get("wandb_url") or ""),
                str(record.get("_record_path") or ""),
            ]
        ).lower()
        tokens = {
            token
            for label in labels | {wandb_name}
            for token in re.split(r"[^a-zA-Z0-9]+", label.lower())
            if len(token) >= 3
        }
        if tokens and sum(1 for token in tokens if token in haystack) >= min(2, len(tokens)):
            return True, "fuzzy"

    return False, ""


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{100 * float(value):.1f}%"
    return "-"


def matching_training_records(
    experiment: str,
    config: dict,
    records: list[dict],
    *,
    fuzzy: bool,
) -> list[tuple[str, dict]]:
    matches: list[tuple[str, dict]] = []
    for record in records:
        matched, reason = run_matches_train_experiment(record, experiment, config, fuzzy=fuzzy)
        if matched:
            matches.append((reason, record))
    reason_priority = {"training_job_id": 3, "wandb_name": 2, "fuzzy": 1}
    return sorted(
        matches,
        key=lambda item: (
            reason_priority.get(item[0], 0),
            str(item[1].get("wandb_created_at") or ""),
            str(item[1].get("wandb_run_name") or ""),
        ),
        reverse=True,
    )


def checkpoint_dir_from_record(record: dict[str, Any], checkpoint_kind: str) -> Path:
    normalized = checkpoint_kind.replace("_", "-").lower()
    if normalized == "best":
        value = record.get("best_checkpoint_dir")
        fallback = "best"
    elif normalized == "last":
        value = record.get("last_checkpoint_dir")
        fallback = "last"
    elif normalized in {"best-rollout", "rollout"}:
        value = record.get("best_rollout_checkpoint_dir")
        fallback = "best_rollout"
    else:
        raise ValueError("checkpoint must be one of: best, last, best-rollout")

    if value:
        return Path(str(value))
    save_dir = record.get("save_dir")
    if not save_dir:
        raise ValueError("matched training record has no save_dir/checkpoint_dir metadata")
    return Path(str(save_dir)) / fallback


def train_eval_target(
    experiment: str,
    checkpoint_kind: str,
    *,
    fuzzy: bool = True,
) -> TrainEvalTarget:
    _, train_config = load_train_experiment(experiment)
    composed = compose_train_experiment_dict(experiment)
    matches = matching_training_records(experiment, train_config, load_training_records(), fuzzy=fuzzy)
    if not matches:
        raise ValueError(f"no local training record matches train experiment {experiment!r}")

    reason, record = matches[0]
    checkpoint_dir = checkpoint_dir_from_record(record, checkpoint_kind)
    policy_path = checkpoint_dir / "policy.pt"
    lsf_job_id = str(record.get("lsf_job_id") or "")
    run_suffix = f"_{lsf_job_id}" if lsf_job_id else ""
    normalized_checkpoint = checkpoint_kind.replace("_", "-").lower()
    wandb_name = sanitize_job_part(f"eval_{experiment}_{normalized_checkpoint}{run_suffix}")
    rollout_config = composed.get("rollout") or {}
    num_envs = int(rollout_config.get("eval_num_envs") or rollout_config.get("num_envs") or 8)

    return TrainEvalTarget(
        experiment=experiment,
        checkpoint_kind=normalized_checkpoint,
        checkpoint_dir=checkpoint_dir,
        policy_path=policy_path,
        checkpoint=str(record.get("checkpoint") or composed.get("checkpoint") or "HuggingFaceVLA/smolvla_libero"),
        simulator=str(composed.get("simulator") or "libero"),
        suite=str(composed.get("suite") or "spatial"),
        num_episodes=int(composed.get("eval_episodes") or 100),
        max_steps=int(composed["max_steps"]) if composed.get("max_steps") is not None else None,
        seed=int(composed.get("seed") or 42),
        num_envs=num_envs,
        fixed_noise_seed=int(composed.get("seed") or 42),
        wandb_name=wandb_name,
        training_record=record,
        match_reason=reason,
    )


def train_eval_config(target: TrainEvalTarget) -> dict[str, Any]:
    return {
        "checkpoint_dir": cli_path(target.checkpoint_dir),
        "checkpoint": target.checkpoint,
        "simulator": target.simulator,
        "suite": target.suite,
        "num_episodes": target.num_episodes,
        "max_steps": target.max_steps,
        "seed": target.seed,
        "num_envs": target.num_envs,
        "fixed_noise_seed": target.fixed_noise_seed,
        "wandb": True,
        "wandb_name": target.wandb_name,
    }


def train_eval_hydra_overrides(target: TrainEvalTarget) -> list[str]:
    overrides = []
    for key, value in train_eval_config(target).items():
        if value is None:
            continue
        overrides.append(shlex.quote(f"{key}={value}"))
    return overrides


def train_eval_command(target: TrainEvalTarget) -> str:
    return " ".join(
        [
            "uv run --no-sync python scripts/evaluate_hydra.py",
            *train_eval_hydra_overrides(target),
        ]
    )


def validate_train_eval_submit(
    experiment: str,
    profile_name: str,
    checkpoint_kind: str,
    *,
    submit: bool,
    fuzzy: bool = True,
) -> SubmitValidation:
    errors: list[str] = []
    profiles = load_yaml(PROFILES_PATH)
    profile = profiles.get(profile_name)
    if profile is None:
        errors.append(f"unknown profile {profile_name!r}; available: {', '.join(profiles)}")
        profile = {}

    experiment_path = hydra_experiment_dir("train") / f"{experiment}.yaml"
    target: TrainEvalTarget | None = None
    args: list[str] = []
    warnings: list[str] = []
    try:
        target = train_eval_target(experiment, checkpoint_kind, fuzzy=fuzzy)
        from scripts.evaluate_hydra import config_to_evaluate_args

        args = config_to_evaluate_args(train_eval_config(target))
        errors.extend(validate_cli_args("scripts/evaluate.py", args))
        if not target.policy_path.exists():
            message = f"checkpoint policy.pt not visible: {cli_path(target.policy_path)}"
            if submit:
                errors.append(message)
            else:
                warnings.append(message)
    except Exception as exc:
        errors.append(str(exc))

    missing, hpc_warnings = hpc_setup_issues(submit=submit)
    if submit:
        errors.extend(f"missing HPC prerequisite: {item}" for item in missing)

    if target is None:
        target_summary = None
    else:
        target_summary = {
            "label": f"{experiment}:{target.checkpoint_kind}",
            "checkpoint": target.checkpoint,
            "checkpoint_dir": cli_path(target.checkpoint_dir),
            "wandb_name": target.wandb_name,
            "training_job_id": str(target.training_record.get("lsf_job_id") or ""),
            "match_reason": target.match_reason,
            "policy_path": cli_path(target.policy_path),
        }

    return SubmitValidation(
        summary={
            "kind": "eval",
            "experiment": experiment,
            "profile_name": profile_name,
            "profile": profile,
            "experiment_path": experiment_path,
            "args": args,
            "eval_targets": [target_summary] if target_summary else [],
            "hpc_missing": missing,
            "hpc_warnings": [*hpc_warnings, *warnings],
            "checkpoint_kind": checkpoint_kind,
            "train_eval_target": target,
            "command": train_eval_command(target) if target else "",
        },
        errors=errors,
    )


def generate_train_eval_job_script(experiment: str, profile_name: str, checkpoint_kind: str) -> GeneratedJob:
    target = train_eval_target(experiment, checkpoint_kind)
    job_name = sanitize_job_part(f"eval_{experiment}_{target.checkpoint_kind}_{profile_name}")
    command = train_eval_command(target)
    return generated_job_script(
        job_name=job_name,
        profile_name=profile_name,
        command=command,
        kind="eval",
        experiment=experiment,
        config_dir="configs/train_srpo configs/evaluate",
        entrypoint="scripts/evaluate_hydra.py",
    )
