import json
import math
import re
from pathlib import Path
from typing import Any

import typer
import wandb
from tqdm import tqdm

from vla.constants import RESULTS_DIR
from vla.results_registry import sanitize_name

app = typer.Typer(help="Fetch runs from WandB and reconstruct local JSON results.")

_LSF_JOB_ID_RE = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe representation without failing on W&B internals."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _safe_run_metadata(run) -> dict:
    metadata = _safe_attr(run, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _safe_run_summary(run) -> dict:
    summary = _safe_attr(run, "summary", None)
    data = _safe_attr(summary, "_json_dict", {}) if summary is not None else {}
    return data if isinstance(data, dict) else {}


def _first_value(*sources: dict, keys: tuple[str, ...]) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _infer_lsf_job_id(run_name: str, *sources: dict) -> str:
    value = _first_value(
        *sources,
        keys=("lsf_job_id", "LSF_JOBID", "job_id", "scheduler_job_id", "slurm_job_id", "SLURM_JOB_ID"),
    )
    if value not in (None, ""):
        return str(value)

    # Run names commonly end with the LSF id, e.g. "..._seed42_28263586".
    matches = _LSF_JOB_ID_RE.findall(run_name or "")
    return matches[-1] if matches else ""


def _extract_git_metadata(config: dict, summary: dict, metadata: dict) -> dict:
    git_meta = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
    commit = (
        _first_value(config, summary, metadata, git_meta, keys=("git_commit", "commit", "git.commit", "sha"))
        or _safe_attr(metadata, "commit", None)
    )
    branch = _first_value(config, summary, metadata, git_meta, keys=("git_branch", "branch", "git.branch"))
    remote = _first_value(config, summary, metadata, git_meta, keys=("git_remote", "remote", "git.remote", "remote_url"))
    return {
        "git_commit": str(commit or ""),
        "git_branch": str(branch or ""),
        "git_remote": str(remote or ""),
    }


def reconstruct_run_provenance(run, *, config: dict | None = None, summary: dict | None = None) -> dict:
    """Collect reproducibility metadata that W&B exposes for both train and eval runs."""
    cfg = config if config is not None else dict(run.config)
    summ = summary if summary is not None else _safe_run_summary(run)
    metadata = _safe_run_metadata(run)
    git = _extract_git_metadata(cfg, summ, metadata)

    record = {
        "wandb_id": str(_safe_attr(run, "id", "") or ""),
        "wandb_project": str(_safe_attr(run, "project", "") or ""),
        "wandb_entity": str(_safe_attr(run, "entity", "") or ""),
        "wandb_url": str(_safe_attr(run, "url", "") or ""),
        "wandb_state": str(_safe_attr(run, "state", "") or ""),
        "wandb_created_at": str(_safe_attr(run, "created_at", "") or ""),
        "wandb_updated_at": str(_safe_attr(run, "updated_at", "") or ""),
        "wandb_tags": _jsonable(_safe_attr(run, "tags", []) or []),
        "lsf_job_id": _infer_lsf_job_id(str(run.name), cfg, summ, metadata),
        "lsf_job_name": str(_first_value(cfg, summ, metadata, keys=("lsf_job_name", "LSB_JOBNAME", "job_name")) or ""),
        "hostname": str(_first_value(cfg, summ, metadata, keys=("hostname", "host", "HOSTNAME")) or ""),
    }
    record.update(git)
    return record


def _training_history_keys(mode: str) -> tuple[list[str], list[str]]:
    resolved_mode = str(mode or "").lower()

    iteration_keys = []
    eval_keys = []

    for candidate in (
        resolved_mode,
        "sparse_rl",
        "srpo",
        "sft",
    ):
        if not candidate:
            continue
        if candidate == "sft":
            iteration_keys.append("sft/epoch")
            eval_keys.append("sft/success_rate")
            eval_keys.append("sft/eval/success_rate")
        else:
            iteration_keys.append(f"{candidate}/iteration")
            eval_keys.append(f"{candidate}/eval/success_rate")

    # Preserve order while removing duplicates.
    iteration_keys = list(dict.fromkeys(iteration_keys))
    eval_keys = list(dict.fromkeys(eval_keys))
    return iteration_keys, eval_keys


def _history_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, str):
        return value
    return None


def fetch_training_history(run) -> list[dict]:
    """Return scalar training history rows suitable for local metrics plotting."""
    history_rows = []
    for row in run.scan_history():
        if not isinstance(row, dict):
            continue

        compact = {}
        for key, value in row.items():
            if key.startswith("_"):
                if key in {"_step", "_runtime", "_timestamp"}:
                    scalar = _history_scalar(value)
                    if scalar is not None:
                        compact[key] = scalar
                continue
            scalar = _history_scalar(value)
            if scalar is not None:
                compact[key] = scalar

        if not compact:
            continue

        history_rows.append(compact)

    return history_rows


def summarize_training_history(history_rows: list[dict], mode: str) -> dict:
    """Compute best/final eval stats from fetched training history rows."""
    if not history_rows:
        return {}

    iteration_keys, eval_keys = _training_history_keys(mode)
    eval_rows = []
    for row in history_rows:
        iteration = None
        for key in iteration_keys:
            iteration = _as_int(row.get(key))
            if iteration is not None:
                break

        eval_success = None
        eval_metric_key = None
        for key in eval_keys:
            eval_success = _as_float(row.get(key))
            if eval_success is not None:
                eval_metric_key = key
                break

        if iteration is None or eval_success is None:
            continue

        eval_rows.append(
            {
                "iteration": iteration,
                "eval_success_rate": eval_success,
                "eval_metric_key": eval_metric_key,
            }
        )

    if not eval_rows:
        return {"history_points": len(history_rows)}

    best_row = max(eval_rows, key=lambda row: row["eval_success_rate"])
    final_row = eval_rows[-1]
    return {
        "history_points": len(history_rows),
        "best_eval_metric_key": best_row.get("eval_metric_key", ""),
        "best_eval_metric_value": best_row["eval_success_rate"],
        "best_eval_iteration": best_row["iteration"],
        "final_eval_metric_key": final_row.get("eval_metric_key", ""),
        "final_eval_metric_value": final_row["eval_success_rate"],
        "final_eval_iteration": final_row["iteration"],
    }


def reconstruct_eval_record(run) -> dict:
    """Map a WandB eval run back to the local eval_record JSON format."""
    cfg = dict(run.config)
    summary = _safe_run_summary(run)
    suite = cfg.get("suite", "spatial")

    # Reconstruct the task_metrics list from the flat summary keys
    task_metrics = []
    for k, v in summary.items():
        if k.startswith(f"eval/{suite}/task_") and k.endswith("/success_rate"):
            # extract task_id string "0", "1", etc.
            part = k.split("/")[2]
            task_id_str = part.replace("task_", "")
            try:
                task_id = int(task_id_str)
                task_metrics.append({"task_id": task_id, "success_rate": v})
            except ValueError:
                pass

    # Sort for consistency
    task_metrics = sorted(task_metrics, key=lambda x: x["task_id"])

    record = {
        "record_type": "evaluation",
        "wandb_run_name": run.name,
        "checkpoint": cfg.get("checkpoint", ""),
        "simulator": cfg.get("simulator", "libero"),
        "suite": suite,
        "success_rate": summary.get(f"eval/{suite}/overall/success_rate", 0.0),
        "mean_reward": summary.get(f"eval/{suite}/overall/mean_reward", 0.0),
        "task_metrics": task_metrics,
        # Plot_results relies on these for labeling
        "training_save_dir": cfg.get("checkpoint_dir") or "",
        "training_method": _determine_eval_method(run.name, cfg),
    }
    record.update(reconstruct_run_provenance(run, config=cfg, summary=summary))

    training_git_commit = (
        cfg.get("training_git_commit")
        or cfg.get("git_commit")
        or summary.get("training_git_commit")
        or summary.get("git_commit")
        or record.get("git_commit", "")
    )
    if training_git_commit:
        record["training_git_commit"] = training_git_commit

    return record


def _determine_eval_method(run_name: str, cfg: dict) -> str:
    """Robustly determine if an eval run is SFT or Sparse RL based on wandb metadata."""
    name_lower = run_name.lower()
    ckpt_dir = str(cfg.get("checkpoint_dir", "")).lower()

    if cfg.get("method"):
        return cfg.get("method")

    if "sft" in name_lower or "sft" in ckpt_dir or not ckpt_dir:
        return "sft"
    elif "srpo" in name_lower or "srpo" in ckpt_dir:
        return "srpo"

    return "sparse_rl"


def reconstruct_training_record(run, history_path: Path | None = None, history_summary: dict | None = None) -> dict:
    """Map a WandB training run back to the local training_record JSON format."""
    cfg = dict(run.config)
    summary = _safe_run_summary(run)
    record = {
        "record_type": "training",
        "wandb_run_name": run.name,
        "method": cfg.get("mode", cfg.get("method", "sparse_rl")),
        "update_method": cfg.get("update_method", ""),
        "save_dir": cfg.get("save_dir", ""),
        "checkpoint": cfg.get("checkpoint", ""),
        "sft_checkpoint": cfg.get("sft_checkpoint", ""),
        "suite": cfg.get("suite", ""),
        "seed": cfg.get("seed"),
        "num_tasks": cfg.get("num_tasks"),
        "task_ids": cfg.get("tasks", cfg.get("task_ids", [])),
        "libero_task_indices": cfg.get("libero_task_indices", []),
        "include_demos_in_update": cfg.get("include_demos_in_update", False),
        "success_replay_total_size": cfg.get("success_replay_total_size", 0),
        "trajs_per_task_per_iter": cfg.get("trajs_per_task_per_iter", cfg.get("trajs_per_task")),
        "config": cfg,
    }
    record.update(reconstruct_run_provenance(run, config=cfg, summary=summary))
    if history_path is not None:
        record["metrics_jsonl"] = str(history_path)
    if history_summary:
        record.update(history_summary)
    return record


@app.command()
def sync(
    project: str = typer.Argument(..., help="WandB project (e.g., vla-libero-eval or srpo-smolvla)"),
    entity: str = typer.Option(None, "--entity", "-e", help="WandB entity/username"),
    record_type: str = typer.Option("eval", "--type", "-t", help="Dataset type: 'eval' or 'training'"),
    with_history: bool = typer.Option(
        True,
        "--with-history/--no-history",
        help="For training runs, also fetch scalar training history for local plotting.",
    ),
):
    """
    Fetch all runs from a WandB project and recreate them as local JSON files
    in results/evals/ or results/training/ for offline plotting.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    typer.echo(f"Fetching {record_type} runs from {path}...")

    try:
        runs = api.runs(path)
    except Exception as e:
        typer.secho(f"Failed to fetch runs: {e}", fg=typer.colors.RED)
        raise typer.Exit(1) from e

    if not runs:
        typer.secho("No runs found.", fg=typer.colors.YELLOW)
        return

    out_folder = RESULTS_DIR / ("evals" if record_type == "eval" else "training")
    out_folder.mkdir(parents=True, exist_ok=True)
    history_folder = RESULTS_DIR / "training_curves"
    if record_type == "training" and with_history:
        history_folder.mkdir(parents=True, exist_ok=True)
    count = 0

    for run in tqdm(runs, desc=f"Writing JSONs to {out_folder.name}/"):
        name = sanitize_name(run.name)
        if not name:
            continue

        json_path = out_folder / f"{name}.json"
        history_path = None
        history_summary = None
        if record_type == "training" and with_history:
            history_rows = fetch_training_history(run)
            if history_rows:
                history_path = history_folder / f"{name}.jsonl"
                _write_jsonl(history_path, history_rows)
                mode = str(run.config.get("mode", run.config.get("method", "sparse_rl")))
                history_summary = summarize_training_history(history_rows, mode=mode)

        record = (
            reconstruct_eval_record(run)
            if record_type == "eval"
            else reconstruct_training_record(run, history_path=history_path, history_summary=history_summary)
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        count += 1

    typer.secho(f"\nSuccessfully synced {count} {record_type} runs cleanly to:", fg=typer.colors.GREEN)
    typer.echo(f"  {out_folder}/")
    if record_type == "training" and with_history:
        typer.echo(f"  {history_folder}/")


if __name__ == "__main__":
    app()
