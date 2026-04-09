from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vla.constants import RESULTS_DIR
from vla.utils.serialization import to_json_serializable

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"git_commit": commit, "git_branch": branch}
    except Exception:
        return {"git_commit": "", "git_branch": ""}


def get_scheduler_info() -> dict[str, str]:
    return {
        "lsf_job_id": os.environ.get("LSB_JOBID", ""),
        "lsf_job_name": os.environ.get("LSB_JOBNAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": os.environ.get("HOSTNAME", os.environ.get("COMPUTERNAME", "")),
    }


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_json_serializable(payload), indent=2) + "\n", encoding="utf-8")
    return out_path


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def append_jsonl(path: Path | str, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with _file_lock(lock_path):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_json_serializable(payload)) + "\n")
    return out_path


def append_csv_row(path: Path | str, row: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    serializable_row = {k: to_json_serializable(v) for k, v in row.items()}

    with _file_lock(lock_path):
        existing_rows: list[dict[str, Any]] = []
        fieldnames: list[str] = []
        if out_path.exists():
            with open(out_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                existing_rows = list(reader)

        for key in serializable_row:
            if key not in fieldnames:
                fieldnames.append(key)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow(existing)
            writer.writerow(serializable_row)
    return out_path


def flatten_task_metrics(task_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for payload in task_payloads:
        task_id = payload.get("task_id")
        if task_id is None:
            continue
        prefix = f"task_{task_id}"
        flat[f"{prefix}_success_rate"] = payload.get("success_rate")
        flat[f"{prefix}_successes"] = payload.get("successes")
        flat[f"{prefix}_mean_reward"] = payload.get("mean_reward")
        flat[f"{prefix}_mean_episode_length"] = payload.get("mean_episode_length")
        if payload.get("task_description"):
            flat[f"{prefix}_description"] = payload.get("task_description")
    return flat


def summarize_metrics_jsonl(jsonl_path: Path | str, eval_key_suffixes: list[str]) -> dict[str, Any]:
    path = Path(jsonl_path)
    if not path.exists():
        return {}

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}

    best_eval_value: float | None = None
    best_eval_key = ""
    best_eval_iteration: int | None = None
    final_eval_value: float | None = None
    final_eval_key = ""
    final_iteration: int | None = None

    for row in lines:
        for key, value in row.items():
            if not any(key.endswith(suffix) for suffix in eval_key_suffixes):
                continue
            if not isinstance(value, (int, float)):
                continue
            iteration = row.get("sparse_rl/iteration", row.get("srpo/iteration", row.get("sft/epoch")))
            if best_eval_value is None or value > best_eval_value:
                best_eval_value = float(value)
                best_eval_key = key
                best_eval_iteration = int(iteration) if isinstance(iteration, (int, float)) else None
            final_eval_value = float(value)
            final_eval_key = key
            final_iteration = int(iteration) if isinstance(iteration, (int, float)) else None

    return {
        "metrics_jsonl": str(path),
        "best_eval_metric_key": best_eval_key,
        "best_eval_metric_value": best_eval_value,
        "best_eval_iteration": best_eval_iteration,
        "final_eval_metric_key": final_eval_key,
        "final_eval_metric_value": final_eval_value,
        "final_eval_iteration": final_iteration,
    }


def write_training_registry(record: dict[str, Any]) -> None:
    append_jsonl(RESULTS_DIR / "training_runs.jsonl", record)
    append_csv_row(RESULTS_DIR / "training_runs.csv", record)


def write_eval_registry(record: dict[str, Any]) -> None:
    append_jsonl(RESULTS_DIR / "eval_runs.jsonl", record)
    append_csv_row(RESULTS_DIR / "eval_runs.csv", record)


def find_training_metadata(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    candidates = [
        checkpoint_dir / "training_run.json",
        checkpoint_dir.parent / "training_run.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load JSON metadata from %s", path, exc_info=True)
        return None
