"""Audit per-task demo-replay success rate from training history JSONL files.

The trainer logs ``sparse_rl/<task>/demo_replay_success_rate`` and
``sparse_rl/demo_replay_success_rate`` at iter 0 (see
``src/vla/rl/trainer.py`` lin. 865-870).  This script extracts those
values from one or more local training history files and flags tasks
below a configurable threshold.

Typical use:
    uv run python scripts/audit_demo_replay.py results/training_curves
    uv run python scripts/audit_demo_replay.py results/training_curves --threshold 0.8

A value < threshold on any task means the recorded demo actions do not
reach the goal when replayed in the freshly-seeded env (init-state
mismatch).  ``fallback_to_raw_demo=True`` in ``train_srpo.py`` keeps
those demos in the data pipeline as raw LeRobot frames, so the
success_bc/replay path still works, but the env-observation-grounded
view of the demo is lost on those tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app = typer.Typer(add_completion=False, help=__doc__)


def _load_first_row(path: Path) -> dict | None:
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        return None
    try:
        return json.loads(first)
    except json.JSONDecodeError:
        return None


def _extract_demo_replay(row: dict) -> tuple[dict[str, float], float | None]:
    per_task: dict[str, float] = {}
    suite_avg: float | None = None
    for key, value in row.items():
        if not isinstance(value, (int, float)):
            continue
        if not key.endswith("demo_replay_success_rate"):
            continue
        parts = key.split("/")
        if len(parts) >= 3 and parts[-2].startswith(("spatial_task", "object_task", "goal_task", "long_task", "task_")):
            per_task[parts[-2]] = float(value)
        elif parts[-1] == "demo_replay_success_rate":
            suite_avg = float(value)
    return per_task, suite_avg


def _format_row(name: str, per_task: dict[str, float], suite_avg: float | None, threshold: float) -> list[str]:
    if not per_task and suite_avg is None:
        return [f"{name}: NO demo_replay_success_rate present (was --no-demo-replay used?)"]

    lines = [name]
    if suite_avg is not None:
        flag = "BROKEN" if suite_avg < threshold else "OK"
        lines.append(f"  suite-avg = {suite_avg:.3f}  [{flag}]")
    if per_task:
        broken = [t for t, v in sorted(per_task.items()) if v < threshold]
        ok = [t for t, v in sorted(per_task.items()) if v >= threshold]
        lines.append(f"  tasks below {threshold:.2f}: {len(broken)}/{len(per_task)}")
        if broken:
            lines.append("    " + ", ".join(f"{t}={per_task[t]:.2f}" for t in broken))
        if ok:
            lines.append("    OK: " + ", ".join(f"{t}={per_task[t]:.2f}" for t in ok))
    return lines


@app.command()
def main(
    target: Path = typer.Argument(..., help="JSONL file or directory of JSONL files to audit."),
    threshold: float = typer.Option(0.8, "--threshold", help="Minimum acceptable per-task demo_replay_success_rate."),
    glob: str = typer.Option("*.jsonl", "--glob", help="Glob pattern when target is a directory."),
    exit_code: bool = typer.Option(False, "--exit-code", help="Exit with status 1 if any task is below threshold."),
) -> None:
    target = target.resolve()
    if target.is_file():
        paths = [target]
    elif target.is_dir():
        paths = sorted(target.glob(glob))
    else:
        typer.echo(f"Target not found: {target}", err=True)
        raise typer.Exit(code=2)

    any_broken = False
    audited = 0
    for path in paths:
        row = _load_first_row(path)
        if row is None:
            continue
        per_task, suite_avg = _extract_demo_replay(row)
        if not per_task and suite_avg is None:
            continue
        audited += 1
        lines = _format_row(path.name, per_task, suite_avg, threshold)
        if any(v < threshold for v in per_task.values()) or (suite_avg is not None and suite_avg < threshold):
            any_broken = True
        for line in lines:
            typer.echo(line)
        typer.echo("")

    if audited == 0:
        typer.echo(
            "No file contained sparse_rl/.../demo_replay_success_rate.  "
            "Run with demo_replay=True (the default in scripts/train_srpo.py) to populate this metric.",
            err=True,
        )
        raise typer.Exit(code=1 if exit_code else 0)

    typer.echo(f"Audited {audited} file(s).  threshold={threshold:.2f}.  any_broken={any_broken}.")
    if exit_code and any_broken:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
