"""Select a bounded, reproducible LIBERO-Plus task subset.

LIBERO-Plus stores perturbation variants as ordinary LIBERO task ids inside
the same suite names used by the base benchmark. The classification JSON uses
1-based ids; this helper prints the 0-based ids expected by the evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

DEFAULT_CLASSIFICATION_PATH = (
    Path(".libero-plus-src") / "libero" / "libero" / "benchmark" / "task_classification.json"
)

SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "libero_spatial": "libero_spatial",
    "object": "libero_object",
    "objects": "libero_object",
    "libero_object": "libero_object",
    "goal": "libero_goal",
    "libero_goal": "libero_goal",
    "long": "libero_10",
    "10": "libero_10",
    "libero_10": "libero_10",
}

CATEGORY_ALIASES = {
    "all": None,
    "total": None,
    "background": "background textures",
    "backgrounds": "background textures",
    "texture": "background textures",
    "textures": "background textures",
    "camera": "camera viewpoints",
    "viewpoint": "camera viewpoints",
    "viewpoints": "camera viewpoints",
    "language": "language instructions",
    "instruction": "language instructions",
    "instructions": "language instructions",
    "light": "light conditions",
    "lighting": "light conditions",
    "layout": "objects layout",
    "object-layout": "objects layout",
    "objects-layout": "objects layout",
    "robot": "robot initial states",
    "robot-state": "robot initial states",
    "robot-states": "robot initial states",
    "noise": "sensor noise",
    "sensor-noise": "sensor noise",
}


@dataclass(frozen=True)
class LiberoPlusTask:
    task_id: int
    source_id: int
    name: str
    category: str
    difficulty_level: int | None


def load_classification(path: Path) -> dict[str, list[dict[str, Any]]]:
    resolved = resolve_classification_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"LIBERO-Plus task classification not found: {path}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top level of {resolved}")
    return data


def resolve_classification_path(path: Path) -> Path:
    if path.exists():
        return path

    spec = importlib.util.find_spec("libero")
    if spec is None:
        return path

    locations = list(spec.submodule_search_locations or [])
    if not locations:
        return path

    installed = Path(locations[0]) / "libero" / "benchmark" / "task_classification.json"
    return installed if installed.exists() else path


def resolve_suite_key(suite: str, data: dict[str, list[dict[str, Any]]]) -> str:
    suite_key = SUITE_ALIASES.get(suite.strip().lower(), suite.strip().lower())
    if suite_key not in data:
        available = ", ".join(sorted(data))
        raise ValueError(f"Unknown LIBERO-Plus suite {suite!r}; available suites: {available}")
    return suite_key


def resolve_category(category: str, categories: set[str]) -> str | None:
    raw = category.strip().lower()
    if raw in CATEGORY_ALIASES:
        expected = CATEGORY_ALIASES[raw]
        if expected is None:
            return None
        for item in categories:
            if item.lower() == expected:
                return item
        raise ValueError(f"Category alias {category!r} resolved to {expected!r}, which is not present")

    exact = [item for item in categories if item.lower() == raw]
    if len(exact) == 1:
        return exact[0]

    contains = [item for item in categories if raw in item.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise ValueError(f"Ambiguous category {category!r}; matches: {', '.join(sorted(contains))}")

    available = ", ".join(sorted(categories))
    raise ValueError(f"Unknown LIBERO-Plus category {category!r}; available categories: {available}")


def _task_from_payload(payload: dict[str, Any]) -> LiberoPlusTask:
    source_id = int(payload["id"])
    return LiberoPlusTask(
        task_id=source_id - 1,
        source_id=source_id,
        name=str(payload.get("name", "")),
        category=str(payload.get("category", "")),
        difficulty_level=payload.get("difficulty_level"),
    )


def select_tasks(
    classification_path: Path,
    suite: str,
    category: str = "all",
    difficulty: int | None = None,
    max_tasks: int | None = 100,
    seed: int = 42,
    shuffle: bool = True,
) -> list[LiberoPlusTask]:
    data = load_classification(classification_path)
    suite_key = resolve_suite_key(suite, data)
    suite_tasks = data[suite_key]
    categories = {str(item.get("category", "")) for item in suite_tasks}
    resolved_category = resolve_category(category, categories)

    tasks = [_task_from_payload(item) for item in suite_tasks]
    if resolved_category is not None:
        tasks = [task for task in tasks if task.category == resolved_category]
    if difficulty is not None:
        tasks = [task for task in tasks if task.difficulty_level == difficulty]

    if not tasks:
        raise ValueError(
            f"No LIBERO-Plus tasks matched suite={suite!r}, category={category!r}, difficulty={difficulty!r}"
        )

    if shuffle:
        tasks = list(tasks)
        random.Random(seed).shuffle(tasks)
    else:
        tasks = sorted(tasks, key=lambda task: task.source_id)

    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    return tasks


def main(
    suite: str = typer.Option("spatial", "--suite", help="LIBERO suite: spatial, object, goal, or long"),
    category: str = typer.Option(
        "all",
        "--category",
        help="Perturbation category alias or exact name. Use all/total for an unstratified suite sample.",
    ),
    difficulty: int | None = typer.Option(None, "--difficulty", help="Optional LIBERO-Plus difficulty level"),
    max_tasks: int = typer.Option(100, "--max-tasks", help="Maximum tasks to print. Use 0 for no cap."),
    seed: int = typer.Option(42, "--seed", help="Deterministic sampling seed"),
    classification_path: Path = typer.Option(
        DEFAULT_CLASSIFICATION_PATH,
        "--classification-path",
        path_type=Path,
        help="Path to task_classification.json from LIBERO-Plus",
    ),
    shuffle: bool = typer.Option(True, "--shuffle/--ordered", help="Shuffle before taking --max-tasks"),
    show_summary: bool = typer.Option(True, "--show-summary/--quiet", help="Print selection summary to stderr"),
) -> None:
    try:
        tasks = select_tasks(
            classification_path=classification_path,
            suite=suite,
            category=category,
            difficulty=difficulty,
            max_tasks=None if max_tasks <= 0 else max_tasks,
            seed=seed,
            shuffle=shuffle,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if show_summary:
        categories = sorted({task.category for task in tasks})
        print(
            f"Selected {len(tasks)} LIBERO-Plus tasks for suite={suite}, "
            f"category={category}, categories={categories}",
            file=sys.stderr,
        )

    print(",".join(str(task.task_id) for task in tasks))


if __name__ == "__main__":
    typer.run(main)
