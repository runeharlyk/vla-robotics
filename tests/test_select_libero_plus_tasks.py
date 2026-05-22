from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.select_libero_plus_tasks import select_tasks
from vla.diagnostics.eval import _resolve_task_ids


def _write_classification(tmp_path: Path) -> Path:
    path = tmp_path / "task_classification.json"
    path.write_text(
        json.dumps(
            {
                "libero_spatial": [
                    {
                        "id": 1,
                        "name": "task_camera_a",
                        "category": "Camera Viewpoints",
                        "difficulty_level": 1,
                    },
                    {
                        "id": 2,
                        "name": "task_noise",
                        "category": "Sensor Noise",
                        "difficulty_level": 2,
                    },
                    {
                        "id": 3,
                        "name": "task_camera_b",
                        "category": "Camera Viewpoints",
                        "difficulty_level": 2,
                    },
                    {
                        "id": 4,
                        "name": "task_layout",
                        "category": "Objects Layout",
                        "difficulty_level": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_select_tasks_uses_zero_based_libero_task_ids(tmp_path: Path) -> None:
    classification = _write_classification(tmp_path)

    tasks = select_tasks(classification, suite="spatial", category="camera", max_tasks=10, shuffle=False)

    assert [task.source_id for task in tasks] == [1, 3]
    assert [task.task_id for task in tasks] == [0, 2]


def test_select_tasks_filters_difficulty_and_category_alias(tmp_path: Path) -> None:
    classification = _write_classification(tmp_path)

    tasks = select_tasks(
        classification,
        suite="spatial",
        category="viewpoint",
        difficulty=2,
        max_tasks=10,
        shuffle=False,
    )

    assert [task.name for task in tasks] == ["task_camera_b"]


def test_select_tasks_seeded_subset_is_deterministic(tmp_path: Path) -> None:
    classification = _write_classification(tmp_path)

    first = select_tasks(classification, suite="spatial", category="all", max_tasks=2, seed=7)
    second = select_tasks(classification, suite="spatial", category="total", max_tasks=2, seed=7)

    assert first == second
    assert len(first) == 2


def test_resolve_task_ids_rejects_invalid_or_ambiguous_selection() -> None:
    assert _resolve_task_ids(num_tasks=4, task_ids=[0, 3]) == [0, 3]

    with pytest.raises(ValueError, match="out of range"):
        _resolve_task_ids(num_tasks=4, task_ids=[4])

    with pytest.raises(ValueError, match="either task_id or task_ids"):
        _resolve_task_ids(num_tasks=4, task_id=1, task_ids=[2])
