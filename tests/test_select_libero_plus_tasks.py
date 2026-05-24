from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.select_libero_plus_tasks import resolve_classification_path, select_tasks
from vla.constants import is_libero_simulator, resolve_libero_suite_name
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


def test_resolve_classification_path_falls_back_to_installed_package(tmp_path: Path, monkeypatch) -> None:
    installed_root = tmp_path / "site-packages" / "libero"
    installed_classification = installed_root / "libero" / "benchmark" / "task_classification.json"
    installed_classification.parent.mkdir(parents=True)
    installed_classification.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.select_libero_plus_tasks.importlib.util.find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(installed_root)]),
    )

    missing_default = tmp_path / "missing" / "task_classification.json"
    assert resolve_classification_path(missing_default) == installed_classification


def test_resolve_task_ids_rejects_invalid_or_ambiguous_selection() -> None:
    assert _resolve_task_ids(num_tasks=4, task_ids=[0, 3]) == [0, 3]

    with pytest.raises(ValueError, match="out of range"):
        _resolve_task_ids(num_tasks=4, task_ids=[4])

    with pytest.raises(ValueError, match="either task_id or task_ids"):
        _resolve_task_ids(num_tasks=4, task_id=1, task_ids=[2])


def test_libero_plus_aliases_reuse_libero_backend_names() -> None:
    assert is_libero_simulator("libero_plus")
    assert is_libero_simulator("libero-pro")
    assert resolve_libero_suite_name("spatial") == "libero_spatial"
    assert resolve_libero_suite_name("libero_spatial") == "libero_spatial"
    assert resolve_libero_suite_name("long") == "libero_10"
