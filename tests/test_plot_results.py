from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from vla.utils.plot_results import _extract_eval_run_id, _extract_filter_run_id, load_eval_records


def _evaluation_record(name: str, *, method: str = "sparse_rl") -> dict:
    return {
        "record_type": "evaluation",
        "eval_name": name,
        "wandb_run_name": name,
        "suite": "spatial",
        "success_rate": 0.5,
        "task_metrics": [{"task_id": 0, "success_rate": 0.5}],
        "training_method": method,
        "training_save_dir": "",
    }


def test_extract_filter_run_id_uses_training_id_from_eval_name() -> None:
    record = _evaluation_record("eval_rl_spatial_l40s_spatial_task_2_seed42_28123898_28248486")

    assert _extract_filter_run_id(record) == 28123898


def test_extract_eval_run_id_uses_trailing_id_from_eval_name() -> None:
    record = _evaluation_record("eval_rl_spatial_l40s_spatial_task_2_seed42_28123898_28248486")

    assert _extract_eval_run_id(record) == 28248486


class _FakeJsonPath:
    def __init__(self, name: str, record: dict) -> None:
        self.name = name
        self._record = record

    def read_text(self, *, encoding: str) -> str:
        return json.dumps(self._record)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _FakeJsonPath):
            return NotImplemented
        return self.name < other.name


class _FakeResultsDir:
    def __init__(self, paths: list[_FakeJsonPath]) -> None:
        self._paths = paths

    def glob(self, pattern: str) -> list[_FakeJsonPath]:
        assert pattern == "*.json"
        return self._paths


def test_load_eval_records_filters_by_eval_id_not_training_id() -> None:
    old_training_new_eval = _evaluation_record("eval_rl_spatial_l40s_spatial_task_2_seed42_28123898_28248486")
    new_training = _evaluation_record("eval_rl_spatial_l40s_spatial_task_5_seed42_28188629_28192830")
    results_dir = _FakeResultsDir(
        [
            _FakeJsonPath("old_training_new_eval.json", old_training_new_eval),
            _FakeJsonPath("new_training.json", new_training),
        ]
    )

    records = load_eval_records(cast(Path, results_dir), suite="spatial", min_eval_run_id=28161033)

    assert [record["_source"] for record in records] == ["new_training.json", "old_training_new_eval.json"]


def test_load_eval_records_can_filter_by_training_id() -> None:
    old_training_new_eval = _evaluation_record("eval_rl_spatial_l40s_spatial_task_2_seed42_28123898_28248486")
    new_training = _evaluation_record("eval_rl_spatial_l40s_spatial_task_5_seed42_28188629_28192830")
    results_dir = _FakeResultsDir(
        [
            _FakeJsonPath("old_training_new_eval.json", old_training_new_eval),
            _FakeJsonPath("new_training.json", new_training),
        ]
    )

    records = load_eval_records(cast(Path, results_dir), suite="spatial", min_training_run_id=28161033)

    assert [record["_source"] for record in records] == ["new_training.json"]
