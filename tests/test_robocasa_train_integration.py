from __future__ import annotations

from scripts.train_srpo import _build_tasks
from vla.constants import LiberoSuite, Simulator


def test_build_tasks_robocasa_uses_probe_and_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        "vla.envs.robocasa.list_robocasa_tasks",
        lambda: ["PickPlaceCounterToCabinet", "OpenSingleDoor"],
    )
    monkeypatch.setattr(
        "vla.envs.robocasa.probe_robocasa_task",
        lambda env_id, **kwargs: {
            "env_id": env_id,
            "instruction": f"instruction for {env_id}",
            "state_dim": 16,
            "action_dim": 12,
            "layout_id": kwargs.get("layout_id"),
            "style_id": kwargs.get("style_id"),
            "split": kwargs.get("split"),
        },
    )

    task_specs, demo_trajectories, state_dim, action_dim = _build_tasks(
        data_path=None,
        data_dir=None,
        libero_suite=None,
        num_demos=0,
        seed=0,
        simulator=Simulator.ROBOCASA,
        suite=LiberoSuite.SPATIAL,
        task_ids=[1],
        include_demos=False,
        env_id_override=None,
        instruction_override=None,
        robocasa_layout_id=20,
        robocasa_style_id=58,
        robocasa_split="all",
    )

    assert demo_trajectories is None
    assert state_dim == 16
    assert action_dim == 12
    assert len(task_specs) == 1
    spec = task_specs[0]
    assert spec.task_id == "OpenSingleDoor_layout20_style58"
    assert spec.env_id == "OpenSingleDoor"
    assert spec.layout_id == 20
    assert spec.style_id == 58
    assert spec.split == "all"
    assert spec.instruction == "instruction for OpenSingleDoor"
