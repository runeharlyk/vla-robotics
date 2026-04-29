from __future__ import annotations

from pathlib import Path

from vla.utils import experiment_jobs as jobs
from vla.utils.experiment_jobs import (
    generate_hydra_job_script,
    matching_training_records,
    run_matches_train_experiment,
    validate_hydra_submit,
)


def test_generate_train_hydra_job_uses_experiment_and_profile() -> None:
    job = generate_hydra_job_script("train", "success_bc_t5_chunk5", "l40s-16")

    assert job.name == "train_success_bc_t5_chunk5_l40s-16"
    assert "#BSUB -q gpul40s" in job.script
    assert "#BSUB -n 16" in job.script
    assert "uv run --no-sync python scripts/train_srpo_hydra.py experiment=success_bc_t5_chunk5" in job.script


def test_generate_eval_hydra_job_uses_experiment_and_profile() -> None:
    job = generate_hydra_job_script("eval", "spatial_rl_28263586_seeded", "a100-12")

    assert job.name == "eval_spatial_rl_28263586_seeded_a100-12"
    assert "#BSUB -q gpua100" in job.script
    assert "#BSUB -n 12" in job.script
    assert "uv run --no-sync python scripts/evaluate_hydra.py experiment=spatial_rl_28263586_seeded" in job.script


def test_generate_eval_hydra_job_supports_a10_eval_profile() -> None:
    job = generate_hydra_job_script("eval", "spatial_rl_28263586_seeded", "a10-10h")

    assert job.name == "eval_spatial_rl_28263586_seeded_a10-10h"
    assert "#BSUB -q gpua10" in job.script
    assert "#BSUB -W 10:00" in job.script


def test_validate_hydra_submit_accepts_known_experiment_and_profile() -> None:
    validation = validate_hydra_submit("train", "fpo_t5_v28_control", "l40s-16", submit=False)

    assert validation.errors == []
    assert validation.summary["profile"]["queue"] == "gpul40s"
    assert validation.summary["experiment"] == "fpo_t5_v28_control"
    assert "--update-method" in validation.summary["args"]


def test_validate_eval_submit_summarizes_single_checkpoint_target() -> None:
    validation = validate_hydra_submit("eval", "spatial_rl_28263586_seeded", "a10-10h", submit=False)

    assert validation.errors == []
    assert validation.summary["eval_targets"] == [
        {
            "label": "spatial-rl-28263586-seeded",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "checkpoint_dir": "/work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_5_seed42_28263586/best",
            "wandb_name": "eval_rl_spatial_28263586_best_current_seed42",
            "training_job_id": "28263586",
        }
    ]


def test_validate_eval_submit_summarizes_protocol_targets() -> None:
    validation = validate_hydra_submit("eval", "spatial_current_protocol", "a10-10h", submit=False)

    assert validation.errors == []
    assert len(validation.summary["eval_targets"]) == 3
    assert validation.summary["eval_targets"][0]["label"] == "spatial-sft-current-seeded"
    assert validation.summary["eval_targets"][0]["checkpoint_dir"] == ""
    assert validation.summary["eval_targets"][1]["training_job_id"] == "28188629"
    assert validation.summary["eval_targets"][2]["training_job_id"] == "28263586"


def test_validate_train_eval_submit_resolves_checkpoint_from_training_experiment(monkeypatch) -> None:
    checkpoint_dir = Path(".tmp") / "test_hydra_job_generation" / "run" / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "policy.pt").write_text("stub", encoding="utf-8")
    train_config = {"wandb_name": "unit-train", "metadata": {"training_job_ids": [123]}}
    composed_config = {
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "simulator": "libero",
        "suite": "spatial",
        "eval_episodes": 8,
        "max_steps": 220,
        "seed": 42,
        "rollout": {"eval_num_envs": 2, "num_envs": 8},
    }
    record = {
        "wandb_run_name": "unit-train_sparse_rl_spatial_task_5_seed42_123",
        "lsf_job_id": "123",
        "best_checkpoint_dir": str(checkpoint_dir),
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
    }

    monkeypatch.setattr(jobs, "load_train_experiment", lambda _experiment: (Path("unit.yaml"), train_config))
    monkeypatch.setattr(jobs, "compose_train_experiment_dict", lambda _experiment: composed_config)
    monkeypatch.setattr(jobs, "load_training_records", lambda: [record])

    validation = jobs.validate_train_eval_submit("unit_train", "a10-10h", "best", submit=False)

    assert validation.errors == []
    assert validation.summary["eval_targets"][0]["checkpoint_dir"] == str(checkpoint_dir).replace("\\", "/")
    assert validation.summary["eval_targets"][0]["training_job_id"] == "123"
    assert "--checkpoint-dir" in validation.summary["args"]
    assert str(checkpoint_dir).replace("\\", "/") in validation.summary["args"]


def test_training_run_matches_config_by_wandb_prefix_or_job_id() -> None:
    config = {
        "wandb_name": "v28-control",
        "metadata": {"training_job_ids": [28188629]},
    }

    by_name = {"wandb_run_name": "v28-control_sparse_rl_spatial_task_5_seed42_1"}
    by_job = {"wandb_run_name": "renamed-run", "lsf_job_id": "28188629"}
    miss = {"wandb_run_name": "other-run", "lsf_job_id": "1"}

    assert run_matches_train_experiment(by_name, "fpo_t5_v28_control", config, fuzzy=False) == (True, "wandb_name")
    assert run_matches_train_experiment(by_job, "fpo_t5_v28_control", config, fuzzy=False) == (
        True,
        "training_job_id",
    )
    assert run_matches_train_experiment(miss, "fpo_t5_v28_control", config, fuzzy=False) == (False, "")


def test_matching_training_records_returns_empty_for_unrun_config() -> None:
    config = {"wandb_name": "new-experiment", "metadata": {"label": "new experiment"}}
    records = [{"wandb_run_name": "old-experiment_sparse_rl_spatial_task_5_seed42_1"}]

    assert matching_training_records("new_experiment", config, records, fuzzy=False) == []
