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
        "rollout": {"eval_num_envs": 2, "num_envs": 8, "n_action_steps": 5},
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
    assert validation.summary["eval_targets"][0]["n_action_steps"] == "5"
    assert validation.summary["eval_targets"][0]["num_episodes"] == "8"
    assert validation.summary["eval_targets"][0]["match_reason"] == "wandb_name"
    assert "--checkpoint-dir" in validation.summary["args"]
    assert str(checkpoint_dir).replace("\\", "/") in validation.summary["args"]
    assert "--n-action-steps" in validation.summary["args"]
    assert "5" in validation.summary["args"]


def test_submit_eval_overrides_n_action_steps_and_num_episodes(monkeypatch) -> None:
    checkpoint_dir = Path(".tmp") / "test_hydra_job_generation" / "run_overrides" / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "policy.pt").write_text("stub", encoding="utf-8")
    train_config = {"wandb_name": "unit-train"}
    composed_config = {
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "simulator": "libero",
        "suite": "spatial",
        "eval_episodes": 10,
        "max_steps": 220,
        "seed": 42,
        "rollout": {"num_envs": 8, "n_action_steps": 5},
    }
    record = {
        "wandb_run_name": "unit-train_sparse_rl_spatial_task_5_seed42_999",
        "lsf_job_id": "999",
        "best_checkpoint_dir": str(checkpoint_dir),
    }

    monkeypatch.setattr(jobs, "load_train_experiment", lambda _experiment: (Path("unit.yaml"), train_config))
    monkeypatch.setattr(jobs, "compose_train_experiment_dict", lambda _experiment: composed_config)
    monkeypatch.setattr(jobs, "load_training_records", lambda: [record])

    validation = jobs.validate_train_eval_submit(
        "unit_train",
        "a10-10h",
        "best",
        submit=False,
        n_action_steps=1,
        num_episodes=100,
    )

    target = validation.summary["eval_targets"][0]
    assert target["n_action_steps"] == "1"
    assert target["num_episodes"] == "100"
    args = validation.summary["args"]
    assert "--n-action-steps" in args
    n_idx = args.index("--n-action-steps")
    assert args[n_idx + 1] == "1"
    assert "--num-episodes" in args
    e_idx = args.index("--num-episodes")
    assert args[e_idx + 1] == "100"


def test_submit_eval_with_explicit_checkpoint_dir_bypasses_record_match(monkeypatch) -> None:
    explicit_dir = Path(".tmp") / "test_hydra_job_generation" / "explicit" / "best"
    explicit_dir.mkdir(parents=True, exist_ok=True)
    (explicit_dir / "policy.pt").write_text("stub", encoding="utf-8")
    train_config = {"wandb_name": "unit-train"}
    composed_config = {
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "simulator": "libero",
        "suite": "spatial",
        "eval_episodes": 10,
        "seed": 42,
        "rollout": {"num_envs": 8, "n_action_steps": 5},
    }

    def _fail_load_records() -> list[dict]:
        raise AssertionError("checkpoint_dir override should bypass record matching")

    monkeypatch.setattr(jobs, "load_train_experiment", lambda _experiment: (Path("unit.yaml"), train_config))
    monkeypatch.setattr(jobs, "compose_train_experiment_dict", lambda _experiment: composed_config)
    monkeypatch.setattr(jobs, "load_training_records", _fail_load_records)

    validation = jobs.validate_train_eval_submit(
        "unit_train",
        "a10-10h",
        "best",
        submit=False,
        n_action_steps=1,
        num_episodes=100,
        checkpoint_dir=str(explicit_dir),
        training_job_id="28338903",
    )

    target = validation.summary["eval_targets"][0]
    assert target["checkpoint_dir"] == str(explicit_dir).replace("\\", "/")
    assert target["match_reason"] == "explicit_checkpoint_dir"
    assert target["training_job_id"] == "28338903"
    assert "28338903" in target["wandb_name"]
    assert validation.errors == []


def test_submit_eval_pinned_training_job_id_filters_records(monkeypatch) -> None:
    wrong_dir = Path(".tmp") / "test_hydra_job_generation" / "wrong" / "best"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    (wrong_dir / "policy.pt").write_text("stub", encoding="utf-8")
    right_dir = Path(".tmp") / "test_hydra_job_generation" / "right" / "best"
    right_dir.mkdir(parents=True, exist_ok=True)
    (right_dir / "policy.pt").write_text("stub", encoding="utf-8")

    train_config = {"wandb_name": "unit-train"}
    composed_config = {
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "simulator": "libero",
        "suite": "spatial",
        "eval_episodes": 100,
        "seed": 42,
        "rollout": {"num_envs": 8, "n_action_steps": 1},
    }
    records = [
        {
            "wandb_run_name": "unit-train_sparse_rl_spatial_task_5_seed42_111",
            "lsf_job_id": "111",
            "best_checkpoint_dir": str(wrong_dir),
        },
        {
            "wandb_run_name": "unit-train_sparse_rl_spatial_task_5_seed42_222",
            "lsf_job_id": "222",
            "best_checkpoint_dir": str(right_dir),
        },
    ]

    monkeypatch.setattr(jobs, "load_train_experiment", lambda _experiment: (Path("unit.yaml"), train_config))
    monkeypatch.setattr(jobs, "compose_train_experiment_dict", lambda _experiment: composed_config)
    monkeypatch.setattr(jobs, "load_training_records", lambda: records)

    validation = jobs.validate_train_eval_submit(
        "unit_train",
        "a10-10h",
        "best",
        submit=False,
        training_job_id="222",
    )

    target = validation.summary["eval_targets"][0]
    assert target["training_job_id"] == "222"
    assert target["checkpoint_dir"] == str(right_dir).replace("\\", "/")


def test_submit_eval_pinned_training_job_id_unknown_errors(monkeypatch) -> None:
    train_config = {"wandb_name": "unit-train"}
    composed_config = {"rollout": {"n_action_steps": 1}, "eval_episodes": 100, "seed": 42}
    records = [{"wandb_run_name": "unit-train_x", "lsf_job_id": "111", "save_dir": str(Path(".tmp"))}]
    monkeypatch.setattr(jobs, "load_train_experiment", lambda _experiment: (Path("unit.yaml"), train_config))
    monkeypatch.setattr(jobs, "compose_train_experiment_dict", lambda _experiment: composed_config)
    monkeypatch.setattr(jobs, "load_training_records", lambda: records)

    validation = jobs.validate_train_eval_submit(
        "unit_train",
        "a10-10h",
        "best",
        submit=False,
        training_job_id="222",
    )

    assert validation.errors
    assert "lsf_job_id='222'" in validation.errors[0]


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
