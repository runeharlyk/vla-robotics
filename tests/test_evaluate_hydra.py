from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.evaluate_hydra import config_to_evaluate_args, expand_eval_configs


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "evaluate"


def test_eval_hydra_config_to_evaluate_args_uses_cli_flags() -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "/tmp/checkpoint/best",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "simulator": "libero",
            "suite": "spatial",
            "num_episodes": 100,
            "fixed_noise_seed": 42,
            "wandb": False,
            "metadata": {"label": "ignored"},
        }
    )

    args = config_to_evaluate_args(cfg)

    assert "--checkpoint-dir" in args
    assert "/tmp/checkpoint/best" in args
    assert "--num-episodes" in args
    assert "--fixed-noise-seed" in args
    assert "--no-wandb" in args
    assert "--metadata.label" not in args


def test_eval_hydra_experiment_overrides_apply_at_top_level() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=spatial_rl_28188629_seeded"])

    assert "experiment" not in cfg
    assert cfg.checkpoint_dir.endswith("spatial_task_5_seed42_28188629/best")
    assert cfg.wandb_name == "eval_rl_spatial_28188629_best_current_seed42"
    assert cfg.fixed_noise_seed == 42


def test_eval_hydra_protocol_expands_multiple_eval_runs() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=spatial_current_protocol"])

    expanded = expand_eval_configs(cfg)

    assert len(expanded) == 3
    assert expanded[0]["checkpoint_dir"] is None
    assert expanded[1]["checkpoint_dir"].endswith("spatial_task_5_seed42_28188629/best")
    assert expanded[2]["checkpoint_dir"].endswith("spatial_task_5_seed42_28263586/best")
    assert {item["fixed_noise_seed"] for item in expanded} == {42}
