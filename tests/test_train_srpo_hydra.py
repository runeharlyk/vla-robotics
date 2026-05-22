from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train_srpo_hydra import config_to_train_srpo_args

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train_srpo"


def test_hydra_config_to_train_srpo_args_uses_dotted_cli_flags() -> None:
    cfg = OmegaConf.create(
        {
            "update_method": "success_bc",
            "lr": 1e-6,
            "wandb": False,
            "rollout": {
                "num_envs": 8,
                "n_action_steps": 5,
                "gradient_checkpointing": True,
            },
            "awr": {
                "minibatch_trajs": 3,
            },
            "fpo": {
                "full_chunk_target": False,
            },
            "success_bc": {
                "minibatch_trajs": 2,
            },
            "kl": {
                "coeff": 0.0,
                "sft_coeff": 0.005,
            },
            "replay": {
                "success_total_size": 320,
            },
            "metadata": {"label": "ignored"},
        }
    )

    args = config_to_train_srpo_args(cfg)

    assert "--update-method" in args
    assert "success_bc" in args
    assert "--rollout.num-envs" in args
    assert "--rollout.n-action-steps" in args
    assert "--rollout.gradient-checkpointing" in args
    assert "--awr.minibatch-trajs" in args
    assert "3" in args
    assert "--no-fpo.full-chunk-target" in args
    assert "--success-bc.minibatch-trajs" in args
    assert "2" in args
    assert "--kl.sft-coeff" in args
    assert "--replay.success-total-size" in args
    assert "--no-wandb" in args
    assert "--metadata.label" not in args


def test_hydra_experiment_overrides_apply_at_top_level() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=success_bc_t5_chunk5"])

    assert "experiment" not in cfg
    assert cfg.update_method == "success_bc"
    assert cfg.lr == 1e-6
    assert cfg.kl.coeff == 0.0
    assert cfg.replay.success_total_size == 320
    assert cfg.rollout.n_action_steps == 5


def test_all_train_hydra_experiments_compose_and_convert_to_cli() -> None:
    experiment_dir = CONFIG_DIR / "experiment"
    experiment_names = sorted(path.stem for path in experiment_dir.glob("*.yaml"))

    assert experiment_names

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        for experiment in experiment_names:
            cfg = compose(config_name="base", overrides=[f"experiment={experiment}"])
            args = config_to_train_srpo_args(cfg)

            assert "experiment" not in cfg
            assert "--metadata.label" not in args
            assert "--checkpoint" in args
            assert "--simulator" in args
            assert "--wandb-name" in args
