"""CPU-only tests for the WiSE-FT merge helper and config plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.evaluate_hydra import config_to_evaluate_args, expand_eval_configs
from vla.utils.wise_ft import wise_ft_merge_into_policy

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "evaluate"


class _ToyVLA(torch.nn.Module):
    """Minimal stand-in for VLAFlowMatching with parameters and an int buffer."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.register_buffer("position_ids", torch.arange(4, dtype=torch.long), persistent=True)


def _make_toy_policy(weight: torch.Tensor, bias: torch.Tensor) -> SimpleNamespace:
    """A SmolVLAPolicy stand-in with just the attribute the helper touches."""
    return SimpleNamespace(model=_ToyVLA(weight, bias))


def _save_rl_ckpt(tmp_path: Path, weight: torch.Tensor, bias: torch.Tensor) -> Path:
    """Mimic the policy.pt layout that load_checkpoint reads."""
    ckpt_dir = tmp_path / "rl_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rl_module = _ToyVLA(weight, bias)
    torch.save(
        {
            "model_state_dict": rl_module.state_dict(),
            "action_dim": 7,
            "state_dim": 8,
            "env_metadata": {},
        },
        ckpt_dir / "policy.pt",
    )
    return ckpt_dir


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_wise_ft_merge_linearly_interpolates_weights(tmp_path, alpha) -> None:
    sft_w = torch.zeros(3, 4)
    sft_b = torch.zeros(3)
    rl_w = torch.ones(3, 4)
    rl_b = torch.ones(3) * 2.0

    policy = _make_toy_policy(sft_w, sft_b)
    rl_dir = _save_rl_ckpt(tmp_path, rl_w, rl_b)

    diag = wise_ft_merge_into_policy(policy, rl_dir, alpha)

    expected_w = (1.0 - alpha) * sft_w + alpha * rl_w
    expected_b = (1.0 - alpha) * sft_b + alpha * rl_b

    torch.testing.assert_close(policy.model.linear.weight.detach(), expected_w)
    torch.testing.assert_close(policy.model.linear.bias.detach(), expected_b)
    assert diag["alpha"] == pytest.approx(alpha)
    assert diag["n_merged_keys"] == 2  # weight + bias
    assert diag["n_copied_keys"] == 1  # position_ids buffer
    assert diag["max_abs_delta"] == pytest.approx(alpha * 2.0)


def test_wise_ft_merge_alpha_zero_is_identity_on_sft(tmp_path) -> None:
    sft_w = torch.randn(3, 4)
    sft_b = torch.randn(3)
    policy = _make_toy_policy(sft_w, sft_b)
    rl_dir = _save_rl_ckpt(tmp_path, torch.full_like(sft_w, 99.0), torch.full_like(sft_b, -7.0))

    diag = wise_ft_merge_into_policy(policy, rl_dir, alpha=0.0)

    torch.testing.assert_close(policy.model.linear.weight.detach(), sft_w)
    torch.testing.assert_close(policy.model.linear.bias.detach(), sft_b)
    assert diag["max_abs_delta"] == pytest.approx(0.0)


def test_wise_ft_merge_alpha_one_matches_rl_state_dict(tmp_path) -> None:
    sft_w = torch.zeros(3, 4)
    sft_b = torch.zeros(3)
    rl_w = torch.randn(3, 4)
    rl_b = torch.randn(3)
    policy = _make_toy_policy(sft_w, sft_b)
    rl_dir = _save_rl_ckpt(tmp_path, rl_w, rl_b)

    wise_ft_merge_into_policy(policy, rl_dir, alpha=1.0)

    torch.testing.assert_close(policy.model.linear.weight.detach(), rl_w)
    torch.testing.assert_close(policy.model.linear.bias.detach(), rl_b)


def test_wise_ft_merge_rejects_alpha_out_of_range(tmp_path) -> None:
    policy = _make_toy_policy(torch.zeros(3, 4), torch.zeros(3))
    rl_dir = _save_rl_ckpt(tmp_path, torch.zeros(3, 4), torch.zeros(3))

    with pytest.raises(ValueError, match="wise_ft_alpha"):
        wise_ft_merge_into_policy(policy, rl_dir, alpha=-0.1)
    with pytest.raises(ValueError, match="wise_ft_alpha"):
        wise_ft_merge_into_policy(policy, rl_dir, alpha=1.5)


def test_wise_ft_merge_rejects_mismatched_state_dict_keys(tmp_path) -> None:
    policy = _make_toy_policy(torch.zeros(3, 4), torch.zeros(3))

    bad_dir = tmp_path / "rl_bad"
    bad_dir.mkdir()
    torch.save(
        {
            "model_state_dict": {
                "linear.weight": torch.zeros(3, 4),
                "linear.bias": torch.zeros(3),
                "extra_param.weight": torch.zeros(2, 2),
            },
            "action_dim": 7,
            "state_dim": 8,
        },
        bad_dir / "policy.pt",
    )

    with pytest.raises(ValueError, match="state dicts have different keys"):
        wise_ft_merge_into_policy(policy, bad_dir, alpha=0.5)


def test_wise_ft_merge_missing_checkpoint_file(tmp_path) -> None:
    policy = _make_toy_policy(torch.zeros(3, 4), torch.zeros(3))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="policy.pt"):
        wise_ft_merge_into_policy(policy, empty_dir, alpha=0.5)


def test_eval_hydra_wise_ft_alpha_propagates_to_cli() -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "/tmp/checkpoint/best",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "simulator": "libero",
            "suite": "spatial",
            "num_episodes": 100,
            "n_action_steps": 1,
            "fixed_noise_seed": 42,
            "wandb": True,
            "wise_ft_alpha": 0.5,
            "metadata": {"label": "ignored"},
        }
    )

    args = config_to_evaluate_args(cfg)

    assert "--wise-ft-alpha" in args
    assert "0.5" in args
    assert "--checkpoint-dir" in args
    assert "/tmp/checkpoint/best" in args


def test_eval_hydra_wise_ft_alpha_null_is_omitted() -> None:
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "/tmp/checkpoint/best",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "simulator": "libero",
            "suite": "spatial",
            "num_episodes": 100,
            "n_action_steps": 1,
            "fixed_noise_seed": 42,
            "wandb": True,
            "wise_ft_alpha": None,
            "metadata": {"label": "ignored"},
        }
    )

    args = config_to_evaluate_args(cfg)

    assert "--wise-ft-alpha" not in args


def test_wise_ft_v12_alpha_sweep_expands_to_five_runs() -> None:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=wise_ft_v12_alpha_sweep"])

    expanded = expand_eval_configs(cfg)

    assert len(expanded) == 5
    alphas = [item["wise_ft_alpha"] for item in expanded]
    assert alphas == [0.00, 0.25, 0.50, 0.75, 1.00]
    for item in expanded:
        assert item["checkpoint"] == "HuggingFaceVLA/smolvla_libero"
        assert item["checkpoint_dir"].endswith("10tasks_spatial_seed42_28403771/best")
        assert item["n_action_steps"] == 1
        assert item["fixed_noise_seed"] == 42
        assert item["wandb_name"].startswith("eval_wise_ft_v12_28403771_alpha")
