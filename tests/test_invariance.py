"""Unit tests for the latent-invariance objective (vla.training.invariance).

These exercise the new code on synthetic tensors and the real instruction-variant
JSON, with no model download or GPU required, so they run fast in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vla.training.invariance import (
    EmaEncoder,
    InvarianceConfig,
    Predictor,
    build_nuisance_views,
    feature_drift,
    invariance_loss,
    load_instruction_variants,
    paraphrase,
)

VARIANTS_JSON = Path("smolvla_language_pilot/instruction_variants.json")


def test_load_instruction_variants_repo_json():
    if not VARIANTS_JSON.exists():
        pytest.skip("instruction_variants.json not present")
    variants = load_instruction_variants(VARIANTS_JSON, ("politeness", "verb_paraphrase"))
    assert variants
    base = next(iter(variants))
    assert isinstance(variants[base], list) and variants[base]
    # The canonical/"original" instruction is never offered as a nuisance target.
    assert base not in variants[base]


def test_paraphrase_returns_known_variant():
    gen = torch.Generator().manual_seed(0)
    variants = {"do x": ["please do x", "kindly do x"]}
    assert paraphrase("do x", variants, gen) in {"please do x", "kindly do x"}


def test_paraphrase_template_fallback_for_unlisted_instruction():
    """The variants JSON covers only 5/10 Spatial tasks; unlisted instructions
    must get templated (politeness / verb-swap) paraphrases, never the base."""
    gen = torch.Generator().manual_seed(0)
    variants: dict[str, list[str]] = {}
    base = "pick up the black bowl and put it on the plate"
    for _ in range(20):
        out = paraphrase(base, variants, gen)
        assert out != base
        assert "bowl" in out and "plate" in out  # meaning-preserving
    # memoized into the table, held-out types (verbosity etc.) not generated
    assert base in variants
    assert not any("for this task" in v or "in this task" in v for v in variants[base])


def test_corrupt_images_preserves_shape_and_changes_pixels():
    # fog is implemented without the optional cv2 dependency, so this runs anywhere.
    cfg = InvarianceConfig(noise_types=("fog",), severities=(1, 3))
    gen = torch.Generator().manual_seed(0)
    from vla.training.invariance import corrupt_images

    x4 = torch.rand(2, 3, 32, 32)
    y4 = corrupt_images(x4, cfg, gen)
    assert y4.shape == x4.shape
    assert 0.0 <= float(y4.min()) and float(y4.max()) <= 1.0
    assert not torch.allclose(y4, x4)  # corruption actually altered the image

    x5 = torch.rand(2, 2, 3, 32, 32)  # (B, V, C, H, W)
    assert corrupt_images(x5, cfg, gen).shape == x5.shape


def test_build_nuisance_views_respects_axis_toggles():
    gen = torch.Generator().manual_seed(0)
    imgs = torch.rand(2, 3, 16, 16)
    instrs = ["do x", "do y"]
    variants = {"do x": ["please do x"], "do y": ["kindly do y"]}

    # vision-only: instructions unchanged
    cfg_v = InvarianceConfig(noise_types=("fog",), apply_vision=True, apply_language=False)
    _, instr_v = build_nuisance_views(imgs, instrs, cfg_v, variants, gen)
    assert instr_v == instrs

    # language-only: images unchanged
    cfg_l = InvarianceConfig(apply_vision=False, apply_language=True)
    img_l, instr_l = build_nuisance_views(imgs, instrs, cfg_l, variants, gen)
    assert torch.allclose(img_l, imgs.clamp(0, 1))
    assert instr_l == ["please do x", "kindly do y"]


def test_predictor_shape():
    pred = Predictor(16, bottleneck=4)
    assert pred(torch.randn(5, 16)).shape == (5, 16)


def test_nuisance_prob_gates_per_sample():
    """nuisance_prob=0 must pass clean inputs through; =1 must perturb all
    (the augment arm uses 0.5 for a fair clean/augmented mix)."""
    gen = torch.Generator().manual_seed(0)
    imgs = torch.rand(4, 3, 16, 16)
    instrs = ["do x"] * 4
    variants = {"do x": ["please do x"]}

    cfg0 = InvarianceConfig(noise_types=("fog",), nuisance_prob=0.0)
    img0, instr0 = build_nuisance_views(imgs, instrs, cfg0, variants, gen)
    assert torch.allclose(img0, imgs.clamp(0, 1)) and instr0 == instrs

    cfg1 = InvarianceConfig(noise_types=("fog",), nuisance_prob=1.0)
    img1, instr1 = build_nuisance_views(imgs, instrs, cfg1, variants, gen)
    assert not torch.allclose(img1, imgs.clamp(0, 1))
    assert instr1 == ["please do x"] * 4


def test_default_config_has_no_predictor():
    """v2 default: direct alignment. The v1 predictor absorbed the invariance
    mapping (inv_loss -> 0 while representation drift rose), so the backbone
    never became invariant — regression guard on the default."""
    assert InvarianceConfig().use_predictor is False


def test_invariance_loss_zero_when_identical():
    z = torch.randn(4, 16)
    assert abs(invariance_loss(z, z, predictor=None).item()) < 1e-5


def test_invariance_loss_decreases_under_optimization():
    target = torch.randn(4, 16)
    z = torch.nn.Parameter(torch.randn(4, 16))
    opt = torch.optim.SGD([z], lr=0.5)
    first = invariance_loss(z, target, predictor=None).item()
    for _ in range(50):
        opt.zero_grad()
        invariance_loss(z, target, predictor=None).backward()
        opt.step()
    assert invariance_loss(z, target, predictor=None).item() < first


def test_variance_loss_relative_scale():
    """Zero when nuisance spread matches clean spread; positive when the
    nuisance rep collapses relative to it — and scale-free (natural backbone
    scale must not be penalized)."""
    from vla.training.invariance import variance_loss

    z = torch.randn(32, 16) * 7.3  # arbitrary natural scale
    assert variance_loss(z, z).item() < 1e-4
    assert variance_loss(z * 0.1, z).item() > 0.5  # collapsed nuisance rep
    assert variance_loss(z.clone(), z * 3.0).item() > 0.1  # mismatch either way


def test_feature_drift_zero_for_identical_positive_otherwise():
    z = torch.randn(4, 16)
    assert feature_drift(z, z) < 1e-5
    assert feature_drift(z, torch.randn(4, 16)) > 0.0


def test_sftconfig_to_dict_serializes_nested_invariance():
    """to_dict() must recursively convert the nested InvarianceConfig (else the
    training-run JSON write fails — regression guard for the HPC smoke bug)."""
    from dataclasses import is_dataclass

    from vla.training.sft_smolvla import SFTConfig

    d = SFTConfig(invariance=InvarianceConfig(enabled=True)).to_dict()
    assert isinstance(d["invariance"], dict)
    assert d["invariance"]["enabled"] is True
    assert not any(is_dataclass(v) and not isinstance(v, type) for v in d.values())


def test_ema_update_math():
    online = torch.nn.Linear(4, 4)
    ema = EmaEncoder(online)
    before = [p.clone() for p in ema.model.parameters()]
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    ema.update(online, decay=0.9)
    for b, e, o in zip(before, ema.model.parameters(), online.parameters(), strict=True):
        assert torch.allclose(e, 0.9 * b + 0.1 * o, atol=1e-6)
