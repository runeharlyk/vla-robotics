"""Integration checks for the latent-invariance objective against the real SmolVLA.

Exercises ``encode_prefix_pooled``, the EMA target encoder, and one full
combined-loss optimizer step on the actual 0.45B model with a synthetic batch
(so no LIBERO simulator/dataset is required).  Skipped automatically when CUDA
or the cached checkpoint is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CKPT = "HuggingFaceVLA/smolvla_libero"
_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceVLA--smolvla_libero"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _CACHE.exists(),
    reason="requires CUDA and a cached SmolVLA checkpoint",
)


@pytest.fixture(scope="module")
def policy():
    from vla.models.smolvla import SmolVLAPolicy

    return SmolVLAPolicy(checkpoint=CKPT, action_dim=7, state_dim=8, device="cuda")


def _batch(policy, b: int = 2):
    return {
        "images": torch.rand(b, 3, 256, 256),
        "instr": ["pick up the black bowl and place it on the plate"] * b,
        "states": torch.zeros(b, 8, device="cuda"),
        "chunks": torch.zeros(b, policy.chunk_size, 7, device="cuda"),
        "mask": torch.ones(b, policy.chunk_size, device="cuda"),
    }


def test_encode_prefix_pooled_and_ema_invariance(policy):
    from vla.training.invariance import (
        InvarianceConfig,
        InvarianceModule,
        feature_drift,
        invariance_loss,
    )

    d = _batch(policy)
    with torch.autocast("cuda", dtype=policy.dtype):
        z = policy.encode_prefix_pooled(d["images"].to("cuda"), d["instr"], d["states"])
    assert z.shape == (2, policy.prefix_dim)
    assert torch.isfinite(z).all()

    cfg = InvarianceConfig(
        enabled=True, target="ema", apply_vision=True, apply_language=True,
        noise_types=("gaussian_blur",), severities=(3,),
    )
    inv = InvarianceModule(policy, cfg)
    nz_imgs, nz_instr = inv.make_views(d["images"], d["instr"])
    with torch.autocast("cuda", dtype=policy.dtype):
        z_pert = policy.encode_prefix_pooled(nz_imgs.to("cuda"), nz_instr, d["states"])
    z_clean = inv.encode_clean(policy, d["images"].to("cuda"), d["instr"], d["states"])

    assert torch.isfinite(invariance_loss(z_pert, z_clean, inv.predictor))
    assert feature_drift(z_clean, z_pert) > 0.0  # visual corruption => reps differ
    inv.ema_step(policy)  # EMA update runs on the real model


def test_one_combined_training_step_direct_alignment(policy):
    """One SFT + invariance step with the v2 objective (no predictor): the
    invariance gradient must reach the VLM backbone directly — this is the
    mechanism the v1 predictor absorbed."""
    from vla.training.invariance import InvarianceConfig, InvarianceModule, invariance_loss

    cfg = InvarianceConfig(enabled=True, target="online", noise_types=("gaussian_blur",), severities=(3,))
    inv = InvarianceModule(policy, cfg)
    assert inv.predictor is None and inv.trainable_parameters() == []

    # NOTE: module-scoped policy is loaded with defaults (frozen VLM), so grads
    # only reach the action expert there; use a fresh check on requires_grad'd
    # params generically: unfreeze the backbone for this test.
    for p in policy.model.vlm_with_expert.parameters():
        p.requires_grad_(True)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-5)

    d = _batch(policy)
    nz_imgs, nz_instr = inv.make_views(d["images"], d["instr"])
    opt.zero_grad()
    with torch.autocast("cuda", dtype=policy.dtype):
        out = policy(d["images"].to("cuda"), d["instr"], d["chunks"], d["mask"], states=d["states"])
        z_pert = policy.encode_prefix_pooled(nz_imgs.to("cuda"), nz_instr, d["states"])
    z_clean = inv.encode_clean(policy, d["images"].to("cuda"), d["instr"], d["states"])
    total = out["loss"] + cfg.lambda_inv * invariance_loss(z_pert, z_clean, inv.predictor)
    assert torch.isfinite(total)
    total.backward()

    vlm_grads = sum(
        1
        for p in policy.model.vlm_with_expert.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    assert vlm_grads > 0, "invariance gradient did not reach the VLM backbone"

    # Decisive locus check: the gradient must reach the LLM *transformer layers*
    # (where vision-language fusion happens). Pooling input embeddings — the
    # v1/v2 bug — leaves these at exactly zero for the language pathway.
    llm_layer_grads = sum(
        1
        for name, p in policy.model.vlm_with_expert.named_parameters()
        if "text_model" in name
        and ".layers." in name
        and p.requires_grad
        and p.grad is not None
        and p.grad.abs().sum() > 0
    )
    assert llm_layer_grads > 0, "invariance gradient did not reach the LLM transformer (fusion locus)"
    opt.step()
