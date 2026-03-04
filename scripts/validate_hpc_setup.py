"""Validate HPC environment before submitting SRPO training jobs.

Run inside an interactive GPU session to check that all models can be
downloaded and loaded, CUDA works, and the SRPO pipeline imports are
functional.

Usage (on HPC, after sourcing _env.sh):
    # Download models to HF cache first (login node, no GPU needed):
    uv run huggingface-cli download facebook/dinov2-large
    uv run huggingface-cli download Sylvest/vjepa2-vit-g

    # Then validate on a GPU node:
    uv run python scripts/validate_hpc_setup.py
    uv run python scripts/validate_hpc_setup.py --world-model vjepa2
    uv run python scripts/validate_hpc_setup.py --world-model all
"""

from __future__ import annotations

import sys
import time

import typer

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def _ok(msg: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str, exc: Exception | None = None) -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    detail = f" ({exc})" if exc else ""
    print(f"  [FAIL] {msg}{detail}")


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_imports() -> None:
    _section("1. Core imports")
    try:
        import torch
        _ok(f"torch {torch.__version__}")
    except Exception as e:
        _fail("import torch", e)
        return

    try:
        import transformers
        _ok(f"transformers {transformers.__version__}")
    except Exception as e:
        _fail("import transformers", e)

    try:
        import sklearn  # noqa: F401
        _ok("scikit-learn available (DBSCAN)")
    except Exception as e:
        _fail("import sklearn", e)

    try:
        import wandb  # noqa: F401
        _ok("wandb available")
    except Exception as e:
        _fail("import wandb", e)


def check_cuda() -> None:
    _section("2. CUDA / GPU")
    import torch

    if torch.cuda.is_available():
        _ok(f"CUDA available — {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            _ok(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        _fail("CUDA not available — world-model encoding requires GPU")


def check_srpo_pipeline() -> None:
    _section("3. SRPO pipeline imports")
    try:
        from vla.models.world_model import DINOv2Encoder, VJEPA2Encoder, build_world_model  # noqa: F401
        _ok("vla.models.world_model")
    except Exception as e:
        _fail("vla.models.world_model", e)

    try:
        from vla.rl.srpo_reward import WorldProgressReward, ClusterDiagnostics  # noqa: F401
        _ok("vla.rl.srpo_reward (with ClusterDiagnostics)")
    except Exception as e:
        _fail("vla.rl.srpo_reward", e)

    try:
        from vla.rl.trainer import train_srpo, SRPOConfig  # noqa: F401
        _ok("vla.rl.trainer")
    except Exception as e:
        _fail("vla.rl.trainer", e)

    try:
        from vla.models.smolvla import SmolVLAPolicy  # noqa: F401
        _ok("vla.models.smolvla")
    except Exception as e:
        _fail("vla.models.smolvla", e)


def check_dinov2() -> None:
    _section("4. DINOv2 encoder")
    import torch

    try:
        from vla.models.world_model import DINOv2Encoder, DINOV2_MODEL_ID

        print(f"  Model ID: {DINOV2_MODEL_ID}")
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = DINOv2Encoder(device=device)
        _ok(f"DINOv2 loaded in {time.time() - t0:.1f}s — embed_dim={encoder.embed_dim()}")

        dummy = torch.randn(2, 3, 224, 224)
        embs = encoder.encode_frames(dummy)
        _ok(f"Forward pass: input (2,3,224,224) → output {tuple(embs.shape)}")
    except Exception as e:
        _fail("DINOv2 load/forward", e)


def check_vjepa2() -> None:
    _section("5. V-JEPA 2 encoder")
    import torch

    try:
        from vla.models.world_model import VJEPA2Encoder, VJEPA2_MODEL_ID

        print(f"  Model ID: {VJEPA2_MODEL_ID}")
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = VJEPA2Encoder(device=device)
        elapsed = time.time() - t0

        if encoder._fallback is not None:
            _fail(f"V-JEPA 2 fell back to DINOv2 after {elapsed:.1f}s — checkpoint not available?")
            print("       Try: uv run huggingface-cli download Sylvest/vjepa2-vit-g")
            return

        proc_type = "video" if encoder._is_video_processor else "image"
        _ok(f"V-JEPA 2 loaded in {elapsed:.1f}s — embed_dim={encoder.embed_dim()}, processor={proc_type}")

        dummy_traj = torch.randn(20, 3, 384, 384)
        emb = encoder.encode_trajectory(dummy_traj, subsample_every=5)
        _ok(f"encode_trajectory: (20,3,384,384) → {tuple(emb.shape)}")
    except Exception as e:
        _fail("V-JEPA 2 load/forward", e)


def check_hf_cache() -> None:
    _section("6. HuggingFace cache")
    import os
    from pathlib import Path

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub_dir = Path(hf_home) / "hub"
    print(f"  HF_HOME = {hf_home}")

    if hub_dir.exists():
        models = sorted(p.name for p in hub_dir.iterdir() if p.name.startswith("models--"))
        if models:
            _ok(f"{len(models)} cached model(s):")
            for m in models:
                print(f"       {m.replace('models--', '').replace('--', '/')}")
        else:
            _fail("Hub cache exists but no models found. Run uv run huggingface-cli download first.")
    else:
        _fail(f"Hub directory not found: {hub_dir}")

    dinov2_cached = any("dinov2" in m for m in (models if hub_dir.exists() else []))
    vjepa2_cached = any("vjepa2" in m for m in (models if hub_dir.exists() else []))
    if dinov2_cached:
        _ok("DINOv2 checkpoint cached")
    else:
        _fail("DINOv2 not cached — run: uv run huggingface-cli download facebook/dinov2-large")
    if vjepa2_cached:
        _ok("V-JEPA 2 checkpoint cached")
    else:
        _fail("V-JEPA 2 not cached — run: uv run huggingface-cli download Sylvest/vjepa2-vit-g")


def main(
    world_model: str = typer.Option("dinov2", "--world-model", "-w", help="dinov2, vjepa2, or all"),
) -> None:
    """Validate HPC environment for SRPO training."""
    check_imports()
    check_cuda()
    check_srpo_pipeline()
    check_hf_cache()

    wm = world_model.lower()
    if wm in ("dinov2", "all"):
        check_dinov2()
    if wm in ("vjepa2", "all"):
        check_vjepa2()

    _section("Summary")
    total = CHECKS_PASSED + CHECKS_FAILED
    print(f"  {CHECKS_PASSED}/{total} checks passed, {CHECKS_FAILED} failed")
    if CHECKS_FAILED > 0:
        print("  Fix the issues above before submitting jobs.")
        sys.exit(1)
    else:
        print("  All checks passed — ready to submit!")
        sys.exit(0)


if __name__ == "__main__":
    typer.run(main)
