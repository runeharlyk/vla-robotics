"""Unified latent-invariance objective for robust VLA backbones.

Adds a JEPA-style representation-consistency loss on the *fused* vision+language
conditioning features that the SmolVLA action expert receives (the output of
``VLAFlowMatching.embed_prefix``).  The loss aligns the representation of a
**nuisance view** (ImageNet-C corrupted image + paraphrased instruction) to that
of the matched **clean/canonical view** (clean image + canonical instruction),
so robustness is installed in the representation rather than left for the action
decoder to compensate.

See ``RESEARCH_DIRECTION.md`` for the motivation, novelty positioning and the
5-arm experimental ladder this module supports:

    A  baseline            invariance disabled (stock SFT)
    A' augment-only        SFT loss on the nuisance view, no invariance loss
    B  vision-invariance   clean SFT + invariance loss, visual nuisance only
    C  language-invariance clean SFT + invariance loss, language nuisance only
    D  both (unified)      clean SFT + invariance loss, visual + language nuisance

Design choices (kept deliberately simple for the first runnable version):
  * Target = EMA copy of the backbone (JEPA) or the online backbone with
    stop-grad (SimSiam-style).  A predictor head + stop-grad prevents collapse.
  * Visual corruptions reuse ``visual_diagnostic/noise.py`` (ImageNet-C family).
  * Language paraphrases reuse ``smolvla_language_pilot/instruction_variants.json``.
  * Invariance is only ever applied across *nuisance-equivalent* pairs (same
    sample, same task/state) — never across tasks — so task-relevant content is
    preserved (invariance to nuisance, not to meaning).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default split of paraphrase types: train on these, hold the rest out for eval
# so the language-robustness result measures transfer, not memorisation.
DEFAULT_TRAIN_VARIANT_TYPES = ("politeness", "verb_paraphrase")
DEFAULT_HELDOUT_VARIANT_TYPES = ("sentence_structure", "verbosity")

# ImageNet-C corruption families available in visual_diagnostic/noise.py.
DEFAULT_TRAIN_NOISE_TYPES = ("gaussian_blur", "motion_blur", "fog")
DEFAULT_HELDOUT_NOISE_TYPES = ("glass_blur", "zoom_blur")


@dataclass
class InvarianceConfig:
    """Configuration for the latent-invariance objective.

    Disabled by default so existing SFT runs are byte-for-byte unaffected.
    """

    enabled: bool = False
    # Arm selection ---------------------------------------------------------
    apply_vision: bool = True          # corrupt the image in the nuisance view
    apply_language: bool = True        # paraphrase the instruction in the nuisance view
    augment_only: bool = False         # arm A': SFT on nuisance view, no invariance loss
    # Loss ------------------------------------------------------------------
    lambda_inv: float = 1.0            # weight of the invariance term
    target: str = "ema"                # "ema" (JEPA) or "online" (SimSiam stop-grad)
    ema_decay: float = 0.999
    use_predictor: bool = True
    predictor_bottleneck: int = 4      # hidden = D // predictor_bottleneck
    # Visual nuisance -------------------------------------------------------
    noise_types: tuple[str, ...] = DEFAULT_TRAIN_NOISE_TYPES
    severities: tuple[int, ...] = (1, 2, 3)
    # Language nuisance -----------------------------------------------------
    variants_path: str = "smolvla_language_pilot/instruction_variants.json"
    variant_types: tuple[str, ...] = DEFAULT_TRAIN_VARIANT_TYPES
    seed: int = 0


# ---------------------------------------------------------------------------
# Instruction-variant lookup
# ---------------------------------------------------------------------------


def load_instruction_variants(
    path: str | Path,
    variant_types: tuple[str, ...] = DEFAULT_TRAIN_VARIANT_TYPES,
) -> dict[str, list[str]]:
    """Load paraphrases keyed by base instruction, restricted to *variant_types*.

    Handles the repo's ``{"rollouts": [{base_instruction, variants: {type: [...]}}]}``
    schema as well as a flat ``{base: [paraphrases]}`` mapping or a bare list.
    Returns ``{base_instruction: [paraphrase, ...]}`` (the canonical/"original"
    form is excluded — only genuine rewordings are kept).
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Instruction-variants file not found at %s; language nuisance disabled.", p)
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(loaded, list):  # bare list of paraphrases -> wildcard
        return {"*": [str(v) for v in loaded]}

    if isinstance(loaded, dict) and "rollouts" in loaded:
        out: dict[str, list[str]] = {}
        for entry in loaded["rollouts"]:
            base = str(entry["base_instruction"])
            variants = entry.get("variants", {})
            paraphrases: list[str] = []
            for vtype in variant_types:
                paraphrases.extend(str(v) for v in variants.get(vtype, []))
            # never include the canonical "original" form as a nuisance target
            paraphrases = [s for s in paraphrases if s.strip() != base.strip()]
            if paraphrases:
                out[base] = paraphrases
        return out

    if isinstance(loaded, dict):  # flat {base: [paraphrases]}
        return {str(k): [str(v) for v in vs] for k, vs in loaded.items() if isinstance(vs, list)}

    raise ValueError(f"Unrecognised instruction-variants schema in {p}")


def paraphrase(base: str, variants: dict[str, list[str]], gen: torch.Generator) -> str:
    """Return a random paraphrase of *base* (or *base* itself if none available)."""
    choices = variants.get(base) or variants.get("*") or []
    if not choices:
        return base
    idx = int(torch.randint(0, len(choices), (1,), generator=gen).item())
    return choices[idx]


# ---------------------------------------------------------------------------
# Nuisance-view construction
# ---------------------------------------------------------------------------


def _to_float01(images: torch.Tensor) -> torch.Tensor:
    out = images.float()
    if out.numel() and out.max().item() > 2.0:
        out = out / 255.0
    return out.clamp(0.0, 1.0)


def _import_noise():
    """Import the ImageNet-C corruptions from the top-level ``visual_diagnostic``.

    That directory lives at the repo root, not inside the installed ``vla``
    package, so it is not importable when launched as ``python scripts/...``
    (whose ``sys.path[0]`` is ``scripts/``).  Locate the repo root relative to
    this module and add it to ``sys.path`` so the import succeeds regardless of
    how the process was started.
    """
    try:
        from visual_diagnostic.noise import NoiseConfig, apply_noise
    except ModuleNotFoundError:
        import sys

        repo_root = Path(__file__).resolve().parents[3]  # src/vla/training/ -> repo root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from visual_diagnostic.noise import NoiseConfig, apply_noise
    return NoiseConfig, apply_noise


def corrupt_images(
    images: torch.Tensor,
    cfg: InvarianceConfig,
    gen: torch.Generator,
) -> torch.Tensor:
    """Apply a random ImageNet-C corruption per sample (and per view).

    Accepts ``(B, C, H, W)`` or ``(B, V, C, H, W)`` and returns a float[0,1]
    tensor of the same shape on CPU.  One (type, severity) is drawn per sample
    and shared across that sample's views to keep the corruption coherent.
    """
    NoiseConfig, apply_noise = _import_noise()

    imgs = _to_float01(images).cpu()
    squeeze_view = imgs.ndim == 4
    if squeeze_view:
        imgs = imgs.unsqueeze(1)  # (B, 1, C, H, W)
    b, v = imgs.shape[:2]
    out = imgs.clone()
    n_types, n_sev = len(cfg.noise_types), len(cfg.severities)
    for i in range(b):
        nt = cfg.noise_types[int(torch.randint(0, n_types, (1,), generator=gen).item())]
        sev = cfg.severities[int(torch.randint(0, n_sev, (1,), generator=gen).item())]
        nconf = NoiseConfig(noise_type=nt, severity=int(sev))
        for j in range(v):
            out[i, j] = apply_noise(imgs[i, j], nconf)
    if squeeze_view:
        out = out.squeeze(1)
    return out


def build_nuisance_views(
    images: torch.Tensor,
    instructions: list[str],
    cfg: InvarianceConfig,
    variants: dict[str, list[str]],
    gen: torch.Generator,
) -> tuple[torch.Tensor, list[str]]:
    """Return (nuisance_images, nuisance_instructions) for the context view.

    Honours ``cfg.apply_vision`` / ``cfg.apply_language``: an axis that is off
    passes the clean input through unchanged, so arms B (vision-only) and C
    (language-only) are obtained purely from config.
    """
    nuisance_images = corrupt_images(images, cfg, gen) if cfg.apply_vision else _to_float01(images).cpu()
    if cfg.apply_language and variants:
        nuisance_instructions = [paraphrase(instr, variants, gen) for instr in instructions]
    else:
        nuisance_instructions = list(instructions)
    return nuisance_images, nuisance_instructions


# ---------------------------------------------------------------------------
# Invariance head (predictor + optional EMA target encoder) and loss
# ---------------------------------------------------------------------------


class Predictor(nn.Module):
    """SimSiam-style bottleneck MLP predictor (LayerNorm for batch-size robustness)."""

    def __init__(self, dim: int, bottleneck: int = 4) -> None:
        super().__init__()
        hidden = max(dim // max(bottleneck, 1), 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmaEncoder:
    """Holds an EMA copy of ``policy.model`` used to encode the clean view.

    Only the ``embed_prefix`` path is used; parameters are updated by EMA from
    the online model after each optimizer step and never receive gradients.
    """

    def __init__(self, online_model: nn.Module) -> None:
        self.model = copy.deepcopy(online_model)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, online_model: nn.Module, decay: float) -> None:
        for ema_p, online_p in zip(self.model.parameters(), online_model.parameters(), strict=True):
            ema_p.mul_(decay).add_(online_p.detach(), alpha=1.0 - decay)
        for ema_b, online_b in zip(self.model.buffers(), online_model.buffers(), strict=True):
            ema_b.copy_(online_b)


def invariance_loss(
    z_pert: torch.Tensor,
    z_clean: torch.Tensor,
    predictor: nn.Module | None,
) -> torch.Tensor:
    """Negative-cosine invariance loss with stop-grad on the clean target.

    ``z_clean`` is detached (the gradient flows only through the nuisance view +
    predictor), which — together with the predictor — is the SimSiam recipe that
    avoids representational collapse.
    """
    # The invariance head runs in fp32 (it is tiny) regardless of the autocast
    # dtype of the pooled features, so cast before the predictor.
    pred_in = z_pert.float()
    pred = predictor(pred_in) if predictor is not None else pred_in
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(z_clean.detach().float(), dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


@torch.no_grad()
def feature_drift(z_clean: torch.Tensor, z_pert: torch.Tensor) -> float:
    """Diagnostic probe: mean cosine distance between clean and nuisance reps.

    Independent of the predictor — measures how invariant the representation
    already is.  This is the load-bearing metric that separates our *mechanism*
    (features are invariant) from emergent-outcome baselines.
    """
    a = F.normalize(z_clean.float(), dim=-1)
    b = F.normalize(z_pert.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean().item()


class InvarianceModule:
    """Bundles the predictor, optional EMA target, and the variant table.

    Usage in the SFT loop (per micro-batch, clean inputs already on device)::

        nz_imgs, nz_instr = inv.make_views(cpu_images, instructions)
        z_clean = inv.encode_clean(policy, clean_images, clean_instr, states)
        z_pert  = policy.encode_prefix_pooled(nz_imgs, nz_instr, states)
        loss_inv = invariance_loss(z_pert, z_clean, inv.predictor)
        drift = feature_drift(z_clean, z_pert)
        ... total = sft_loss + cfg.lambda_inv * loss_inv ...
        inv.ema_step(policy)   # after optimizer.step()
    """

    def __init__(self, policy, cfg: InvarianceConfig) -> None:
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.variants = (
            load_instruction_variants(cfg.variants_path, cfg.variant_types) if cfg.apply_language else {}
        )
        dim = policy.prefix_dim
        # Keep the invariance head in fp32 for stable cosine alignment even when
        # the backbone runs under bf16 autocast (inputs are cast to fp32 in
        # ``invariance_loss``).
        self.predictor = (
            Predictor(dim, cfg.predictor_bottleneck).to(policy.device)
            if cfg.use_predictor
            else None
        )
        self.ema = EmaEncoder(policy.model) if cfg.target == "ema" else None

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.predictor.parameters()) if self.predictor is not None else []

    def make_views(self, cpu_images: torch.Tensor, instructions: list[str]) -> tuple[torch.Tensor, list[str]]:
        return build_nuisance_views(cpu_images, instructions, self.cfg, self.variants, self.gen)

    def encode_clean(self, policy, images, instructions, states) -> torch.Tensor:
        """Encode the clean/canonical view to the target representation (no grad)."""
        model = self.ema.model if self.ema is not None else policy.model
        with torch.no_grad():
            return policy.encode_prefix_pooled(images, instructions, states, model=model)

    def ema_step(self, policy) -> None:
        if self.ema is not None:
            self.ema.update(policy.model, self.cfg.ema_decay)
