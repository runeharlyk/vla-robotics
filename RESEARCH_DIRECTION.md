# Research Direction — Unified Latent-Invariance Backbone for a Robust VLA

**Status:** active, starting (master's continuation of the bachelor thesis). **Last updated:** 2026-06-29.
**Novelty:** deep-research verified NOVEL (mid-2026, 25/25 verification claims confirmed). See *Novelty & positioning* below.

---

## The idea (one line)

Make robustness come from the **backbone representation**, not from the action decoder compensating: add a single **JEPA-style / EMA-target representation-consistency loss** on the VLA's **fused vision+language conditioning features** (what the flow-matching action expert receives), so those features are invariant to **both** visual corruption **and** instruction paraphrase. **SFT-scale, no RL.**

## Why (from the bachelor findings)

The bachelor (`Towards Robust VLA Models`) diagnosed two brittleness modes in SmolVLA on the calibrated LIBERO protocol:
- **Visual:** ImageNet-C corruptions cause large action divergence and closed-loop collapse (Object 91% → ~0% at severity 5). Cause: SigLIP encoder trained on clean frames → corrupted frames are OOD for the action expert.
- **Language:** meaning-preserving rewording (verbosity, sentence-structure) drops success to ~0%; LIBERO uses only 0.16% of the token space, so surface form becomes a predictive feature.

Both are the *same disease*: the conditioning representation is not invariant to nuisance, so the action expert must compensate from scarce robot data. RL post-training gave only within-noise gains (saturated clean tasks). This project fixes the representation instead, supervised.

## Method sketch

For each training sample (same task/state), build two views:
- **Target view** = clean image + canonical instruction → **EMA target encoder** (frozen, slow-updated copy of the backbone) → `z_clean` (stop-grad).
- **Context view** = ImageNet-C-corrupted image + paraphrased instruction → online backbone → `z_pert`.

Loss: `L_total = L_SFT(CFM flow-matching) + λ · L_inv`, where `L_inv` pulls `z_pert → stop-grad(z_clean)` in latent space (cosine / smooth-L1, optional small predictor head, JEPA-style) on the **fused conditioning features**.

**Critical constraint (invariance ≠ information loss):** align **only nuisance-equivalent pairs** (same task, same state). Never align across tasks — that collapses the discriminative info the policy needs and tanks success. Corruptions stay photometric; paraphrases stay meaning-preserving.

## Novelty & positioning (verified)

Unique on **three load-bearing axes** — state all three explicitly:
1. **Mechanism** — JEPA/EMA representation-*consistency* (not decoder-output consistency, not info-bottleneck, not curriculum SFT, not temporal prediction).
2. **Locus** — the **fused post-backbone conditioning features** the action expert receives (not the projector, not the action output).
3. **Scope** — a **single objective over both** visual corruption *and* instruction paraphrase.

### Nearest neighbors / baselines to beat

| Paper | arXiv | What it does | How we differ |
|---|---|---|---|
| **VLA-JEPA** ⚠️ most dangerous | 2602.10098 | JEPA/EMA on a VLA; LIBERO-Plus gains incl. strong **language** numbers (85.4 vs 69.6) | Theirs = *temporal future-frame* prediction, **visual-only**, internet-scale **pretraining**; language robustness is *emergent*. Ours = *static nuisance-pair* invariance, **explicit cross-modal**, **SFT add-on**, explicit paraphrase objective. **Must benchmark head-to-head.** |
| RobustVLA (ICLR'26) | 2510.00037 | Robustness in the **action decoder** (output consistency + adversarial training) | The approach we argue against ("don't make the decoder compensate"). Foil. |
| StableVLA | 2605.18287 | Info-bottleneck adapter at the projector | **Visual-only**; explicitly *"no JEPA/EMA/contrastive"*. We add cross-modal + JEPA/EMA. |
| STRONG-VLA | 2604.10055 | Two-stage **curriculum SFT**, separate per-modality | Data/schedule-level, separate modalities. Ours = single fused-feature invariance loss. |
| LangGap | 2603.00592 | Language **data augmentation** (no new loss) | Complementary; "no novel loss introduced." |

**Problem confirmed:** LIBERO-Para (2603.28301) 22–52pp paraphrase drop; LIBERO-PRO (2510.03827) 19.8–51pp; LIBERO-Plus (2510.13626) "models largely ignore language."

**The invariance probe is load-bearing** — it separates our *mechanism* (features are invariant) from VLA-JEPA's *emergent outcome*.

## Course 1 (first special course) — pilot plan

**Hypothesis:** the invariance loss shrinks the clean→perturbed success gap on *both* axes more than augmentation alone, without hurting clean LIBERO.

**Arms (causal ladder):**
| Arm | Invariance on | Isolates |
|---|---|---|
| A | none (stock SmolVLA SFT) | baseline |
| A′ | none, trained on augmented data | "is it just augmentation?" |
| B | vision only | visual-axis contribution |
| C | language only | language-axis contribution |
| **D** | both (unified) | the full claim |

**Controls:** gate on clean-LIBERO parity before reading perturbed scores; held-out perturbations disjoint from training augmentations; ≥3 seeds (±3.5pp noise floor); depth ablation (projector-only vs +LoRA backbone, start light).

**Evaluation:** visual axis (LIBERO-Plus / ImageNet-C harness) + language axis (instruction-perturbation harness) — primary metric = reduction in clean→perturbed gap vs A and A′. Plus the fused-feature **invariance probe** (feature drift under held-out nuisance).

**Baselines:** stock SmolVLA SFT (A); augmentation-only SFT (A′, the real bar); VLA-JEPA / RobustVLA / StableVLA / STRONG-VLA on LIBERO-Plus.

**Compute:** single 24–48 GB GPU (DTU L40S/A100), hours–days/run, existing LIBERO data + on-the-fly augmentation; ~5 arms × 3 seeds ≈ 1–2 weeks. No rollouts.

**Go/no-go:** (Q1) does the invariance loss beat A′ on the perturbed gap? If only A′-level → objective adds nothing, report & pivot. (Q2) does clean parity hold? If D tanks clean → over-invariance; reduce λ / tighten pairs.

**Success:** D ties A on clean, significantly shrinks the gap on both axes beyond A′, and the probe shows lower fused-feature drift.

## Implementation status (2026-06-29)

Scaffold built and tested on branch `latent-invariance-backbone` (not yet committed):
- `src/vla/training/invariance.py` — config, nuisance-view builders (reuse `visual_diagnostic/noise.py` + `smolvla_language_pilot/instruction_variants.json`), EMA target encoder, SimSiam predictor, invariance loss (fp32 head, stop-grad), `feature_drift` probe, `InvarianceModule`.
- `src/vla/models/smolvla.py` — `prefix_dim` + `encode_prefix_pooled()` (masked mean-pool of `embed_prefix`).
- `src/vla/training/sft_smolvla.py` — `L_SFT + λ·L_inv` wired behind `invariance.enabled` (baseline untouched), EMA step, logging.
- `scripts/train_sft.py` — `--arm baseline|augment|vision|language|both`, `--inv-lambda`, `--inv-target ema|online`.
- Tests: `tests/test_invariance.py` (9 unit) + `tests/test_invariance_integration.py` (real-model: encode/EMA + one combined training step). **11/11 pass.**

**CRITICAL (found during HPC bring-up):** SmolVLA defaults to `train_expert_only=True`, which **freezes the whole VLM backbone**. The invariance loss on the fused prefix then can only train the predictor (inv_loss falls) but cannot make the representation invariant (drift stays flat). The objective therefore REQUIRES a trainable backbone — pass `--unfreeze-backbone` (sets train_expert_only/freeze_vision_encoder False). Validated on GPU: unfreeze → 557M trainable, invariance gradient reaches 199 VLM tensors. Risk: full unfreeze can forget pretrained features → gate on clean-LIBERO parity; if clean drops, switch to **LoRA on the VLM** (the lower-risk variant) or lower lr/epochs. Sweep uses lr 2.5e-5, 5 epochs to limit forgetting.

Run sweep: `bsub < jobs/inv_sweep_spatial_l40s.sh` (5-arm array, unfreeze, eval off, deterministic save-tags). Eval: `jobs/inv_eval_clean_spatial_l40s.sh` (clean) + `jobs/inv_eval_plus_spatial_l40s.sh` (LIBERO-Plus), both 5-arm arrays over checkpoints/sft/spatial_<arm>_seed42${SUFFIX}/last.

**v1 sweep finding (2026-06-30/07-02): PREDICTOR ABSORPTION.** All 5 arms trained (~8-10h each, curves in `results/training_curves/`). In arms B/C/D, `inv_loss` collapsed 0.70→0.002 within 2 epochs while raw representation `drift` ROSE (vision 0.042→0.078, language 0.009→0.022, both 0.042→0.079): the SimSiam-style predictor learned the nuisance→clean mapping itself, removing gradient pressure on the backbone — the representation never became invariant. The drift probe did its job (caught the false victory). Also: v1's probe compared online-nuisance vs EMA-clean, conflating EMA lag with nuisance sensitivity.
**v2 objective:** direct alignment `z_pert → stop-grad(z_clean)` with NO predictor (`--no-inv-predictor`, now the default). Collapse-safe because the SFT action loss anchors the representation and EMA targets are sample-specific. Probe fixed to online-vs-online, sampled every `probe_every` micro-batches.

**v3 (code review before relaunch) — three more fixes:**
1. **Wrong locus (critical):** `encode_prefix_pooled` pooled the VLM's *input* embeddings — language tokens were raw embedding-table lookups that never passed through the LLM, so the language arm had no fusion and no LLM gradient. Now pools the **contextual** prefix (full VLM-transformer pass, the same prefix whose KVs the action expert cross-attends to). GPU-verified: invariance gradient reaches 288 LLM-layer tensors (488 total).
2. **Variant coverage:** the paraphrase JSON covers only 5/10 Spatial tasks → half the language-arm data silently trained clean. Added deterministic templated fallback (politeness/verb-swap only — held-out types stay unseen).
3. **Unfair A′:** augment arm trained on 100% nuisance views; now a per-sample 50/50 clean/nuisance mix (`nuisance_prob`).
Baseline (arm 1) v1 checkpoint stays valid; v3 retrains arms 2–5: `bsub -J "inv_sweep[2-5]" < jobs/inv_sweep_spatial_l40s.sh` (SUFFIX defaults `_v3`). Eval: `SUFFIX="" …[1]` for baseline, `SUFFIX=_v3 …[2-5]`.
Known vendored quirk (harmless, single-GPU): `set_requires_grad`'s "freeze last VLM layers" list uses `text_model.model.layers.*` which doesn't match real names (`vlm.model.text_model.layers.*`) — those layers simply stay trainable in unfreeze mode.
Success criterion for v3 training: **drift falls** over steps.

## Repo integration notes

- Backbone / fused-feature attach point, SFT entry + CFM loss, eval harness, ImageNet-C + language perturbation generators, V-JEPA2 path: **to be filled in from the repo map** (the `vjepa2` world-model path and `libero_plus_*` configs already exist — reuse them).
- Training configs follow Hydra under `configs/train_srpo/experiment/`; eval under `configs/evaluate/experiment/`. New arms = new experiment configs.

## Key references

LIBERO-Plus `2510.13626` · LIBERO-PRO `2510.03827` · LIBERO-Para `2603.28301` · VLA-JEPA `2602.10098` · RobustVLA `2510.00037` · StableVLA `2605.18287` · STRONG-VLA `2604.10055` · LangGap `2603.00592` · SmolVLA `2506.01844` · SRPO `2511.15605`.
