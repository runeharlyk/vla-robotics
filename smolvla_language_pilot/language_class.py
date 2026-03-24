from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from vla.models.smolvla import SmolVLAPolicy


@dataclass
class LanguageRunResult:
    rollout_path: str
    base_instruction: str
    labels: list[str]
    variants: list[str]
    divergence_curves: torch.Tensor
    mean_curve: torch.Tensor
    std_curve: torch.Tensor
    variant_type_means: dict[str, torch.Tensor]
    variant_type_stds: dict[str, torch.Tensor]
    boxplot_data: dict[str, list[float]]
    lss_scores: dict[str, float]
    overall_mean_curve: torch.Tensor
    peak_timestep: int
    peak_frame: np.ndarray


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rollout(path: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    with h5py.File(path, "r") as f:
        images = torch.tensor(f["observation/image"][:])
        states = torch.tensor(f["observation/state"][:])
        instruction = f["instruction"][()].decode("utf-8")
    return images, states, instruction


def load_policy_bundle(
    checkpoint: str = "HuggingFaceVLA/smolvla_libero",
    device: str = "cuda",
) -> dict:
    policy_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    policy = SmolVLAPolicy(checkpoint, action_dim=7, device=str(policy_device))
    policy.eval()

    return {
        "policy": policy,
        "device": policy_device,
        "model_dtype": policy.dtype,
    }


def run_with_instruction(
    instruction: str,
    images: torch.Tensor,
    states: torch.Tensor,
    policy_bundle: dict,
    seed: int = 0,
) -> torch.Tensor:
    _set_seed(seed)

    policy = policy_bundle["policy"]
    device_obj = policy_bundle["device"]

    if hasattr(policy, "reset"):
        policy.reset()

    actions = []

    with torch.inference_mode():
        for img, state in zip(images, states):
            action = policy.predict_action(img, instruction, state)
            actions.append(action.cpu())

    return torch.stack(actions)


def _normalize_path_for_match(path: str) -> str:
    return str(Path(path).expanduser().resolve()).lower()


def load_variants_for_rollout(
    rollout_path: str,
    base_instruction: str,
    variants_json_path: str,
) -> tuple[list[str], list[str]]:
    json_path = Path(variants_json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Variants JSON not found: {variants_json_path}. "
            "Run instruction_variants.py first."
        )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rollouts = payload.get("rollouts", [])
    if not rollouts:
        raise ValueError(f"No rollout entries found in variants JSON: {variants_json_path}")

    target_norm = _normalize_path_for_match(rollout_path)
    base_norm = " ".join(base_instruction.lower().strip().split())

    for entry in rollouts:
        entry_path = entry.get("rollout_path", "")
        entry_instruction = entry.get("base_instruction", "")
        labels = entry.get("labels", [])
        variants = entry.get("variants", [])

        path_match = False
        if entry_path:
            try:
                path_match = _normalize_path_for_match(entry_path) == target_norm
            except Exception:
                path_match = entry_path == rollout_path

        instruction_match = " ".join(entry_instruction.lower().strip().split()) == base_norm

        if path_match or instruction_match:
            if not isinstance(labels, list) or not isinstance(variants, list):
                raise ValueError("Malformed rollout entry in variants JSON: labels/variants must be lists.")
            if len(labels) != len(variants):
                raise ValueError("Malformed rollout entry in variants JSON: labels and variants length mismatch.")
            if len(labels) == 0:
                raise ValueError("Malformed rollout entry in variants JSON: no variants provided.")
            return labels, variants

    raise ValueError(
        f"No matching rollout found in variants JSON for rollout_path={rollout_path}. "
        "Generate/update the JSON with this rollout first."
    )


def run_language_sensitivity_for_rollout(
    rollout_path: str,
    policy_bundle: dict,
    variants_json_path: str,
    seed: int = 0,
) -> LanguageRunResult:
    images, states, base_instruction = _load_rollout(rollout_path)

    labels, variants = load_variants_for_rollout(
        rollout_path=rollout_path,
        base_instruction=base_instruction,
        variants_json_path=variants_json_path,
    )

    base_actions = run_with_instruction(
        instruction=base_instruction,
        images=images,
        states=states,
        policy_bundle=policy_bundle,
        seed=seed,
    )

    motion_scale = torch.norm(base_actions, dim=-1)
    eps = 1e-8

    divergence_curves = []
    for variant in variants:
        actions_v = run_with_instruction(
            instruction=variant,
            images=images,
            states=states,
            policy_bundle=policy_bundle,
            seed=seed,
        )
        abs_l2 = torch.norm(actions_v - base_actions, dim=-1)
        rel_l2 = abs_l2 / (motion_scale + eps)
        divergence_curves.append(rel_l2)

    divergence_curves = torch.stack(divergence_curves)
    mean_curve = divergence_curves.mean(dim=0)
    std_curve = divergence_curves.std(dim=0)

    variant_type_means: dict[str, torch.Tensor] = {}
    variant_type_stds: dict[str, torch.Tensor] = {}
    grouped: dict[str, list[torch.Tensor]] = {}

    for label, curve in zip(labels, divergence_curves):
        variant_type = label.split("_")[0]
        grouped.setdefault(variant_type, []).append(curve)

    for variant_type, curves in grouped.items():
        stacked = torch.stack(curves)
        variant_type_means[variant_type] = stacked.mean(dim=0)
        variant_type_stds[variant_type] = stacked.std(dim=0)

    trajectory_means = divergence_curves.mean(dim=1).cpu().numpy()
    boxplot_data: dict[str, list[float]] = {}
    for label, score in zip(labels, trajectory_means):
        variant_type = label.split("_")[0]
        boxplot_data.setdefault(variant_type, []).append(float(score))

    lss_scores = {k: float(np.mean(v)) for k, v in boxplot_data.items()}

    all_type_curves = torch.stack(list(variant_type_means.values()))
    overall_mean_curve = all_type_curves.mean(dim=0)
    peak_timestep = int(overall_mean_curve.argmax().item())
    peak_frame = images[peak_timestep].permute(1, 2, 0).cpu().numpy()

    return LanguageRunResult(
        rollout_path=rollout_path,
        base_instruction=base_instruction,
        labels=labels,
        variants=variants,
        divergence_curves=divergence_curves,
        mean_curve=mean_curve,
        std_curve=std_curve,
        variant_type_means=variant_type_means,
        variant_type_stds=variant_type_stds,
        boxplot_data=boxplot_data,
        lss_scores=lss_scores,
        overall_mean_curve=overall_mean_curve,
        peak_timestep=peak_timestep,
        peak_frame=peak_frame,
    )
