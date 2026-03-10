from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from smolvla_language_pilot.language_class import (
        LanguageRunResult,
        load_policy_bundle,
        run_language_sensitivity_for_rollout,
    )


@dataclass
class MultiTaskAggregateResult:
    task_results: list[LanguageRunResult]
    per_task_overall_mean_curves: torch.Tensor
    global_mean_curve: torch.Tensor
    global_std_curve: torch.Tensor
    per_task_lss: dict[str, list[float]]
    global_lss_mean: dict[str, float]


MANUAL_ROLLOUT_PATHS: list[str] = [
    "smolvla_language_pilot/rollout.h5",
    "",
    "",
    "",
    "",
]

VARIANTS_JSON_PATH = "smolvla_language_pilot/instruction_variants.json"


def _get_manual_rollouts(expected_count: int = 5) -> list[str]:
    rollouts = [p.strip() for p in MANUAL_ROLLOUT_PATHS if p.strip()]
    if len(rollouts) != expected_count:
        raise ValueError(
            f"Please set exactly {expected_count} non-empty entries in MANUAL_ROLLOUT_PATHS. "
            f"Current non-empty entries: {len(rollouts)}"
        )
    return rollouts


def run_multi_task_pipeline(
    checkpoint: str = "lerobot/smolvla_base",
    device: str = "cuda",
    variants_json_path: str = VARIANTS_JSON_PATH,
    seed: int = 0,
) -> MultiTaskAggregateResult:
    selected_paths = _get_manual_rollouts(expected_count=5)

    policy_bundle = load_policy_bundle(checkpoint=checkpoint, device=device)

    task_results = []
    for idx, rollout_path in enumerate(selected_paths):
        task_seed = seed + idx
        result = run_language_sensitivity_for_rollout(
            rollout_path=rollout_path,
            policy_bundle=policy_bundle,
            variants_json_path=variants_json_path,
            seed=task_seed,
        )
        task_results.append(result)

    per_task_overall_mean_curves = torch.stack([r.overall_mean_curve for r in task_results])
    global_mean_curve = per_task_overall_mean_curves.mean(dim=0)
    global_std_curve = per_task_overall_mean_curves.std(dim=0)

    per_task_lss: dict[str, list[float]] = {}
    for result in task_results:
        for variant_type, score in result.lss_scores.items():
            per_task_lss.setdefault(variant_type, []).append(score)

    global_lss_mean = {k: float(np.mean(v)) for k, v in per_task_lss.items()}

    return MultiTaskAggregateResult(
        task_results=task_results,
        per_task_overall_mean_curves=per_task_overall_mean_curves,
        global_mean_curve=global_mean_curve,
        global_std_curve=global_std_curve,
        per_task_lss=per_task_lss,
        global_lss_mean=global_lss_mean,
    )
