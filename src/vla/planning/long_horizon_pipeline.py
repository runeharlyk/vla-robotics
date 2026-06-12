from __future__ import annotations

import contextlib
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla.envs import make_env_factory
from vla.models import load_policy
from vla.models.smolvla import DEFAULT_CHECKPOINT
from vla.planning.smolvlm_decompose import (
    DecompositionResult,
    Subgoal,
    _build_model_inputs,
    _move_inputs_to_device,
    _normalize_result,
    _parse_model_response,
    resolve_vlm_model_id,
)

DEFAULT_THIRD_PERSON_CAMERA = "agentview_image"
DEFAULT_EXECUTOR_CAMERAS = "agentview_image,robot0_eye_in_hand_image"
DEFAULT_CONTROL_HZ = 20.0
YES_TOKEN_IDS = (10539, 9805)
NO_TOKEN_IDS = (5230, 787)


@dataclass(frozen=True)
class JudgeDecision:
    predicted_label: str
    yes_prob: float
    no_prob: float
    threshold: float
    solved: bool
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanEvent:
    step: int
    completed_before: list[str]
    task_summary: str
    subgoals: list[str]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeEvent:
    step: int
    active_subgoal: str
    completed_before: list[str]
    predicted_label: str
    yes_prob: float
    no_prob: float
    solved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LongHorizonResult:
    task_instruction: str
    success: bool
    terminated: bool
    truncated: bool
    steps: int
    solved_subgoals: list[str]
    remaining_subgoals: list[str]
    judge_interval_steps: int
    planner_camera: str
    judge_camera: str
    visual_token_estimate: int
    plan_events: list[PlanEvent] = field(default_factory=list)
    judge_events: list[JudgeEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_instruction": self.task_instruction,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "steps": self.steps,
            "solved_subgoals": self.solved_subgoals,
            "remaining_subgoals": self.remaining_subgoals,
            "judge_interval_steps": self.judge_interval_steps,
            "planner_camera": self.planner_camera,
            "judge_camera": self.judge_camera,
            "visual_token_estimate": self.visual_token_estimate,
            "plan_events": [event.to_dict() for event in self.plan_events],
            "judge_events": [event.to_dict() for event in self.judge_events],
        }


class Planner(Protocol):
    def plan(
        self,
        *,
        task_instruction: str,
        image: Image.Image | None,
        completed_subgoals: list[str],
        max_subgoals: int,
    ) -> DecompositionResult: ...


class Judge(Protocol):
    def judge(
        self,
        *,
        task_instruction: str,
        active_subgoal: str,
        image: Image.Image | None,
        completed_subgoals: list[str],
        threshold: float,
    ) -> JudgeDecision: ...


class Executor(Protocol):
    def reset(self) -> None: ...

    def act(self, *, env: Any, raw_obs: dict, instruction: str) -> np.ndarray: ...


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


def _subgoal_text(subgoal: Subgoal) -> str:
    return (subgoal.description or subgoal.title).strip()


def _filter_remaining_subgoals(subgoals: list[Subgoal], completed_subgoals: list[str]) -> list[Subgoal]:
    seen = {_normalize_text(item) for item in completed_subgoals}
    kept: list[Subgoal] = []
    for subgoal in subgoals:
        text = _subgoal_text(subgoal)
        norm = _normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        kept.append(subgoal)
    return kept


def extract_camera_image(
    raw_obs: dict,
    camera_name: str = DEFAULT_THIRD_PERSON_CAMERA,
    image_size: int = 512,
) -> Image.Image | None:
    pixels = raw_obs.get("pixels")
    if not isinstance(pixels, dict) or not pixels:
        return None
    frame = pixels.get(camera_name)
    if frame is None:
        first = next(iter(pixels.values()))
        frame = first
    flipped = np.flip(frame, axis=(0, 1)).copy()
    image = Image.fromarray(flipped)
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return image.convert("RGB")


def _estimate_visual_tokens(config: dict[str, Any]) -> int:
    vision = config.get("vision_config", {})
    image_size = int(vision.get("image_size") or 512)
    patch_size = int(vision.get("patch_size") or 16)
    side = max(1, image_size // patch_size)
    return side * side


class SmolVLMReasoner:
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_id: str | None = None,
        device: str = "cuda",
        offline_only: bool = True,
    ) -> None:
        resolved_model_id = resolve_vlm_model_id(checkpoint=checkpoint, model_id=model_id)
        resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32

        if offline_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.model_id = resolved_model_id
        self.device = resolved_device
        self.processor = AutoProcessor.from_pretrained(resolved_model_id, local_files_only=offline_only)
        self.model = AutoModelForImageTextToText.from_pretrained(
            resolved_model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=offline_only,
        )
        self.model.to(resolved_device)
        self.model.eval()
        self.visual_token_estimate = _estimate_visual_tokens(self.model.config.to_dict())

    def _prepare_inputs(self, prompt: str, image: Image.Image | None) -> dict[str, Any]:
        inputs = _build_model_inputs(self.processor, prompt, image)
        return _move_inputs_to_device(inputs, self.device)

    def generate_text(
        self,
        *,
        prompt: str,
        image: Image.Image | None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        inputs = self._prepare_inputs(prompt, image)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs.get("input_ids")
        generated_tokens = outputs[:, input_ids.shape[1] :] if torch.is_tensor(input_ids) else outputs
        return self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    def plan(
        self,
        *,
        task_instruction: str,
        image: Image.Image | None,
        completed_subgoals: list[str],
        max_subgoals: int,
    ) -> DecompositionResult:
        prompt_lines = [
            "Break this robot task into short numbered subgoals.",
            "Use the same object names as the task.",
            "Each subgoal should be a concrete action for a single-arm tabletop robot.",
        ]
        if completed_subgoals:
            prompt_lines.append(f"Already solved: {'; '.join(completed_subgoals)}.")
            prompt_lines.append("Only list the remaining subgoals.")
        prompt_lines.extend(
            [
                f"Return between 1 and {max_subgoals} subgoals.",
                f"Task: {task_instruction}",
                "Return only this format:",
                "Summary: <one sentence>",
                "1. <first subgoal>",
                "2. <second subgoal>",
                "3. <third subgoal>",
            ]
        )
        prompt = "\n".join(prompt_lines)
        raw_response = self.generate_text(prompt=prompt, image=image, max_new_tokens=120, temperature=0.0)
        payload = _parse_model_response(raw_response)
        result = _normalize_result(
            payload,
            instruction=task_instruction,
            model_id=self.model_id,
            raw_response=raw_response,
        )
        filtered = _filter_remaining_subgoals(result.subgoals, completed_subgoals)
        return DecompositionResult(
            instruction=result.instruction,
            model_id=result.model_id,
            task_summary=result.task_summary,
            subgoals=filtered,
            raw_response=result.raw_response,
        )

    def judge(
        self,
        *,
        task_instruction: str,
        active_subgoal: str,
        image: Image.Image | None,
        completed_subgoals: list[str],
        threshold: float,
    ) -> JudgeDecision:
        prompt_lines = [
            "Look at the third-person robot image.",
            f"Overall task: {task_instruction}",
        ]
        if completed_subgoals:
            prompt_lines.append(f"Already solved subgoals: {'; '.join(completed_subgoals)}")
        prompt_lines.extend(
            [
                f"Current subgoal: {active_subgoal}",
                "Has the current subgoal already been solved correctly?",
                "Answer with exactly Yes or No.",
            ]
        )
        prompt = "\n".join(prompt_lines)
        inputs = self._prepare_inputs(prompt, image)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)[0]
        yes_mass = float(probs[list(YES_TOKEN_IDS)].sum().item())
        no_mass = float(probs[list(NO_TOKEN_IDS)].sum().item())
        denom = max(1e-12, yes_mass + no_mass)
        yes_prob = yes_mass / denom
        no_prob = no_mass / denom
        predicted = "Yes" if yes_prob >= no_prob else "No"
        return JudgeDecision(
            predicted_label=predicted,
            yes_prob=yes_prob,
            no_prob=no_prob,
            threshold=threshold,
            solved=yes_prob >= threshold,
            prompt=prompt,
        )


class SmolVLAExecutor:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda") -> None:
        loaded = load_policy("smolvla", checkpoint, device)
        self.policy = loaded.policy
        self.policy.eval()
        self.preprocessor = loaded.preprocessor
        self.postprocessor = loaded.postprocessor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, *, env: Any, raw_obs: dict, instruction: str) -> np.ndarray:
        batch = env.obs_to_batch(raw_obs, device=self.device)
        batch["task"] = [instruction]
        batch = self.preprocessor(batch)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        action_np = action.detach().to("cpu").numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]
        return action_np


def run_orchestrated_episode(
    *,
    env: Any,
    task_instruction: str,
    executor: Executor,
    planner: Planner,
    judge: Judge,
    seed: int = 0,
    max_steps: int | None = None,
    max_subgoals: int = 5,
    judge_interval_seconds: float = 5.0,
    control_hz: float = DEFAULT_CONTROL_HZ,
    yes_threshold: float = 0.9,
    planner_camera: str = DEFAULT_THIRD_PERSON_CAMERA,
    judge_camera: str = DEFAULT_THIRD_PERSON_CAMERA,
    planner_image_size: int = 512,
    judge_image_size: int = 512,
    replan_on_completion: bool = True,
    visual_token_estimate: int = 1024,
) -> LongHorizonResult:
    raw_obs, _info = env.reset(seed=seed)
    executor.reset()

    judge_interval_steps = max(1, int(round(judge_interval_seconds * control_hz)))
    completed: list[str] = []
    plan_events: list[PlanEvent] = []
    judge_events: list[JudgeEvent] = []

    initial_image = extract_camera_image(raw_obs, planner_camera, planner_image_size)
    active_plan = planner.plan(
        task_instruction=task_instruction,
        image=initial_image,
        completed_subgoals=completed,
        max_subgoals=max_subgoals,
    )
    remaining = list(active_plan.subgoals)
    plan_events.append(
        PlanEvent(
            step=0,
            completed_before=[],
            task_summary=active_plan.task_summary,
            subgoals=[_subgoal_text(step) for step in active_plan.subgoals],
            raw_response=active_plan.raw_response,
        )
    )

    terminated = False
    truncated = False
    success = False
    step_count = 0

    max_episode_steps = max_steps or env.max_episode_steps
    for step_idx in range(max_episode_steps):
        step_count = step_idx + 1
        if not remaining:
            break

        active_subgoal = _subgoal_text(remaining[0])
        action = executor.act(env=env, raw_obs=raw_obs, instruction=active_subgoal)
        raw_obs, _reward, terminated, truncated, info = env.step(action)

        if env.is_success(info):
            success = True
            terminated = True
            completed.append(active_subgoal)
            remaining = remaining[1:]
            break

        if step_count % judge_interval_steps == 0:
            judge_image = extract_camera_image(raw_obs, judge_camera, judge_image_size)
            decision = judge.judge(
                task_instruction=task_instruction,
                active_subgoal=active_subgoal,
                image=judge_image,
                completed_subgoals=completed,
                threshold=yes_threshold,
            )
            judge_events.append(
                JudgeEvent(
                    step=step_count,
                    active_subgoal=active_subgoal,
                    completed_before=list(completed),
                    predicted_label=decision.predicted_label,
                    yes_prob=decision.yes_prob,
                    no_prob=decision.no_prob,
                    solved=decision.solved,
                )
            )
            if decision.solved:
                completed.append(active_subgoal)
                remaining = remaining[1:]
                if remaining and replan_on_completion:
                    replan_image = extract_camera_image(raw_obs, planner_camera, planner_image_size)
                    replanned = planner.plan(
                        task_instruction=task_instruction,
                        image=replan_image,
                        completed_subgoals=completed,
                        max_subgoals=max_subgoals,
                    )
                    remaining = list(replanned.subgoals)
                    plan_events.append(
                        PlanEvent(
                            step=step_count,
                            completed_before=list(completed),
                            task_summary=replanned.task_summary,
                            subgoals=[_subgoal_text(step) for step in replanned.subgoals],
                            raw_response=replanned.raw_response,
                        )
                    )

        if terminated or truncated:
            break

    return LongHorizonResult(
        task_instruction=task_instruction,
        success=success,
        terminated=terminated,
        truncated=truncated,
        steps=step_count,
        solved_subgoals=completed,
        remaining_subgoals=[_subgoal_text(subgoal) for subgoal in remaining],
        judge_interval_steps=judge_interval_steps,
        planner_camera=planner_camera,
        judge_camera=judge_camera,
        visual_token_estimate=visual_token_estimate,
        plan_events=plan_events,
        judge_events=judge_events,
    )


def run_libero_long_horizon_trial(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    suite: str = "long",
    task_id: int = 0,
    instruction: str = "",
    seed: int = 0,
    max_steps: int | None = None,
    max_subgoals: int = 5,
    judge_interval_seconds: float = 5.0,
    control_hz: float = DEFAULT_CONTROL_HZ,
    yes_threshold: float = 0.9,
    planner_camera: str = DEFAULT_THIRD_PERSON_CAMERA,
    judge_camera: str = DEFAULT_THIRD_PERSON_CAMERA,
    executor_cameras: str = DEFAULT_EXECUTOR_CAMERAS,
    offline_only: bool = True,
    replan_on_completion: bool = True,
) -> LongHorizonResult:
    planner_judge = SmolVLMReasoner(
        checkpoint=checkpoint,
        device=device,
        offline_only=offline_only,
    )
    executor = SmolVLAExecutor(checkpoint=checkpoint, device=device)

    env_factory = make_env_factory("libero", suite=suite, state_dim=8, task_id=task_id)
    env = env_factory(
        0,
        camera_name=executor_cameras,
    )
    resolved_instruction = instruction.strip() or env.task_description
    try:
        return run_orchestrated_episode(
            env=env,
            task_instruction=resolved_instruction,
            executor=executor,
            planner=planner_judge,
            judge=planner_judge,
            seed=seed,
            max_steps=max_steps,
            max_subgoals=max_subgoals,
            judge_interval_seconds=judge_interval_seconds,
            control_hz=control_hz,
            yes_threshold=yes_threshold,
            planner_camera=planner_camera,
            judge_camera=judge_camera,
            replan_on_completion=replan_on_completion,
            visual_token_estimate=planner_judge.visual_token_estimate,
        )
    finally:
        with contextlib.suppress(Exception):
            env.close()


def format_long_horizon_result(result: LongHorizonResult) -> str:
    lines = [
        f"Task: {result.task_instruction}",
        f"Success: {result.success}",
        f"Steps: {result.steps}",
        f"Third-person visual tokens: {result.visual_token_estimate}",
        f"Solved subgoals: {len(result.solved_subgoals)}",
    ]
    if result.solved_subgoals:
        lines.append("Completed:")
        for idx, subgoal in enumerate(result.solved_subgoals, start=1):
            lines.append(f"{idx}. {subgoal}")
    if result.remaining_subgoals:
        lines.append("Remaining:")
        for idx, subgoal in enumerate(result.remaining_subgoals, start=1):
            lines.append(f"{idx}. {subgoal}")
    if result.judge_events:
        last = result.judge_events[-1]
        lines.append(
            f"Last judge: step={last.step} label={last.predicted_label} yes={last.yes_prob:.3f} no={last.no_prob:.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONTROL_HZ",
    "DEFAULT_EXECUTOR_CAMERAS",
    "DEFAULT_THIRD_PERSON_CAMERA",
    "JudgeDecision",
    "JudgeEvent",
    "LongHorizonResult",
    "PlanEvent",
    "SmolVLAExecutor",
    "SmolVLMReasoner",
    "extract_camera_image",
    "format_long_horizon_result",
    "run_libero_long_horizon_trial",
    "run_orchestrated_episode",
]
