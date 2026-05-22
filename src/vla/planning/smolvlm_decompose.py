from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla.models.smolvla import DEFAULT_CHECKPOINT, SmolVLAPolicy


@dataclass(frozen=True)
class Subgoal:
    index: int
    title: str
    description: str
    success_signal: str


@dataclass(frozen=True)
class DecompositionResult:
    instruction: str
    model_id: str
    task_summary: str
    subgoals: list[Subgoal]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "model_id": self.model_id,
            "task_summary": self.task_summary,
            "subgoals": [asdict(step) for step in self.subgoals],
            "raw_response": self.raw_response,
        }


def resolve_vlm_model_id(checkpoint: str = DEFAULT_CHECKPOINT, model_id: str | None = None) -> str:
    if model_id:
        return model_id
    config = SmolVLAPolicy._load_ckpt_config(checkpoint)
    resolved = str(config.get("vlm_model_name", "")).strip()
    if not resolved:
        raise ValueError(f"Could not resolve 'vlm_model_name' from checkpoint {checkpoint!r}.")
    return resolved


def build_decomposition_prompt(
    instruction: str,
    max_subgoals: int = 6,
    benchmark_hint: str = "LIBERO Long",
    scene_context: str = "",
) -> str:
    prompt_lines = [
        f"Benchmark context: {benchmark_hint}.",
        f"Break this robot task into 2 to {max_subgoals} short numbered subgoals.",
        "Use the same object names as the task.",
        "Each subgoal should be a concrete action for a single-arm tabletop robot.",
    ]
    if scene_context.strip():
        prompt_lines.append(f"Scene context: {scene_context.strip()}")
    prompt_lines.extend(
        [
            f"Task: {instruction.strip()}",
            "Return only this format:",
            "Summary: <one sentence>",
            "1. <first subgoal>",
            "2. <second subgoal>",
            "3. <third subgoal>",
        ]
    )
    return "\n".join(prompt_lines)


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(candidate):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[idx:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not find a valid JSON object in model response: {text!r}")


def _extract_numbered_payload(text: str) -> dict[str, Any]:
    summary = ""
    subgoals: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        summary_match = re.match(r"^summary:\s*(.+)$", line, flags=re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            continue
        step_match = re.match(r"^\d+[\.\)]\s+(.+)$", line)
        if step_match:
            subgoals.append(step_match.group(1).strip())

    if not subgoals:
        raise ValueError(f"Could not find numbered subgoals in model response: {text!r}")

    return {
        "task_summary": summary,
        "subgoals": subgoals,
    }


def _extract_imperative_payload(text: str) -> dict[str, Any]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        raise ValueError("Empty model response.")

    starts = list(
        re.finditer(
            r"(?i)\b(put|pick|place|open|close|turn|move|stack|insert|take|push)\b",
            cleaned,
        )
    )
    if not starts:
        raise ValueError(f"Could not find imperative subgoals in model response: {text!r}")

    spans: list[str] = []
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(cleaned)
        phrase = cleaned[start:end].strip(" ,.;")
        if phrase:
            spans.append(phrase)

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in spans:
        norm = phrase.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(phrase)

    if not deduped:
        raise ValueError(f"Could not derive imperative subgoals from model response: {text!r}")

    return {
        "task_summary": deduped[0] if len(deduped) == 1 else cleaned,
        "subgoals": deduped,
    }


def _parse_model_response(text: str) -> dict[str, Any]:
    try:
        return _extract_json_payload(text)
    except ValueError:
        try:
            return _extract_numbered_payload(text)
        except ValueError:
            return _extract_imperative_payload(text)


def _normalize_subgoals(raw_subgoals: Any) -> list[Subgoal]:
    if not isinstance(raw_subgoals, list) or not raw_subgoals:
        raise ValueError("Expected a non-empty 'subgoals' list in model output.")

    normalized: list[Subgoal] = []
    for idx, item in enumerate(raw_subgoals, start=1):
        if isinstance(item, str):
            title = item.strip()
            description = title
            success_signal = ""
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or item.get("subgoal") or f"step {idx}").strip()
            description = str(item.get("description") or item.get("action") or title).strip()
            success_signal = str(
                item.get("success_signal") or item.get("success") or item.get("done_when") or ""
            ).strip()
        else:
            raise ValueError(f"Unsupported subgoal type at index {idx}: {type(item).__name__}")

        if not title:
            title = f"step {idx}"
        if not description:
            description = title

        normalized.append(
            Subgoal(
                index=idx,
                title=title,
                description=description,
                success_signal=success_signal,
            )
        )
    return normalized


def _normalize_result(
    payload: dict[str, Any],
    instruction: str,
    model_id: str,
    raw_response: str,
) -> DecompositionResult:
    task_summary = str(payload.get("task_summary") or payload.get("summary") or instruction).strip()
    if task_summary.lower() in {"<one sentence>", "one sentence"}:
        task_summary = instruction.strip()
    subgoals = _normalize_subgoals(payload.get("subgoals"))
    return DecompositionResult(
        instruction=instruction,
        model_id=model_id,
        task_summary=task_summary,
        subgoals=subgoals,
        raw_response=raw_response,
    )


def _move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    return Image.open(image_path).convert("RGB")


def _build_model_inputs(
    processor: Any,
    prompt: str,
    image: Image.Image | None,
) -> dict[str, Any]:
    if hasattr(processor, "apply_chat_template"):
        content = [{"type": "text", "text": prompt}]
        images: list[Image.Image] | None = None
        if image is not None:
            content.insert(0, {"type": "image"})
            images = [image]
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        if images is not None:
            return processor(text=chat_prompt, images=images, return_tensors="pt")
        return processor(text=chat_prompt, return_tensors="pt")
    return processor(text=prompt, return_tensors="pt")


def decompose_instruction(
    instruction: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_id: str | None = None,
    image_path: str | None = None,
    scene_context: str = "",
    device: str = "cuda",
    max_subgoals: int = 6,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    benchmark_hint: str = "LIBERO Long",
    offline_only: bool = True,
) -> DecompositionResult:
    if not instruction.strip():
        raise ValueError("instruction must be non-empty")

    resolved_model_id = resolve_vlm_model_id(checkpoint=checkpoint, model_id=model_id)
    resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32

    if offline_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    processor = AutoProcessor.from_pretrained(resolved_model_id, local_files_only=offline_only)
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=offline_only,
    )
    model.to(resolved_device)
    model.eval()

    prompt = build_decomposition_prompt(
        instruction=instruction,
        max_subgoals=max_subgoals,
        benchmark_hint=benchmark_hint,
        scene_context=scene_context,
    )
    image = _load_image(image_path)
    inputs = _build_model_inputs(processor, prompt, image)
    inputs = _move_inputs_to_device(inputs, resolved_device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    input_ids = inputs.get("input_ids")
    generated_tokens = outputs[:, input_ids.shape[1] :] if torch.is_tensor(input_ids) else outputs
    raw_response = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    payload = _parse_model_response(raw_response)
    return _normalize_result(payload, instruction=instruction, model_id=resolved_model_id, raw_response=raw_response)


def format_decomposition(result: DecompositionResult) -> str:
    lines = [
        f"Instruction: {result.instruction}",
        f"Backbone: {result.model_id}",
        f"Summary: {result.task_summary}",
        "",
        "Subgoals:",
    ]
    for step in result.subgoals:
        lines.append(f"{step.index}. {step.title}")
        lines.append(f"   Action: {step.description}")
        if step.success_signal:
            lines.append(f"   Done when: {step.success_signal}")
    return "\n".join(lines)


__all__ = [
    "DecompositionResult",
    "Subgoal",
    "build_decomposition_prompt",
    "decompose_instruction",
    "format_decomposition",
    "resolve_vlm_model_id",
]
