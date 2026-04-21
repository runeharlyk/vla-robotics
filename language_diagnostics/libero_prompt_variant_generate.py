"""Generate LIBERO prompt variants and save a prompt-plan JSON.

This script only performs prompt generation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vla.envs import make_env_factory
from vla.utils import get_device, seed_everything

PROMPT_VARIANT_TYPES: tuple[str, ...] = (
    "original",
    "politeness",
    "verb_paraphrase",
    "sentence_structure",
    "verbosity",
    "carefully",
    "quickly",
    "contrastive_negation",
    "do_not_move",
)

LLM_PROMPT_TEMPLATE = """You generate one controlled instruction variant for each requested category.

Base instruction:
"{instruction}"

Return valid JSON only, with this exact schema:
{{
  "original": "...",
  "politeness": "...",
  "verb_paraphrase": "...",
  "sentence_structure": "...",
  "verbosity": "...",
  "carefully": "...",
  "quickly": "...",
  "contrastive_negation": "...",
  "do_not_move": "..."
}}

CRITICAL INSTRUCTION: Each variant value MUST be a COMPLETE, FULL-LENGTH instruction that matches the base instruction's level of detail.

Variant rules:
- original: output the base instruction exactly unchanged
- politeness: add "please" at the start, then output the complete rest of the instruction.
- verb_paraphrase: replace ONLY one main action verb (e.g., pick->grasp, put->place, push->move) but keep everything else identical. MUST have all same objects/locations.
- sentence_structure: rearrange word order but keep ALL words and meaning. Example: if base is "A B C D", output could be "D C A B" but must be complete.
- verbosity: insert "for this task" or similar phrase within the instruction, but keep the full base meaning. MUST include all original details.
- carefully: append " carefully" to the end of the complete instruction.
- quickly: append " quickly" to the end of the complete instruction.
- contrastive_negation: output the complete instruction but ADD a contrastive element like "not the red one, the blue one" while keeping core instruction.
- do_not_move: start with "do not move" but include the full object reference and goal from base. MUST be complete and executable.

Global rules:
- Every output MUST be a complete sentence that can be executed as a standalone instruction.
- NEVER output fragments, placeholders, or abbreviated forms.
- Each variant should be roughly the same length as the base (within 60-150% of base word count).
- Preserve all object names, spatial relationships, and goal details from the original.
- Output valid JSON only. No explanatory text outside JSON.
"""


@dataclass
class TaskSpec:
    suite: str
    task_id: int
    task_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", dest="suites", default=None)
    parser.add_argument("--tasks-per-suite", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument(
        "--prompt-plan-path",
        default="language_diagnostics/outputs/libero_prompt_variant_prompt_plan.json",
        help="Path to the saved prompt plan JSON.",
    )
    return parser.parse_args()


def _fmt_seconds(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _strip_leading_please(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("please "):
        return cleaned[7:].strip()
    return cleaned


def _replace_first_action_verb(base_instruction: str) -> str:
    lowered = base_instruction.lower()
    replacements = [
        ("pick up ", "grasp "),
        ("pick ", "grasp "),
        ("put ", "place "),
        ("push ", "move "),
        ("pull ", "drag "),
        ("insert ", "place "),
        ("plug ", "insert "),
    ]
    for src, dst in replacements:
        if lowered.startswith(src):
            return f"{dst}{base_instruction[len(src):]}"
    return f"perform {base_instruction}"


def _reorder_existing_words(base_instruction: str) -> str:
    words = base_instruction.split()
    if len(words) <= 3:
        return base_instruction
    offset = max(1, len(words) // 3)
    return " ".join(words[offset:] + words[:offset])


def _repair_variants(base_instruction: str, raw_variants: dict[str, str]) -> dict[str, str]:
    base = base_instruction.strip()
    repaired: dict[str, str] = {
        "original": base,
        "politeness": f"please {base}",
        "verb_paraphrase": _replace_first_action_verb(base),
        "sentence_structure": _reorder_existing_words(base),
        "verbosity": f"{base} for this task",
        "carefully": f"{base} carefully",
        "quickly": f"{base} quickly",
        "contrastive_negation": _strip_leading_please(raw_variants.get("contrastive_negation", base)),
        "do_not_move": _strip_leading_please(raw_variants.get("do_not_move", "do not move the target object in this scene")),
    }

    if "not" not in _normalize_text(repaired["contrastive_negation"]):
        repaired["contrastive_negation"] = f"{base}, not a different object"

    if not _normalize_text(repaired["do_not_move"]).startswith("do not move"):
        repaired["do_not_move"] = f"do not move the target object; {base}"

    return repaired


def _task_description_for(suite: str, task_id: int, state_dim: int) -> str:
    env_factory = make_env_factory("libero", suite=suite, state_dim=state_dim, task_id=task_id)
    env = env_factory(0)
    try:
        return env.task_description
    finally:
        env.close()


def _select_task_ids(suite: str, tasks_per_suite: int, state_dim: int) -> list[int]:
    env_factory = make_env_factory("libero", suite=suite, state_dim=state_dim)
    n = env_factory.num_tasks
    return list(range(min(tasks_per_suite, n)))


def _collect_task_specs(suites: list[str], tasks_per_suite: int, state_dim: int) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    for suite in suites:
        task_ids = _select_task_ids(suite=suite, tasks_per_suite=tasks_per_suite, state_dim=state_dim)
        print(f"Suite={suite}: selected task_ids={task_ids}")
        for task_id in task_ids:
            specs.append(
                TaskSpec(
                    suite=suite,
                    task_id=task_id,
                    task_description=_task_description_for(suite=suite, task_id=task_id, state_dim=state_dim),
                )
            )
    return specs


def load_llm_bundle(llm_model: str) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_device = get_device("cuda")
    dtype = torch.bfloat16 if llm_device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=dtype, device_map="auto")
    llm.eval()
    return {"tokenizer": tokenizer, "llm": llm, "model": llm_model}


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start : end + 1])


def _generate_one(prompt: str, tokenizer, llm, temperature: float) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    output = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _variant_generation_prompt(instruction: str) -> str:
    return LLM_PROMPT_TEMPLATE.format(instruction=instruction)


def _validate_variants(data: dict[str, object], base_instruction: str) -> dict[str, str]:
    missing = [key for key in PROMPT_VARIANT_TYPES if key not in data]
    if missing:
        raise ValueError(f"Missing variant keys: {missing}")

    variants: dict[str, str] = {}
    for key in PROMPT_VARIANT_TYPES:
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Variant '{key}' must be a non-empty string")
        variants[key] = value.strip()

    return _repair_variants(base_instruction, variants)


def _generate_task_variants(
    instruction: str,
    llm_bundle: dict[str, object],
    temperature: float,
    max_retries: int,
) -> dict[str, str]:
    prompt = _variant_generation_prompt(instruction)
    last_error: Exception | None = None
    tokenizer = llm_bundle["tokenizer"]
    llm = llm_bundle["llm"]

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = _generate_one(prompt, tokenizer, llm, temperature)
            data = _extract_json(raw_output)
            return _validate_variants(data, instruction)
        except (ValueError, KeyError) as exc:
            last_error = exc
            print(f"Attempt {attempt}/{max_retries} failed for '{instruction}': {exc}")

    raise RuntimeError(f"Failed to generate prompt variants for '{instruction}' after {max_retries} attempts") from last_error


def generate_and_save_prompts(
    task_specs: list[TaskSpec],
    llm_model: str,
    llm_temperature: float,
    llm_max_retries: int,
    prompt_plan_path: Path,
) -> dict[str, object]:
    llm_bundle = load_llm_bundle(llm_model)
    plan_tasks: list[dict[str, object]] = []

    for spec in task_specs:
        variants = _generate_task_variants(
            spec.task_description,
            llm_bundle=llm_bundle,
            temperature=llm_temperature,
            max_retries=llm_max_retries,
        )
        plan_tasks.append(
            {
                "suite": spec.suite,
                "task_id": spec.task_id,
                "task_description": spec.task_description,
                "prompts": variants,
            }
        )

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_model": llm_model,
        "variant_types": list(PROMPT_VARIANT_TYPES),
        "tasks": plan_tasks,
    }
    prompt_plan_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved prompt plan to: {prompt_plan_path}")
    return payload


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    seed_everything(args.seed)

    suites = args.suites or ["spatial", "object", "goal"]
    prompt_plan_path = Path(args.prompt_plan_path)

    task_specs = _collect_task_specs(suites=suites, tasks_per_suite=args.tasks_per_suite, state_dim=args.state_dim)
    generate_and_save_prompts(
        task_specs=task_specs,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_retries=args.llm_max_retries,
        prompt_plan_path=prompt_plan_path,
    )

    print(f"Total wall time: {_fmt_seconds(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
