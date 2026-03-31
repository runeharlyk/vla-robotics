from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vla.utils.seed import seed_everything

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from dataclasses import dataclass
from typing import Dict, List


PROMPT_TEMPLATE = """You generate controlled instruction variations for robotic manipulation tasks.

Your job is to rewrite a base instruction into category-specific variants while preserving the same underlying action and goal.

Base instruction:
"{instruction}"

Generate variations for these categories:

1. politeness
Definition:
- Make the instruction more polite
- Keep wording as close as possible to the original
- Do not restructure unless absolutely necessary
- Do not add extra context
- Do not change the verb unless required for politeness

2. sentence_structure
Definition:
- Change the syntactic structure of the sentence
- Preserve the same meaning
- Do not add politeness markers
- Do not add unnecessary context
- Do not change the core verb unless needed by the structure

3. verb_paraphrase
Definition:
- Replace or minimally modify the action verb with an equivalent alternative
- Preserve the same meaning
- Keep sentence structure as close as possible to the original
- Do not add politeness markers
- Do not add unnecessary context

4. verbosity
Definition:
- Make the instruction longer with unnecessary but meaning-preserving wording
- Preserve the same action and goal
- Do not add politeness markers
- Do not substantially restructure the sentence
- Do not change the core verb unless unavoidable

5. original
Definition:
- Keep the original instruction unchanged

Global rules:
- The underlying action must remain exactly the same
- Variants across categories should overlap as little as possible
- Each variant should belong to only one category
- Keep edits minimal for the target category
- Do not introduce new objects, relations, or goals
- Do not change drawer position, object identity, or cabinet reference
- All outputs must be natural English
- Avoid duplicate variants
- Create exactly 3 variants for each non-original category
- Create exactly 1 variant for original

Output format:
Return valid JSON only.
Use this schema:

{{
  "instruction": "<base instruction>",
  "variants": {{
    "original": [
      "<original instruction>"
    ],
    "politeness": [
      "<variant 1>",
      "<variant 2>",
      "<variant 3>"
    ],
    "sentence_structure": [
      "<variant 1>",
      "<variant 2>",
      "<variant 3>"
    ],
    "verb_paraphrase": [
      "<variant 1>",
      "<variant 2>",
      "<variant 3>"
    ],
    "verbosity": [
      "<variant 1>",
      "<variant 2>",
      "<variant 3>"
    ]
  }}
}}
"""


@dataclass
class InstructionVariants:
    instruction: str
    variants: Dict[str, List[str]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "instruction": self.instruction,
            "variants": self.variants,
        }


def build_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction)


def extract_json(text: str) -> Dict[str, object]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start:end + 1])


def validate_output(data: Dict[str, object]) -> InstructionVariants:
    required_categories = {
        "original": 1,
        "politeness": 3,
        "sentence_structure": 3,
        "verb_paraphrase": 3,
        "verbosity": 3,
    }

    if "instruction" not in data or "variants" not in data:
        raise ValueError("Output must contain 'instruction' and 'variants'")

    instruction = data["instruction"]
    variants = data["variants"]

    if not isinstance(instruction, str):
        raise ValueError("'instruction' must be a string")

    if not isinstance(variants, dict):
        raise ValueError("'variants' must be a dictionary")

    for category, expected_count in required_categories.items():
        if category not in variants:
            raise ValueError(f"Missing category: {category}")
        if not isinstance(variants[category], list):
            raise ValueError(f"Category '{category}' must be a list")
        if len(variants[category]) != expected_count:
            raise ValueError(
                f"Category '{category}' must contain exactly {expected_count} items"
            )
        if not all(isinstance(item, str) and item.strip() for item in variants[category]):
            raise ValueError(f"Category '{category}' contains invalid entries")

    return InstructionVariants(instruction=instruction, variants=variants)


def generate_variants(instruction: str, llm_bundle: dict, max_retries: int = 3) -> InstructionVariants:
    prompt = build_prompt(instruction)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_output = _generate_one(prompt, llm_bundle["tokenizer"], llm_bundle["llm"])
            data = extract_json(raw_output)
            return validate_output(data)
        except (ValueError, KeyError) as exc:
            last_error = exc
            print(f"Attempt {attempt}/{max_retries} failed for '{instruction}': {exc}")
    raise RuntimeError(
        f"Failed to generate valid variants for '{instruction}' after {max_retries} attempts"
    ) from last_error


def print_variants(result: InstructionVariants) -> None:
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


def load_llm_bundle(llm_model: str = "Qwen/Qwen2.5-3B-Instruct") -> dict:
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    llm.eval()
    return {"tokenizer": tokenizer, "llm": llm, "model": llm_model}


def _generate_one(prompt: str, tokenizer, llm) -> str:
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
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    candidate = tokenizer.decode(generated, skip_special_tokens=True)
    return candidate


def _load_task_instructions(path: str) -> list[dict]:
    with h5py.File(path, "r") as f:
        # Combined LIBERO export format: demonstrations/* with task attrs.
        if "demonstrations" in f:
            demos = f["demonstrations"]
            tasks: dict[int, str] = {}
            for demo_key in demos.keys():
                demo = demos[demo_key]
                if "task_index" not in demo.attrs or "task" not in demo.attrs:
                    continue
                task_index = int(demo.attrs["task_index"])
                task_text = str(demo.attrs["task"])
                tasks.setdefault(task_index, task_text)

            ordered = []
            for task_index in sorted(tasks):
                ordered.append(
                    {
                        "task_index": task_index,
                        "base_instruction": tasks[task_index],
                    }
                )
            if ordered:
                return ordered

        # Single-rollout format fallback.
        if "instruction" in f:
            return [{"task_index": 0, "base_instruction": f["instruction"][()].decode("utf-8")}]

    raise ValueError(
        f"Could not find task instructions in {path}. Expected 'demonstrations/*' attrs or root 'instruction'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate instruction variants and save to JSON for reuse."
    )
    parser.add_argument(
        "--rollout",
        dest="rollouts",
        action="append",
        required=True,
        help="Rollout path. Repeat this flag exactly once per rollout.",
    )
    parser.add_argument("--output-json", default="smolvla_language_pilot/instruction_variants.json")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    llm_bundle = load_llm_bundle(llm_model=args.llm_model)

    rollouts = [str(Path(r).as_posix()) for r in args.rollouts]
    result_rollouts: list[dict] = []

    for rollout_path in rollouts:
        task_entries = _load_task_instructions(rollout_path)
        print(f"Found {len(task_entries)} task instructions in {rollout_path}")

        for i, task_entry in enumerate(task_entries):
            instruction = task_entry["base_instruction"]
            task_index = int(task_entry["task_index"])

            print(f"[{i + 1}/{len(task_entries)}] Generating variants for task_index={task_index}: '{instruction}' …", flush=True)
            result = generate_variants(instruction, llm_bundle)
            print(f"  Done — {sum(len(v) for v in result.variants.values())} variants generated.", flush=True)

            result_rollouts.append(
                {
                    "rollout_path": rollout_path,
                    "task_index": task_index,
                    "base_instruction": instruction,
                    "variants": result.variants,
                }
            )

            print(f"Generated variants for task_index={task_index}: {instruction}")

    payload = {
        "metadata": {
            "llm_model": args.llm_model,
            "seed": args.seed,
        },
        "rollouts": result_rollouts,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved variants JSON: {output_path}")


if __name__ == "__main__":
    main()
