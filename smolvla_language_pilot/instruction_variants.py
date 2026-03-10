from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import difflib

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_CONTROLLED_VARIANTS = {
    "politeness": (
        "Rewrite the instruction with polite phrasing while keeping the task unchanged. "
        "You may add 'please', 'kindly', 'could you', 'would you', 'if possible', or 'I'd like you to'. "
        "Do not change the action, object, location, or constraints. "
        "Do not add execution modifiers (e.g., carefully, slowly, gently, precisely)."
    ),
    "sentence_structure": (
        "Rewrite the instruction with a different sentence structure while keeping the task unchanged. "
        "You may reorder constituents or use an imperative with a clause. "
        "Do not change the action, object, location, or constraints."
    ),
    "verb_paraphrase": (
        "Rewrite the instruction using a different verb or verb phrase with the same meaning. "
        "Do not change the action target, object, location, or constraints."
    ),
    "verbosity": (
        "Rewrite the instruction with slightly more verbose wording while keeping the task unchanged. "
        "Begin the sentence with phrases like 'for this task', 'your task is', or 'the goal is'. "
        "Do not change the action, object, location, or constraints."
    ),
}


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_instruction(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _cleanup_candidate(text: str) -> str:

    text = text.strip().strip('"').strip("'")

    stop_phrases = [
        "Sure",
        "Here",
        "To adhere",
        "Following",
        "Task:",
        "Instruction:",
        "Rewrite:",
    ]

    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0]

    # cut after first sentence
    text = re.split(r"[.!?\n]", text)[0]

    # remove bullet markers
    text = re.sub(r"^[\-\*\•]+\s*", "", text)

    # collapse repeated segments
    words = text.split()
    if len(words) > 6 and words[:3] == words[3:6]:
        words = words[: len(words) // 2]

    return " ".join(words).strip()


def _preserves_core_words(text: str, original: str) -> bool:
    """Ensure core nouns remain unchanged."""
    text = text.lower()
    original = original.lower()

    core_words = ["drawer", "cabinet", "middle"]

    return all(word in text for word in core_words)


def _too_similar(text: str, seen: set[str]) -> bool:
    """Reject near duplicates."""
    for s in seen:
        if difflib.SequenceMatcher(None, text, s).ratio() > 0.9:
            return True
    return False


def _is_valid_candidate(text, original, seen, variant_type):

    if not text:
        return False

    text = text.strip()
    text_norm = _normalize_instruction(text)

    if text_norm == _normalize_instruction(original):
        return False

    if text_norm in seen:
        return False

    # avoid extremely long outputs
    if len(text.split()) > 18:
        return False

    lower = text.lower()

    # preserve required nouns
    required = ["drawer", "cabinet", "middle"]
    if not all(w in lower for w in required):
        return False

    # reject execution modifiers
    forbidden_modifiers = [
        "carefully",
        "gently",
        "slowly",
        "precisely",
        "softly"
    ]

    if any(m in lower for m in forbidden_modifiers):
        return False

    # reject malformed grammar
    if "would you could you" in lower:
        return False

    # enforce correct actions
    allowed_actions = [
        "open",
        "pull",
        "slide",
        "draw"
    ]

    if variant_type == "verb_paraphrase":
        if not any(lower.startswith(a) for a in allowed_actions):
            return False

    # politeness check
    if variant_type == "politeness":
        if not any(p in lower for p in ["please", "kindly", "could", "would"]):
            return False

    # verbosity must be longer
    if variant_type == "verbosity":
        if len(text.split()) <= len(original.split()):
            return False

    return True


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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    output = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    candidate = tokenizer.decode(generated, skip_special_tokens=True)
    return candidate

def generate_llm_variants_batch(instruction: str, rule: str, llm_bundle: dict, n: int) -> str:

    prompt = f"""
Rewrite the robot instruction in {n} different ways.

Original instruction:
{instruction}

Rewrite rule:
{rule}

Strict constraints:
- Preserve the object "drawer"
- Preserve the container "cabinet"
- Preserve the location "middle"
- Output exactly {n} sentences
- Each sentence must be different
- Do not explain anything
- Avoid repeating the phrase "Please open the middle drawer of the cabinet"

Return the result as a numbered list.

Example format:
1. ...
2. ...
3. ...
"""

    return _generate_one(prompt, llm_bundle["tokenizer"], llm_bundle["llm"])


def parse_numbered_list(text: str):

    variants = []

    for line in text.split("\n"):

        line = line.strip()

        match = re.match(r"^\d+[\.\)]\s*(.+)", line)

        if match:
            candidate = _cleanup_candidate(match.group(1))
            variants.append(candidate)

    return variants


def generate_llm_variant(instruction: str, rule: str, llm_bundle: dict) -> str:
    prompt = f"""Rewrite the robot instruction while keeping the meaning exactly the same.

Original instruction:
{instruction}

Rewrite rule:
{rule}

Strict constraints:
- Output exactly ONE sentence
- Keep the object "drawer"
- Keep the container "cabinet"
- Keep the location "middle"
- Do NOT replace cabinet with cupboard
- Do NOT replace drawer with compartment
- Do NOT repeat the sentence
- Do NOT add explanations

Rewritten instruction:"""

    return _generate_one(prompt, llm_bundle["tokenizer"], llm_bundle["llm"])


def generate_instruction_variants(
    instruction: str,
    llm_bundle: dict,
    controlled_variants: dict[str, str] | None = None,
    n_variants: int = 10,
    tries_per_variant: int = 5,
    seed: int = 0,
) -> tuple[list[str], list[str]]:

    rules = controlled_variants or DEFAULT_CONTROLLED_VARIANTS
    variants: list[str] = []
    labels: list[str] = []

    seen_per_type: dict[str, set[str]] = {
        name: {_normalize_instruction(instruction)}
        for name in rules
    }

    for type_index, (variant_name, rule) in enumerate(rules.items()):

        seen = seen_per_type[variant_name]

        if tqdm:
            print(f"\nGenerating {variant_name} variants")

        attempts = 0

        while len(seen) - 1 < n_variants and attempts < tries_per_variant:

            local_seed = seed + (type_index * 10000) + attempts
            _set_seed(local_seed)

            batch_output = generate_llm_variants_batch(
                instruction,
                rule,
                llm_bundle,
                n_variants
            )

            candidates = parse_numbered_list(batch_output)

            for candidate in candidates:

                if not _is_valid_candidate(candidate, instruction, seen, variant_name):
                    continue

                seen.add(_normalize_instruction(candidate))

                variants.append(candidate)
                labels.append(f"{variant_name}_{len(seen)-1}")

                if len(seen) - 1 == n_variants:
                    break

            attempts += 1

        print(f"{variant_name}: generated {len(seen)-1}/{n_variants}")

    return variants, labels


def _load_rollout_instruction(path: str) -> str:
    with h5py.File(path, "r") as f:
        return f["instruction"][()].decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic instruction variants and save to JSON for reuse."
    )
    parser.add_argument(
        "--rollout",
        dest="rollouts",
        action="append",
        required=True,
        help="Rollout path. Repeat this flag exactly once per rollout.",
    )
    parser.add_argument("--output-json", default="smolvla_language_pilot/instruction_variants.json")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _set_seed(args.seed)
    llm_bundle = load_llm_bundle(llm_model=args.llm_model)

    rollouts = [str(Path(r).as_posix()) for r in args.rollouts]
    result_rollouts: list[dict] = []

    for idx, rollout_path in enumerate(rollouts):
        instruction = _load_rollout_instruction(rollout_path)

        # Deterministic per rollout while keeping unique streams.
        variants, labels = generate_instruction_variants(
            instruction=instruction,
            llm_bundle=llm_bundle,
            controlled_variants=DEFAULT_CONTROLLED_VARIANTS,
            n_variants=args.n_variants,
            tries_per_variant=5,
            seed=args.seed + idx,
        )

        result_rollouts.append(
            {
                "rollout_path": rollout_path,
                "base_instruction": instruction,
                "labels": labels,
                "variants": variants,
            }
        )

        print(f"Generated {len(variants)} variants for {rollout_path}")

    payload = {
        "metadata": {
            "llm_model": args.llm_model,
            "seed": args.seed,
            "n_variants_per_type": args.n_variants,
            "variant_types": list(DEFAULT_CONTROLLED_VARIANTS.keys()),
        },
        "rollouts": result_rollouts,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved variants JSON: {output_path}")


if __name__ == "__main__":
    main()
