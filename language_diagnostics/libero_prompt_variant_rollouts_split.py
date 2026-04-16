"""Two-stage LIBERO prompt-variant evaluation pipeline.

Stage 1:
    Generate prompt variants per task and save them to a prompt plan file.

Stage 2:
    Load that prompt plan file and run rollouts for each prompt variant.

This keeps prompt generation reproducible and decoupled from rollout execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vla.envs import make_env_factory
from vla.models.smolvla import SmolVLAPolicy
from vla.utils import get_device, seed_everything
from vla.utils.tensor import action_to_numpy

DEFAULT_SFT_CHECKPOINT = "HuggingFaceVLA/smolvla_libero"

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
- politeness: add "please" at the start, then output the complete rest of the instruction. MUST be 80%+ as long as base.
- verb_paraphrase: replace ONLY one main action verb (e.g., pick→grasp, put→place, push→move) but keep everything else identical. MUST have all same objects/locations.
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
class PromptSpec:
    variant_type: str
    variant_index: int
    prompt: str


@dataclass
class TaskSpec:
    suite: str
    task_id: int
    task_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "generate", "run"], default="all")
    parser.add_argument("--suite", action="append", dest="suites", default=None)
    parser.add_argument("--tasks-per-suite", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--output-dir", default="language_diagnostics/outputs")
    parser.add_argument(
        "--prompt-plan-path",
        default="language_diagnostics/outputs/libero_prompt_variant_prompt_plan.json",
        help="Path to the saved prompt plan JSON used between generate and run stages.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--step-progress-every", type=int, default=0)
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
        "do_not_move": _strip_leading_please(raw_variants.get("do_not_move", f"do not move the target object in this scene")),
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


def load_prompt_plan(prompt_plan_path: Path) -> dict[str, object]:
    return json.loads(prompt_plan_path.read_text(encoding="utf-8"))


def _episode_seeds_for_task(global_seed: int, suite: str, task_id: int, num_episodes: int) -> list[int]:
    suite_offset = {"spatial": 0, "object": 100_000, "goal": 200_000, "long": 300_000}.get(suite, 400_000)
    base_seed = global_seed + suite_offset + task_id * 1_000
    return [base_seed + episode_index for episode_index in range(num_episodes)]


def _evaluate_prompt(
    policy: SmolVLAPolicy,
    suite: str,
    task_id: int,
    prompt: str,
    episode_seeds: list[int],
    max_steps: int,
    progress_every: int,
    step_progress_every: int,
    marker_label: str,
) -> list[dict[str, object]]:
    env_factory = make_env_factory("libero", suite=suite, state_dim=policy.state_dim, task_id=task_id)

    def _policy_fn(batch: dict) -> torch.Tensor:
        image_keys = sorted(k for k in batch if k.startswith("observation.images."))
        if not image_keys:
            raise ValueError(f"No image keys in batch. Keys: {list(batch.keys())}")
        cam_views: list[torch.Tensor] = []
        for key in image_keys:
            img = batch[key]
            if img.ndim in (4, 5):
                img = img[0]
            if img.ndim == 2:
                img = img.unsqueeze(0)
            cam_views.append(img)
        image = torch.stack(cam_views, dim=0) if len(cam_views) > 1 else cam_views[0]
        state = batch.get("observation.state")
        if state is not None and state.ndim == 2:
            state = state[0]
        return policy.predict_action(image, prompt, state)

    env = env_factory(0)
    try:
        capped_max_steps = min(max_steps, env.max_episode_steps)
        t0 = time.perf_counter()
        rollout_rows: list[dict[str, object]] = []

        for rollout_index, episode_seed in enumerate(episode_seeds):
            raw_obs, _info = env.reset(seed=episode_seed)
            ep_len = 0
            success = False
            ep_t0 = time.perf_counter()

            for step_idx in range(capped_max_steps):
                batch = env.obs_to_batch(raw_obs, device=policy.device)
                action_np = action_to_numpy(_policy_fn(batch))
                raw_obs, reward, terminated, truncated, info = env.step(action_np)
                _ = reward
                ep_len += 1

                step_done = step_idx + 1
                if step_progress_every > 0 and (
                    step_done == 1
                    or step_done == capped_max_steps
                    or step_done % step_progress_every == 0
                ):
                    ep_elapsed = time.perf_counter() - ep_t0
                    print(
                        f"        step-progress [{marker_label}] rollout={rollout_index + 1}/{len(episode_seeds)}, "
                        f"step={step_done}/{capped_max_steps}, elapsed={_fmt_seconds(ep_elapsed)}"
                    )

                if env.is_success(info):
                    success = True
                    break
                if terminated or truncated:
                    break

            rollout_rows.append(
                {
                    "rollout_index": rollout_index,
                    "episode_seed": episode_seed,
                    "success": success,
                    "episode_length": ep_len,
                }
            )

            done = rollout_index + 1
            if done == 1 or done == len(episode_seeds) or (progress_every > 0 and done % progress_every == 0):
                elapsed = time.perf_counter() - t0
                print(f"      progress [{marker_label}] {done}/{len(episode_seeds)} rollouts, elapsed={_fmt_seconds(elapsed)}")

        return rollout_rows
    finally:
        env.close()


def run_rollouts_from_prompt_file(
    policy: SmolVLAPolicy,
    prompt_payload: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []

    tasks = cast(list[dict[str, object]], prompt_payload.get("tasks", []))
    total_prompt_runs = len(tasks) * len(PROMPT_VARIANT_TYPES)
    prompt_runs_done = 0

    for task in tasks:
        suite = cast(str, task["suite"])
        task_id = cast(int, task["task_id"])
        task_description = cast(str, task["task_description"])
        prompts = cast(dict[str, str], task["prompts"])

        episode_seeds = _episode_seeds_for_task(args.seed, suite, task_id, args.episodes)
        prompt_specs = [PromptSpec(variant_type=k, variant_index=0, prompt=prompts[k]) for k in PROMPT_VARIANT_TYPES]

        print(f"  Task {task_id} ({suite}): evaluating {len(prompt_specs)} prompt(s)")

        for variant_type, prompt in prompts.items():
            prompt_rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_description": task_description,
                    "variant_type": variant_type,
                    "prompt": prompt,
                }
            )

        for spec in prompt_specs:
            print(f"    [{spec.variant_type}:{spec.variant_index}] {spec.prompt}")
            prompt_start = time.perf_counter()
            marker_label = f"{suite}/task{task_id}/{spec.variant_type}"
            rollout_rows = _evaluate_prompt(
                policy=policy,
                suite=suite,
                task_id=task_id,
                prompt=spec.prompt,
                episode_seeds=episode_seeds,
                max_steps=args.max_steps,
                progress_every=args.progress_every,
                step_progress_every=args.step_progress_every,
                marker_label=marker_label,
            )

            for rollout in rollout_rows:
                rows.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode_seed_base": episode_seeds[0] if episode_seeds else None,
                        "variant_type": spec.variant_type,
                        "variant_index": spec.variant_index,
                        "prompt": spec.prompt,
                        "rollout_index": rollout["rollout_index"],
                        "episode_seed": rollout["episode_seed"],
                        "success": rollout["success"],
                        "episode_length": rollout["episode_length"],
                    }
                )

            prompt_runs_done += 1
            successes = sum(1 for r in rollout_rows if bool(r["success"]))
            lengths = [cast(int, r["episode_length"]) for r in rollout_rows]
            mean_len = sum(lengths) / max(len(lengths), 1)
            prompt_elapsed = time.perf_counter() - prompt_start
            print(
                f"    done [{marker_label}] prompt_run={prompt_runs_done}/{total_prompt_runs}, "
                f"success={successes}/{len(rollout_rows)}, mean_len={mean_len:.1f}, elapsed={_fmt_seconds(prompt_elapsed)}"
            )

    return rows, prompt_rows


def _save_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    seed_everything(args.seed)
    suites = args.suites or ["spatial", "object", "goal"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_plan_path = Path(args.prompt_plan_path)

    if args.stage in ("all", "generate"):
        task_specs = _collect_task_specs(suites=suites, tasks_per_suite=args.tasks_per_suite, state_dim=args.state_dim)
        generate_and_save_prompts(
            task_specs=task_specs,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            llm_max_retries=args.llm_max_retries,
            prompt_plan_path=prompt_plan_path,
        )

    if args.stage in ("all", "run"):
        if not prompt_plan_path.exists():
            raise FileNotFoundError(f"Prompt plan file not found: {prompt_plan_path}")

        prompt_payload = load_prompt_plan(prompt_plan_path)
        device = get_device(args.device)
        policy = SmolVLAPolicy(
            checkpoint=DEFAULT_SFT_CHECKPOINT,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            device=str(device),
        )
        policy.eval()

        rows, prompt_rows = run_rollouts_from_prompt_file(policy=policy, prompt_payload=prompt_payload, args=args)

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "variant_types": list(PROMPT_VARIANT_TYPES),
            "prompt_plan_path": str(prompt_plan_path),
            "num_rollout_rows": len(rows),
            "num_prompt_rows": len(prompt_rows),
        }

        rows_path = output_dir / "libero_prompt_variant_rollouts_raw.csv"
        prompts_csv_path = output_dir / "libero_prompt_variant_prompts.csv"
        summary_path = output_dir / "libero_prompt_variant_rollouts_summary.json"
        _save_rows_csv(rows_path, rows)
        _save_rows_csv(prompts_csv_path, prompt_rows)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Saved raw results to: {rows_path}")
        print(f"Saved prompts to: {prompts_csv_path}")
        print(f"Saved summary to: {summary_path}")

    print(f"Total wall time: {_fmt_seconds(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
