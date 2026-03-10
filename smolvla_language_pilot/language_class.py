from __future__ import annotations

from dataclasses import dataclass
import re

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
	from tqdm.auto import tqdm
except Exception:
	tqdm = None

from lerobot.policies.factory import make_pre_post_processors
from vla.models.smolvla import smolvla


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
		"You may add neutral framing such as 'for this task' or 'your task is to'. "
		"Do not add new constraints or execution modifiers."
	),
	"context": (
		"Rewrite the instruction by adding neutral context phrases such as "
		"'in this scene', 'for this task', or 'in this situation'. "
		"Do not introduce new objects, locations, colors, or attributes."
	),
}


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

	parts = re.split(r"[.!?\n]", text)
	if parts:
		text = parts[0]

	while text.startswith(("-", "*", "•")):
		text = text[1:].strip()

	return text.strip()


def _is_valid_candidate(text: str, original_instruction: str, seen: set[str]) -> bool:
	if not text:
		return False
	if len(text.split()) < 3:
		return False
	norm = _normalize_instruction(text)
	if norm == _normalize_instruction(original_instruction):
		return False
	if norm in seen:
		return False
	return True


def _load_rollout(path: str) -> tuple[torch.Tensor, torch.Tensor, str]:
	with h5py.File(path, "r") as f:
		images = torch.tensor(f["observation/image"][:])
		states = torch.tensor(f["observation/state"][:])
		instruction = f["instruction"][()].decode("utf-8")
	return images, states, instruction


def load_policy_bundle(
	checkpoint: str = "lerobot/smolvla_base",
	device: str = "cuda",
) -> dict:
	policy_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
	policy, model_id, _ = smolvla(checkpoint, str(policy_device))
	policy.eval()

	preprocessor, postprocessor = make_pre_post_processors(
		policy.config,
		model_id,
		preprocessor_overrides={"device_processor": {"device": str(policy_device)}},
	)

	return {
		"policy": policy,
		"model_id": model_id,
		"preprocessor": preprocessor,
		"postprocessor": postprocessor,
		"device": policy_device,
		"model_dtype": next(policy.parameters()).dtype,
	}


def load_llm_bundle(
	llm_model: str = "Qwen/Qwen2.5-3B-Instruct",
) -> dict:
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
		max_new_tokens=20,
		temperature=0.8,
		top_p=0.9,
		do_sample=True,
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)

	generated = output[0][inputs["input_ids"].shape[-1] :]
	candidate = tokenizer.decode(generated, skip_special_tokens=True)
	return _cleanup_candidate(candidate)


def generate_llm_variant(instruction: str, rule: str, llm_bundle: dict) -> str:
	prompt = f"""Task: Rewrite a robot instruction.

Original instruction:
{instruction}

Rewrite rule:
{rule}

Constraints:
- Output exactly one sentence
- Do not add explanations
- Do not ask questions
- Do not continue the conversation

Rewritten instruction:"""

	return _generate_one(prompt, llm_bundle["tokenizer"], llm_bundle["llm"])


def generate_instruction_variants(
	instruction: str,
	llm_bundle: dict,
	controlled_variants: dict[str, str] | None = None,
	n_variants: int = 10,
	tries_per_variant: int = 5,
) -> tuple[list[str], list[str]]:
	rules = controlled_variants or DEFAULT_CONTROLLED_VARIANTS
	variants: list[str] = []
	labels: list[str] = []
	seen = {_normalize_instruction(instruction)}

	for variant_name, rule in rules.items():
		iterator = range(n_variants)
		if tqdm is not None:
			iterator = tqdm(iterator, desc=f"Generating {variant_name}", dynamic_ncols=True)

		for i in iterator:
			candidate = ""
			for _ in range(tries_per_variant):
				trial = generate_llm_variant(instruction, rule, llm_bundle)
				if _is_valid_candidate(trial, instruction, seen):
					candidate = trial
					break

			if not candidate:
				candidate = f"Please {instruction}"

			seen.add(_normalize_instruction(candidate))
			variants.append(candidate)
			labels.append(f"{variant_name}_{i + 1}")

	return variants, labels


def run_with_instruction(
	instruction: str,
	images: torch.Tensor,
	states: torch.Tensor,
	policy_bundle: dict,
	seed: int = 0,
) -> torch.Tensor:
	_set_seed(seed)

	policy = policy_bundle["policy"]
	preprocessor = policy_bundle["preprocessor"]
	postprocessor = policy_bundle["postprocessor"]
	device_obj = policy_bundle["device"]
	model_dtype = policy_bundle["model_dtype"]

	if hasattr(policy, "reset"):
		policy.reset()

	actions = []
	use_amp = device_obj.type == "cuda"

	with torch.inference_mode():
		for img, state in zip(images, states):
			batch = {
				"observation.state": state.unsqueeze(0).to(device_obj, dtype=model_dtype),
				"task": [instruction],
			}

			for key in policy.config.input_features:
				if key.startswith("observation.images."):
					batch[key] = img.unsqueeze(0).to(device_obj, dtype=model_dtype)

			batch = preprocessor(batch)

			with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
				action = policy.select_action(batch)

			action = postprocessor(action)
			actions.append(action.squeeze(0).cpu())

	return torch.stack(actions)


def run_language_sensitivity_for_rollout(
	rollout_path: str,
	policy_bundle: dict,
	llm_bundle: dict,
	controlled_variants: dict[str, str] | None = None,
	n_variants: int = 10,
	seed: int = 0,
) -> LanguageRunResult:
	images, states, base_instruction = _load_rollout(rollout_path)

	variants, labels = generate_instruction_variants(
		instruction=base_instruction,
		llm_bundle=llm_bundle,
		controlled_variants=controlled_variants,
		n_variants=n_variants,
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

