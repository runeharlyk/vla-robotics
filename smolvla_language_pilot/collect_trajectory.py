from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import typer
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.policies.factory import make_pre_post_processors

from vla.constants import SUITE_MAP
from vla.evaluation.evaluate import _obs_to_batch
from vla.models.smolvla import smolvla

app = typer.Typer(no_args_is_help=True)


def _first_image_tensor(batch: dict) -> torch.Tensor:
	image_keys = sorted(k for k in batch if k.startswith("observation.images."))
	if not image_keys:
		raise RuntimeError("No image tensor found in batch under 'observation.images.*'.")

	img = batch[image_keys[0]]
	if img.ndim != 4:
		raise RuntimeError(f"Expected image tensor [B,C,H,W], got shape: {tuple(img.shape)}")
	return img[0].detach().to("cpu", dtype=torch.float32)


def _state_tensor(batch: dict) -> torch.Tensor:
	if "observation.state" not in batch:
		raise RuntimeError("No state tensor found in batch under 'observation.state'.")

	state = batch["observation.state"]
	if state.ndim != 2:
		raise RuntimeError(f"Expected state tensor [B,D], got shape: {tuple(state.shape)}")
	return state[0].detach().to("cpu", dtype=torch.float32)


@app.command()
def main(
	checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="SmolVLA checkpoint path or HF model id."),
	suite: str = typer.Option("libero_goal", "--suite", "-s", help="LIBERO suite name (e.g. libero_goal)."),
	task_id: int = typer.Option(0, "--task-id", "-t", help="Task id inside the suite."),
	seed: int = typer.Option(0, "--seed", help="Episode seed."),
	device: str = typer.Option("cuda", "--device", "-d", help="Device for policy inference."),
	max_steps: int = typer.Option(100, "--max-steps", "-n", min=1, max=150, help="Number of steps to record (1-150)."),
	output: Path = typer.Option(
		Path("smolvla_language_pilot/trajectory.h5"),
		"--output",
		"-o",
		help="Output H5 file path.",
	),
) -> None:
	"""Record image/state/instruction from a SmolVLA rollout for distillation."""

	device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
	policy, model_id, _action_dim = smolvla(checkpoint, str(device_obj))
	policy.eval()

	state_feature = policy.config.input_features.get("observation.state")
	state_dim = state_feature.shape[0] if state_feature else 8

	preprocessor, postprocessor = make_pre_post_processors(
		policy.config,
		model_id,
		preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
	)

	suite_name = SUITE_MAP.get(suite, suite)
	benchmark_suite = _get_suite(suite_name)
	if task_id < 0 or task_id >= len(benchmark_suite.tasks):
		raise typer.BadParameter(f"task-id must be in [0, {len(benchmark_suite.tasks) - 1}] for suite '{suite_name}'.")

	env = LiberoEnv(
		task_suite=benchmark_suite,
		task_id=task_id,
		task_suite_name=suite_name,
		obs_type="pixels_agent_pos",
	)

	try:
		obs_raw, _info = env.reset(seed=seed)
		instruction = env.task_description
		policy.reset()

		images: list[np.ndarray] = []
		states: list[np.ndarray] = []

		use_amp = device_obj.type == "cuda"
		for step_idx in range(max_steps):
			batch = _obs_to_batch(obs_raw, instruction, state_dim, device=device_obj)
			batch = preprocessor(batch)

			images.append(_first_image_tensor(batch).numpy())
			states.append(_state_tensor(batch).numpy())

			with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
				action = policy.select_action(batch)

			action = postprocessor(action)
			action_np = action.to("cpu").numpy()
			if action_np.ndim == 2:
				action_np = action_np[0]

			obs_raw, _reward, terminated, truncated, info = env.step(action_np)
			if terminated or truncated or info.get("is_success", False):
				print(f"Episode ended at step {step_idx + 1}.")
				break

		if not images:
			raise RuntimeError("No steps were recorded.")

		image_arr = np.stack(images, axis=0).astype(np.float32)
		state_arr = np.stack(states, axis=0).astype(np.float32)

		output.parent.mkdir(parents=True, exist_ok=True)
		with h5py.File(output, "w") as f:
			f.create_dataset("observation/image", data=image_arr)
			f.create_dataset("observation/state", data=state_arr)
			f.create_dataset("instruction", data=np.array(instruction, dtype=h5py.string_dtype("utf-8")))

			f.attrs["suite"] = suite_name
			f.attrs["task_id"] = task_id
			f.attrs["seed"] = seed
			f.attrs["steps"] = image_arr.shape[0]

		print(f"Saved {image_arr.shape[0]} steps to: {output}")
		print(f"image shape: {image_arr.shape}, state shape: {state_arr.shape}")
		print(f"instruction: {instruction}")
	finally:
		env.close()


if __name__ == "__main__":
	app()
