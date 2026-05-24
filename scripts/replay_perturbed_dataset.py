"""Replay collected LIBERO actions under sim/image/text perturbations."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import typer

from vla.constants import resolve_libero_suite_name
from vla.rl.libero_rollout import _pack_obs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)
DEFAULT_LIBERO_CAMERAS = "agentview_image,robot0_eye_in_hand_image"


def _load_instruction_variants(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return {"*": [str(v) for v in loaded]}
    if isinstance(loaded, dict):
        return {str(k): [str(v) for v in values] for k, values in loaded.items() if isinstance(values, list)}
    raise typer.BadParameter("--instruction-variants must be a JSON list or object")


def _choose_instruction(base: str, variants: dict[str, list[str]], rng: np.random.Generator) -> str:
    choices = variants.get(base, []) or variants.get("*", [])
    if not choices:
        return base
    all_choices = [base, *choices]
    return all_choices[int(rng.integers(0, len(all_choices)))]


def _parse_task_id(episode: dict, metadata: dict) -> int:
    raw = str(episode.get("task_id", metadata.get("task_id", metadata.get("libero_task_id", "0"))))
    if raw and raw[-1].isdigit():
        try:
            return int(raw.split("_")[-1])
        except ValueError:
            pass
    return int(metadata.get("libero_task_id", 0))


def _flatten_replay_actions(episode: dict) -> torch.Tensor:
    if "action_chunks" in episode and "action_masks" in episode:
        chunks = episode["action_chunks"].float()
        masks = episode["action_masks"].bool()
        return chunks[masks]
    return episode["actions"].float()


def _robosuite_env(env: Any) -> Any | None:
    lerobot_env = getattr(env, "_env", None)
    offscreen = getattr(lerobot_env, "_env", None)
    return getattr(offscreen, "env", offscreen)


def _camera_names(raw_names: str) -> list[str]:
    names = []
    for name in raw_names.split(","):
        stripped = name.strip()
        if stripped.endswith("_image"):
            stripped = stripped[: -len("_image")]
        if stripped:
            names.append(stripped)
    return names


def _snapshot_cameras(env: Any, camera_names: list[str]) -> dict[str, tuple[np.ndarray, float]]:
    rs_env = _robosuite_env(env)
    sim = getattr(rs_env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return {}
    snapshot = {}
    for name in camera_names:
        try:
            cam_id = model.camera_name2id(name)
            snapshot[name] = (np.array(model.cam_pos[cam_id], dtype=np.float64).copy(), float(model.cam_fovy[cam_id]))
        except Exception:
            continue
    return snapshot


def _restore_cameras(env: Any, snapshot: dict[str, tuple[np.ndarray, float]]) -> None:
    rs_env = _robosuite_env(env)
    sim = getattr(rs_env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return
    for name, (pos, fovy) in snapshot.items():
        try:
            cam_id = model.camera_name2id(name)
            model.cam_pos[cam_id] = pos
            model.cam_fovy[cam_id] = fovy
        except Exception:
            continue
    with contextlib.suppress(Exception):
        sim.forward()


def _apply_camera_jitter(
    env: Any,
    camera_names: list[str],
    rng: np.random.Generator,
    pos_std: float,
    fovy_std: float,
) -> bool:
    if pos_std <= 0 and fovy_std <= 0:
        return False
    rs_env = _robosuite_env(env)
    sim = getattr(rs_env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return False
    changed = False
    for name in camera_names:
        try:
            cam_id = model.camera_name2id(name)
            if pos_std > 0:
                model.cam_pos[cam_id] = model.cam_pos[cam_id] + rng.normal(0.0, pos_std, size=3)
            if fovy_std > 0:
                model.cam_fovy[cam_id] = float(model.cam_fovy[cam_id]) * float(
                    np.exp(rng.normal(0.0, fovy_std))
                )
            changed = True
        except Exception:
            continue
    if changed:
        with contextlib.suppress(Exception):
            sim.forward()
    return changed


def _refresh_observation(env: Any, fallback: dict) -> dict:
    lerobot_env = getattr(env, "_env", None)
    offscreen = getattr(lerobot_env, "_env", None)
    rs_env = getattr(offscreen, "env", None)
    try:
        raw_obs = rs_env._get_observations()
        return lerobot_env._format_raw_obs(raw_obs)
    except Exception:
        return fallback


def _obs_to_tensors(raw_obs: dict, image_size: int, state_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    packed = _pack_obs(raw_obs, image_size=image_size, state_dim=state_dim)
    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in packed["images"]], dim=0)
    state = torch.from_numpy(packed["state"]).float()
    return images, state


def _postprocess_images(
    images: torch.Tensor,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
    noise_std: float,
    motion_blur_max: int,
) -> torch.Tensor:
    arr = images.float()
    arr = arr / 255.0 if arr.max().item() > 2.0 else arr
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        mean = arr.mean(dim=(-2, -1), keepdim=True)
        arr = (arr - mean) * factor + mean
    if brightness > 0:
        arr = arr + rng.uniform(-brightness, brightness)
    if noise_std > 0:
        noise = torch.from_numpy(rng.normal(0.0, noise_std, size=tuple(arr.shape))).to(arr.dtype)
        arr = arr + noise
    out = (arr.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    if motion_blur_max > 1:
        k = int(rng.integers(1, motion_blur_max + 1))
        if k % 2 == 0:
            k += 1
        if k > 1:
            blurred = []
            for frame in out:
                views = []
                for view in frame:
                    hwc = view.permute(1, 2, 0).numpy()
                    kernel = np.zeros((k, k), dtype=np.float32)
                    kernel[k // 2, :] = 1.0 / k
                    bhwc = cv2.filter2D(hwc, -1, kernel)
                    views.append(torch.from_numpy(bhwc).permute(2, 0, 1))
                blurred.append(torch.stack(views))
            out = torch.stack(blurred)
    return out


@app.command()
def main(
    data: Path = typer.Option(..., "--data", exists=True, file_okay=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output"),
    suite: str = typer.Option("spatial", "--suite"),
    variants: int = typer.Option(5, "--variants", min=1),
    seed: int = typer.Option(42, "--seed"),
    image_size: int = typer.Option(256, "--image-size", min=1),
    state_dim: int = typer.Option(8, "--state-dim", min=1),
    camera_names: str = typer.Option(DEFAULT_LIBERO_CAMERAS, "--camera-names"),
    camera_pos_std: float = typer.Option(0.01, "--camera-pos-std", min=0.0),
    camera_fovy_std: float = typer.Option(0.03, "--camera-fovy-std", min=0.0),
    brightness: float = typer.Option(0.08, "--brightness", min=0.0),
    contrast: float = typer.Option(0.12, "--contrast", min=0.0),
    noise_std: float = typer.Option(0.005, "--noise-std", min=0.0),
    motion_blur_max: int = typer.Option(5, "--motion-blur-max", min=1),
    instruction_variants: Path | None = typer.Option(
        None,
        "--instruction-variants",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    require_replay_success: bool = typer.Option(False, "--require-success/--keep-failures"),
) -> None:
    """Replay stored actions and record perturbed observations as a new SFT ``.pt`` dataset."""
    from vla.envs.libero import LiberoEnv

    payload = torch.load(data, map_location="cpu", weights_only=False)
    metadata = dict(payload["metadata"])
    episodes: list[dict] = payload["episodes"]
    suite_key = suite.lower()
    suite_name = resolve_libero_suite_name(suite_key)
    variants_by_instruction = _load_instruction_variants(instruction_variants)
    mujoco_camera_names = _camera_names(camera_names)

    out_episodes: list[dict] = []
    env: Any | None = None
    current_task_id: int | None = None
    camera_snapshot: dict[str, tuple[np.ndarray, float]] = {}
    try:
        for episode_idx, episode in enumerate(episodes):
            task_id = _parse_task_id(episode, metadata)
            if env is None or current_task_id != task_id:
                if env is not None:
                    env.close()
                env = LiberoEnv(suite_name=suite_name, task_id=task_id, state_dim=state_dim, camera_name=camera_names)
                current_task_id = task_id
                camera_snapshot = _snapshot_cameras(env, mujoco_camera_names)

            actions = _flatten_replay_actions(episode)
            base_seed = int(episode.get("reset_seed", seed + episode_idx))
            base_instruction = str(episode.get("instruction", metadata.get("instruction", env.task_description)))
            episode_init_state_id = episode.get("init_state_id")

            for variant_idx in range(variants):
                assert env is not None
                rng = np.random.default_rng(seed + episode_idx * 10_000 + variant_idx)
                if episode_init_state_id is not None:
                    try:
                        raw_obs, _ = env.reset(seed=base_seed, init_state_id=int(episode_init_state_id))
                    except TypeError:
                        raw_obs, _ = env.reset(seed=base_seed)
                else:
                    raw_obs, _ = env.reset(seed=base_seed)
                _restore_cameras(env, camera_snapshot)
                camera_changed = _apply_camera_jitter(
                    env,
                    mujoco_camera_names,
                    rng,
                    pos_std=camera_pos_std,
                    fovy_std=camera_fovy_std,
                )
                if camera_changed:
                    raw_obs = _refresh_observation(env, raw_obs)

                images: list[torch.Tensor] = []
                states: list[torch.Tensor] = []
                replayed_actions: list[torch.Tensor] = []
                success = False
                for action in actions:
                    img, state = _obs_to_tensors(raw_obs, image_size=image_size, state_dim=state_dim)
                    action_np = action.detach().cpu().numpy().astype(np.float32)
                    images.append(img)
                    states.append(state)
                    replayed_actions.append(torch.from_numpy(action_np.copy()))
                    raw_obs, _reward, terminated, truncated, info = env.step(action_np)
                    success = success or bool(info.get("is_success", False))
                    if terminated or truncated:
                        break

                if require_replay_success and not success:
                    continue
                if not replayed_actions:
                    continue

                ep_images = _postprocess_images(
                    torch.stack(images),
                    rng,
                    brightness=brightness,
                    contrast=contrast,
                    noise_std=noise_std,
                    motion_blur_max=motion_blur_max,
                )
                out_episode: dict[str, Any] = {
                    "images": ep_images,
                    "states": torch.stack(states).float(),
                    "actions": torch.stack(replayed_actions).float(),
                    "instruction": _choose_instruction(base_instruction, variants_by_instruction, rng),
                    "success": bool(success),
                    "task_id": episode.get("task_id", f"{suite_key}_task_{task_id}"),
                    "source_episode": episode_idx,
                    "variant": variant_idx,
                    "reset_seed": base_seed,
                }
                if episode_init_state_id is not None:
                    out_episode["init_state_id"] = int(episode_init_state_id)
                out_episodes.append(out_episode)
                logger.info(
                    "Replayed episode=%d variant=%d len=%d success=%s",
                    episode_idx,
                    variant_idx,
                    len(replayed_actions),
                    success,
                )
    finally:
        if env is not None:
            env.close()

    if not out_episodes:
        raise RuntimeError("No replayed episodes were written")

    out_metadata = {
        **metadata,
        "source_dataset": str(data),
        "num_episodes": len(out_episodes),
        "replay_perturbation": {
            "variants": variants,
            "camera_pos_std": camera_pos_std,
            "camera_fovy_std": camera_fovy_std,
            "brightness": brightness,
            "contrast": contrast,
            "noise_std": noise_std,
            "motion_blur_max": motion_blur_max,
            "instruction_variants": str(instruction_variants or ""),
            "seed": seed,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": out_metadata, "episodes": out_episodes}, output)
    logger.info("Saved %d replayed perturbed episodes to %s", len(out_episodes), output)


if __name__ == "__main__":
    app()
