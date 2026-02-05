"""
Train RT-1 on ManiSkill demonstrations.

Usage:
    uv run python src/vla/train_rt1.py train --env PickCube-v1 --epochs 100
    uv run python src/vla/train_rt1.py evaluate --env PickCube-v1 --model models/rt1_pickcube.pt
"""
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MODELS_DIR = PROJECT_ROOT / "models"


class PreprocessedDataset(Dataset):
    """
    Dataset that loads preprocessed .pt files with images, states, and actions.
    
    Each preprocessed file contains episodes with:
        - images: (T, 3, H, W) tensor
        - states: (T, state_dim) tensor
        - actions: (T, action_dim) tensor
        - instruction: str
    """

    def __init__(
        self,
        data_path: Path,
        image_size: int = 256,
        sequence_length: int = 1,
        action_horizon: int = 1,
    ):
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.samples = []

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading preprocessed data from {data_path}")
        data = torch.load(data_path, weights_only=False)
        self.metadata = data["metadata"]
        episodes = data["episodes"]

        for ep in episodes:
            images = ep["images"]
            states = ep["states"]
            actions = ep["actions"]
            instruction = ep["instruction"]
            T = len(actions)

            for t in range(T - action_horizon + 1):
                start_idx = max(0, t - sequence_length + 1)
                img_seq = images[start_idx : t + 1]

                if img_seq.shape[0] < sequence_length:
                    pad_size = sequence_length - img_seq.shape[0]
                    padding = img_seq[0:1].repeat(pad_size, 1, 1, 1)
                    img_seq = torch.cat([padding, img_seq], dim=0)

                action_seq = actions[t : t + action_horizon]

                self.samples.append({
                    "images": img_seq,
                    "state": states[t],
                    "actions": action_seq,
                    "instruction": instruction,
                })

        print(f"Loaded {len(self.samples)} samples from {len(episodes)} episodes")
        print(f"  Action dim: {self.metadata['action_dim']}")
        print(f"  State dim: {self.metadata['state_dim']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        images = sample["images"].float() / 255.0
        return {
            "images": images,
            "state": sample["state"],
            "actions": sample["actions"],
            "instruction": sample["instruction"],
        }


class RawH5Dataset(Dataset):
    """
    Fallback dataset that loads directly from h5 files (uses dummy images).
    Use PreprocessedDataset when possible for real images.
    """

    def __init__(self, demo_path: Path, env_id: str, max_samples: Optional[int] = None):
        import h5py

        self.episodes = []
        demo_file = self._find_h5_file(demo_path, env_id)

        print(f"Loading demos from: {demo_file}")

        with h5py.File(demo_file, "r") as f:
            for ep_key in f.keys():
                if not ep_key.startswith("traj"):
                    continue
                ep = f[ep_key]
                actions = ep["actions"][:]

                obs = ep["obs"]
                if "agent" in obs and len(list(obs["agent"].keys())) > 0:
                    qpos = obs["agent"]["qpos"][:]
                    qvel = obs["agent"]["qvel"][:]
                    state = np.concatenate([qpos, qvel], axis=-1)
                elif "state" in obs:
                    state = obs["state"][:]
                else:
                    env_states = ep["env_states"]
                    articulations = env_states["articulations"]
                    robot_key = list(articulations.keys())[0]
                    state = articulations[robot_key][:]

                for i in range(len(actions)):
                    self.episodes.append({
                        "state": state[i],
                        "action": actions[i],
                    })

        if max_samples and len(self.episodes) > max_samples:
            indices = np.random.choice(len(self.episodes), max_samples, replace=False)
            self.episodes = [self.episodes[i] for i in indices]

        print(f"Loaded {len(self.episodes)} transitions")

    def _find_h5_file(self, demo_path: Path, env_id: str) -> Path:
        candidates = [
            demo_path / env_id / "motionplanning" / "trajectory.h5",
            demo_path / env_id / "trajectory.h5",
        ]
        for c in candidates:
            if c.exists():
                return c

        env_dir = demo_path / env_id
        if env_dir.exists():
            for subdir in env_dir.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "trajectory.h5"
                    if candidate.exists():
                        return candidate

        raise FileNotFoundError(f"Could not find trajectory.h5 in {demo_path / env_id}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        dummy_image = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        return {
            "images": dummy_image,
            "state": torch.tensor(ep["state"], dtype=torch.float32),
            "actions": torch.tensor(ep["action"], dtype=torch.float32).unsqueeze(0),
            "instruction": "pick up the cube",
        }


def create_rt1_model(
    action_dim: int = 8,
    device: str = "cuda",
    model_size: str = "small",
) -> nn.Module:
    """
    Create RT-1 model.
    
    Args:
        action_dim: Dimension of action space
        device: Device to place model on
        model_size: 'tiny', 'small', or 'base'
    """
    from robotic_transformer_pytorch import RT1, MaxViT

    configs = {
        "tiny": {
            "dim_conv_stem": 16,
            "dim": 32,
            "dim_head": 16,
            "depth": (1, 1, 1, 1),
            "rt1_depth": 2,
            "rt1_heads": 2,
            "rt1_dim_head": 16,
        },
        "small": {
            "dim_conv_stem": 32,
            "dim": 48,
            "dim_head": 16,
            "depth": (1, 1, 2, 1),
            "rt1_depth": 4,
            "rt1_heads": 4,
            "rt1_dim_head": 32,
        },
        "base": {
            "dim_conv_stem": 64,
            "dim": 96,
            "dim_head": 32,
            "depth": (2, 2, 5, 2),
            "rt1_depth": 6,
            "rt1_heads": 8,
            "rt1_dim_head": 64,
        },
    }

    cfg = configs.get(model_size, configs["small"])

    vit = MaxViT(
        num_classes=1000,
        dim_conv_stem=cfg["dim_conv_stem"],
        dim=cfg["dim"],
        dim_head=cfg["dim_head"],
        depth=cfg["depth"],
        window_size=8,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    model = RT1(
        vit=vit,
        num_actions=action_dim,
        action_bins=256,
        depth=cfg["rt1_depth"],
        heads=cfg["rt1_heads"],
        dim_head=cfg["rt1_dim_head"],
        cond_drop_prob=0.2,
    ).to(device)

    return model


def discretize_actions(actions: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    actions_clipped = torch.clamp(actions, -1, 1)
    bins = ((actions_clipped + 1) / 2 * (num_bins - 1)).long()
    return bins


def bins_to_continuous(bins: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    return (bins.float() / (num_bins - 1)) * 2 - 1


def get_skill_filename(env_id: str) -> str:
    return env_id.replace("-v1", "").replace("-", "_").lower() + ".pt"


@app.command()
def train(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(3e-5, "--lr", help="Learning rate"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    save_path: Optional[str] = typer.Option(None, "--save", "-s", help="Path to save model"),
    model_size: str = typer.Option("small", "--model-size", "-m", help="Model size: tiny/small/base"),
    eval_interval: int = typer.Option(10, "--eval-interval", help="Evaluate every N epochs"),
    num_eval_episodes: int = typer.Option(5, "--num-eval", help="Number of evaluation episodes"),
    sequence_length: int = typer.Option(1, "--seq-len", help="Number of frames per sample"),
    gradient_clip: float = typer.Option(1.0, "--grad-clip", help="Gradient clipping value"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use mixed precision training (faster)"),
) -> None:
    """Train RT-1 on preprocessed ManiSkill demonstrations."""
    skill_file = get_skill_filename(env_id)
    data_path = PREPROCESSED_DIR / skill_file

    if not data_path.exists():
        typer.echo(f"Preprocessed data not found at {data_path}")
        typer.echo("Run preprocessing first:")
        typer.echo(f"  uv run invoke preprocess-data --skill {env_id}")
        raise typer.Exit(1)

    dataset = PreprocessedDataset(
        data_path,
        sequence_length=sequence_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    action_dim = dataset.metadata["action_dim"]
    instruction = dataset.metadata["instruction"]

    print(f"\nCreating RT-1 model ({model_size})...")
    model = create_rt1_model(
        action_dim=action_dim,
        device=device,
        model_size=model_size,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)

    if save_path is None:
        save_path = str(MODELS_DIR / f"rt1_{env_id.lower().replace('-', '_')}.pt")
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining RT-1 on {env_id}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Action dim: {action_dim}")
    if device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Device: {device} ({gpu_name}, {gpu_memory:.1f} GB)")
        else:
            print(f"  Device: {device} (WARNING: CUDA not available, will fail!)")
    else:
        print(f"  Device: {device}")
    print(f"  Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Save path: {save_path}")

    best_loss = float("inf")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)

            B, T, C, H, W = images.shape
            video = images.permute(0, 2, 1, 3, 4)

            target_bins = discretize_actions(actions[:, 0, :])

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                texts = [instruction] * B
                logits = model(video, texts=texts)

                logits_flat = logits[:, 0, :, :].reshape(-1, 256)
                target_flat = target_bins.reshape(-1)

                loss = criterion(logits_flat, target_flat)

            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({"loss": "nan/inf (skipped)"})
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "config": {
                    "action_dim": action_dim,
                    "model_size": model_size,
                    "env_id": env_id,
                    "instruction": instruction,
                },
            }, save_path)
            print(f"  Saved best model (loss={avg_loss:.4f})")

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            print(f"\nRunning evaluation at epoch {epoch + 1}...")
            success_rate = run_evaluation(
                model,
                env_id,
                device,
                num_episodes=num_eval_episodes,
                instruction=instruction,
            )
            print(f"  Success rate: {success_rate * 100:.1f}%\n")
            model.train()

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")


def run_evaluation(
    model: nn.Module,
    env_id: str,
    device: str,
    num_episodes: int = 5,
    max_steps: int = 100,
    instruction: str = "pick up the cube",
) -> float:
    import gymnasium as gym

    import mani_skill.envs

    model.eval()

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
    )

    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_success = False

        for step in range(max_steps):
            rgb = extract_rgb_from_obs(obs, env)

            img_tensor = torch.from_numpy(rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            video = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            video = video.permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                logits = model(video, texts=[instruction])
                action_bins = logits.argmax(dim=-1)[0, 0]

            action = bins_to_continuous(action_bins).cpu().numpy()
            action = action.reshape(1, -1)

            obs, reward, terminated, truncated, info = env.step(action)

            success = info.get("success", False)
            if hasattr(success, "item"):
                success = success.item()
            if success:
                episode_success = True
                break

            done = terminated.any() if hasattr(terminated, "any") else terminated
            if done:
                break

        successes.append(episode_success)

    env.close()
    return np.mean(successes)


def extract_rgb_from_obs(obs: dict, env) -> np.ndarray:
    from PIL import Image

    sensor_data = obs.get("sensor_data", {})
    for cam_name in ["base_camera", "hand_camera", "sensor_camera", "3rd_view_camera"]:
        if cam_name in sensor_data:
            cam_data = sensor_data[cam_name]
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                img = Image.fromarray(rgb).resize((256, 256))
                return np.array(img)

    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    img = Image.fromarray(frame).resize((256, 256))
    return np.array(img)


@app.command()
def evaluate(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to trained model"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n", help="Number of test episodes"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
    render: bool = typer.Option(False, "--render/--no-render", help="Render environment"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    save_video: bool = typer.Option(False, "--save-video", help="Save video of episodes"),
    output_dir: str = typer.Option("outputs/rt1_eval", "--output-dir", "-o", help="Output directory"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed (lower = slower)"),
    loop: bool = typer.Option(False, "--loop", "-l", help="Keep running episodes indefinitely"),
) -> None:
    """Evaluate trained RT-1 model on ManiSkill environment."""
    import gymnasium as gym

    import mani_skill.envs

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        typer.echo(f"Model not found: {model_path}")
        raise typer.Exit(1)

    import sys

    print(f"Loading model from {model_path}")
    sys.stdout.flush()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Checkpoint loaded!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    config = checkpoint["config"]
    action_dim = config["action_dim"]
    model_size = config["model_size"]
    instruction = config["instruction"]
    print(f"Config: action_dim={action_dim}, model_size={model_size}")
    sys.stdout.flush()

    try:
        model = create_rt1_model(
            action_dim=action_dim,
            device=device,
            model_size=model_size,
        )
        print("Model created!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Weights loaded!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

    model.eval()

    print(f"Creating environment: {env_id}")
    sys.stdout.flush()
    try:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            render_backend="cpu",
        )
        print("Environment created successfully!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise typer.Exit(1)

    if render:
        import cv2
        cv2.namedWindow("RT1 Evaluation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RT1 Evaluation", 512, 512)

    if save_video:
        from mani_skill.utils.wrappers import RecordEpisode

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(output_path), save_video=True, video_fps=30)

    print(f"\nEvaluating RT-1 on {env_id}")
    print(f"  Episodes: {'infinite (Ctrl+C to stop)' if loop else num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Instruction: '{instruction}'")

    total_rewards = []
    successes = []
    ep = 0

    try:
        while True:
            print(f"Resetting environment (episode {ep + 1})...")
            obs, info = env.reset(seed=ep)
            print("Environment reset complete.")
            episode_reward = 0.0

            for step in range(max_steps):
                try:
                    frame = env.render()
                    if hasattr(frame, "cpu"):
                        frame = frame.cpu().numpy()
                    if frame.ndim == 4:
                        frame = frame[0]
                    
                    from PIL import Image as PILImage
                    rgb = np.array(PILImage.fromarray(frame).resize((256, 256)))

                    img_tensor = torch.from_numpy(rgb).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    video = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
                    video = video.permute(0, 2, 1, 3, 4)

                    with torch.no_grad():
                        logits = model(video, texts=[instruction])
                        action_bins = logits.argmax(dim=-1)[0, 0]

                    action = bins_to_continuous(action_bins).cpu().numpy()
                    action = action.reshape(1, -1)

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward.item() if hasattr(reward, "item") else reward)

                    if render:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("RT1 Evaluation", frame_bgr)
                        key = cv2.waitKey(int(20 / speed))
                        if key & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    done = terminated.any() if hasattr(terminated, "any") else terminated
                    if done:
                        break
                except Exception as e:
                    print(f"\nError at step {step}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise

            success = info.get("success", False)
            if hasattr(success, "item"):
                success = success.item()

            total_rewards.append(episode_reward)
            successes.append(success)
            ep += 1
            print(f"Episode {ep}: Reward={episode_reward:.2f}, Success={success}")

            if not loop and ep >= num_episodes:
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    env.close()
    if render:
        cv2.destroyAllWindows()

    if total_rewards:
        print("\n=== Evaluation Results ===")
        print(f"Episodes: {len(total_rewards)}")
        print(f"Mean Reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
        print(f"Success Rate: {np.mean(successes) * 100:.1f}%")

    if save_video:
        print(f"Videos saved to: {output_dir}")


@app.command()
def test_render(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed"),
    loop: bool = typer.Option(False, "--loop", "-l", help="Keep running episodes"),
) -> None:
    """Test environment rendering with random actions (no model needed)."""
    import sys
    import gymnasium as gym
    import mani_skill.envs

    print(f"Creating environment: {env_id}")
    sys.stdout.flush()

    try:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_pos",
            render_mode="human",
        )
        print("Environment created!")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    ep = 0
    try:
        while True:
            print(f"Episode {ep + 1}...")
            sys.stdout.flush()
            env.reset(seed=ep)

            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.02 / speed)

                done = terminated.any() if hasattr(terminated, "any") else terminated
                if done:
                    break

            ep += 1
            if not loop:
                break

    except KeyboardInterrupt:
        print("\nStopped.")

    env.close()
    print("Done!")


if __name__ == "__main__":
    app()
