import gymnasium as gym
import mani_skill.envs  # noqa: F401 - Required to register ManiSkill environments
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def visualize(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    num_envs: int = typer.Option(1, "--num-envs", "-n", help="Number of parallel environments"),
    obs_mode: str = typer.Option("state", "--obs-mode", "-o", help="Observation mode"),
    control_mode: str = typer.Option("pd_joint_delta_pos", "--control-mode", "-c", help="Control mode"),
    render_mode: str = typer.Option("human", "--render-mode", "-r", help="Render mode (human/rgb_array)"),
    max_steps: int = typer.Option(1000000, "--max-steps", "-s", help="Maximum steps per episode"),
    shader: str = typer.Option("default", "--shader", help="Shader type (default/rt-fast/rt)"),
) -> None:
    print(f"Creating environment: {env_id}")
    print(f"  control_mode={control_mode}, obs_mode={obs_mode}, render_mode={render_mode}")

    kwargs = dict(
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )
    if shader != "default":
        kwargs["shader_dir"] = shader

    env = gym.make(env_id, **kwargs)
    print("Environment created successfully")

    obs, _ = env.reset(seed=0)
    print(f"Environment reset. Starting {max_steps} steps...")

    step = 0
    while step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1

        if terminated.any() if hasattr(terminated, "any") else terminated:
            print(f"Episode terminated at step {step}, resetting...")
            obs, _ = env.reset()

    print("Done!")
    env.close()


@app.command()
def record(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    output: str = typer.Option("output.mp4", "--output", "-o", help="Output video path"),
    max_steps: int = typer.Option(200, "--max-steps", "-s", help="Maximum steps"),
    control_mode: str = typer.Option("pd_joint_delta_pos", "--control-mode", "-c", help="Control mode"),
) -> None:
    from mani_skill.utils.wrappers import RecordEpisode

    print(f"Recording environment: {env_id} -> {output}")

    output_dir = Path(output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        render_mode="rgb_array",
    )
    env = RecordEpisode(env, output_dir=str(output_dir), save_video=True, video_fps=30)

    obs, _ = env.reset(seed=0)

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated.any() if hasattr(terminated, "any") else terminated:
            break

    env.close()
    print("Video saved!")


@app.command()
def clip_policy(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    control_mode: str = typer.Option("pd_joint_delta_pos", "--control-mode", "-c", help="Control mode"),
    render_mode: str = typer.Option("human", "--render-mode", "-r", help="Render mode (human/rgb_array)"),
    max_steps: int = typer.Option(1000, "--max-steps", "-s", help="Maximum steps per episode"),
    shader: str = typer.Option("default", "--shader", help="Shader type (default/rt-fast/rt)"),
    clip_model: str = typer.Option("ViT-B/32", "--clip-model", "-m", help="CLIP model variant"),
    task_description: str = typer.Option(
        "Insert the peg into the hole", "--task", "-t", help="Task description for CLIP"
    ),
) -> None:
    """Run environment with CLIP action model policy (untrained - for visualization only)."""
    import torch
    import numpy as np

    from diffusion_policy import create_clip_action_model

    print(f"Creating environment: {env_id}")
    print(f"  control_mode={control_mode}, render_mode={render_mode}")

    kwargs = dict(
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        render_mode=render_mode,
    )
    if shader != "default":
        kwargs["shader_dir"] = shader

    env = gym.make(env_id, **kwargs)
    action_dim = env.action_space.shape[-1]
    print(f"Environment created. Action dim: {action_dim}")

    print(f"Creating CLIP action model: {clip_model}")
    model = create_clip_action_model(
        action_dim=action_dim,
        clip_model=clip_model,
        use_history=False,
        freeze_clip=True,
    )
    model.eval()
    print(f"Model created on device: {model.device}")
    print(f"Task description: '{task_description}'")

    obs, _ = env.reset(seed=0)
    print(f"Environment reset. Running {max_steps} steps...")

    step = 0
    while step < max_steps:
        dummy_image = torch.randn(1, 3, 224, 224).to(model.device)
        with torch.no_grad():
            action = model(dummy_image, text=[task_description])
        action = action.cpu().numpy().squeeze()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1

        if step % 100 == 0:
            print(f"Step {step}, reward: {reward}")

        if terminated.any() if hasattr(terminated, "any") else terminated:
            print(f"Episode terminated at step {step}, resetting...")
            obs, _ = env.reset()

    print("Done!")
    env.close()


@app.command()
def list_envs() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    for env_id in sorted(REGISTERED_ENVS.keys()):
        print(env_id)


if __name__ == "__main__":
    app()
