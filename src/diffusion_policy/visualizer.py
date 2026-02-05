import gymnasium as gym
import mani_skill.envs
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def visualize(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    num_envs: int = typer.Option(1, "--num-envs", "-n", help="Number of parallel environments"),
    obs_mode: str = typer.Option("state", "--obs-mode", "-o", help="Observation mode"),
    control_mode: str = typer.Option("pd_joint_delta_pos", "--control-mode", "-c", help="Control mode"),
    render_mode: str = typer.Option("human", "--render-mode", "-r", help="Render mode (human/rgb_array)"),
    max_steps: int = typer.Option(float("inf"), "--max-steps", "-s", help="Maximum steps per episode"),
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
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
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
    print(f"Video saved!")


@app.command()
def list_envs() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    for env_id in sorted(REGISTERED_ENVS.keys()):
        print(env_id)


if __name__ == "__main__":
    app()
