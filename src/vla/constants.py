from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "videos"

ACTION_DIM = 7

SIMULATORS = ("libero", "maniskill")

LIBERO_SUITES = {
    "spatial": "lerobot/libero_spatial_image",
    "object": "lerobot/libero_object_image",
    "goal": "lerobot/libero_goal_image",
    "long": "lerobot/libero_10_image",
}

SUITE_MAP = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}

MANISKILL_TASKS: dict[str, dict] = {
    "PegInsertionSide-v1": {
        "instruction": "insert the peg into the hole",
        "action_dim": 8,
        "max_episode_steps": 100,
    },
    "PickCube-v1": {
        "instruction": "pick up the cube",
        "action_dim": 8,
        "max_episode_steps": 200,
    },
    "StackCube-v1": {
        "instruction": "stack the red cube on top of the green cube",
        "action_dim": 8,
        "max_episode_steps": 200,
    },
    "PushCube-v1": {
        "instruction": "push the cube to the target",
        "action_dim": 8,
        "max_episode_steps": 200,
    },
}


def resolve_suites(suite: str) -> list[str]:
    if suite.lower() == "all":
        return list(SUITE_MAP.keys())
    return [s.strip().lower() for s in suite.split(",")]
