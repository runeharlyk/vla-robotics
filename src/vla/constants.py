import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# On HPC, VLA_WORK3 points to /work3/s234814/vla-robotics (fast scratch).
# Locally it falls back to the git checkout directory.
WORK_DIR = Path(os.environ.get("VLA_WORK3", str(PROJECT_ROOT)))

DATA_DIR = WORK_DIR / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINTS_DIR = WORK_DIR / "checkpoints"
MODELS_DIR = WORK_DIR / "models"
VIDEOS_DIR = WORK_DIR / "videos"
OUTPUTS_DIR = WORK_DIR / "outputs"

ACTION_DIM = 7


@dataclass
class WorldModelTypes:
    dinov2: str = "dinov2"
    vjepa2: str = "vjepa2"


@dataclass
class DistanceMetrics:
    normalized_l2: str = "normalized_l2"
    cosine: str = "cosine"
    l2: str = "l2"


@dataclass
class UpdateMethods:
    awr: str = "awr"
    ppo: str = "ppo"


@dataclass
class Mode:
    srpo: str = "srpo"
    sparse_rl: str = "sparse_rl"


@dataclass
class Simulators:
    libero: str = "libero"
    maniskill: str = "maniskill"


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
    # ── Two-camera tasks (base_camera + hand_camera) ─────────────────
    "PegInsertionSide-v1": {
        "instruction": "insert the peg into the hole",
        "action_dim": 8,
        "max_episode_steps": 100,
        "num_cameras": 2,
    },
    "StackCube-v1": {
        "instruction": "stack the red cube on top of the green cube",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
    },
    "AssemblingKits-v1": {
        "instruction": "assemble the kit by placing the objects in the correct poses",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
    },
    "PickSingleYCB-v1": {
        "instruction": "pick up the object",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
    },
    "PlugCharger-v1": {
        "instruction": "plug the charger into the receptacle",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
    },
    "StackPyramid-v1": {
        "instruction": "stack the cubes into a pyramid",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
    },
    # ── Articulated-object tasks (require PartNet Mobility assets) ──
    # Download with: python -m mani_skill.utils.download_asset <uid>
    "OpenCabinetDoor-v1": {  # Fetch robot - cameras: fetch_head, fetch_hand (sensor_data)
        "instruction": "open the cabinet door",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
        "requires_assets": True,
    },
    "OpenCabinetDrawer-v1": {  # Fetch robot - cameras: fetch_head, fetch_hand (sensor_data)
        "instruction": "open the cabinet drawer",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
        "requires_assets": True,
    },
    "TurnFaucet-v1": {  # Panda robot - cameras: base_camera, hand_camera (sensor_data)
        "instruction": "turn on the faucet",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 2,
        "requires_assets": True,
    },
    # ── One-camera tasks (base_camera only) ──────────────────────────
    "PickCube-v1": {
        "instruction": "pick up the cube",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 1,
    },
    "PushCube-v1": {
        "instruction": "push the cube to the target",
        "action_dim": 8,
        "max_episode_steps": 200,
        "num_cameras": 1,
    },
}


def resolve_suites(suite: str) -> list[str]:
    if suite.lower() == "all":
        return list(SUITE_MAP.keys())
    return [s.strip().lower() for s in suite.split(",")]
