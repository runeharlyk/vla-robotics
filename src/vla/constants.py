import os
from enum import StrEnum
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
RESULTS_DIR = WORK_DIR / "results"

ACTION_DIM = 7


class AdvantageMode(StrEnum):
    SRPO_ZSCORE = "srpo_zscore"
    LEAVE_ONE_OUT = "leave_one_out"


class UpdateMethod(StrEnum):
    AWR = "awr"
    FPO = "fpo"
    PPO = "ppo"


class WorldModelType(StrEnum):
    DINOV2 = "dinov2"
    VJEPA2 = "vjepa2"


class DistanceMetric(StrEnum):
    L2 = "l2"
    NORMALIZED_L2 = "normalized_l2"
    COSINE = "cosine"


class Mode(StrEnum):
    SRPO = "srpo"
    SPARSE_RL = "sparse_rl"


class Simulator(StrEnum):
    LIBERO = "libero"
    MANISKILL = "maniskill"


class LiberoSuite(StrEnum):
    SPATIAL = "spatial"
    OBJECT = "object"
    GOAL = "goal"
    LONG = "long"


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
