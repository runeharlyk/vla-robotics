"""
Official RT-1 TensorFlow model wrapper.

Loads the official Google RT-1 checkpoints (TensorFlow SavedModel format).
These checkpoints can be downloaded from:
    gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .

Available checkpoints:
    - rt_1_x_tf_trained_for_002272480_step (RT-1-X)
    - rt_1_tf_trained_for_000400120 (RT-1-Converged) 
    - rt_1_tf_trained_for_000058240 (RT-1-15%)
    - rt_1_tf_trained_for_000001120 (RT-1-Begin)

Requirements:
    pip install tensorflow tensorflow-hub tf-agents transforms3d
"""
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


class RT1TFPolicy:
    """
    Wrapper for official Google RT-1 TensorFlow checkpoints.
    
    Args:
        checkpoint_path: Path to the TensorFlow SavedModel checkpoint directory
        image_width: Width to resize images to (default: 320)
        image_height: Height to resize images to (default: 256)
        action_scale: Scale factor for actions
        policy_setup: Robot configuration ("google_robot" or "widowx_bridge")
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ):
        self.checkpoint_path = checkpoint_path
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        
        self.model = None
        self.lang_embed_model = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None

    def load(self) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            from tf_agents.policies import py_tf_eager_policy
            from tf_agents import specs as tf_specs
        except ImportError:
            raise ImportError(
                "TensorFlow dependencies not installed. Install with:\n"
                "  pip install tensorflow tensorflow-hub tf-agents transforms3d"
            )

        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Download with:\n"
                "  gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .\n"
                "  unzip rt_1_x_tf_trained_for_002272480_step.zip"
            )

        print(f"Loading language embedding model...")
        self.lang_embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        print(f"Loading RT-1 policy from {checkpoint_path}...")
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=str(checkpoint_path),
            load_specs_from_pbtxt=True,
            use_tf_function=True,
        )

        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise ValueError(f"Unknown policy_setup: {self.policy_setup}")

        self._initialize_model()
        print("RT-1 TF model loaded successfully!")

    def _initialize_model(self) -> None:
        import tensorflow as tf
        import tf_agents
        from tf_agents.trajectories import time_step as ts

        self.observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation)
        )
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
        _ = self.tfa_policy.action(self.tfa_time_step, self.policy_state)

    def reset(self, task_description: str) -> None:
        import tensorflow as tf

        self._initialize_model()
        self.task_description = task_description
        self.task_description_embedding = self.lang_embed_model([task_description])[0]

    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray,
        low: float,
        high: float,
        safety_margin: float = 0.0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
        return np.clip(resc_actions, post_scaling_min + safety_margin, post_scaling_max - safety_margin)

    def _unnormalize_action_widowx_bridge(self, action: dict) -> dict:
        action["world_vector"] = self._rescale_action_with_bound(
            action["world_vector"], low=-1.75, high=1.75, post_scaling_max=0.05, post_scaling_min=-0.05
        )
        action["rotation_delta"] = self._rescale_action_with_bound(
            action["rotation_delta"], low=-1.4, high=1.4, post_scaling_max=0.25, post_scaling_min=-0.25
        )
        return action

    def _resize_image(self, image: np.ndarray) -> "tf.Tensor":
        import tensorflow as tf
        image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        image = tf.cast(image, tf.uint8)
        return image

    @staticmethod
    def _small_action_filter(raw_action: dict, arm_movement: bool = False, gripper: bool = True) -> dict:
        import tensorflow as tf

        if arm_movement:
            raw_action["world_vector"] = tf.where(
                tf.abs(raw_action["world_vector"]) < 5e-3,
                tf.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = tf.where(
                tf.abs(raw_action["rotation_delta"]) < 5e-3,
                tf.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = tf.where(
                tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                tf.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def predict_action(
        self,
        image: np.ndarray,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict robot action from image and language instruction.
        
        Args:
            image: RGB image (H, W, 3) as numpy array, uint8
            instruction: Natural language task instruction
        
        Returns:
            Action array with format depending on policy_setup:
            - For ManiSkill: (7,) array [x, y, z, rx, ry, rz, gripper]
        """
        import tensorflow as tf
        from tf_agents.trajectories import time_step as ts
        from transforms3d.euler import euler2axangle

        if self.tfa_policy is None:
            self.load()

        if instruction is not None and instruction != self.task_description:
            self.reset(instruction)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = self._resize_image(image)
        self.observation["image"] = image
        self.observation["natural_language_embedding"] = self.task_description_embedding

        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        policy_step = self.tfa_policy.action(self.tfa_time_step, self.policy_state)
        raw_action = policy_step.action

        if self.policy_setup == "google_robot":
            raw_action = self._small_action_filter(raw_action, arm_movement=False, gripper=True)
        if self.unnormalize_action:
            raw_action = self.unnormalize_action_fxn(raw_action)

        for k in raw_action.keys():
            raw_action[k] = np.asarray(raw_action[k])

        world_vector = np.asarray(raw_action["world_vector"], dtype=np.float64) * self.action_scale

        if self.action_rotation_mode == "axis_angle":
            rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            rotation_angle = np.linalg.norm(rotation_delta)
            rotation_ax = rotation_delta / rotation_angle if rotation_angle > 1e-6 else np.array([0.0, 1.0, 0.0])
            rot_axangle = rotation_ax * rotation_angle * self.action_scale
        elif self.action_rotation_mode in ["rpy", "ypr", "pry"]:
            if self.action_rotation_mode == "rpy":
                roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            elif self.action_rotation_mode == "ypr":
                yaw, pitch, roll = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            elif self.action_rotation_mode == "pry":
                pitch, roll, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            rotation_ax, rotation_angle = euler2axangle(roll, pitch, yaw)
            rot_axangle = rotation_ax * rotation_angle * self.action_scale
        else:
            raise NotImplementedError()

        raw_gripper = raw_action["gripper_closedness_action"]
        if self.invert_gripper_action:
            raw_gripper = -raw_gripper
        gripper = np.asarray(raw_gripper, dtype=np.float64)
        if self.policy_setup == "widowx_bridge":
            gripper = 2.0 * (gripper > 0.0) - 1.0

        self.policy_state = policy_step.state

        action = np.concatenate([world_vector.flatten(), rot_axangle.flatten(), gripper.flatten()])
        return action.astype(np.float32)

    def get_terminate_episode(self) -> bool:
        return False
