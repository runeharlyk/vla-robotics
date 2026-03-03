from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def smolvla(checkpoint: str, device: str):

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
        print(f"Loading SmolVLA from local checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        config = ckpt["config"]
        model_id = config["model_id"]
        action_dim = config.get("action_dim", 7)
        image_size = config.get("image_size", 256)
        chunk_size = config.get("chunk_size", 50)

        policy = SmolVLAPolicy.from_pretrained(model_id)

        policy.config.input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
        }
        policy.config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        }
        policy.config.empty_cameras = 0
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = chunk_size

        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Loading SmolVLA from HuggingFace: {checkpoint}")
        model_id = checkpoint
        action_dim = 7
        policy = SmolVLAPolicy.from_pretrained(checkpoint)

    dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
    policy = policy.to(device_obj, dtype=dtype)

    return policy, model_id, action_dim
