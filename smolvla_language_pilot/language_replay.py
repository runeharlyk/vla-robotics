import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from lerobot.policies.factory import make_pre_post_processors
from vla.models.smolvla import smolvla

CHECKPOINT = "lerobot/smolvla_base"
DEVICE = "cuda"

# ----------------------------
# Load rollout
# ----------------------------
with h5py.File("smolvla_language_pilot/rollout.h5", "r") as f:
    images = torch.tensor(f["observation/image"][:])
    states = torch.tensor(f["observation/state"][:])
    base_instruction = f["instruction"][()].decode("utf-8")

print("Loaded rollout")
print("Images:", images.shape)
print("States:", states.shape)
print("Instruction:", base_instruction)

device_obj = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load policy
# ----------------------------
policy, model_id, _ = smolvla(CHECKPOINT, str(device_obj))
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
)

state_dim = policy.config.input_features["observation.state"].shape[0]
model_dtype = next(policy.parameters()).dtype

# ----------------------------
# Replay function
# ----------------------------
def run_with_instruction(instruction):

    # Force deterministic sampling
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    if hasattr(policy, "reset"):
        policy.reset()

    actions = []
    use_amp = device_obj.type == "cuda"

    with torch.inference_mode():
        for img, state in zip(images, states):

            batch = {
                "observation.state": state.unsqueeze(0).to(device_obj, dtype=model_dtype),
                "task": [instruction],
            }

            # Fill all required image keys dynamically
            for key in policy.config.input_features:
                if key.startswith("observation.images."):
                    batch[key] = img.unsqueeze(0).to(device_obj, dtype=model_dtype)

            batch = preprocessor(batch)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                action = policy.select_action(batch)
            action = postprocessor(action)

            actions.append(action.squeeze(0).cpu())

    return torch.stack(actions)


# ----------------------------
# Language variants
# ----------------------------
variants = [
    "please " + base_instruction,
    base_instruction + " carefully",
    base_instruction.replace("open", "pull open"),
    "do not " + base_instruction,
]

# ----------------------------
# Compute divergences
# ----------------------------
base_actions = run_with_instruction(base_instruction)
motion_scale = torch.norm(base_actions, dim=-1)
eps = 1e-8

divergences = {}

for v in variants:
    actions_v = run_with_instruction(v)
    abs_l2 = torch.norm(actions_v - base_actions, dim=-1)
    mask = motion_scale > 1e-3
    rel_l2 = abs_l2[mask] / (motion_scale[mask] + eps)
    divergences[v] = rel_l2
    print(f"{v[:40]} | mean relative L2: {rel_l2.mean().item():.6f}")

baseline_test = run_with_instruction(base_instruction)
print("Baseline self L2:",
      torch.norm(baseline_test - base_actions))
print("Mean baseline action norm:",
      torch.norm(base_actions, dim=-1).mean())

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 5))

for v, d in divergences.items():
    plt.plot(d.numpy(), label=v[:50])

plt.xlabel("Timestep")
plt.ylabel("Relative L2 Action Difference")
plt.title("SmolVLA Language Sensitivity (Replay, Relative to Motion Scale)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("language_divergence.png", dpi=200)
plt.show()

print("Saved plot to language_divergence.png")