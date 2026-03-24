import h5py
import torch
import re
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from vla.models.smolvla import SmolVLAPolicy
from vla.utils.device import get_device
from vla.utils.seed import seed_everything




# ------------------------------------------------
# CONFIG
# ------------------------------------------------

CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
DEVICE = "cuda"

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ROLLOUT_PATH = "smolvla_language_pilot/rollout.h5"

# Number of variants PER variant type
N_VARIANTS = 3

CONTROLLED_VARIANTS = {
    "politeness": (
        "Rewrite the instruction using a polite phrasing while keeping the task exactly the same. "
        "Allowed politeness strategies include: adding 'please' or 'kindly', using a polite request "
        "such as 'could you' or 'would you', softening with phrases like 'when ready' or 'if possible', "
        "or using deferential phrasing like 'I'd like you to'. "
        "Do not change the object, location, action, or task constraints. "
        "Do not add words like carefully, slowly, gently, precisely, or other modifiers that change execution."
    ),

    "sentence_structure": (
        "Rewrite the instruction with a different sentence structure while keeping the task exactly the same. "
        "Allowed structure changes include: turning it into a question, fronting a phrase, using an imperative "
        "with a trailing clause, or changing constituent order. "
        "Do not change the action, object, location, or task constraints. "
        "Do not add politeness unless it is necessary for the grammatical form."
    ),

    "verb_paraphrase": (
        "Rewrite the instruction using a different verb or verb phrase with the same meaning. "
        "Allowed examples include alternatives such as 'pull open', 'open up', or other natural paraphrases "
        "that preserve the task. "
        "Do not change the object, location, or task constraints. "
        "Do not replace the verb with one that changes the physical action or intent."
    ),

    "verbosity": (
        "Rewrite the instruction with slightly more verbose wording while keeping the task exactly the same. "
        "Allowed changes include adding harmless framing phrases such as 'for this task' or 'your task is to'. "
        "Do not add new semantic constraints or execution modifiers like carefully, slowly, gently, firmly, or precisely. "
        "Do not change the action, object, or location."
    ),

    "context": (
    "Rewrite the instruction by adding neutral task context phrases such as "
    "'in this scene', 'for this task', or 'in this situation'. "
    "Do not introduce new locations, objects, colors, or attributes. "
    "The action, object, and location must remain exactly the same."
    )   
}

# ------------------------------------------------
# SEED LLM
# ------------------------------------------------
SEED = 0

seed_everything(SEED)

# ------------------------------------------------
# Load rollout
# ------------------------------------------------

with h5py.File(ROLLOUT_PATH, "r") as f:
    images = torch.tensor(f["observation/image"][:])
    states = torch.tensor(f["observation/state"][:])
    base_instruction = f["instruction"][()].decode("utf-8")

print("Loaded rollout")
print("Images:", images.shape)
print("States:", states.shape)
print("Instruction:", base_instruction)

device_obj = get_device(DEVICE)


# ------------------------------------------------
# Load VLA policy
# ------------------------------------------------

policy = SmolVLAPolicy(CHECKPOINT, action_dim=7, device=str(device_obj))
policy.eval()

model_dtype = policy.dtype


# ------------------------------------------------
# Load LLM
# ------------------------------------------------

print("Loading local LLM...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# ------------------------------------------------
# Text helpers
# ------------------------------------------------


def _cleanup_candidate(text: str):

    text = text.strip().strip('"').strip("'")

    # remove instruction-style explanations
    stop_phrases = [
        "Sure",
        "Here",
        "To adhere",
        "Following",
        "The rewritten instruction",
        "Task:",
        "Instruction:",
        "Rewrite:",
    ]

    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0]

    # keep only the first sentence
    parts = re.split(r'[.!?\n]', text)
    if parts:
        text = parts[0]

    # remove bullet prefixes
    while text.startswith(("-", "*", "•")):
        text = text[1:].strip()

    return text.strip()


def _normalize_instruction(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _is_valid_candidate(text: str, original_instruction: str, seen: set) -> bool:
    if not text:
        return False

    if len(text.split()) < 3:
        return False

    if _normalize_instruction(text) == _normalize_instruction(original_instruction):
        return False
    
    if _normalize_instruction(text) in seen:
        return False

    return True


def _generate_one(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: v.to(llm.device) for k, v in inputs.items()}

    output = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    candidate = tokenizer.decode(generated, skip_special_tokens=True)

    return _cleanup_candidate(candidate)


def generate_llm_variant(instruction: str, rule: str):
    prompt = f"""Task: Rewrite a robot instruction.

Original instruction:
{instruction}

Rewrite rule:
{rule}

Constraints:
- Output exactly one sentence
- Do not add explanations
- Do not ask questions
- Do not continue the conversation

Rewritten instruction:"""

    return _generate_one(prompt)


# ------------------------------------------------
# Generate instruction variants
# ------------------------------------------------

def generate_variants(instruction):
    variants = []
    labels = []

    seen = {_normalize_instruction(instruction)}
    print("\nGenerating variants...\n")

    for name, rule in CONTROLLED_VARIANTS.items():
        pbar = tqdm(range(N_VARIANTS), desc=f"Generating {name}") if tqdm else range(N_VARIANTS)
        for i in pbar:
            candidate = ""
            for _ in range(5):
                trial = generate_llm_variant(instruction, rule)
                if _is_valid_candidate(trial, instruction, seen):
                    norm = _normalize_instruction(trial)
                    if norm not in seen:
                        candidate = trial
                        break

            if not candidate:
                candidate = f"Please {instruction}"
            
            seen.add(_normalize_instruction(candidate))
            variants.append(candidate)
            labels.append(f"{name}_{i+1}")

            print(f"{name}_{i+1}: {candidate}")

    return variants, labels


print("Generating instruction variants...")

variants, labels = generate_variants(base_instruction)

print("\nGenerated variants:\n")

for l, v in zip(labels, variants):
    print(l, ":", v)


# ------------------------------------------------
# Replay function
# ------------------------------------------------

def run_with_instruction(instruction):

    seed_everything(0)

    if hasattr(policy, "reset"):
        policy.reset()

    actions = []

    use_amp = device_obj.type == "cuda"

    with torch.inference_mode():

        for img, state in zip(images, states):

            action = policy.predict_action(img, instruction, state)

            actions.append(action.cpu())

    return torch.stack(actions)


# ------------------------------------------------
# Run baseline
# ------------------------------------------------

print("\nRunning base instruction replay...")

base_actions = run_with_instruction(base_instruction)

motion_scale = torch.norm(base_actions, dim=-1)

eps = 1e-8


# ------------------------------------------------
# Run variants
# ------------------------------------------------

divergence_curves = []

for label, variant in zip(labels, variants):

    print("Running:", label)

    actions_v = run_with_instruction(variant)

    abs_l2 = torch.norm(actions_v - base_actions, dim=-1)

    rel_l2 = abs_l2 / (motion_scale + eps)

    divergence_curves.append(rel_l2)

divergence_curves = torch.stack(divergence_curves)

print("Divergence matrix:", divergence_curves.shape)

# ------------------------------------------------
# Compute mean divergence per variant type
# ------------------------------------------------

variant_type_means = {}
variant_type_stds = {}

for label, curve in zip(labels, divergence_curves):

    variant_type = label.split("_")[0]

    if variant_type not in variant_type_means:
        variant_type_means[variant_type] = []

    variant_type_means[variant_type].append(curve)

for k in variant_type_means:
    curves = torch.stack(variant_type_means[k])
    variant_type_means[k] = curves.mean(dim=0)
    variant_type_stds[k] = curves.std(dim=0)

type_labels = list(variant_type_means.keys())

type_curves = torch.stack([
    variant_type_means[k] for k in type_labels
])


# ------------------------------------------------
# Statistics
# ------------------------------------------------

mean_curve = divergence_curves.mean(dim=0)

std_curve = divergence_curves.std(dim=0)

# ------------------------------------------------
# Trajectory-level mean divergence per variant
# ------------------------------------------------

trajectory_means = []

for curve in divergence_curves:
    trajectory_means.append(curve.mean().item())

trajectory_means = np.array(trajectory_means)


# ------------------------------------------------
# Heatmap
# ------------------------------------------------

plt.figure(figsize=(10,5))

plt.imshow(type_curves.numpy(), aspect="auto")
plt.yticks(range(len(type_labels)), type_labels)

plt.colorbar(label="Relative L2")

plt.xlabel("Timestep")
plt.ylabel("Variant")

plt.title("Language Variant Sensitivity Heatmap")

plt.tight_layout()

plt.savefig("language_heatmap.png", dpi=200)


plt.show()

# ------------------------------------------------
# Plot mean divergence per variant type
# ------------------------------------------------

plt.figure(figsize=(8,5))

x = np.arange(len(mean_curve))

for variant_type, curve in variant_type_means.items():

    std = variant_type_stds[variant_type]

    plt.plot(x, curve.numpy(), label=variant_type)

    plt.fill_between(
        x,
        (curve - std).numpy(),
        (curve + std).numpy(),
        alpha=0.2
    )

plt.xlabel("Timestep")
plt.ylabel("Relative Action Divergence")
plt.title("Mean Divergence per Instruction Variant Type")

plt.legend()

plt.tight_layout()

plt.savefig("language_variant_type_mean.png", dpi=200)

plt.show()

# ------------------------------------------------
# Boxplot per instruction type
# ------------------------------------------------

boxplot_data = {}
for label, val in zip(labels, trajectory_means):

    variant_type = label.split("_")[0]

    if variant_type not in boxplot_data:
        boxplot_data[variant_type] = []

    boxplot_data[variant_type].append(val)

plt.figure(figsize=(6,5))

plt.boxplot(
    boxplot_data.values(),
    labels=boxplot_data.keys(),
)

plt.ylabel("Trajectory Mean Divergence")
plt.title("Language Robustness per Instruction Type")

plt.tight_layout()

plt.savefig("language_boxplot.png", dpi=200)

plt.show()

# ------------------------------------------------
# Language Sensitivity Score (LSS)
# ------------------------------------------------

print("\nLanguage Sensitivity Score (LSS):")

lss_scores = {}

for variant_type, values in boxplot_data.items():

    lss = np.mean(values)

    lss_scores[variant_type] = lss

    print(f"{variant_type}: {lss:.3f}")

    # ------------------------------------------------
# LSS bar chart
# ------------------------------------------------

plt.figure(figsize=(6,5))

plt.bar(
    lss_scores.keys(),
    lss_scores.values()
)

plt.ylabel("Language Sensitivity Score")
plt.title("Language Sensitivity by Instruction Type")

plt.tight_layout()

plt.savefig("language_lss.png", dpi=200)
plt.show()


# ------------------------------------------------
# Visualize divergence spike with rollout frame
# ------------------------------------------------

# Compute overall mean curve across variant types
all_curves = torch.stack(list(variant_type_means.values()))
overall_mean_curve = all_curves.mean(dim=0)

# Find peak timestep
peak_timestep = overall_mean_curve.argmax().item()

print("Peak divergence timestep:", peak_timestep)

# Get rollout image
frame = images[peak_timestep].permute(1,2,0).cpu().numpy()

# Create figure
fig = plt.figure(figsize=(10,8))

# -------------------------
# Top: divergence curves
# -------------------------
ax1 = fig.add_subplot(2,1,1)

x = np.arange(len(overall_mean_curve))

for variant_type, curve in variant_type_means.items():
    ax1.plot(x, curve.numpy(), label=variant_type)

# mark peak
ax1.axvline(peak_timestep, color="black", linestyle="--", label="peak divergence")

ax1.set_ylabel("Relative Action Divergence")
ax1.set_title("Language Sensitivity with Peak Frame")
ax1.legend()

# -------------------------
# Bottom: image frame
# -------------------------
ax2 = fig.add_subplot(2,1,2)

ax2.imshow(frame)
ax2.set_title(f"Rollout frame at timestep {peak_timestep}")
ax2.axis("off")

plt.tight_layout()

plt.savefig("language_divergence_with_frame.png", dpi=200)

plt.show()


