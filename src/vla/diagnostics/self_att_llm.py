import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from collections import Counter

from vla.models.smolvla import smolvla
from vla.utils import get_device, seed_everything
from vla.constants import PROJECT_ROOT


def get_multi_layer_attention(model, pil_image, task_description, device="cuda"):
    """
    Hooks into multiple percentiles of the LLM to track attention evolution.
    """
    attention_maps = {}

    # 1. Drill down to the Language Model
    hf_vlm = model.model.vlm_with_expert.vlm
    llm = hf_vlm.model.text_model
    total_layers = len(llm.layers)
    print(f"Isolated the Language Model successfully. Total layers: {total_layers}")

    # 2. Calculate the target layers (10%, 25%, 50%, 75%, Final)
    # percentiles = [0.10, 0.25, 0.50, 0.75, 1.0]
    # layer_indices = [max(0, int(total_layers * p) - 1) for p in percentiles]
    layer_indices = [1, 2, 3, 4, 5, 6, 7]
    print(f"Targeting layer indices: {layer_indices}")

    # Helper function to create a unique hook for each layer
    def get_forward_hook(layer_idx):
        def hook(module, input, output):
            print(f"Hook triggered for layer {layer_idx}!")
            # output[1] contains the attention weights
            attention_maps[layer_idx] = output[1].detach().cpu()

        return hook

    # 3. Register hooks on all target layers
    hook_handles = []
    for idx in layer_indices:
        target_layer = llm.layers[idx].self_attn
        handle = target_layer.register_forward_hook(get_forward_hook(idx))
        hook_handles.append(handle)

    # Force the LLM to output attention
    llm.config._attn_implementation = "eager"
    llm.config.output_attentions = True

    # 4. Process the Image and Text together
    print("Processing image and text prompt...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

    # Ensure image dtype matches vision model
    vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

    # 5. Run forward pass on the FULL VLM
    print("Running forward pass through the VLM...")
    with torch.no_grad():
        _ = hf_vlm(**inputs)

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    return attention_maps, inputs, layer_indices, processor


def visualize_multi_layer_heatmap(attention_maps_dict, inputs, pil_image, layer_indices, processor, target_word):
    """
    Plots the original image alongside the heatmaps from the different layer percentiles.
    """
    # Find the target token index
    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    target_idx = -1
    for i, t in enumerate(tokens):
        if target_word.lower() in t.lower():
            target_idx = i
            print(f"Tracking attention for word '{t}' (token index {i})")
            break

    if target_idx == -1:
        print(f"Warning: Could not find '{target_word}'. Defaulting to the last token.")
        target_idx = -1

    # Find image token indices
    most_common_token = Counter(input_ids).most_common(1)[0][0]
    image_token_indices = [i for i, token in enumerate(input_ids) if token == most_common_token]
    num_crops = inputs["pixel_values"].shape[1]

    # Setup the plot (1 row, 6 columns: Original + 5 Layers)
    fig, axes = plt.subplots(1, 8, figsize=(24, 4))

    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    percentile_labels = ["1", "2", "3", "4", "5", "6", "7"]

    # Process and plot each layer
    for plot_idx, layer_idx in enumerate(layer_indices):
        attn = attention_maps_dict[layer_idx].squeeze(0).mean(dim=0)
        word_attn = attn[target_idx, :]
        image_attn = word_attn[image_token_indices]

        # Isolate the global crop
        tokens_per_crop = len(image_attn) // num_crops
        global_crop_attn = image_attn[:tokens_per_crop]

        # Reshape to 2D grid
        grid_size = int(np.sqrt(tokens_per_crop))
        salience_grid = global_crop_attn.reshape(grid_size, grid_size)

        # Normalize
        salience_grid = (salience_grid - salience_grid.min()) / (salience_grid.max() - salience_grid.min() + 1e-8)

        # Convert bfloat16 to float32 if needed for numpy
        if salience_grid.dtype == torch.bfloat16:
            salience_grid = salience_grid.to(torch.float32)
        salience_np = salience_grid.numpy()

        # Plot with BICUBIC interpolation for smoothness
        ax = axes[plot_idx + 1]
        ax.imshow(pil_image)
        ax.imshow(salience_np, cmap="jet", alpha=0.5, extent=[0, pil_image.width, pil_image.height, 0])
        ax.set_title(f"Layer {layer_idx} ({percentile_labels[plot_idx]})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    checkpoint = "HuggingFaceVLA/smolvla_libero"

    episode_id = "ep0000_pick_up_the_orange_juice_and_place_it_in_the_basket"
    frame_idx = 0
    data_dir = PROJECT_ROOT / "data/images/libero/object"
    img_path = data_dir / episode_id / f"frame{frame_idx:04d}.png"
    task_path = data_dir / episode_id / "task.txt"

    print(f"Loading model from checkpoint: {checkpoint} on device: {device}")
    model, model_id, action_dim = smolvla(checkpoint, device)

    image = Image.open(img_path).convert("RGB")

    with open(task_path, "r") as f:
        task_description = f.read().strip()

    print(f"Task: {task_description}")

    target_word = "orange juice"

    attention_maps_dict, model_inputs, layer_indices, processor = get_multi_layer_attention(
        model, image, task_description, device
    )
    
    print("Generating multi-layer heatmap overlay...")
    visualize_multi_layer_heatmap(attention_maps_dict, model_inputs, image, layer_indices, processor, target_word)