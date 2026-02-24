import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from collections import Counter

# Your custom imports
from vla.models.smolvla import smolvla 

def get_llm_attention(model, pil_image, task_description, device="cuda"):
    """
    Hooks into the Language Model to see how the text prompt attends to the image.
    """
    attention_maps = []
    
    def forward_hook(module, input, output):
        # output[1] contains the attention weights: [batch, heads, seq_len, seq_len]
        print(f"LLM Hook triggered! Attention shape: {output[1].shape}")
        attn_weights = output[1].detach().cpu()
        attention_maps.append(attn_weights)
    
    # 1. Drill down to the Language Model
    hf_vlm = model.model.vlm_with_expert.vlm
    llm = hf_vlm.model.text_model
    print("Isolated the Language Model successfully.")

    # 2. Target the final self-attention layer of the LLM
    target_layer = llm.layers[-1].self_attn
    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    # 3. Force the LLM to output attention (using the SDPA override hack)
    llm.config._attn_implementation = "eager"
    llm.config.output_attentions = True
    
    # 4. Process the Image and Text together
    print("Processing image and text prompt...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": task_description}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
    
    # Ensure image dtype matches vision model
    vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)
    
    # 5. Run forward pass on the FULL VLM
    print("Running forward pass through the VLM...")
    with torch.no_grad():
        _ = hf_vlm(**inputs)
        
    hook_handle.remove()
    
    return attention_maps[0], inputs


def visualize_text_to_image_heatmap(attention_map, inputs, pil_image):
    """
    Extracts the text-to-image attention and overlays it as a heatmap.
    """
    # 1. Average across the attention heads
    # Shape goes from [1, num_heads, seq_len, seq_len] -> [seq_len, seq_len]
    attn = attention_map.squeeze(0).mean(dim=0)
    
    # 2. Get the attention from the LAST text token to all previous tokens
    # This token represents the model's final understanding of the prompt
    last_token_attn = attn[-1, :]
    
    # 3. Find the image tokens in the sequence
    # Hugging Face processors repeat a specific placeholder ID for image patches
    input_ids_list = inputs["input_ids"][0].tolist()
    most_common_token = Counter(input_ids_list).most_common(1)[0][0]
    image_token_indices = [i for i, token in enumerate(input_ids_list) if token == most_common_token]
    
    # Extract the attention weights ONLY for the image tokens
    image_attn = last_token_attn[image_token_indices]
    
    # 4. Isolate the global crop
    # The image is split into multiple crops. We want the first block (the global view).
    num_crops = inputs["pixel_values"].shape[1]
    tokens_per_crop = len(image_attn) // num_crops
    global_crop_attn = image_attn[:tokens_per_crop]
    
    # 5. Reshape into a 2D grid
    grid_size = int(np.sqrt(tokens_per_crop))
    salience_grid = global_crop_attn.reshape(grid_size, grid_size)
    
    # 6. Normalize for plotting
    salience_grid = (salience_grid - salience_grid.min()) / (salience_grid.max() - salience_grid.min())
    if salience_grid.dtype == torch.bfloat16: # same error with image_attn, so convert to float32 
        salience_grid = salience_grid.to(torch.float32)
    salience_np = salience_grid.numpy()
    
    # 7. Plot it
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(pil_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(pil_image)
    im = ax[1].imshow(salience_np, cmap='jet', alpha=0.5, extent=[0, pil_image.width, pil_image.height, 0])
    ax[1].set_title("Text-Conditioned Attention (LLM)")
    ax[1].axis('off')
    
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    checkpoint = "HuggingFaceVLA/smolvla_libero"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from checkpoint: {checkpoint} on device: {device}")
    model, model_id, action_dim = smolvla(checkpoint, device)
    
    # Load Image
    img_path = "data/images/libero/spatial/ep0000_task_0/frame0000.png" 
    image = Image.open(img_path).convert("RGB")
    
    # Load Task Description
    task_path = "data/images/libero/spatial/ep0000_pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate/task.txt"
    with open(task_path, "r") as f:
        task_description = f.read().strip()
        
    print(f"Task: {task_description}")
    
    # Run the diagnostic
    attention_map, model_inputs = get_llm_attention(model, image, "cookie box", device)
    
    print("Generating heatmap overlay...")
    visualize_text_to_image_heatmap(attention_map, model_inputs, image)