import torch
from PIL import Image
from transformers import AutoProcessor

# Your custom imports
from vla.models.smolvla import smolvla 

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def get_vision_encoder_attention(model, pil_image, device="cuda"):
    """
    Extracts the self-attention map from the final layer of the isolated vision encoder.
    """
    attention_maps = []
    
    def forward_hook(module, input, output):
        # output[1] contains the attention weights when output_attentions=True
        print(f"Hook triggered! Attention weights shape: {output[1].shape}")
        attn_weights = output[1].detach().cpu()
        attention_maps.append(attn_weights)
    
    # 1. Drill down through the LeRobot wrapper to get the HF Vision Model
    hf_vlm = model.model.vlm_with_expert.vlm
    vision_model = hf_vlm.model.vision_model
    print("Isolated the vision model successfully.")

    # 2. Target the final self-attention layer of the vision encoder
    target_layer = vision_model.encoder.layers[-1].self_attn
    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    # 3. Force the vision model to output attention tensors
    vision_model.config._attn_implementation = "eager"  # Ensure we get the attention weights in the output
    vision_model.config.output_attentions = True
    
    # 4. Use the official processor to format the image perfectly
    print("Processing image...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.ndim == 5:
            inputs[k] = v[:, 0, ...]
            
    # Ensure the image tensor dtype matches the model's weights
    dtype = next(vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    
    # 5. Run the forward pass ONLY on the vision model
    print("Running forward pass through vision encoder...")
    print(f"Input pixel_values shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
    with torch.no_grad():
        _ = vision_model(**inputs)
        
    hook_handle.remove()
    
    return attention_maps[0]


def visualize_attention_heatmap(attention_map, pil_image):
    """
    Converts a [1, num_heads, seq_len, seq_len] attention map into a heatmap
    and overlays it on the original image.
    """
    # 1. Remove batch dimension and average across the 12 attention heads
    # Shape goes from [1, 12, 1024, 1024] -> [1024, 1024]
    attn = attention_map.squeeze(0).mean(dim=0)
    
    # 2. Average the attention each patch *receives* from all other patches
    # Shape goes from [1024, 1024] -> [1024]
    salience = attn.mean(dim=0)
    
    # 3. Reshape the 1024 flat patches back into a 2D grid (32x32)
    grid_size = int(np.sqrt(salience.shape[0]))
    salience_grid = salience.reshape(grid_size, grid_size)
    
    # 4. Normalize the values to be between 0 and 1 for coloring
    salience_grid = (salience_grid - salience_grid.min()) / (salience_grid.max() - salience_grid.min())
    
    # 5. Convert to a numpy array for matplotlib
    # first convert from bfloat16 to float32 if necessary
    print(f"Salience grid dtype before conversion: {salience_grid.dtype}")
    if salience_grid.dtype == torch.bfloat16:
        salience_grid = salience_grid.to(torch.float32)
    salience_np = salience_grid.numpy()
    
    # 6. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    ax[0].imshow(pil_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Show overlay
    # We use extent to stretch the 32x32 grid over the full original image dimensions
    ax[1].imshow(pil_image)
    im = ax[1].imshow(salience_np, cmap='jet', alpha=0.5, extent=[0, pil_image.width, pil_image.height, 0])
    ax[1].set_title("Self-Attention Heatmap")
    ax[1].axis('off')
    
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    checkpoint = "HuggingFaceVLA/smolvla_libero"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- MODEL DEFINITION ---
    print(f"Loading model from checkpoint: {checkpoint} on device: {device}")
    model, model_id, action_dim = smolvla(checkpoint, device)
    print(f"Model loaded: {model_id}, Action dim: {action_dim}")

    # --- IMAGE LOADING ---
    img_path = "data/images/libero/spatial/ep0000_task_0/frame0000.png" 
    
    try:
        # Load the raw PIL Image
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Could not find {img_path}. Make sure you are in the correct root directory.")
        exit(1)
    
    # --- RUN DIAGNOSTIC ---
    attention_map = get_vision_encoder_attention(model, image, device)
    
    print("Final output shape:", attention_map.shape)

    print("Generating heatmap overlay")
    visualize_attention_heatmap(attention_map, image)