print("Running self-attention diagnostics for VLM...")
import torch
print(f"PyTorch version: {torch.__version__}")
from vla.data.dataset import load_libero_suite, make_dataloader, split_dataset
from vla.constants import ACTION_DIM, LIBERO_SUITES, MODELS_DIR, resolve_suites
from vla.training.train_custom import _prepare_batch
from vla.models.smolvla import smolvla  
print("Imports successful. Starting diagnostics...")

def get_vlm_self_attention(model, image_tensor, text_input):
    """
    Hooks into the VLM to extract the final layer's self-attention map.

    Goal: Generate a heatmap to show the VLM's visual focus on the raw image pixels.

    Output: A Python function that hooks into the final self-attention block of the vision encoder and outputs the attention weights.
    """
    attention_maps = []
    
    def forward_hook(module, input, output):
        # output is a tuple: (hidden_states, attention_weights)
        # We extract the attention_weights
        print("Hook triggered. Output shape:", output[1].shape)
        print("Attention weights sample:", output[1][0, 0, :5, :5])  # Print a small sample of the attention weights
        print("Input shape:", input[0].shape)  # Print the shape of the input to the attention layer
        print("Output type:", type(output[0]))  # Print the type of the hidden states output
        attn_weights = output[1].detach().cpu()
        attention_maps.append(attn_weights)
    
    # Access the final self-attention layer of the vision encoder.
    # Architecture mapping depends on the specific Hugging Face SmolVLM implementation.
    print(f"model type: {type(model)}")
    if hasattr(model, "vlm") and hasattr(model.vlm, "vision_model") and hasattr(model.vlm.vision_model, "encoder"):
        print("Model architecture seems correct. Accessing vision encoder layers.")

    target_layer = model.vlm.vision_model.encoder.layers[-1].self_attn
    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    # Ensure the model outputs attention tensors
    model.vlm.config.output_attentions = True
    
    with torch.no_grad():
        _ = model(image_tensor, text_input)
        
    hook_handle.remove()
    
    # Returns a tensor of shape (batch, num_heads, sequence_length, sequence_length)
    return attention_maps[0]


if __name__ == "__main__":
    checkpoint = "HuggingFaceVLA/smolvla_libero"
    device = "cuda"  # or "cpu"
    print(f"Loading model from checkpoint: {checkpoint} on device: {device}")
    model, model_id, action_dim = smolvla(checkpoint, device)
    print(f"Model loaded: {model_id}, Action dim: {action_dim}")

    # Dummy inputs for testing
    suite = "lerobot/libero_10_image"
    suite_names = resolve_suites(suite)
    print(f"Resolved suite names: {suite_names}")
    action_dim = ACTION_DIM
    suite_label = suite_names[0] if len(suite_names) == 1 else f"{len(suite_names)}_suites"
    full_dataset = load_libero_suite(suite_names[0])
    print(f"Loaded dataset for suite: {suite_names[0]}, Number of episodes: {len(full_dataset)}")

    dataloader = make_dataloader(full_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Created dataloader with batch. Number of batches: {len(dataloader)}")
    data_iter = iter(dataloader)
    batch = next(data_iter)
    prepared_batch = _prepare_batch(batch, model.config, device)
    print(f"Prepared batch keys: {prepared_batch.keys()}")
    text_input = [""]  # Adjust format as needed
    
    attention_map = get_vlm_self_attention(model, batch["image"], text_input)
    print(attention_map.shape)