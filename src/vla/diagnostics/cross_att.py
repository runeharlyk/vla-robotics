import torch


def get_action_cross_attention(model, image_tensor, text_input):
    """
    Hooks into the VLM to extract the final layer's cross-attention map.

    Goal: Generate a heatmap to show the VLM's visual focus on the raw image pixels when processing the text input.

    Output: A Python function that hooks into the final cross-attention block of the vision encoder
    and outputs the attention weights.
    """
    attention_maps = []

    def forward_hook(module, input, output):
        # output is a tuple: (hidden_states, attention_weights)
        # We extract the attention_weights
        attn_weights = output[1].detach().cpu()
        attention_maps.append(attn_weights)

    # Access the final cross-attention layer of the vision encoder.
    # Architecture mapping depends on the specific Hugging Face SmolVLM implementation.
    target_layer = model.vlm.vision_model.encoder.layers[-1].cross_attn
    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Ensure the model outputs attention tensors
    model.vlm.config.output_attentions = True

    with torch.no_grad():
        _ = model(image_tensor, text_input)

    hook_handle.remove()

    # Returns a tensor of shape (batch, num_heads, sequence_length, sequence_length)
    return attention_maps[0]
