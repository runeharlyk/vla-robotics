from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoProcessor

from vla.constants import PROJECT_ROOT
from vla.models.smolvla import smolvla
from vla.utils import get_device, seed_everything


def _find_token_indices(tokens, target_word):
    """
    Find token indices that correspond to a multi-token word.
    Handles subword tokenization (e.g. "basket" → ["▁bas", "ket"]).
    Returns list of matching token indices.
    """
    target_lower = target_word.lower().replace(" ", "")
    n = len(tokens)

    for start in range(n):
        merged = ""
        for end in range(start, n):
            fragment = tokens[end].lower().replace("▁", "").replace("Ġ", "").replace("##", "")
            merged += fragment
            if merged == target_lower:
                return list(range(start, end + 1))
            if len(merged) > len(target_lower):
                break

    for i, t in enumerate(tokens):
        if target_lower in t.lower().replace("▁", "").replace("Ġ", ""):
            return [i]

    return []


def get_multi_layer_attention(model, pil_image, task_description, device="cuda", layer_indices=None):
    """
    Hooks into the LLM layers to extract attention maps.
    If layer_indices is None, samples layers across the full depth.
    """
    attention_maps = {}

    hf_vlm = model.model.vlm_with_expert.vlm
    llm = hf_vlm.model.text_model
    total_layers = len(llm.layers)
    print(f"Total LLM layers: {total_layers}")

    if layer_indices is None:
        percentiles = [0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 1.0]
        layer_indices = sorted(set(max(0, int(total_layers * p) - 1) for p in percentiles))

    layer_indices = [i for i in layer_indices if 0 <= i < total_layers]
    print(f"Targeting layer indices: {layer_indices}")

    def get_forward_hook(layer_idx):
        def hook(module, input, output):
            attention_maps[layer_idx] = output[1].detach().cpu()

        return hook

    hook_handles = []
    for idx in layer_indices:
        target_layer = llm.layers[idx].self_attn
        handle = target_layer.register_forward_hook(get_forward_hook(idx))
        hook_handles.append(handle)

    llm.config._attn_implementation = "eager"
    llm.config.output_attentions = True

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

    vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

    with torch.no_grad():
        _ = hf_vlm(**inputs)

    for handle in hook_handles:
        handle.remove()

    return attention_maps, inputs, layer_indices, processor


def _extract_image_attention_grid(attn_map, token_indices, image_token_indices, num_crops, aggregate="mean"):
    """
    Given an attention map [num_heads, seq, seq], extract the attention from
    token_indices → image patches and reshape into a 2D grid.
    Returns (salience_grid_np, per_head_grids_np) both as float32 numpy.
    """
    attn = attn_map.squeeze(0)  # [heads, seq, seq]

    word_attn = attn[:, token_indices, :]  # [heads, n_tokens, seq]
    word_attn = word_attn.mean(dim=1) if aggregate == "mean" else word_attn.max(dim=1).values

    image_attn = word_attn[:, image_token_indices]  # [heads, n_image_tokens]

    tokens_per_crop = len(image_token_indices) // num_crops
    global_crop_attn = image_attn[:, :tokens_per_crop]  # [heads, tokens_per_crop]

    grid_size = int(np.sqrt(tokens_per_crop))
    per_head_grids = global_crop_attn.reshape(-1, grid_size, grid_size)

    head_avg = per_head_grids.mean(dim=0)

    def to_np(t):
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.float().numpy()

    return to_np(head_avg), to_np(per_head_grids)


def _normalize(arr):
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _smooth_and_resize(grid, target_size, sigma=1.5):
    smoothed = gaussian_filter(grid, sigma=sigma)
    smoothed = _normalize(smoothed)
    from PIL import Image as _Img

    resized = (
        np.array(_Img.fromarray((smoothed * 255).astype(np.uint8)).resize(target_size, resample=Image.BICUBIC)) / 255.0
    )
    return resized


def compute_attention_rollout(attention_maps, layer_indices, token_indices, image_token_indices, num_crops):
    """
    Compute attention rollout: propagate attention through layers accounting
    for residual connections.  R = prod_l (0.5*I + 0.5*A_l) for each layer l.
    Returns a 2D numpy grid of cumulative attention from token_indices to image patches.
    """
    sorted_layers = sorted(attention_maps.keys())

    first_attn = attention_maps[sorted_layers[0]].squeeze(0)  # [heads, seq, seq]
    seq_len = first_attn.shape[-1]

    rollout = torch.eye(seq_len)

    for layer_idx in sorted_layers:
        attn = attention_maps[layer_idx].squeeze(0).mean(dim=0)  # [seq, seq]
        if attn.dtype == torch.bfloat16:
            attn = attn.float()
        identity = torch.eye(seq_len)
        attn_with_residual = 0.5 * identity + 0.5 * attn
        rollout = attn_with_residual @ rollout

    word_rollout = rollout[token_indices, :].mean(dim=0)  # [seq]
    image_rollout = word_rollout[image_token_indices]

    tokens_per_crop = len(image_token_indices) // num_crops
    global_crop = image_rollout[:tokens_per_crop]

    grid_size = int(np.sqrt(tokens_per_crop))
    grid = global_crop.reshape(grid_size, grid_size).numpy()
    return _normalize(grid)


def _add_contour(ax, smooth, extent, min_contrast=0.3):
    contrast = smooth.max() - smooth.min()
    if contrast >= min_contrast:
        ax.contour(
            smooth,
            levels=[np.percentile(smooth, 90)],
            colors="cyan",
            linewidths=1.0,
            extent=extent,
        )


def visualize_multi_layer_heatmap(
    attention_maps_dict,
    inputs,
    pil_image,
    layer_indices,
    processor,
    target_word,
    sigma=1.5,
    cmap="inferno",
    top_k_contour=True,
):
    """
    Enhanced multi-layer heatmap: smooth overlays, attention rollout, and top-K contours.
    Uses a 2-row grid so panels are large enough to read.
    """
    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    target_indices = _find_token_indices(tokens, target_word)
    if not target_indices:
        print(f"Could not find '{target_word}' in tokens. Defaulting to last token.")
        target_indices = [len(tokens) - 1]
    else:
        matched = [tokens[i] for i in target_indices]
        print(f"Matched '{target_word}' → tokens {matched} at indices {target_indices}")

    most_common_token = Counter(input_ids).most_common(1)[0][0]
    image_token_indices = [i for i, token in enumerate(input_ids) if token == most_common_token]
    num_crops = inputs["pixel_values"].shape[1]

    rollout_grid = compute_attention_rollout(
        attention_maps_dict, layer_indices, target_indices, image_token_indices, num_crops
    )

    panels = ["original"] + [f"layer_{li}" for li in layer_indices] + ["rollout"]
    n_panels = len(panels)
    cols = min(n_panels, 5)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for flat_idx, panel in enumerate(panels):
        r, c = divmod(flat_idx, cols)
        ax = axes[r, c]

        if panel == "original":
            ax.imshow(pil_image)
            ax.set_title("Original", fontsize=11, fontweight="bold")
        elif panel == "rollout":
            rollout_smooth = _smooth_and_resize(rollout_grid, img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(rollout_smooth, cmap=cmap, alpha=0.6, extent=extent)
            if top_k_contour:
                _add_contour(ax, rollout_smooth, extent)
            ax.set_title("Attention Rollout", fontsize=11, fontweight="bold")
        else:
            layer_idx = int(panel.split("_")[1])
            avg_grid, _ = _extract_image_attention_grid(
                attention_maps_dict[layer_idx], target_indices, image_token_indices, num_crops
            )
            smooth = _smooth_and_resize(avg_grid, img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            if top_k_contour:
                _add_contour(ax, smooth, extent)
            ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.axis("off")

    for flat_idx in range(n_panels, rows * cols):
        r, c = divmod(flat_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f'Attention for "{target_word}"', fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_per_head(
    attention_maps_dict,
    inputs,
    pil_image,
    processor,
    target_word,
    layer_idx,
    top_n_heads=8,
    min_score_ratio=0.05,
    sigma=1.5,
    cmap="inferno",
):
    """
    Show the top-N most focused attention heads at a given layer for target_word.
    Heads with score < min_score_ratio * best_score are dropped as noise.
    """
    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    target_indices = _find_token_indices(tokens, target_word)
    if not target_indices:
        target_indices = [len(tokens) - 1]

    most_common_token = Counter(input_ids).most_common(1)[0][0]
    image_token_indices = [i for i, token in enumerate(input_ids) if token == most_common_token]
    num_crops = inputs["pixel_values"].shape[1]

    _, per_head_grids = _extract_image_attention_grid(
        attention_maps_dict[layer_idx], target_indices, image_token_indices, num_crops
    )

    head_scores = per_head_grids.max(axis=(1, 2)) - per_head_grids.mean(axis=(1, 2))
    sorted_heads = np.argsort(head_scores)[::-1]

    best_score = head_scores[sorted_heads[0]]
    threshold = best_score * min_score_ratio
    top_heads = [h for h in sorted_heads[:top_n_heads] if head_scores[h] >= threshold]

    if not top_heads:
        top_heads = [sorted_heads[0]]

    n = len(top_heads)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for i, head_idx in enumerate(top_heads):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        grid = per_head_grids[head_idx]
        smooth = _smooth_and_resize(grid, img_size, sigma=sigma)

        ax.imshow(pil_image)
        ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
        _add_contour(ax, smooth, extent)
        ax.set_title(f"Head {head_idx}  (score: {head_scores[head_idx]:.4f})", fontsize=10)
        ax.axis("off")

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(f'Top-{n} heads at Layer {layer_idx} for "{target_word}"', fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def _is_natural_word(token_str):
    import re

    if "<" in token_str or ">" in token_str:
        return False
    cleaned = token_str.replace("▁", "").replace("Ġ", "").replace("##", "").strip()
    if not cleaned:
        return False
    return not re.fullmatch(r"[\W_]+", cleaned)


def visualize_token_attention_summary(attention_maps_dict, inputs, pil_image, processor, layer_idx):
    """
    Bar chart showing how much each text token attends to image patches vs. other text.
    Only shows natural-language tokens (filters out special/template tokens).
    """
    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    num_tokens = len(tokens)

    most_common_token = Counter(input_ids).most_common(1)[0][0]
    image_token_indices = set(i for i, token in enumerate(input_ids) if token == most_common_token)

    attn = attention_maps_dict[layer_idx].squeeze(0).mean(dim=0)
    if attn.dtype == torch.bfloat16:
        attn = attn.float()

    image_attn_per_token = []
    labels = []
    for idx in range(num_tokens):
        if idx in image_token_indices:
            continue
        if not _is_natural_word(tokens[idx]):
            continue
        row = attn[idx]
        img_attn = row[list(image_token_indices)].sum().item()
        total = row.sum().item()
        frac = img_attn / (total + 1e-12)
        image_attn_per_token.append(frac)
        label = tokens[idx].replace("▁", " ").replace("Ġ", " ").replace("##", "").strip()
        labels.append(label)

    if not labels:
        print("No natural-language tokens found to plot.")
        return

    median_val = np.median(image_attn_per_token)
    colors = ["#e74c3c" if v > median_val else "#3498db" for v in image_attn_per_token]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    ax.bar(range(len(labels)), image_attn_per_token, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of attention → image", fontsize=11)
    ax.set_title(f"Token visual grounding (Layer {layer_idx})", fontsize=13, fontweight="bold")
    ax.axhline(median_val, color="gray", ls="--", lw=0.8, label="median")
    ax.legend(fontsize=9)
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

    # Load Image
    img_path = "data/images/libero/spatial/ep0000_task_0/frame0000.png"
    image = Image.open(img_path).convert("RGB")

    # Load Full Task Description
    task_path = (
        "data/images/libero/spatial/"
        "ep0000_pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate/task.txt"
    )
    with open(task_path) as f:
        task_description = f.read().strip()

    print(f"Task: {task_description}")

    # Pick a specific word to track (e.g., "bowl" or "box")
    target_word = "box"

    # Run the multi-layer diagnostic
    attention_maps_dict, model_inputs, layer_indices, processor = get_multi_layer_attention(
        model, image, task_description, device
    )

    print("Generating multi-layer heatmap overlay...")
    visualize_multi_layer_heatmap(attention_maps_dict, model_inputs, image, layer_indices, processor, target_word)

    print("Generating per-head breakdown...")
    visualize_per_head(attention_maps_dict, model_inputs, image, processor, target_word, layer_idx=layer_indices[-1])

    print("Generating token attention summary...")
    visualize_token_attention_summary(attention_maps_dict, model_inputs, image, processor, layer_idx=layer_indices[-1])
