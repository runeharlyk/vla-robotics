import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor
from collections import Counter
from vla.models.smolvla import smolvla
from vla.utils import get_device, seed_everything
from vla.constants import PROJECT_ROOT
from vla.diagnostics.self_att_llm import _find_token_indices, _normalize, _smooth_and_resize, _add_contour


def grad_cam_vision_encoder(model, pil_image, task_description, target_word, device="cuda", layer_idx=-1):
    """
    Grad-CAM on the vision encoder: which image patches most influence the
    prediction of target_word, according to gradients flowing back from the
    LLM's logit for that token.

    Returns (cam_grid, inputs, processor) where cam_grid is a 2D numpy array.
    """
    hf_vlm = model.model.vlm_with_expert.vlm
    vision_model = hf_vlm.model.vision_model

    target_layer = vision_model.encoder.layers[layer_idx]
    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        activations["value"] = output[0]

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

    vision_dtype = next(vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

    was_training = hf_vlm.training
    hf_vlm.eval()

    inputs["pixel_values"].requires_grad_(True)

    outputs = hf_vlm(**inputs)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    target_indices = _find_token_indices(tokens, target_word)
    if not target_indices:
        print(f"Could not find '{target_word}' in tokens. Using last token.")
        target_indices = [len(tokens) - 1]
    else:
        matched = [tokens[i] for i in target_indices]
        print(f"Grad-CAM target: '{target_word}' → tokens {matched} at indices {target_indices}")

    target_logits = logits[0, target_indices, :].mean(dim=0)
    target_score = target_logits.max()
    target_score.backward(retain_graph=False)

    fwd_handle.remove()
    bwd_handle.remove()

    if was_training:
        hf_vlm.train()

    grad = gradients["value"]  # [batch*num_crops, patches_per_crop, channels]
    act_val = activations["value"]  # [batch*num_crops, patches_per_crop, channels]

    if grad.dtype == torch.bfloat16:
        grad = grad.float()
    if act_val.dtype == torch.bfloat16:
        act_val = act_val.float()

    weights = grad.mean(dim=1, keepdim=True)  # [batch*num_crops, 1, channels]
    cam_all = (weights * act_val).sum(dim=-1)  # [batch*num_crops, patches_per_crop]
    cam_all = torch.relu(cam_all).detach().cpu().numpy()

    cam_global = cam_all[0]  # first crop (global view)
    patches_per_crop = cam_global.shape[0]

    grid_size = int(np.sqrt(patches_per_crop))
    cam_grid = cam_global[: grid_size * grid_size].reshape(grid_size, grid_size)
    cam_grid = _normalize(cam_grid)

    return cam_grid, inputs, processor


def grad_cam_llm_layers(model, pil_image, task_description, target_word, device="cuda", layer_indices=None):
    """
    Grad-CAM at LLM layers: measures gradient-weighted activation of image
    token hidden states with respect to the target word's logit.

    Returns (cam_grids_dict, inputs, layer_indices, processor).
    cam_grids_dict maps layer_idx → 2D numpy grid.
    """
    hf_vlm = model.model.vlm_with_expert.vlm
    llm = hf_vlm.model.text_model
    total_layers = len(llm.layers)

    if layer_indices is None:
        percentiles = [0.25, 0.50, 0.75, 1.0]
        layer_indices = sorted(set(max(0, int(total_layers * p) - 1) for p in percentiles))
    layer_indices = [i for i in layer_indices if 0 <= i < total_layers]
    print(f"Grad-CAM LLM layers: {layer_indices}")

    activations = {}
    gradients = {}

    def _extract_hidden(output):
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t

    def make_fwd_hook(idx):
        def hook(module, input, output):
            activations[idx] = _extract_hidden(output).detach()
        return hook

    def make_bwd_hook(idx):
        def hook(module, grad_input, grad_output):
            gradients[idx] = _extract_hidden(grad_output).detach()
        return hook

    handles = []
    for idx in layer_indices:
        layer = llm.layers[idx]
        handles.append(layer.register_forward_hook(make_fwd_hook(idx)))
        handles.append(layer.register_full_backward_hook(make_bwd_hook(idx)))

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

    vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

    was_training = hf_vlm.training
    hf_vlm.eval()

    inputs["pixel_values"].requires_grad_(True)

    outputs = hf_vlm(**inputs)
    logits = outputs.logits

    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    target_token_indices = _find_token_indices(tokens, target_word)
    if not target_token_indices:
        print(f"Could not find '{target_word}'. Using last token.")
        target_token_indices = [len(tokens) - 1]
    else:
        matched = [tokens[i] for i in target_token_indices]
        print(f"Grad-CAM target: '{target_word}' → tokens {matched}")

    target_logits = logits[0, target_token_indices, :].mean(dim=0)
    target_score = target_logits.max()
    target_score.backward(retain_graph=False)

    for h in handles:
        h.remove()
    if was_training:
        hf_vlm.train()

    most_common_token = Counter(input_ids).most_common(1)[0][0]
    image_token_indices = [i for i, tid in enumerate(input_ids) if tid == most_common_token]
    num_crops = inputs["pixel_values"].shape[1]
    tokens_per_crop = len(image_token_indices) // num_crops

    cam_grids = {}
    for idx in layer_indices:
        if idx not in gradients or idx not in activations:
            continue
        grad = gradients[idx].float()
        act = activations[idx].float()

        grad_img = grad[0, image_token_indices, :]  # [n_image_tokens, hidden]
        act_img = act[0, image_token_indices, :]

        grad_global = grad_img[:tokens_per_crop]
        act_global = act_img[:tokens_per_crop]

        weights = grad_global.mean(dim=0, keepdim=True)  # [1, hidden]
        cam = (weights * act_global).sum(dim=-1)  # [tokens_per_crop]
        cam = torch.relu(cam).detach().cpu().numpy()

        grid_size = int(np.sqrt(tokens_per_crop))
        grid = cam.reshape(grid_size, grid_size)
        cam_grids[idx] = _normalize(grid)

    return cam_grids, inputs, layer_indices, processor


def grad_cam_vision_encoder_multi_layer(
    model, pil_image, task_description, target_word, device="cuda", layer_indices=None
):
    """
    Grad-CAM at multiple vision encoder layers.
    Earlier layers preserve more spatial detail; later layers are more semantic.
    Returns dict mapping layer_idx → 2D cam grid.
    """
    hf_vlm = model.model.vlm_with_expert.vlm
    vision_model = hf_vlm.model.vision_model
    total_layers = len(vision_model.encoder.layers)

    if layer_indices is None:
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
        layer_indices = sorted(set(max(0, int(total_layers * p) - 1) for p in percentiles))
    layer_indices = [i for i in layer_indices if 0 <= i < total_layers]
    print(f"Vision encoder has {total_layers} layers. Hooking: {layer_indices}")

    activations = {}
    gradients_store = {}

    def make_fwd(idx):
        def hook(module, inp, out):
            activations[idx] = out[0]

        return hook

    def make_bwd(idx):
        def hook(module, grad_in, grad_out):
            gradients_store[idx] = grad_out[0]

        return hook

    handles = []
    for idx in layer_indices:
        layer = vision_model.encoder.layers[idx]
        handles.append(layer.register_forward_hook(make_fwd(idx)))
        handles.append(layer.register_full_backward_hook(make_bwd(idx)))

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)

    vision_dtype = next(vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)

    was_training = hf_vlm.training
    hf_vlm.eval()
    inputs["pixel_values"].requires_grad_(True)

    outputs = hf_vlm(**inputs)
    logits = outputs.logits

    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    target_indices = _find_token_indices(tokens, target_word)
    if not target_indices:
        print(f"Could not find '{target_word}'. Using last token.")
        target_indices = [len(tokens) - 1]
    else:
        print(f"Target: '{target_word}' → {[tokens[i] for i in target_indices]}")

    target_logits = logits[0, target_indices, :].mean(dim=0)
    target_logits.max().backward(retain_graph=False)

    for h in handles:
        h.remove()
    if was_training:
        hf_vlm.train()

    cam_grids = {}
    for idx in layer_indices:
        if idx not in gradients_store or idx not in activations:
            continue
        grad = gradients_store[idx].float()
        act = activations[idx].float()
        weights = grad.mean(dim=1, keepdim=True)
        cam = (weights * act).sum(dim=-1)
        cam = torch.relu(cam).detach().cpu().numpy()
        cam_global = cam[0]
        grid_size = int(np.sqrt(cam_global.shape[0]))
        grid = cam_global[: grid_size * grid_size].reshape(grid_size, grid_size)
        cam_grids[idx] = _normalize(grid)

    return cam_grids, layer_indices


def occlusion_sensitivity(model, pil_image, task_description, target_word, device="cuda", grid_n=8):
    """
    Gradient-free occlusion sensitivity: mask each NxN region with gray,
    measure logit drop for target_word.  Regions that cause the largest drop
    are the most important.
    Returns a 2D importance grid (grid_n × grid_n), normalized 0-1.
    """
    hf_vlm = model.model.vlm_with_expert.vlm
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")

    def _get_logit(img):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)
        with torch.no_grad():
            logits = hf_vlm(**inputs).logits
        input_ids = inputs["input_ids"][0].tolist()
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
        idx_list = _find_token_indices(tokens, target_word)
        if not idx_list:
            idx_list = [len(tokens) - 1]
        return logits[0, idx_list, :].mean(dim=0).max().item()

    w, h = pil_image.size
    cell_w, cell_h = w // grid_n, h // grid_n

    print(f"  Baseline forward pass ...")
    baseline = _get_logit(pil_image)
    print(f"  Baseline logit = {baseline:.4f}")

    importance = np.zeros((grid_n, grid_n), dtype=np.float32)
    img_array = np.array(pil_image)
    gray_val = img_array.mean(axis=(0, 1)).astype(np.uint8)

    total = grid_n * grid_n
    for ri in range(grid_n):
        for ci in range(grid_n):
            masked = img_array.copy()
            y0, y1 = ri * cell_h, min((ri + 1) * cell_h, h)
            x0, x1 = ci * cell_w, min((ci + 1) * cell_w, w)
            masked[y0:y1, x0:x1] = gray_val
            masked_img = Image.fromarray(masked)
            score = _get_logit(masked_img)
            importance[ri, ci] = max(baseline - score, 0)
            done = ri * grid_n + ci + 1
            if done % 8 == 0 or done == total:
                print(f"  Occlusion: {done}/{total} patches", end="\r")

    print()
    return _normalize(importance)


def visualize_occlusion_multi_word(
    model, pil_image, description, words, device="cuda", grid_n=8, sigma=2.0, cmap="inferno"
):
    """
    Occlusion sensitivity for each word, displayed in a single grid.
    """
    panels = ["original"] + list(words)
    n = len(panels)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for flat_idx, panel in enumerate(panels):
        r, c = divmod(flat_idx, cols)
        ax = axes[r, c]
        if panel == "original":
            ax.imshow(pil_image)
            ax.set_title("Original", fontsize=12, fontweight="bold")
        else:
            print(f'  Occlusion map for "{panel}" ...')
            occ_grid = occlusion_sensitivity(model, pil_image, description, panel, device, grid_n=grid_n)
            smooth = _smooth_and_resize(occ_grid, img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            _add_contour(ax, smooth, extent)
            ax.set_title(f'"{panel}"', fontsize=12, fontweight="bold")
        ax.axis("off")

    for flat_idx in range(n, rows * cols):
        r, c = divmod(flat_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("Occlusion Sensitivity per word (gradient-free)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_vision_encoder_multi_layer(cam_grids, pil_image, layer_indices, target_word, sigma=1.5, cmap="inferno"):
    """
    Grad-CAM across vision-encoder layers: early layers keep spatial detail,
    later layers are more abstract.
    """
    valid = [i for i in layer_indices if i in cam_grids]
    panels = ["original"] + valid
    n = len(panels)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
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
        else:
            smooth = _smooth_and_resize(cam_grids[panel], img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            _add_contour(ax, smooth, extent)
            ax.set_title(f"ViT Layer {panel}", fontsize=11)
        ax.axis("off")

    for flat_idx in range(n, rows * cols):
        r, c = divmod(flat_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f'Vision Encoder Grad-CAM for "{target_word}" (early → late)', fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def _prepare_model_inputs(model, pil_image, task_description, device):
    hf_vlm = model.model.vlm_with_expert.vlm
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task_description}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
    vision_dtype = next(hf_vlm.model.vision_model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(vision_dtype)
    return hf_vlm, processor, inputs


def contrastive_grad_cam(model, pil_image, task_description,
                         target_word, foil_word, device="cuda", layer_idx=-1):
    """
    Contrastive Grad-CAM: backprop from (logit_target − logit_foil).
    Highlights regions uniquely important for target_word *versus* foil_word,
    suppressing shared background activation.
    """
    hf_vlm = model.model.vlm_with_expert.vlm
    vision_model = hf_vlm.model.vision_model
    target_layer = vision_model.encoder.layers[layer_idx]

    activations, gradients_s = {}, {}

    def fwd_hook(module, inp, out):
        activations["v"] = out[0]

    def bwd_hook(module, grad_in, grad_out):
        gradients_s["v"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    _, processor, inputs = _prepare_model_inputs(model, pil_image, task_description, device)
    was_training = hf_vlm.training
    hf_vlm.eval()
    inputs["pixel_values"].requires_grad_(True)

    outputs = hf_vlm(**inputs)
    logits = outputs.logits

    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    tgt_idx = _find_token_indices(tokens, target_word) or [len(tokens) - 1]
    foil_idx = _find_token_indices(tokens, foil_word) or [len(tokens) - 1]

    tgt_logit = logits[0, tgt_idx, :].mean(dim=0).max()
    foil_logit = logits[0, foil_idx, :].mean(dim=0).max()
    contrastive_score = tgt_logit - foil_logit
    print(f"  Contrastive: logit({target_word})={tgt_logit.item():.2f}  "
          f"logit({foil_word})={foil_logit.item():.2f}  Δ={contrastive_score.item():.2f}")
    contrastive_score.backward(retain_graph=False)

    h1.remove()
    h2.remove()
    if was_training:
        hf_vlm.train()

    grad = gradients_s["v"].float()
    act = activations["v"].float()
    weights = grad.mean(dim=1, keepdim=True)
    cam = (weights * act).sum(dim=-1)
    cam = torch.relu(cam).detach().cpu().numpy()
    cam_global = cam[0]
    gs = int(np.sqrt(cam_global.shape[0]))
    grid = cam_global[:gs * gs].reshape(gs, gs)
    return _normalize(grid)


def input_gradient_saliency(model, pil_image, task_description, target_word, device="cuda"):
    """
    Pixel-level saliency: ∂(logit_target) / ∂(input_pixels).
    Much sharper than Grad-CAM because it operates at full image resolution.
    Returns a 2D saliency map at the original image size.
    """
    hf_vlm, processor, inputs = _prepare_model_inputs(model, pil_image, task_description, device)
    was_training = hf_vlm.training
    hf_vlm.eval()

    pv = inputs["pixel_values"]  # [1, num_crops, C, H, W]
    pv.requires_grad_(True)

    outputs = hf_vlm(**inputs)
    logits = outputs.logits

    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    tgt_idx = _find_token_indices(tokens, target_word) or [len(tokens) - 1]

    target_logit = logits[0, tgt_idx, :].mean(dim=0).max()
    target_logit.backward(retain_graph=False)

    if was_training:
        hf_vlm.train()

    grad = pv.grad[0, 0].float()  # first crop: [C, H, W]
    saliency = grad.abs().mean(dim=0).cpu().numpy()  # [H, W]
    return _normalize(saliency)


def smoothgrad_saliency(model, pil_image, task_description, target_word,
                        device="cuda", n_samples=20, noise_std=0.15):
    """
    SmoothGrad: average pixel saliency over n_samples noisy copies.
    Dramatically reduces gradient noise and produces cleaner maps.
    """
    hf_vlm, processor, inputs = _prepare_model_inputs(model, pil_image, task_description, device)
    was_training = hf_vlm.training
    hf_vlm.eval()

    pv_clean = inputs["pixel_values"].clone()  # [1, num_crops, C, H, W]
    _, nc, C, H, W = pv_clean.shape
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    tgt_idx = _find_token_indices(tokens, target_word) or [len(tokens) - 1]

    accum = torch.zeros(H, W, device="cpu")
    stdev = noise_std * (pv_clean.max() - pv_clean.min()).item()

    for i in range(n_samples):
        noise = torch.randn_like(pv_clean) * stdev
        inputs["pixel_values"] = (pv_clean + noise).requires_grad_(True)
        outputs = hf_vlm(**inputs)
        logits = outputs.logits
        score = logits[0, tgt_idx, :].mean(dim=0).max()
        score.backward(retain_graph=False)
        grad = inputs["pixel_values"].grad[0, 0].float()  # [C, H, W]
        accum += grad.abs().mean(dim=0).cpu()
        hf_vlm.zero_grad(set_to_none=True)
        if (i + 1) % 5 == 0:
            print(f"  SmoothGrad: {i + 1}/{n_samples}", end="\r")

    if was_training:
        hf_vlm.train()

    print()
    avg = accum / n_samples
    return _normalize(avg.numpy())


def visualize_contrastive_grid(model, pil_image, description, word_pairs,
                               device="cuda", sigma=1.5, cmap="inferno"):
    """
    Contrastive Grad-CAM for multiple (target, foil) pairs.
    word_pairs: list of (target, foil) tuples.
    """
    panels = ["original"] + [f'"{t}" vs "{f}"' for t, f in word_pairs]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        if panel == "original":
            ax.imshow(pil_image)
            ax.set_title("Original", fontsize=12, fontweight="bold")
        else:
            t, f = word_pairs[idx - 1]
            cam = contrastive_grad_cam(model, pil_image, description, t, f, device)
            smooth = _smooth_and_resize(cam, img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            _add_contour(ax, smooth, extent)
            ax.set_title(f'"{t}" vs "{f}"', fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Contrastive Grad-CAM", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_saliency_comparison(model, pil_image, description, words,
                                  device="cuda", n_smooth=20, sigma=1.0, cmap="inferno"):
    """
    Side-by-side: Input×Gradient vs SmoothGrad vs Grad-CAM for each word.
    """
    methods = ["Input Gradient", "SmoothGrad", "Grad-CAM"]
    rows = len(words)
    cols = len(methods) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for ri, word in enumerate(words):
        axes[ri, 0].imshow(pil_image)
        axes[ri, 0].set_title(f'"{word}" — Original', fontsize=11, fontweight="bold")
        axes[ri, 0].axis("off")

        print(f'\n--- "{word}" ---')
        print(f"  Input Gradient ...")
        ig = input_gradient_saliency(model, pil_image, description, word, device)
        ig_smooth = _smooth_and_resize(ig, img_size, sigma=sigma)
        axes[ri, 1].imshow(pil_image)
        axes[ri, 1].imshow(ig_smooth, cmap=cmap, alpha=0.6, extent=extent)
        _add_contour(axes[ri, 1], ig_smooth, extent)
        axes[ri, 1].set_title("Input Gradient", fontsize=11)
        axes[ri, 1].axis("off")

        print(f"  SmoothGrad ({n_smooth} samples) ...")
        sg = smoothgrad_saliency(model, pil_image, description, word, device, n_samples=n_smooth)
        sg_smooth = _smooth_and_resize(sg, img_size, sigma=sigma)
        axes[ri, 2].imshow(pil_image)
        axes[ri, 2].imshow(sg_smooth, cmap=cmap, alpha=0.6, extent=extent)
        _add_contour(axes[ri, 2], sg_smooth, extent)
        axes[ri, 2].set_title("SmoothGrad", fontsize=11)
        axes[ri, 2].axis("off")

        print(f"  Grad-CAM ...")
        gc, _, _ = grad_cam_vision_encoder(model, pil_image, description, word, device)
        gc_smooth = _smooth_and_resize(gc, img_size, sigma=1.5)
        axes[ri, 3].imshow(pil_image)
        axes[ri, 3].imshow(gc_smooth, cmap=cmap, alpha=0.6, extent=extent)
        _add_contour(axes[ri, 3], gc_smooth, extent)
        axes[ri, 3].set_title("Grad-CAM", fontsize=11)
        axes[ri, 3].axis("off")

    fig.suptitle("Saliency Method Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_grad_cam(cam_grid, pil_image, title="Grad-CAM", sigma=1.5, cmap="inferno"):
    """
    Single Grad-CAM heatmap overlaid on the original image.
    """
    img_size = (pil_image.width, pil_image.height)
    smooth = _smooth_and_resize(cam_grid, img_size, sigma=sigma)
    extent = [0, pil_image.width, pil_image.height, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pil_image)
    axes[1].imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
    _add_contour(axes[1], smooth, extent)
    axes[1].set_title(title, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_grad_cam_multi_layer(cam_grids_dict, pil_image, layer_indices, target_word,
                                   sigma=1.5, cmap="inferno"):
    """
    Grad-CAM heatmaps across multiple LLM layers in a grid layout.
    """
    valid_layers = [li for li in layer_indices if li in cam_grids_dict]
    panels = ["original"] + valid_layers
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
        else:
            smooth = _smooth_and_resize(cam_grids_dict[panel], img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            _add_contour(ax, smooth, extent)
            ax.set_title(f"Layer {panel}", fontsize=11)
        ax.axis("off")

    for flat_idx in range(n_panels, rows * cols):
        r, c = divmod(flat_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f'Grad-CAM for "{target_word}"', fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_multi_word_grad_cam(model, pil_image, description, words, device="cuda", sigma=1.5, cmap="inferno"):
    """
    Run vision-encoder Grad-CAM for each word and show them in one grid.
    First panel is the original image, then one panel per word.
    """
    panels = ["original"] + list(words)
    n = len(panels)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    for flat_idx, panel in enumerate(panels):
        r, c = divmod(flat_idx, cols)
        ax = axes[r, c]

        if panel == "original":
            ax.imshow(pil_image)
            ax.set_title("Original", fontsize=12, fontweight="bold")
        else:
            print(f'  Computing Grad-CAM for "{panel}" ...')
            cam_grid, _, _ = grad_cam_vision_encoder(model, pil_image, description, panel, device)
            smooth = _smooth_and_resize(cam_grid, img_size, sigma=sigma)
            ax.imshow(pil_image)
            ax.imshow(smooth, cmap=cmap, alpha=0.6, extent=extent)
            _add_contour(ax, smooth, extent)
            ax.set_title(f'"{panel}"', fontsize=12, fontweight="bold")
        ax.axis("off")

    for flat_idx in range(n, rows * cols):
        r, c = divmod(flat_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("Grad-CAM per word", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_comparison(attention_grid, gradcam_grid, pil_image, target_word,
                         sigma=1.5, cmap="inferno"):
    """
    Side-by-side comparison: attention-based vs Grad-CAM saliency.
    """
    img_size = (pil_image.width, pil_image.height)
    extent = [0, pil_image.width, pil_image.height, 0]

    att_smooth = _smooth_and_resize(attention_grid, img_size, sigma=sigma)
    gc_smooth = _smooth_and_resize(gradcam_grid, img_size, sigma=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pil_image)
    axes[1].imshow(att_smooth, cmap=cmap, alpha=0.6, extent=extent)
    _add_contour(axes[1], att_smooth, extent)
    axes[1].set_title("Attention Rollout", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pil_image)
    axes[2].imshow(gc_smooth, cmap=cmap, alpha=0.6, extent=extent)
    _add_contour(axes[2], gc_smooth, extent)
    axes[2].set_title("Grad-CAM", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f'Attention vs Grad-CAM for "{target_word}"', fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def download_sample_image(url, save_path):
    import urllib.request

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        print(f"Downloading sample image to {save_path} ...")
        urllib.request.urlretrieve(url, save_path)
    return Image.open(save_path).convert("RGB")


if __name__ == "__main__":
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    checkpoint = "HuggingFaceVLA/smolvla_libero"
    print(f"Loading model: {checkpoint}")
    model, model_id, action_dim = smolvla(checkpoint, device)

    dog_cat_path = PROJECT_ROOT / "data/images/both.png"
    if dog_cat_path.exists():
        print("\n=== Dog & Cat (both.png) ===")
        dog_cat_image = Image.open(dog_cat_path).convert("RGB")
        description = "A dog and a cat standing together indoors"
        words = ["dog", "cat", "standing", "indoors"]
        print(f"Description: {description}")
        print(f"Words: {words}")
        visualize_multi_word_grad_cam(model, dog_cat_image, description, words, device)
    else:
        sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
        sample_path = PROJECT_ROOT / "data/images/samples/dog_grass.jpg"
        image = download_sample_image(sample_url, sample_path)
        description = "A dog sitting on the grass in a garden"
        words = ["dog", "grass", "garden", "sitting"]
        print(f"\nDescription: {description}")
        print(f"Words: {words}")
        visualize_multi_word_grad_cam(model, image, description, words, device)
