"""Demo script for CLIP action model."""

import typer

from diffusion_policy import CLIPActionConfig, create_clip_action_model

app = typer.Typer()


@app.command()
def demo(
    action_dim: int = typer.Option(7, "--action-dim", "-a", help="Action dimension"),
    clip_model: str = typer.Option("ViT-B/32", "--clip-model", "-m", help="CLIP model variant"),
    use_history: bool = typer.Option(False, "--use-history", "-h", help="Use observation history"),
    history_length: int = typer.Option(4, "--history-length", "-l", help="History length"),
) -> None:
    """Demo the CLIP action model with random inputs."""
    import torch

    print("Creating CLIP action model...")
    print(f"  clip_model={clip_model}")
    print(f"  action_dim={action_dim}")
    print(f"  use_history={use_history}")

    model = create_clip_action_model(
        action_dim=action_dim,
        clip_model=clip_model,
        use_history=use_history,
        history_length=history_length,
        freeze_clip=True,
    )
    print(f"Model created on device: {model.device}")

    batch_size = 2
    if use_history:
        images = torch.randn(batch_size, history_length, 3, 224, 224).to(model.device)
    else:
        images = torch.randn(batch_size, 3, 224, 224).to(model.device)

    text = ["pick up the red cube", "move to the target"]

    print("\nRunning forward pass...")
    print(f"  Input shape: {images.shape}")

    with torch.no_grad():
        actions = model(images, text=text)

    print(f"  Output shape: {actions.shape}")
    print(f"  Actions:\n{actions}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")


@app.command()
def info() -> None:
    """Show available CLIP models and configuration options."""
    import clip

    print("Available CLIP models:")
    for model_name in clip.available_models():
        print(f"  - {model_name}")

    print("\nConfiguration options (CLIPActionConfig):")
    for field_name, field in CLIPActionConfig.__dataclass_fields__.items():
        default = field.default
        print(f"  {field_name}: {field.type} = {default}")


if __name__ == "__main__":
    app()
