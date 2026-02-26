import torch


def get_device(device: str = "cuda") -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return dev
