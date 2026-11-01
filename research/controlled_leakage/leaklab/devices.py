"""Optional single-device TPU support for the independent leakage package."""
from __future__ import annotations

import os


def resolve_device(device=None):
    import torch

    choice = str(device or os.environ.get("LEAKLAB_DEVICE", "auto"))
    if choice == "xla":
        try:
            import torch_xla
        except ImportError as error:
            raise RuntimeError("LEAKLAB_DEVICE=xla requires a matching torch_xla installation") from error
        return torch_xla.device()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def step(optimizer, device):
    optimizer.step()
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def release_cache(device):
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()
    elif device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value
