"""Optional accelerator selection; importing this module never initializes XLA."""
from __future__ import annotations

import os


def resolve_device(device=None):
    import torch

    choice = str(device or os.environ.get("PROMPTCOMP_DEVICE", "auto"))
    if choice == "xla":
        try:
            import torch_xla
        except ImportError as error:
            raise RuntimeError("PROMPTCOMP_DEVICE=xla requires a matching torch_xla installation") from error
        return torch_xla.device()
    if choice == "auto":
        choice = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def step(optimizer, device):
    optimizer.step()
    if device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)
    elif device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()


def cpu_tree(value):
    """Move optimizer tensors to CPU before serializing an XLA checkpoint."""
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
