"""Device detection and synchronisation helpers for GPU/TPU/CPU."""

import torch


def get_device():
    """Return the best available device: TPU > CUDA > CPU."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except (ImportError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device):
    """Synchronise the device (barrier)."""
    if str(device).startswith("xla"):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device):
    """Free device memory cache if applicable."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # XLA manages memory automatically -- no-op on TPU/CPU
