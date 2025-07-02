"""Simplified performance utilities."""

import gc
import torch


def cleanup_memory(force: bool = False) -> None:
    """Clear CUDA/MPS caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if force:
        gc.collect()


def get_memory_info() -> dict:
    """Return basic memory usage information."""
    info = {"cpu": {}, "gpu": {}}
    if torch.cuda.is_available():
        info["gpu"]["allocated"] = torch.cuda.memory_allocated()
        info["gpu"]["reserved"] = torch.cuda.memory_reserved()
    return info
