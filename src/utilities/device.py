"""Device selection for the project's mixed CPU/MPS/CUDA development targets.

Primary development hardware is Apple Silicon (M4-class); the MPS backend is
preferred there, with CUDA and CPU as fallbacks (docs/experiment_protocol.md
"Framework and hardware").
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or honor an explicit override
    (e.g. `prefer="cpu"` for debugging a numerics issue)."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
