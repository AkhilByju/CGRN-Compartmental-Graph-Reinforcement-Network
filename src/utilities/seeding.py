"""Deterministic seeding across Python, NumPy, and PyTorch.

Every run must record its seed (docs/experiment_protocol.md) -- call
`set_seed` once at the start of any script that trains or generates data.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs this project touches. `deterministic=True` additionally
    disables cuDNN's non-deterministic algorithm selection, trading some
    speed for exact reproducibility -- appropriate for debug/small-scale
    runs, which is the only regime this project currently targets."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
