import random

import numpy as np
import torch

from src.utilities.seeding import set_seed


def test_set_seed_reproducible() -> None:
    set_seed(123)
    a = (random.random(), float(np.random.rand()), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert a == b
