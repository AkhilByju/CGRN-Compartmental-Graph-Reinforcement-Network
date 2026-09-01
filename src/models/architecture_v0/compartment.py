"""Compartment computation for CellV0 -- NOT YET IMPLEMENTED.

The generic compartment function `d_i,b(t) = f_b(local input, current cell
state, incoming messages, possibly compartment state)` is undefined. See
docs/architecture_v0.md Sec 1 and Sec 7 items 2-4 (compartment inputs,
computation, parameterization). Do not implement until `f_b` is specified.
"""

from __future__ import annotations

import torch


class Compartment(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Compartment computation (f_b) is not yet specified. See "
            "docs/architecture_v0.md Sec 1 and Sec 7 before implementing."
        )
