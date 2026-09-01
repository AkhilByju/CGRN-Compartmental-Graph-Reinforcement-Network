"""Compartment-integration mechanism for CellV0 -- NOT YET IMPLEMENTED.

The integration function `u_i(t) = S(d_i,1, ..., d_i,B)` combining
compartment outputs into a single update signal is undefined. See
docs/architecture_v0.md Sec 1 and Sec 7 item 5. Do not implement until `S`
is specified.
"""

from __future__ import annotations

import torch


class Integration(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "The integration mechanism (S) is not yet specified. See "
            "docs/architecture_v0.md Sec 1 and Sec 7 before implementing."
        )
