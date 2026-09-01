"""CellV0 -- NOT YET IMPLEMENTED.

The basic computational unit: persistent state `h_i(t)`, multiple
compartments `D_i1..D_iB`, an integration step, and a state-update rule. See
docs/architecture_v0.md Sec 1 and Sec 7. Do not implement until that
document specifies `f_b`, `S`, and `Update` concretely -- see CLAUDE.md
Sec 2.
"""

from __future__ import annotations

import torch


class CellV0(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "CellV0's mathematics are not yet specified. See docs/architecture_v0.md "
            "Sec 1 and Sec 7, and CLAUDE.md Sec 2, before implementing."
        )
