"""Iterative refinement dynamics -- NOT YET IMPLEMENTED.

The repeated, shared-parameter update `Z_0 -> Z_1 -> ... -> Z_T` over the
whole system state (docs/architecture_v0.md Sec 2 and Sec 5). The number of
iterations `T` must be externally controlled (a config value), not learned,
until adaptive computation is explicitly taken up as a later extension.
"""

from __future__ import annotations

import torch


class IterativeRefinement(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Iterative refinement dynamics are not yet specified, and depend on "
            "CellV0/Cluster/Graph being specified first. See docs/architecture_v0.md "
            "Sec 2 and Sec 5."
        )
