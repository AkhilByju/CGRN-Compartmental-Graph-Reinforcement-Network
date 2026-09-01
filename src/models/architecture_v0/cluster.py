"""Cluster / local-circuit organization of cells -- NOT YET IMPLEMENTED.

Cells organized into local circuits with relatively rich intra-cluster
communication (docs/architecture_v0.md Sec 3). Gated behind CellV0 itself
being specified and validated first (Experiment 002/003 in
docs/experiment_protocol.md) -- do not build cluster structure before the
single-cell case is understood.
"""

from __future__ import annotations

import torch


class Cluster(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Cluster organization is not yet specified, and is gated behind CellV0 "
            "being validated first. See docs/architecture_v0.md Sec 3."
        )
