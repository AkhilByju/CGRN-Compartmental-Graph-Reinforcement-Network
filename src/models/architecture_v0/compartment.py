"""Intra-cell compartments -- NOT PART OF CellV0.

Earlier drafts of docs/architecture_v0.md envisioned each cell containing
multiple internal computational compartments (`D_i1..D_iB`). The decided
CellV0 design (`cell.py`'s `BeliefCell`) does not use compartments -- a
belief cell holds a single `(mu, evidence, uncertainty)` state, and any
richness comes from how cells combine across a layer/graph, not from
structure inside one cell. See docs/architecture_v0.md Sec 1 and
docs/research_log.md.

This stub is kept only in case a later, richer cell (beyond CellV0)
revisits intra-cell compartments. Do not implement it as part of V0.
"""

from __future__ import annotations

import torch


class Compartment(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Compartments are not part of CellV0 (see cell.py's BeliefCell). "
            "This stub is speculative for a possible later, richer cell -- do not "
            "implement it now."
        )
