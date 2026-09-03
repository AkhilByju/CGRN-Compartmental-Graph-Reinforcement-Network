"""Final `N`-cell population -> fixed-size task output.

**Implementation choice, not a user-specified formula** -- this is
`docs/architecture_v1.md` §6 open question 9 ("Decode step"), which the
2026-09-02 design conversation didn't resolve: a variable, input-dependent
set of cell states has no obvious canonical readout the way a fixed
`hidden_cells -> Linear -> out_features` readout does for CellV0's
`BeliefLayer` stack.

Default chosen: precision-weighted pooling (a cell's confidence, `evidence
/ (uncertainty^2 + eps)`, sets how much it contributes to the pooled
summary -- the same "trust confident cells more" principle the rest of the
architecture uses) over both `mu` and `z`, concatenated and passed through
one linear head. Not a designated fixed subset of "output cells" -- every
cell contributes, weighted by its own confidence.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1


class PopulationDecoder(nn.Module):
    def __init__(self, association_dim: int, out_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.readout = nn.Linear(1 + association_dim, out_features)

    def forward(self, cells: BeliefCellV1) -> torch.Tensor:
        """Returns `(batch, out_features)`."""
        precision = cells.evidence / (cells.uncertainty**2 + self.eps)
        weight = precision / (precision.sum(dim=-1, keepdim=True) + self.eps)

        pooled_mu = (weight * cells.mu).sum(dim=-1, keepdim=True)
        pooled_z = (weight.unsqueeze(-1) * cells.z).sum(dim=-2)

        combined = torch.cat([pooled_mu, pooled_z], dim=-1)
        return self.readout(combined)
