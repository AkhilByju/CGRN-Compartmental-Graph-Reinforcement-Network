"""Raw input features -> initial `N`-cell population.

**Implementation choice, not a user-specified formula** -- this is
`docs/architecture_v1.md` §6 open question 10 ("Initialization"), which
the 2026-09-02 design conversation didn't resolve. Any implementation
needs *something* here regardless of which choice is made (`n_cells` is
fixed while input dimensionality varies, so there's no dimension-preserving
"identity" encoding the way `BeliefCell.from_observed_features` used for
CellV0, where `in_features == in_cells` was assumed).

Default chosen: a learned linear projection from raw features to each
cell's initial `mu` and `z`, with `evidence = uncertainty = 1` for every
cell -- mirroring CellV0's own convention (`from_observed_features`) that
every cell starts equally uncertain, so any useful evidence/uncertainty
structure is learned, not hand-given.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1


class PopulationEncoder(nn.Module):
    def __init__(self, in_features: int, n_cells: int, association_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.association_dim = association_dim
        self.eps = eps
        self.mu_proj = nn.Linear(in_features, n_cells)
        self.z_proj = nn.Linear(in_features, n_cells * association_dim)

    def forward(self, x: torch.Tensor) -> BeliefCellV1:
        """`x`: `(batch, in_features)`. Returns a `BeliefCellV1` population
        of shape `(batch, n_cells)` / `(batch, n_cells, association_dim)`."""
        mu = torch.tanh(self.mu_proj(x))
        z_raw = self.z_proj(x).view(x.shape[0], self.n_cells, self.association_dim)
        z = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)
        evidence = torch.ones(x.shape[0], self.n_cells, dtype=x.dtype, device=x.device)
        uncertainty = torch.ones(x.shape[0], self.n_cells, dtype=x.dtype, device=x.device)
        return BeliefCellV1(mu=mu, evidence=evidence, uncertainty=uncertainty, z=z)
