"""`ObjectSeededEncoder` -- a structured alternative to
`encoder.py::PopulationEncoder`, for `experiments/v1_001_dynamic_groups`.

`PopulationEncoder` projects the *entire* raw input to every one of the
`n_cells` cells via one dense linear layer -- every cell can see the whole
input immediately. For an experiment specifically about whether CellV1
discovers local structure before propagating it globally
(docs/architecture_v1.md §9), that's a confound: a dense global-broadcast
adapter can let the model solve the task without ever needing local
organization, partially bypassing the thing being studied. Not deleting
`PopulationEncoder` -- it's still the right default for unstructured,
fixed-length feature vectors (R2/C2/U2) -- but
`experiments/v1_001_dynamic_groups`'s `dynamic_groups` task has an actual
per-object structure to exploit instead: each input is `n_objects`
`(k1, k2, v)` triples (`src.data.synthetic.dynamic_groups`), so each of the
first `n_objects` cells is seeded directly from one object, and only those
cells' own local/global dynamics can propagate information to the rest --
nothing else has global visibility from the start.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1


class ObjectSeededEncoder(nn.Module):
    """`x`: `(batch, n_objects * 3)`, flattened `[k1, k2, v]` per object
    (`src.data.synthetic.dynamic_groups.dynamic_groups`'s flatten order).
    The first `n_objects` cells are seeded one-to-one from the objects;
    the remaining `n_cells - n_objects` cells start neutral (per the
    user's spec: "the other cells can start neutral... and potentially
    become recruited during iterative computation" -- no special
    recruitment machinery, just this initial condition feeding into the
    same dynamics every other cell uses).
    """

    def __init__(
        self,
        n_objects: int,
        n_cells: int,
        association_dim: int,
        semantic_dim: int = 2,
        filler_evidence: float = 0.1,
        filler_uncertainty: float = 5.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_cells < n_objects:
            raise ValueError(f"n_cells ({n_cells}) must be >= n_objects ({n_objects}).")
        self.n_objects = n_objects
        self.n_cells = n_cells
        self.filler_evidence = filler_evidence
        self.filler_uncertainty = filler_uncertainty
        self.eps = eps

        self.seed_projection = nn.Linear(semantic_dim, association_dim)  # F_seed
        # A learned, shared "I am unassigned" address for filler cells --
        # z can't be all-zero (L2-normalizing a zero vector is undefined),
        # and a learned embedding lets the network give "empty" a
        # consistent, distinguishable meaning rather than arbitrary noise.
        self.empty_z = nn.Parameter(torch.randn(association_dim))

    def forward(self, x: torch.Tensor) -> BeliefCellV1:
        batch = x.shape[0]
        objects = x.view(batch, self.n_objects, 3)
        semantic, value = objects[..., :2], objects[..., 2]

        z_seed = self.seed_projection(semantic)
        z_seed = z_seed / (z_seed.norm(p=2, dim=-1, keepdim=True) + self.eps)

        n_filler = self.n_cells - self.n_objects
        mu = torch.cat([value, value.new_zeros(batch, n_filler)], dim=-1)
        evidence = torch.cat(
            [value.new_ones(batch, self.n_objects), value.new_full((batch, n_filler), self.filler_evidence)],
            dim=-1,
        )
        uncertainty = torch.cat(
            [
                value.new_ones(batch, self.n_objects),
                value.new_full((batch, n_filler), self.filler_uncertainty),
            ],
            dim=-1,
        )

        empty_z = self.empty_z / (self.empty_z.norm(p=2) + self.eps)
        z_filler = empty_z.view(1, 1, -1).expand(batch, n_filler, -1)
        z = torch.cat([z_seed, z_filler], dim=1)

        return BeliefCellV1(mu=mu, evidence=evidence, uncertainty=uncertainty, z=z)
