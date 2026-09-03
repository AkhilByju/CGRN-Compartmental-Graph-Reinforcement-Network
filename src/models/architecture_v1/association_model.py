"""`LearnedAssociationField` -- the assembled Learned Association Field
model: encoder -> `T` association-fusion/(optional global)/gated-write
steps -> decoder. Reuses `PopulationEncoder`/`ObjectSeededEncoder` and
`PopulationDecoder` directly, unwrapped -- unlike `field_model.py`'s
`FieldEncoder` (which additionally had to compute the field's initial
routing position `r_0`), this design has no persistent coordinate beyond
plain `BeliefCellV1`, so there's nothing for an encoder wrapper to do.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.association_dynamics import AssociationRefinementCore
from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.decoder import PopulationDecoder
from src.models.architecture_v1.encoder import PopulationEncoder


class LearnedAssociationField(nn.Module):
    """`use_global=False` (default): local association field only.
    `use_global=True`: adds the (frozen, reused-unmodified) linear
    global communication field on top, exactly as
    `field_model.py::SelfOrganizingRefinementField` does."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cells: int = 128,
        association_dim: int = 8,
        assoc_dim: int = 32,
        hidden_dim: int = 32,
        num_steps: int = 1,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
        encoder: nn.Module | None = None,
        use_global: bool = False,
        global_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PopulationEncoder(
            in_features, n_cells, association_dim, eps=eps
        )
        self.core = AssociationRefinementCore(
            n_cells,
            association_dim,
            num_steps=num_steps,
            assoc_dim=assoc_dim,
            hidden_dim=hidden_dim,
            eps=eps,
            gate_init_bias=gate_init_bias,
            use_global=use_global,
            global_dim=global_dim,
        )
        self.decoder = PopulationDecoder(association_dim, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(x)[0]

    def forward_with_state(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1]:
        cells = self.encoder(x)
        cells = self.core(cells)
        prediction = self.decoder(cells)
        return prediction, cells
