"""`SelfOrganizingRefinementField` -- the assembled Self-Organizing
Refinement Field model: encoder -> initial routing position -> `T`
gated-mean-shift/field-fusion steps -> decoder (docs/architecture_v1.md
§12). Reuses the existing `PopulationEncoder`/`ObjectSeededEncoder` and
`PopulationDecoder` unchanged (they only ever touched `(mu, e, u, z)`,
never CellV1.1's routing/graph internals, so nothing about them needed to
change for the field to replace discrete routing) -- only wraps the
encoder to also produce the field's initial persistent routing position
`r_0 = F_route(mu_0, e_0, u_0, z_0)`.

`use_global=False` (default): local field only. `use_global=True`: adds
the linear-attention global communication field on top
(`field_dynamics.py`, `global_field.py`) -- same encoder/decoder either
way, since global communication only changes what happens *inside* each
`FieldRefinementStep`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.decoder import PopulationDecoder
from src.models.architecture_v1.encoder import PopulationEncoder
from src.models.architecture_v1.field_dynamics import FieldCellState, FieldRefinementCore
from src.models.architecture_v1.field_functions import RoutingFunction


class FieldEncoder(nn.Module):
    """Wraps any `(x) -> BeliefCellV1` encoder (`PopulationEncoder`,
    `object_encoder.ObjectSeededEncoder`) and additionally computes the
    field's initial routing position via `RoutingFunction` -- the same
    function `field_dynamics.py` has no further use for after this (later
    steps evolve `r` through the mean-shift update, not by recomputing
    `F_route` from scratch each time)."""

    def __init__(self, base_encoder: nn.Module, association_dim: int, routing_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.routing_fn = RoutingFunction(association_dim, routing_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> FieldCellState:
        cells = self.base_encoder(x)
        r0 = self.routing_fn(cells.mu, cells.evidence, cells.uncertainty, cells.z)
        return FieldCellState(cells=cells, r=r0)


class SelfOrganizingRefinementField(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cells: int = 128,
        association_dim: int = 8,
        routing_dim: int = 8,
        num_features: int = 256,
        hidden_dim: int = 32,
        num_steps: int = 1,
        h_min: float = 0.1,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
        encoder: nn.Module | None = None,
        use_global: bool = False,
        global_dim: int = 16,
    ) -> None:
        super().__init__()
        base_encoder = encoder if encoder is not None else PopulationEncoder(
            in_features, n_cells, association_dim, eps=eps
        )
        self.encoder = FieldEncoder(base_encoder, association_dim, routing_dim, hidden_dim=hidden_dim)
        self.core = FieldRefinementCore(
            n_cells,
            association_dim,
            routing_dim,
            num_steps=num_steps,
            num_features=num_features,
            hidden_dim=hidden_dim,
            h_min=h_min,
            eps=eps,
            gate_init_bias=gate_init_bias,
            use_global=use_global,
            global_dim=global_dim,
        )
        self.decoder = PopulationDecoder(association_dim, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(x)[0]

    def forward_with_state(self, x: torch.Tensor) -> tuple[torch.Tensor, FieldCellState]:
        state = self.encoder(x)
        state = self.core(state)
        prediction = self.decoder(state.cells)
        return prediction, state
