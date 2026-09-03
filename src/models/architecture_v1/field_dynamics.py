"""One Self-Organizing Refinement Field step, and the `T`-step loop that
reuses it (docs/architecture_v1.md §12). CellV1's `(mu, e, u, z)` belief
math (`fusion.py`'s scale-stable precision) is unchanged; what's new is
*how a cell finds and weighs its neighbors* -- a continuous, learned
density field (`field_fusion.py`) instead of discrete graph routing
(`routing.py`/`sparse_routing.py`), using the self-anchored orthogonal
random-Fourier-features approximation validated in
`docs/research_log.md`'s CellV1.3 diagnostic.

`FieldCellState` extends `BeliefCellV1` with a persistent routing
position `r`: `docs/architecture_v1.md`'s role split is "`z` = what the
cell semantically represents (persists, updated by a learned gate),
`r` = where the cell currently sits in the self-organizing field
(persists, updated by a *gated mean-shift step* -- moves toward the
local density mode, but only as much as the cell's own gate allows)."
Unlike `z`, `r` has no analogue in `BeliefCellV1` itself -- it's specific
to how this variant finds structure, not part of the belief being
carried -- hence the wrapper rather than extending `BeliefCellV1`
(`cell.py`) itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.field_functions import (
    BandwidthFunction,
    FieldSemanticUpdateFunction,
    MassFunction,
    RoutingGateFunction,
)
from src.models.architecture_v1.field_fusion import local_field_fusion
from src.models.architecture_v1.random_features import RandomFourierFeatures
from src.models.architecture_v1.shared_functions import WriteGateFunction


@dataclass(frozen=True)
class FieldCellState:
    """`cells`: the usual `BeliefCellV1` `(mu, e, u, z)`. `r`: `(batch,
    n_cells, routing_dim)`, this variant's persistent routing position."""

    cells: BeliefCellV1
    r: torch.Tensor

    def detach(self) -> FieldCellState:
        return FieldCellState(cells=self.cells.detach(), r=self.r.detach())


class FieldRefinementStep(nn.Module):
    """One refinement step: compute this step's bandwidth/mass, evaluate
    the local density field (self-anchored ORFF, `field_fusion.py`),
    gated-write `(mu, e, u)` toward the field's fused proposal, gated
    mean-shift `r` toward the field's density mode, and gated-update `z`
    from `(r, r_bar)` as spatial context.
    """

    def __init__(
        self,
        n_cells: int,
        association_dim: int,
        routing_dim: int,
        num_features: int = 256,
        hidden_dim: int = 32,
        h_min: float = 0.1,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps

        self.bandwidth_fn = BandwidthFunction(association_dim, hidden_dim=hidden_dim, h_min=h_min)
        self.mass_fn = MassFunction(association_dim, hidden_dim=hidden_dim)
        self.metric = nn.Linear(routing_dim, routing_dim, bias=False)

        self.psi_features = RandomFourierFeatures(
            routing_dim, num_features, sigma=1.0, orthogonal=True, generator=generator
        )
        self.psi_sq_features = RandomFourierFeatures(
            routing_dim, num_features, sigma=1.0 / math.sqrt(2.0), orthogonal=True, generator=generator
        )

        self.fuse_bias = nn.Parameter(torch.zeros(n_cells))
        # Reused as-is: (mu, e, u, mu_field, e_field, u_field) -> 4 gates is
        # exactly WriteGateFunction(use_global=False)'s existing interface --
        # "mu_local/e_local/u_local" there and "mu_field/e_field/u_field"
        # here are the same shape playing the same role (a fused proposal to
        # blend the old state against).
        self.write_gate = WriteGateFunction(use_global=False, hidden_dim=hidden_dim, init_bias=gate_init_bias)
        self.routing_gate = RoutingGateFunction(hidden_dim=hidden_dim)
        self.semantic_update = FieldSemanticUpdateFunction(association_dim, routing_dim, hidden_dim=hidden_dim)

    def forward(self, state: FieldCellState) -> FieldCellState:
        mu, evidence, uncertainty, z = state.cells.mu, state.cells.evidence, state.cells.uncertainty, state.cells.z
        r = state.r

        h = self.bandwidth_fn(mu, evidence, uncertainty, z)  # (batch, n)
        m = self.mass_fn(mu, evidence, uncertainty, z)  # (batch, n)

        x_scaled = self.metric(r) / h.sqrt().unsqueeze(-1)  # bandwidth-rescaled coords for kernel evaluation
        psi = self.psi_features(x_scaled)
        psi_sq = self.psi_sq_features(x_scaled)

        mu_field, e_field, u_field, r_bar = local_field_fusion(
            mu, evidence, uncertainty, r, m, psi, psi_sq, self.fuse_bias, self.eps
        )

        beta_mu, beta_e, beta_u, beta_z = self.write_gate(mu, evidence, uncertainty, mu_field, e_field, u_field)
        mu_next = (1 - beta_mu) * mu + beta_mu * mu_field
        evidence_next = (1 - beta_e) * evidence + beta_e * e_field
        uncertainty_next = (1 - beta_u) * uncertainty + beta_u * u_field

        beta_r = self.routing_gate(mu, evidence, uncertainty, mu_field, e_field, u_field)
        r_next = r + beta_r.unsqueeze(-1) * (r_bar - r)

        z_delta = self.semantic_update(z, r, r_bar, mu_next, evidence_next, uncertainty_next)
        z_raw = z + beta_z.unsqueeze(-1) * z_delta
        z_next = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)

        next_cells = BeliefCellV1(mu=mu_next, evidence=evidence_next, uncertainty=uncertainty_next, z=z_next)
        return FieldCellState(cells=next_cells, r=r_next)


class FieldRefinementCore(nn.Module):
    """Applies one shared `FieldRefinementStep` `num_steps` times (default
    `1` -- "one refinement step initially," per the user's spec; the same
    shared-parameters-across-iterations convention as `dynamics.py`'s
    `DynamicBeliefGraphCore` once `num_steps > 1` is tried)."""

    def __init__(
        self, n_cells: int, association_dim: int, routing_dim: int, num_steps: int = 1, **step_kwargs
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.step = FieldRefinementStep(n_cells, association_dim, routing_dim, **step_kwargs)

    def forward(self, state: FieldCellState) -> FieldCellState:
        for _ in range(self.num_steps):
            state = self.step(state)
        return state
