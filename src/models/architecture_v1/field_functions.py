"""The learned functions specific to the Self-Organizing Refinement Field
(docs/architecture_v1.md): where a cell sits in routing space, how broad
its attraction is, how much it weighs in the local density, and (for
global communication) its query/key identity and its stochastic
send/need gates. Small shared MLPs, matching this repo's existing
convention (`shared_functions.py`) for "one shared function, not a
per-cell or per-edge network."
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.hard_concrete import hard_concrete_gate


def _mlp(in_features: int, hidden_dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_features),
    )


class RoutingFunction(nn.Module):
    """`r_i = F_route(mu_i, e_i, u_i, z_i)` -- where this cell currently
    sits in the (learned) semantic space local attraction is computed in.
    Not `z` itself: `z` also carries the cell's own identity/history
    (`shared_functions.py::SemanticUpdateFunction`); `r` is a further,
    task-shaped projection of the full belief specifically for routing."""

    def __init__(self, association_dim: int, routing_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=routing_dim)

    def forward(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([mu.unsqueeze(-1), e.unsqueeze(-1), u.unsqueeze(-1), z], dim=-1)
        return self.net(features)


class BandwidthFunction(nn.Module):
    """`h_i = h_min + softplus(F_scale(B_i))` -- how broad this cell's
    conceptual neighborhood currently is. `softplus` keeps the learned
    part non-negative; `h_min` keeps `h_i` bounded away from 0 (needed
    since `field_fusion.py` divides by `sqrt(h_i)`)."""

    def __init__(self, association_dim: int, hidden_dim: int = 32, h_min: float = 0.1) -> None:
        super().__init__()
        self.h_min = h_min
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=1)

    def forward(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([mu.unsqueeze(-1), e.unsqueeze(-1), u.unsqueeze(-1), z], dim=-1)
        return self.h_min + nn.functional.softplus(self.net(features).squeeze(-1))


class MassFunction(nn.Module):
    """`m_i = softplus(F_mass(B_i))` -- how much this cell's current
    state should weigh in the local density field, independent of how
    close/far other cells consider it."""

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=1)

    def forward(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([mu.unsqueeze(-1), e.unsqueeze(-1), u.unsqueeze(-1), z], dim=-1)
        return nn.functional.softplus(self.net(features).squeeze(-1))


class GlobalQueryKeyFunction(nn.Module):
    """`q_i = F_query(B_i)` / `k_i = F_key(B_i)` -- separate learned
    projections (one shared `F_query`, one shared `F_key`) for
    content-based global matching, analogous to `routing.py::GlobalRouting`'s
    `W_Q`/`W_K` but taking the full `(mu, e, u, z)` state rather than `z`
    alone."""

    def __init__(self, association_dim: int, key_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.query = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=key_dim)
        self.key = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=key_dim)

    @staticmethod
    def _features(mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([mu.unsqueeze(-1), e.unsqueeze(-1), u.unsqueeze(-1), z], dim=-1)

    def query_of(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.query(self._features(mu, e, u, z))

    def key_of(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.key(self._features(mu, e, u, z))


class StochasticGateFunction(nn.Module):
    """`s_i = SendGate(B_i, Local_i)` / `n_i = NeedGate(B_i, Local_i)` --
    a learned `log_alpha_i` from a shared MLP, turned into an actual
    stochastic `{0,1}`-capable gate via `hard_concrete_gate`. One class
    serves both the send and need gates (two separate instances) -- same
    input shape, different learned weights, different meaning."""

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=6, hidden_dim=hidden_dim, out_features=1)

    def forward(
        self,
        mu: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor,
        mu_local: torch.Tensor,
        e_local: torch.Tensor,
        u_local: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([mu, e, u, mu_local, e_local, u_local], dim=-1)
        log_alpha = self.net(features).squeeze(-1)
        return hard_concrete_gate(log_alpha, training=self.training)


class RoutingGateFunction(nn.Module):
    """`beta_r = sigmoid(F_route_gate(mu, e, u, mu_field, e_field,
    u_field))` -- how much of the mean-shift displacement `(r_bar - r)`
    (`field_fusion.py`) a cell actually takes this step, instead of a
    fixed step size. Same input shape/spirit as `WriteGateFunction`'s
    `beta_mu`/`beta_e`/`beta_u`/`beta_z` (this repo's convention: ordinary
    `sigmoid`, not `hard_concrete_gate` -- routing movement doesn't need
    to be exactly sparse the way global send/need decisions do)."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=6, hidden_dim=hidden_dim, out_features=1)

    def forward(
        self,
        mu: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor,
        mu_field: torch.Tensor,
        e_field: torch.Tensor,
        u_field: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([mu, e, u, mu_field, e_field, u_field], dim=-1)
        return torch.sigmoid(self.net(features).squeeze(-1))


class FieldSemanticUpdateFunction(nn.Module):
    """`F_z` for the field: how a cell's semantic identity `z` drifts,
    given its own current `z`, where it currently sits in routing space
    (`r`) and where the local density just pulled it (`r_bar`), and its
    just-updated belief. Deliberately **not** a kernel-weighted average
    of neighbors' `z` (`docs/architecture_v1.md` §12 -- that was the
    original, broken design: `z` is an identity a cell carries and
    updates, not a density to sit at the mean of). `r`/`r_bar` stand in
    for "what's around me" instead of a z-specific kernel summary --
    already computed for the mean-shift step, at no extra cost.
    """

    def __init__(self, association_dim: int, routing_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(
            in_features=association_dim + 2 * routing_dim + 3, hidden_dim=hidden_dim, out_features=association_dim
        )

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        r_bar: torch.Tensor,
        mu_next: torch.Tensor,
        evidence_next: torch.Tensor,
        uncertainty_next: torch.Tensor,
    ) -> torch.Tensor:
        """`z`: `(batch, n_cells, association_dim)`. `r`/`r_bar`:
        `(batch, n_cells, routing_dim)`. `mu_next`/`evidence_next`/
        `uncertainty_next`: `(batch, n_cells)`. Returns the raw `F_z`
        output (before the `beta_z`-gated residual update + normalize,
        which `field_dynamics.py` applies): `(batch, n_cells,
        association_dim)`."""
        scalars = torch.stack([mu_next, evidence_next, uncertainty_next], dim=-1)
        features = torch.cat([z, r, r_bar, scalars], dim=-1)
        return self.net(features)
