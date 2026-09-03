"""The small shared functions `docs/architecture_v1.md` calls out by name
but doesn't pin an exact architecture for: `F_msg`, `F_need`, `F_offer`,
`F_z`. The user's spec is explicit that each must be ONE function reused
by every cell/pair/step ("not: each Cell has its own neural network...
not: every edge has its own transformation matrix"); it does not specify
what's *inside* that function. Implemented here as small 2-layer MLPs
(`Linear -> Tanh -> Linear`), matching this repo's existing convention for
small shared networks (`src/models/baselines/mlp.py`). This is an
implementation choice, not a specified formula -- flagged so it's easy to
find and revisit.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_features: int, hidden_dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_features),
    )


class MessageFunction(nn.Module):
    """`F_msg` (docs/architecture_v1.md Part II): `m_ij = F_msg(B_i, B_j)`
    where `B = (mu, evidence, uncertainty)`. Reused identically for local
    and global message content (§"Then global information uses the SAME
    belief mathematics").

    Input per pair: `[mu_i, e_i, u_i, mu_j, e_j, u_j]` (6 scalars).
    Output: one scalar message `m_ij`.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=6, hidden_dim=hidden_dim, out_features=1)

    def forward(
        self, mu: torch.Tensor, evidence: torch.Tensor, uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """`mu`/`evidence`/`uncertainty`: `(batch, n_cells)`. Returns
        `m`: `(batch, n_cells, n_cells)` with `m[..., i, j] = F_msg(B_i, B_j)`."""
        n = mu.size(-1)
        mu_i = mu.unsqueeze(-1).expand(*mu.shape, n)
        mu_j = mu.unsqueeze(-2).expand(*mu.shape[:-1], n, n)
        e_i = evidence.unsqueeze(-1).expand(*evidence.shape, n)
        e_j = evidence.unsqueeze(-2).expand(*evidence.shape[:-1], n, n)
        u_i = uncertainty.unsqueeze(-1).expand(*uncertainty.shape, n)
        u_j = uncertainty.unsqueeze(-2).expand(*uncertainty.shape[:-1], n, n)
        pair_features = torch.stack([mu_i, e_i, u_i, mu_j, e_j, u_j], dim=-1)
        return self.net(pair_features).squeeze(-1)

    def forward_sparse(
        self,
        mu_i: torch.Tensor,
        e_i: torch.Tensor,
        u_i: torch.Tensor,
        mu_cand: torch.Tensor,
        e_cand: torch.Tensor,
        u_cand: torch.Tensor,
    ) -> torch.Tensor:
        """CellV1.1 (sparse) variant of `forward`: `mu_i`/`e_i`/`u_i` are
        `(batch, n_cells)` (self), `mu_cand`/`e_cand`/`u_cand` are `(batch,
        n_cells, pool)` (already-gathered LSH candidates,
        `src.models.architecture_v1.lsh.gather_scalar`) -- computes the
        identical `F_msg` over `O(n_cells * pool)` pairs instead of
        `forward`'s `O(n_cells^2)`, using the same learned weights."""
        mu_i = mu_i.unsqueeze(-1).expand_as(mu_cand)
        e_i = e_i.unsqueeze(-1).expand_as(e_cand)
        u_i = u_i.unsqueeze(-1).expand_as(u_cand)
        pair_features = torch.stack([mu_i, e_i, u_i, mu_cand, e_cand, u_cand], dim=-1)
        return self.net(pair_features).squeeze(-1)


class NeedFunction(nn.Module):
    """`F_need` (docs/architecture_v1.md Part III, Step 1): how much cell
    `i` needs information from outside its local assembly.
    `need_i = sigmoid(F_need(e_i, u_i, delta_i, z_i))`.
    """

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=1)

    def forward(
        self, evidence: torch.Tensor, uncertainty: torch.Tensor, delta: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """`evidence`/`uncertainty`/`delta`: `(batch, n_cells)`. `z`:
        `(batch, n_cells, association_dim)`. Returns `need`: `(batch,
        n_cells)`, in `(0, 1)`."""
        features = torch.cat([evidence.unsqueeze(-1), uncertainty.unsqueeze(-1), delta.unsqueeze(-1), z], dim=-1)
        return torch.sigmoid(self.net(features).squeeze(-1))


class OfferFunction(nn.Module):
    """`F_offer` (docs/architecture_v1.md Part III, Step 2): how worth
    listening to cell `j` currently is for long-range communication.
    `offer_j = sigmoid(F_offer(e_j, u_j, z_j, local_centrality_j))`.
    """

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=1)

    def forward(
        self,
        evidence: torch.Tensor,
        uncertainty: torch.Tensor,
        z: torch.Tensor,
        local_centrality: torch.Tensor,
    ) -> torch.Tensor:
        """Same shape convention as `NeedFunction.forward`. Returns
        `offer`: `(batch, n_cells)`, in `(0, 1)`."""
        features = torch.cat(
            [evidence.unsqueeze(-1), uncertainty.unsqueeze(-1), local_centrality.unsqueeze(-1), z],
            dim=-1,
        )
        return torch.sigmoid(self.net(features).squeeze(-1))


class SemanticUpdateFunction(nn.Module):
    """`F_z` (docs/architecture_v1.md Part V): how a cell's association
    vector drifts given its own current `z`, the local/global semantic
    summaries it just received, and its just-updated belief.
    `z_i^{t+1} = normalize(z_i^t + beta_z_i * F_z(z_i^t, z_bar_i^L,
    z_bar_i^G, mu_i^{t+1}, e_i^{t+1}, u_i^{t+1}))`, where `beta_z` is
    `WriteGateFunction`'s learned per-cell gate (CellV1.2 -- this was
    originally a fixed scalar `eta`; `beta_z` replaces it, same residual-
    update shape). `normalize` (L2, so `z` stays on the unit hypersphere
    rather than drifting unboundedly) is this module's implementation
    choice, not specified by the user.

    `use_global=False` (the CellV1-Local ablation, `dynamics.py`'s
    `use_global_routing`) drops `z_bar^G` from the input entirely, rather
    than passing zeros for it -- a real structural difference between the
    two ablation arms, not a numerically-suppressed one.
    """

    def __init__(self, association_dim: int, hidden_dim: int = 32, use_global: bool = True) -> None:
        super().__init__()
        self.use_global = use_global
        n_summaries = 3 if use_global else 2
        self.net = _mlp(
            in_features=n_summaries * association_dim + 3,
            hidden_dim=hidden_dim,
            out_features=association_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        z_local: torch.Tensor,
        mu_next: torch.Tensor,
        evidence_next: torch.Tensor,
        uncertainty_next: torch.Tensor,
        z_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z`/`z_local`(/`z_global`): `(batch, n_cells, association_dim)`.
        `mu_next`/`evidence_next`/`uncertainty_next`: `(batch, n_cells)`.
        Returns the raw `F_z` output (before the `beta_z`-gated residual
        update + normalize, which `dynamics.py`/`sparse_dynamics.py`
        apply): `(batch, n_cells, association_dim)`."""
        if self.use_global and z_global is None:
            raise ValueError("This SemanticUpdateFunction was built with use_global=True; z_global is required.")
        if not self.use_global and z_global is not None:
            raise ValueError("This SemanticUpdateFunction was built with use_global=False; it doesn't take z_global.")
        scalars = torch.stack([mu_next, evidence_next, uncertainty_next], dim=-1)
        parts = [z, z_local] + ([z_global] if self.use_global else [])
        features = torch.cat([*parts, scalars], dim=-1)
        return self.net(features)


class WriteGateFunction(nn.Module):
    """`F_gate` (CellV1.2, docs/architecture_v1.md §12) -- the user's fix
    for a diagnosed state-collapse/over-mixing failure mode: repeated
    message-passing + consensus-style fusion + shared dynamics over
    several recurrent steps, with nothing stopping a cell from being fully
    overwritten by the fused proposal every step (the recurrent/message-
    passing analogue of GNN oversmoothing). `beta = sigmoid(F_gate(mu, e,
    u, mu_local, e_local, u_local[, mu_global, e_global, u_global]))`, one
    gate per state channel (`beta_mu`, `beta_e`, `beta_u`, `beta_z`) from
    a shared trunk -- `dynamics.py`/`sparse_dynamics.py` blend the old
    state with the fused proposal as `(1 - beta) * old + beta * proposal`
    instead of fully overwriting it, so "communication proposes an
    update; the cell decides how much to accept."

    Bias-initialized so every gate starts at `sigmoid(-2.0) ~= 0.12` (the
    user's suggested 0.1-0.2 range) -- cells mostly preserve themselves
    early in training and learn when communication is worth accepting,
    rather than starting from "always fully overwrite" (the previous,
    ungated behavior).
    """

    def __init__(self, use_global: bool, hidden_dim: int = 32, init_bias: float = -2.0) -> None:
        super().__init__()
        self.use_global = use_global
        n_in = 9 if use_global else 6
        self.net = _mlp(in_features=n_in, hidden_dim=hidden_dim, out_features=4)
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)

    def forward(
        self,
        mu: torch.Tensor,
        evidence: torch.Tensor,
        uncertainty: torch.Tensor,
        mu_local: torch.Tensor,
        e_local: torch.Tensor,
        u_local: torch.Tensor,
        mu_global: torch.Tensor | None = None,
        e_global: torch.Tensor | None = None,
        u_global: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All inputs `(batch, n_cells)`. Returns `(beta_mu, beta_e,
        beta_u, beta_z)`, each `(batch, n_cells)`, in `(0, 1)`."""
        has_global = mu_global is not None
        if self.use_global != has_global:
            raise ValueError(
                f"This WriteGateFunction was built with use_global={self.use_global}; "
                f"{'mu_global/e_global/u_global are required' if self.use_global else 'it does not take mu_global/e_global/u_global'}."
            )
        parts = [mu, evidence, uncertainty, mu_local, e_local, u_local]
        if self.use_global:
            parts += [mu_global, e_global, u_global]
        features = torch.stack(parts, dim=-1)
        gates = torch.sigmoid(self.net(features))
        beta_mu, beta_e, beta_u, beta_z = gates.unbind(dim=-1)
        return beta_mu, beta_e, beta_u, beta_z
