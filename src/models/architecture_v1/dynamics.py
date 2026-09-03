"""One CellV1 refinement step, and the `T`-step loop that reuses it.

`docs/architecture_v1.md` Part I-V, condensed: local association -> local
precision fusion -> need/offer -> global routing -> global precision
fusion -> fuse self+local+global -> update the semantic address -> repeat.
`DynamicBeliefGraphStep` is a single `nn.Module`; `DynamicBeliefGraphCore`
applies the *same* step `T` times (shared parameters across iterations --
docs/architecture_v1.md's comparison table, "Core update/message rules can
be shared across iterations," contrasted against a Transformer's
layer-specific parameters), not `T` independently-parameterized steps.

`use_global_routing=False` is the CellV1-Local ablation arm for
`experiments/v1_001_dynamic_groups` (docs/architecture_v1.md §9's staging
question): the global-routing module, need/offer functions, and the
global fusion bias are not constructed at all (a genuinely smaller model,
not a full model with its global output suppressed), and the final fuse
and semantic-address update each drop to two sources (self, local) instead
of three.

**CellV1.2 gated write (docs/architecture_v1.md §12).** The self+local+
global fusion (`precision_fusion` below) now produces a *proposal*
(`mu_hat`/`e_hat`/`u_hat`), not the final next state -- a learned
per-channel gate (`WriteGateFunction`) decides how much of it each cell
accepts: `next = (1 - beta) * old + beta * proposal`. Diagnosed fix for a
state-collapse/over-mixing failure mode (repeated message-passing +
consensus fusion + shared dynamics over several recurrent steps, with
nothing previously stopping a cell from being fully overwritten every
step -- the recurrent/message-passing analogue of GNN oversmoothing;
`docs/research_log.md` "CellV1.2" has the diagnostic that found this).
`z`'s update reuses the same mechanism: `beta_z` replaces the old fixed
`eta` scalar in `z + eta * z_delta` -- same residual-update *shape*,
now a learned, per-cell, per-example blend instead of a fixed constant.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.fusion import precision_fusion
from src.models.architecture_v1.routing import GlobalRouting, LocalAssociation
from src.models.architecture_v1.shared_functions import (
    MessageFunction,
    NeedFunction,
    OfferFunction,
    SemanticUpdateFunction,
    WriteGateFunction,
)

Graphs = tuple[torch.Tensor, torch.Tensor | None]  # (a_local, a_global-or-None)


class DynamicBeliefGraphStep(nn.Module):
    """One refinement step over a population of `n_cells` `BeliefCellV1`s."""

    def __init__(
        self,
        n_cells: int,
        association_dim: int,
        key_dim: int | None = None,
        hidden_dim: int = 32,
        lambda_: float = 1.0,
        tau: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-8,
        use_global_routing: bool = True,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.eps = eps
        self.use_global_routing = use_global_routing

        self.local_association = LocalAssociation(association_dim, lambda_=lambda_, tau=tau, eps=eps)
        self.message = MessageFunction(hidden_dim=hidden_dim)
        self.semantic_update = SemanticUpdateFunction(
            association_dim, hidden_dim=hidden_dim, use_global=use_global_routing
        )
        self.write_gate = WriteGateFunction(
            use_global=use_global_routing, hidden_dim=hidden_dim, init_bias=gate_init_bias
        )

        self.local_bias = nn.Parameter(torch.zeros(n_cells))
        self.fuse_bias = nn.Parameter(torch.zeros(n_cells))
        n_sources = 3 if use_global_routing else 2
        # Learned per-source gate for the final fuse -- generalizes V0's
        # always-learned relevance gate `g` to this step's fixed
        # "meta-sources" (self[/local[/global]]).
        self.source_gate_logit = nn.Parameter(torch.zeros(n_sources))

        if use_global_routing:
            self.global_routing = GlobalRouting(association_dim, key_dim=key_dim, gamma=gamma, eps=eps)
            self.need = NeedFunction(association_dim, hidden_dim=hidden_dim)
            self.offer = OfferFunction(association_dim, hidden_dim=hidden_dim)
            self.global_bias = nn.Parameter(torch.zeros(n_cells))
        else:
            self.global_routing = None
            self.need = None
            self.offer = None
            self.global_bias = None

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        return self.forward_with_graphs(cells)[0]

    def forward_with_graphs(self, cells: BeliefCellV1) -> tuple[BeliefCellV1, Graphs]:
        """Same computation as `forward`, additionally returning
        `(a_local, a_global)` for this step (`a_global` is `None` when
        `use_global_routing=False`) -- used for the graph-evolution
        analysis in `experiments/v1_001_dynamic_groups`, not by ordinary
        training."""
        mu, evidence, uncertainty, z = cells.mu, cells.evidence, cells.uncertainty, cells.z
        e_j = evidence.unsqueeze(-2)  # sender-side evidence, broadcasts over the receiver axis
        u_j = uncertainty.unsqueeze(-2)

        # --- Part I-II: local association + local fusion ---
        a_local = self.local_association(mu, uncertainty, z)
        m_local = self.message(mu, evidence, uncertainty)
        mu_local, e_local, u_local, alpha_local = precision_fusion(
            m_local, a_local, e_j, u_j, self.local_bias, self.eps
        )
        z_local = torch.matmul(alpha_local, z)

        if self.use_global_routing:
            # --- Part III, Steps 1-2: need / offer ---
            delta = (mu - mu_local).abs() / torch.sqrt(uncertainty**2 + u_local**2 + self.eps)
            need = self.need(evidence, uncertainty, delta, z)
            local_centrality = a_local.sum(dim=-2)
            offer = self.offer(evidence, uncertainty, z, local_centrality)

            # --- Part III, Step 3 + global fusion ---
            a_global = self.global_routing(z, need, offer, a_local)
            m_global = self.message(mu, evidence, uncertainty)
            mu_global, e_global, u_global, alpha_global = precision_fusion(
                m_global, a_global, e_j, u_j, self.global_bias, self.eps
            )
            z_global = torch.matmul(alpha_global, z)

            # --- Part IV: fuse self + local + global ---
            m_fuse = torch.stack([mu, mu_local, mu_global], dim=-1)
            e_fuse = torch.stack([evidence, e_local, e_global], dim=-1)
            u_fuse = torch.stack([uncertainty, u_local, u_global], dim=-1)
        else:
            a_global = None
            # --- Part IV (Local ablation): fuse self + local only ---
            m_fuse = torch.stack([mu, mu_local], dim=-1)
            e_fuse = torch.stack([evidence, e_local], dim=-1)
            u_fuse = torch.stack([uncertainty, u_local], dim=-1)

        gate = torch.sigmoid(self.source_gate_logit).expand_as(m_fuse)
        mu_hat, e_hat, u_hat, _ = precision_fusion(m_fuse, gate, e_fuse, u_fuse, self.fuse_bias, self.eps)

        # --- CellV1.2: gated write -- proposal, not automatic overwrite ---
        if self.use_global_routing:
            beta_mu, beta_e, beta_u, beta_z = self.write_gate(
                mu, evidence, uncertainty, mu_local, e_local, u_local, mu_global, e_global, u_global
            )
        else:
            beta_mu, beta_e, beta_u, beta_z = self.write_gate(mu, evidence, uncertainty, mu_local, e_local, u_local)

        mu_next = (1 - beta_mu) * mu + beta_mu * mu_hat
        evidence_next = (1 - beta_e) * evidence + beta_e * e_hat
        uncertainty_next = (1 - beta_u) * uncertainty + beta_u * u_hat

        # --- Part V: semantic-address update ---
        if self.use_global_routing:
            z_delta = self.semantic_update(
                z, z_local, mu_next, evidence_next, uncertainty_next, z_global=z_global
            )
        else:
            z_delta = self.semantic_update(z, z_local, mu_next, evidence_next, uncertainty_next)
        z_raw = z + beta_z.unsqueeze(-1) * z_delta
        z_next = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)

        next_cells = BeliefCellV1(mu=mu_next, evidence=evidence_next, uncertainty=uncertainty_next, z=z_next)
        return next_cells, (a_local, a_global)


class DynamicBeliefGraphCore(nn.Module):
    """Applies one shared `DynamicBeliefGraphStep` `num_steps` times
    (docs/architecture_v1.md: "T = 4-8 refinement steps... same cells stay
    alive")."""

    def __init__(self, n_cells: int, association_dim: int, num_steps: int = 6, **step_kwargs) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.step = DynamicBeliefGraphStep(n_cells, association_dim, **step_kwargs)

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        return self.forward_with_graphs(cells)[0]

    def forward_with_graphs(self, cells: BeliefCellV1) -> tuple[BeliefCellV1, list[Graphs]]:
        """Returns `(final_cells, [(a_local, a_global), ...])`, one graph
        pair per step, in order -- the raw material for
        `experiments/v1_001_dynamic_groups`'s graph-evolution analysis."""
        graphs: list[Graphs] = []
        for _ in range(self.num_steps):
            cells, step_graphs = self.step.forward_with_graphs(cells)
            graphs.append(step_graphs)
        return cells, graphs
