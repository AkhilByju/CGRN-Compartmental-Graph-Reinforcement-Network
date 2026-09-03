"""One CellV1.1 (sparse) refinement step, and the `T`-step loop that
reuses it -- the sparse counterpart to `dynamics.py`. Same overall
sequence (local association -> local fusion -> need/offer -> global
routing -> global fusion -> fuse self+local+global -> semantic-address
update), but local/global association and fusion are restricted to an
LSH-selected candidate pool (`sparse_routing.py`) instead of scoring every
other cell -- `O(n_cells * pool_size)` per step, not `O(n_cells^2)`.

Part IV (fuse self+local+global) and Part V (semantic-address update) are
unchanged in *shape*: `mu_local`/`e_local`/`u_local`/`z_local` and their
`_global` counterparts are still plain `(batch, n_cells[, d])` tensors
regardless of whether they came from a dense or sparse computation, so
that part of the step is copied from `dynamics.py` verbatim rather than
reused via an abstraction -- CLAUDE.md's "three similar lines over
premature abstraction," and the two step classes are genuinely different
enough upstream (dense pairwise vs. LSH-gathered) that a shared base class
would mostly be plumbing.

**CellV1.2 gated write** (`dynamics.py`'s docstring has the full
rationale): the self+local+global fusion produces a proposal, and a
learned `WriteGateFunction` decides how much of it each cell accepts --
`next = (1 - beta) * old + beta * proposal` -- instead of a full
overwrite every step. `beta_z` also replaces the old fixed `eta` scalar
in `z`'s residual update.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.fusion import precision_fusion
from src.models.architecture_v1.lsh import gather_scalar, gather_vector
from src.models.architecture_v1.shared_functions import (
    MessageFunction,
    NeedFunction,
    OfferFunction,
    SemanticUpdateFunction,
    WriteGateFunction,
)
from src.models.architecture_v1.sparse_routing import SparseGlobalRouting, SparseLocalAssociation

SparseGraphs = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
# (local_candidate_idx, a_local, global_candidate_idx-or-None, a_global-or-None)


class SparseDynamicBeliefGraphStep(nn.Module):
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
        num_hashes_local: int = 2,
        bits_local: int = 6,
        chunk_size_local: int = 10,
        window_local: int = 0,
        num_hashes_global: int = 2,
        bits_global: int = 6,
        chunk_size_global: int = 4,
        window_global: int = 0,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.eps = eps
        self.use_global_routing = use_global_routing

        self.local_association = SparseLocalAssociation(
            association_dim,
            lambda_=lambda_,
            tau=tau,
            eps=eps,
            num_hashes=num_hashes_local,
            bits=bits_local,
            chunk_size=chunk_size_local,
            window=window_local,
        )
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
        self.source_gate_logit = nn.Parameter(torch.zeros(n_sources))

        if use_global_routing:
            self.global_routing = SparseGlobalRouting(
                association_dim,
                key_dim=key_dim,
                gamma=gamma,
                eps=eps,
                num_hashes=num_hashes_global,
                bits=bits_global,
                chunk_size=chunk_size_global,
                window=window_global,
            )
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

    def forward_with_graphs(self, cells: BeliefCellV1) -> tuple[BeliefCellV1, SparseGraphs]:
        mu, evidence, uncertainty, z = cells.mu, cells.evidence, cells.uncertainty, cells.z

        # --- local association + local fusion, restricted to an LSH pool ---
        a_local, local_idx = self.local_association(mu, uncertainty, z)
        e_cand = gather_scalar(evidence, local_idx)
        u_cand = gather_scalar(uncertainty, local_idx)
        mu_cand = gather_scalar(mu, local_idx)
        z_cand = gather_vector(z, local_idx)

        m_local = self.message.forward_sparse(mu, evidence, uncertainty, mu_cand, e_cand, u_cand)
        mu_local, e_local, u_local, alpha_local = precision_fusion(
            m_local, a_local, e_cand, u_cand, self.local_bias, self.eps
        )
        z_local = torch.einsum("bnp,bnpd->bnd", alpha_local, z_cand)

        if self.use_global_routing:
            delta = (mu - mu_local).abs() / torch.sqrt(uncertainty**2 + u_local**2 + self.eps)
            need = self.need(evidence, uncertainty, delta, z)
            local_centrality = torch.zeros_like(evidence).scatter_add_(
                -1, local_idx.reshape(local_idx.shape[0], -1),
                a_local.reshape(a_local.shape[0], -1),
            )  # per-cell in-degree: sum of weight received from any cell that picked it as a candidate
            offer = self.offer(evidence, uncertainty, z, local_centrality)

            a_global, global_idx = self.global_routing(z, need, offer, local_idx, a_local)
            e_gcand = gather_scalar(evidence, global_idx)
            u_gcand = gather_scalar(uncertainty, global_idx)
            mu_gcand = gather_scalar(mu, global_idx)
            z_gcand = gather_vector(z, global_idx)

            m_global = self.message.forward_sparse(mu, evidence, uncertainty, mu_gcand, e_gcand, u_gcand)
            mu_global, e_global, u_global, alpha_global = precision_fusion(
                m_global, a_global, e_gcand, u_gcand, self.global_bias, self.eps
            )
            z_global = torch.einsum("bnp,bnpd->bnd", alpha_global, z_gcand)

            m_fuse = torch.stack([mu, mu_local, mu_global], dim=-1)
            e_fuse = torch.stack([evidence, e_local, e_global], dim=-1)
            u_fuse = torch.stack([uncertainty, u_local, u_global], dim=-1)
        else:
            a_global, global_idx = None, None
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

        if self.use_global_routing:
            z_delta = self.semantic_update(
                z, z_local, mu_next, evidence_next, uncertainty_next, z_global=z_global
            )
        else:
            z_delta = self.semantic_update(z, z_local, mu_next, evidence_next, uncertainty_next)
        z_raw = z + beta_z.unsqueeze(-1) * z_delta
        z_next = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)

        next_cells = BeliefCellV1(mu=mu_next, evidence=evidence_next, uncertainty=uncertainty_next, z=z_next)
        return next_cells, (local_idx, a_local, global_idx, a_global)


class SparseDynamicBeliefGraphCore(nn.Module):
    def __init__(self, n_cells: int, association_dim: int, num_steps: int = 6, **step_kwargs) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.step = SparseDynamicBeliefGraphStep(n_cells, association_dim, **step_kwargs)

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        return self.forward_with_graphs(cells)[0]

    def forward_with_graphs(self, cells: BeliefCellV1) -> tuple[BeliefCellV1, list[SparseGraphs]]:
        graphs: list[SparseGraphs] = []
        for _ in range(self.num_steps):
            cells, step_graphs = self.step.forward_with_graphs(cells)
            graphs.append(step_graphs)
        return cells, graphs

