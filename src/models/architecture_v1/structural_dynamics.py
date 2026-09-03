"""One CellV1.5 refinement step, and the `T`-step loop that reuses it
(docs/architecture_v1.md §16.5). Unlike every earlier CellV1 variant,
neighbor-*finding* isn't recomputed from scratch each step: `w_ij` comes
from the persistent structural graph (`structural.py`), fixed for the
whole forward pass (addresses don't change within one call, only
between them via backprop) -- only the functional gate `a_ij(t)` (`phi_i
. phi_j`, `learned_association.py::AssociationFunction`, reused
unmodified) and the belief state itself change per step.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.fusion import precision_fusion
from src.models.architecture_v1.learned_association import AssociationFunction
from src.models.architecture_v1.shared_functions import WriteGateFunction
from src.models.architecture_v1.structural import CellActivity, EdgeRegistry, StructuralAddress
from src.models.architecture_v1.structural_fusion import sparse_structural_fusion
from src.models.architecture_v1.structural_plasticity import (
    StructuralPlasticityConfig,
    bootstrap,
    run_plasticity_event,
    should_run_plasticity,
)


def _mlp(in_features: int, hidden_dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_features),
    )


class StructuralSemanticUpdateFunction(nn.Module):
    """`F_z` for CellV1.5 -- not respecified by the user (§16.5's note
    that this "isn't respecified" beyond carrying over the existing
    pattern); same role/shape as `learned_association.py::
    AssociationSemanticUpdateFunction`, fed `Structural_j`'s content as
    "what's around me" instead of the dense association fusion's local
    output."""

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=association_dim + 6, hidden_dim=hidden_dim, out_features=association_dim)

    def forward(
        self,
        z: torch.Tensor,
        mu_structural: torch.Tensor,
        e_structural: torch.Tensor,
        u_structural: torch.Tensor,
        mu_next: torch.Tensor,
        evidence_next: torch.Tensor,
        uncertainty_next: torch.Tensor,
    ) -> torch.Tensor:
        """`z`: `(batch, n_cells, association_dim)`. All other args:
        `(batch, n_cells)`. Returns the raw `F_z` output (before the
        `beta_z`-gated residual update + normalize, applied by
        `StructuralRefinementStep`): `(batch, n_cells, association_dim)`."""
        scalars = torch.stack(
            [mu_structural, e_structural, u_structural, mu_next, evidence_next, uncertainty_next], dim=-1
        )
        features = torch.cat([z, scalars], dim=-1)
        return self.net(features)


class StructuralRefinementStep(nn.Module):
    """One refinement step (§16.5): dynamic gate -> sparse structural
    fusion -> self+structural fuse (learned 2-source gate, §16.1's "no
    separate global source anymore") -> gated write -> `z` update.
    `w`/`edge_index` are forward arguments, not owned here -- they're
    fixed for the whole forward pass (`StructuralRefinementCore` computes
    `w` once via `StructuralAddress.edge_weights`, before the `T`-step
    loop), while this module owns everything that changes every step
    (`assoc_fn`, the fuse/gate/z-update parameters)."""

    def __init__(
        self,
        n_cells: int,
        association_dim: int,
        assoc_dim: int = 32,
        hidden_dim: int = 32,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.eps = eps

        self.assoc_fn = AssociationFunction(association_dim, assoc_dim=assoc_dim, hidden_dim=hidden_dim)
        self.structural_bias = nn.Parameter(torch.zeros(n_cells))
        self.fuse_bias = nn.Parameter(torch.zeros(n_cells))
        # 2 sources: {self, structural} -- no separate global source, §16.1.
        self.source_gate_logit = nn.Parameter(torch.tensor([0.0, 0.0]))

        self.write_gate = WriteGateFunction(use_global=False, hidden_dim=hidden_dim, init_bias=gate_init_bias)
        self.semantic_update = StructuralSemanticUpdateFunction(association_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        cells: BeliefCellV1,
        edge_index: torch.Tensor,
        w: torch.Tensor,
        edge_message_sink: list[torch.Tensor] | None = None,
    ) -> BeliefCellV1:
        mu, evidence, uncertainty, z = cells.mu, cells.evidence, cells.uncertainty, cells.z
        batch = mu.shape[0]

        phi = self.assoc_fn(mu, evidence, uncertainty, z)  # (batch, n_cells, assoc_dim)
        if edge_index.shape[0] > 0:
            phi_source = phi[:, edge_index[:, 0]]
            phi_target = phi[:, edge_index[:, 1]]
            a = (phi_source * phi_target).sum(dim=-1)  # (batch, E) -- a_ij(t), §16.4
        else:
            a = torch.zeros(batch, 0, device=mu.device, dtype=mu.dtype)

        mu_structural, e_structural, u_structural = sparse_structural_fusion(
            mu,
            evidence,
            uncertainty,
            edge_index,
            w,
            a,
            self.n_cells,
            self.structural_bias,
            self.eps,
            edge_message_sink=edge_message_sink,
        )

        m_fuse = torch.stack([mu, mu_structural], dim=-1)
        e_fuse = torch.stack([evidence, e_structural], dim=-1)
        u_fuse = torch.stack([uncertainty, u_structural], dim=-1)
        gate = torch.sigmoid(self.source_gate_logit).expand_as(m_fuse)
        mu_hat, e_hat, u_hat, _ = precision_fusion(m_fuse, gate, e_fuse, u_fuse, self.fuse_bias, self.eps)

        beta_mu, beta_e, beta_u, beta_z = self.write_gate(mu, evidence, uncertainty, mu_hat, e_hat, u_hat)
        mu_next = (1 - beta_mu) * mu + beta_mu * mu_hat
        evidence_next = (1 - beta_e) * evidence + beta_e * e_hat
        uncertainty_next = (1 - beta_u) * uncertainty + beta_u * u_hat

        z_delta = self.semantic_update(
            z, mu_structural, e_structural, u_structural, mu_next, evidence_next, uncertainty_next
        )
        z_raw = z + beta_z.unsqueeze(-1) * z_delta
        z_next = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)

        return BeliefCellV1(mu=mu_next, evidence=evidence_next, uncertainty=uncertainty_next, z=z_next)


class StructuralRefinementCore(nn.Module):
    """Applies one shared `StructuralRefinementStep` `num_steps` times
    (shared parameters across iterations, matching every other CellV1
    variant's `*Core` convention). Owns the persistent structural
    substrate (`StructuralAddress`, `EdgeRegistry`, `CellActivity`) and
    the structural-plasticity schedule config -- `w_ij` is computed once
    per forward pass, before the step loop, since it only depends on
    addresses that don't change within a call (§16.3).

    **Integration contract for structural plasticity/utility (unusual --
    nothing else in this codebase needs this).** A training loop using
    this model should, once per optimizer step:

    1. Call `forward` as normal, compute the loss, call `loss.backward()`.
    2. Call `update_edge_utility()` -- reads `.grad` off the edge
       messages this forward pass retained (only happens in `training`
       mode), EMAs them into `self.edges.utility`. If skipped, utility
       simply stays at its last value (stale), not an error.
    3. Call `optimizer.step()`, `optimizer.zero_grad()`.
    4. Call `maybe_run_structural_plasticity(cells, step, total_steps)`
       with the just-encoded batch of cells -- a no-op unless this
       optimizer step is due for a plasticity event (§16.9).

    Before training starts, call `bootstrap_structural_graph(cells)`
    once (§16.8) -- without it, `edge_index` starts empty and every
    fusion call degenerates to `precision_fusion`'s no-live-edges
    fallback for every cell.
    """

    def __init__(
        self,
        n_cells: int,
        association_dim: int,
        d_s: int = 16,
        num_steps: int = 1,
        plasticity_config: StructuralPlasticityConfig | None = None,
        **step_kwargs,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.structural_address = StructuralAddress(n_cells, d_s)
        self.edges = EdgeRegistry(n_cells)
        self.activity = CellActivity(n_cells)
        self.step = StructuralRefinementStep(n_cells, association_dim, **step_kwargs)
        self.plasticity_config = plasticity_config or StructuralPlasticityConfig()
        self._last_edge_messages: list[torch.Tensor] = []

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        w = self.structural_address.edge_weights(self.edges.edge_index)
        self._last_edge_messages = []
        sink = self._last_edge_messages if self.training else None
        for _ in range(self.num_steps):
            cells = self.step(cells, self.edges.edge_index, w, edge_message_sink=sink)
        self.activity.update(cells.mu, cells.evidence, cells.uncertainty)
        return cells

    def update_edge_utility(self, decay: float = 0.99) -> None:
        """Call after `loss.backward()`, before `optimizer.zero_grad()`
        (see class docstring). `q_ij = |m_ij * dL/dm_ij|` (§16.6),
        batch-meaned then meaned across this forward pass's `num_steps`
        refinement steps (an implementation-choice default extending the
        `mean_batch` convention §16.7's `G_ij` is explicitly given to
        `q_ij`, which the spec doesn't separately pin down), then EMA'd
        into `self.edges.utility`."""
        if not self._last_edge_messages or self.edges.n_edges == 0:
            return
        per_step = []
        for m in self._last_edge_messages:
            if m.grad is None:
                continue
            per_step.append((m * m.grad).abs().mean(dim=0))  # mean over batch -> (E,)
        if not per_step:
            return
        q = torch.stack(per_step, dim=0).mean(dim=0)  # mean over refinement steps -> (E,)
        self.edges.utility = decay * self.edges.utility + (1.0 - decay) * q

    def _phi_mean(self, cells: BeliefCellV1) -> torch.Tensor:
        with torch.no_grad():
            phi = self.step.assoc_fn(cells.mu, cells.evidence, cells.uncertainty, cells.z)
            return phi.mean(dim=0)

    def bootstrap_structural_graph(self, cells: BeliefCellV1, generator: torch.Generator | None = None) -> None:
        """§16.8: one candidate-growth pass filling the edge budget from
        the (meaningless-at-init, disposable) initial addresses, then
        enforcing the in-degree->=1 safety constraint. `cells` should be
        an encoded batch (any batch -- the resulting topology is
        disposable)."""
        phi_mean = self._phi_mean(cells)
        e_max = self.plasticity_config.k_bar * self.edges.n_cells
        bootstrap(self.structural_address, self.edges, self.activity, phi_mean, e_max, generator=generator)

    def maybe_run_structural_plasticity(
        self, cells: BeliefCellV1, step: int, total_steps: int, generator: torch.Generator | None = None
    ) -> bool:
        """§16.9: no-op unless `step` is due for a structural-plasticity
        event (warm-up passed, on the update interval, not yet in the
        end-of-training freeze window). Returns whether an event ran."""
        if not should_run_plasticity(step, total_steps, self.plasticity_config):
            return False
        phi_mean = self._phi_mean(cells)
        run_plasticity_event(self.structural_address, self.edges, self.activity, phi_mean, self.plasticity_config, generator=generator)
        return True
