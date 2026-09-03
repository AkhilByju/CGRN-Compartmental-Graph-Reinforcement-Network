"""One Learned Association Field refinement step, and the `T`-step loop
that reuses it -- the narrower redesign replacing `field_dynamics.py`'s
ORFF-based local field (frozen, kept as a separate experimental line)
with `learned_association.py`'s exact, learned low-rank association
kernel. No routing coordinate `r`, no bandwidth/mass functions, no
mean-shift: the state is plain `BeliefCellV1` (`mu`, `e`, `u`, `z`), the
same state dense/sparse CellV1 already use -- unlike `FieldCellState`,
nothing here needs a persistent extra coordinate, so no wrapper dataclass
is needed at all.

**Global communication is reused unmodified.** Per the user's explicit
instruction ("keep the global mechanism we already developed... let it
consume the learned association state, not an expensive ORFF geometry"):
`global_field.py`'s `GlobalSendFunction`/`GlobalNeedFunction`/
`GlobalQueryKey`/`linear_global_belief_field`, `shared_functions.py`'s
`WriteGateFunction`, and `fusion.py`'s `precision_fusion` are imported
and wired in exactly the same way `field_dynamics.py`'s `use_global=True`
path already does -- the only change is that `mu_local`/`e_local`/
`u_local` (the local-field-flavored ones `GlobalSendFunction`/
`GlobalNeedFunction`/`WriteGateFunction` are fed) now come from
`learned_local_association_fusion` instead of `field_fusion.py::
local_field_fusion`.

**`use_global=False` by default, additive flag, same convention as
`field_dynamics.py`.** Not implemented yet, deliberately deferred (per
the user's "narrower redesign... I would not do another huge sweep"):
an adaptive per-input continue/refine gate deciding `T1` vs `T2` from
task loss, rather than a fixed `num_steps`. `AssociationRefinementCore`
still takes a fixed `num_steps` for now, matching
`FieldRefinementCore`/`DynamicBeliefGraphCore`'s existing convention --
a real gap against the user's full spec, flagged here rather than
half-implemented.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.fusion import precision_fusion
from src.models.architecture_v1.global_field import (
    GlobalNeedFunction,
    GlobalQueryKey,
    GlobalSendFunction,
    linear_global_belief_field,
)
from src.models.architecture_v1.learned_association import (
    AssociationFunction,
    AssociationSemanticUpdateFunction,
    learned_local_association_fusion,
)
from src.models.architecture_v1.shared_functions import WriteGateFunction


class AssociationRefinementStep(nn.Module):
    """One refinement step: compute each cell's association feature
    (`learned_association.py::AssociationFunction`), evaluate the
    learned local association fusion, optionally fold in global
    communication (reusing `global_field.py`/`fusion.py::precision_fusion`
    unmodified), gated-write `(mu, e, u)` toward the resulting proposal,
    and gated-update `z` from the local fusion's content output as
    spatial context (`learned_association.py::
    AssociationSemanticUpdateFunction`)."""

    def __init__(
        self,
        n_cells: int,
        association_dim: int,
        assoc_dim: int = 32,
        hidden_dim: int = 32,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
        use_global: bool = False,
        global_dim: int = 16,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_global = use_global

        self.assoc_fn = AssociationFunction(association_dim, assoc_dim=assoc_dim, hidden_dim=hidden_dim)
        self.local_bias = nn.Parameter(torch.zeros(n_cells))

        # Same "reused as-is" note as field_dynamics.py: WriteGateFunction's
        # interface never changes; it's just fed whichever proposal
        # (local-only or self+local+global-fused) this step computed.
        self.write_gate = WriteGateFunction(use_global=False, hidden_dim=hidden_dim, init_bias=gate_init_bias)
        self.semantic_update = AssociationSemanticUpdateFunction(association_dim, hidden_dim=hidden_dim)

        if use_global:
            self.global_query_key = GlobalQueryKey(association_dim, global_dim)
            self.send_fn = GlobalSendFunction(hidden_dim=hidden_dim, init_bias=gate_init_bias)
            self.need_fn = GlobalNeedFunction(hidden_dim=hidden_dim, init_bias=gate_init_bias)
            self.global_fuse_bias = nn.Parameter(torch.zeros(n_cells))
            self.source_gate_logit = nn.Parameter(torch.tensor([0.0, 0.0, gate_init_bias]))
        else:
            self.global_query_key = None
            self.send_fn = None
            self.need_fn = None
            self.global_fuse_bias = None
            self.source_gate_logit = None

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        mu, evidence, uncertainty, z = cells.mu, cells.evidence, cells.uncertainty, cells.z

        phi = self.assoc_fn(mu, evidence, uncertainty, z)
        mu_local, e_local, u_local = learned_local_association_fusion(
            mu, evidence, uncertainty, phi, self.local_bias, self.eps
        )

        if self.use_global:
            send = self.send_fn(mu, evidence, uncertainty, mu_local, e_local, u_local)
            need = self.need_fn(mu, evidence, uncertainty, mu_local, e_local, u_local)
            q = self.global_query_key.query_of(z)
            k = self.global_query_key.key_of(z)
            mu_global, e_global, u_global = linear_global_belief_field(mu, evidence, uncertainty, send, q, k, self.eps)
            e_global_gated = need * e_global

            m_fuse = torch.stack([mu, mu_local, mu_global], dim=-1)
            e_fuse = torch.stack([evidence, e_local, e_global_gated], dim=-1)
            u_fuse = torch.stack([uncertainty, u_local, u_global], dim=-1)
            gate = torch.sigmoid(self.source_gate_logit).expand_as(m_fuse)
            mu_hat, e_hat, u_hat, _ = precision_fusion(m_fuse, gate, e_fuse, u_fuse, self.global_fuse_bias, self.eps)
        else:
            mu_hat, e_hat, u_hat = mu_local, e_local, u_local

        beta_mu, beta_e, beta_u, beta_z = self.write_gate(mu, evidence, uncertainty, mu_hat, e_hat, u_hat)
        mu_next = (1 - beta_mu) * mu + beta_mu * mu_hat
        evidence_next = (1 - beta_e) * evidence + beta_e * e_hat
        uncertainty_next = (1 - beta_u) * uncertainty + beta_u * u_hat

        z_delta = self.semantic_update(z, mu_local, e_local, u_local, mu_next, evidence_next, uncertainty_next)
        z_raw = z + beta_z.unsqueeze(-1) * z_delta
        z_next = z_raw / (z_raw.norm(p=2, dim=-1, keepdim=True) + self.eps)

        return BeliefCellV1(mu=mu_next, evidence=evidence_next, uncertainty=uncertainty_next, z=z_next)


class AssociationRefinementCore(nn.Module):
    """Applies one shared `AssociationRefinementStep` `num_steps` times
    (shared parameters across iterations, same convention as
    `FieldRefinementCore`/`DynamicBeliefGraphCore`). Fixed `num_steps` --
    the adaptive continue/refine gate described in the module docstring
    is not implemented here yet."""

    def __init__(self, n_cells: int, association_dim: int, num_steps: int = 1, **step_kwargs) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.step = AssociationRefinementStep(n_cells, association_dim, **step_kwargs)

    def forward(self, cells: BeliefCellV1) -> BeliefCellV1:
        for _ in range(self.num_steps):
            cells = self.step(cells)
        return cells
