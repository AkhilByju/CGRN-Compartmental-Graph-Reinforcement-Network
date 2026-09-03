"""Learned low-rank association field -- the user's replacement for the
ORFF-approximated Gaussian self-organizing field (`field_fusion.py`,
frozen, kept as a separate experimental line, not deleted or modified).

**The idea, in the user's own words:** "We shouldn't spend hundreds of
dimensions accurately approximating a similarity function that *we
chose*. The network should learn the similarity function itself."
`field_fusion.py` picked a Gaussian kernel over a routing coordinate `r`
and spent `R=512` random features approximating it as accurately as
possible; nothing about "Gaussian distance in a learned coordinate" was
itself learned from task loss. Here there is no routing coordinate, no
bandwidth, no mass function, no mean-shift -- a single shared function
maps each cell's own belief state directly to a small positive feature
vector, and the *dot product of those vectors is the association kernel
by definition*, not an approximation of one:

    phi_i = softplus(F_assoc(mu_i, e_i, u_i, z_i))     (batch, n, D)
    K_ij  = phi_i . phi_j

`D` (`assoc_dim`) is small (16-64; default 32) and independent of
`n_cells`, so -- exactly like `global_field.py`'s `d_g` -- this doesn't
add per-cell parameters as `n_cells` scales, and (like the global field,
unlike the RFF local field) the kernel is *exact*, not a statistical
approximation: no variance, no "does this converge as R grows" question,
and self-removal is exact subtraction (matching `global_field.py`'s
approach) rather than the RFF local field's self-anchoring-to-a-known-
constant (there is no "true" self value here to anchor to -- `phi_i .
phi_i` is just whatever the learned function currently produces).

Complexity: `O(n_cells * D)`, same class as the global field, an order
of magnitude smaller than the local field's `O(n_cells * R)` at the
`R=512` config that was needed for the field to reliably converge at
scale (`docs/research_log.md`'s global-collapse diagnostic and the
convergence-based complexity sweep).
"""

from __future__ import annotations

import torch
from torch import nn

# Same lesson as global_field.py's _DENOM_FLOOR: float32 catastrophic
# cancellation in the exact self-removal subtraction leaves a residual
# well above eps=1e-8 for O(1)-scale inputs; this floor must dominate
# that noise, while staying far below any realistic aggregate mass once
# more than a couple of cells contribute real signal.
_DENOM_FLOOR = 1e-2


def _mlp(in_features: int, hidden_dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_features),
    )


class AssociationFunction(nn.Module):
    """`phi_i = softplus(F_assoc(mu_i, e_i, u_i, z_i))` -- one shared
    function, every cell. `softplus` (not `ELU(x)+1`, matching this
    module's own spec rather than `global_field.py`'s feature map)
    guarantees `phi_i > 0` elementwise, which is what makes every
    reduction in `learned_local_association_fusion` provably
    non-negative before self-removal, by the same argument
    `global_field.py`'s docstring gives for its own positive feature
    map."""

    def __init__(self, association_dim: int, assoc_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=3 + association_dim, hidden_dim=hidden_dim, out_features=assoc_dim)

    def forward(self, mu: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = torch.cat([mu.unsqueeze(-1), e.unsqueeze(-1), u.unsqueeze(-1), z], dim=-1)
        return nn.functional.softplus(self.net(features))


def learned_local_association_fusion(
    mu: torch.Tensor,
    e: torch.Tensor,
    u: torch.Tensor,
    phi: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mu`/`e`/`u`: `(batch, n_cells)`. `phi`: `(batch, n_cells,
    assoc_dim)`. Returns `(mu_local, evidence_local, uncertainty_local)`,
    each `(batch, n_cells)`.

    Same factorization trick as `global_field.py::linear_global_belief_field`
    -- one set of population-wide reductions (`s_0/s_p/s_mu/s_mu2/s_e`,
    each `(batch, assoc_dim)`), then every receiver reads them through its
    own `phi_i`, `O(n_cells * assoc_dim)` not `O(n_cells^2)`. No `send`/
    `mass` gate: unlike the global field (where a send gate decides
    whether a cell broadcasts) or the old local field (a separate learned
    `MassFunction`), every cell fully participates in local association --
    `phi_i`'s own magnitude already carries that role (a cell whose
    `phi_i` is small in every direction contributes little to any sum
    regardless of alignment).

    Precision `pi_j = e_j / (u_j^2 + eps)` -- reuses CellV0.1's exact
    semantics. `mu_local` is `tanh(c + bias)`, matching
    `field_fusion.py::local_field_fusion`'s exact convention (`c` is the
    pre-`tanh` precision-weighted content; disagreement/uncertainty is
    computed from `c`, not the squashed `mu_local`, for the same reason
    `field_fusion.py`/`fusion.py::precision_fusion` do it that way) --
    unlike `global_field.py`'s global proposal, this *is* the final
    proposal in the local-only case, so it needs its own bounding rather
    than deferring to a downstream fuse.

    Self-removal is *exact* subtraction (`phi` is learned, not an
    RFF estimate of some other target kernel, so there is no "true"
    self-kernel value to anchor to the way the RFF local field has --
    every raw sum simply has its own self term subtracted, and `phi_i .
    phi_j > 0` for all `i, j` guarantees the off-self remainder is
    non-negative *before* the defensive clamp, exactly the argument
    `global_field.py`'s docstring gives for its own self-removal).
    """
    precision = e / (u**2 + eps)  # (batch, n)

    s_0 = phi.sum(dim=-2)  # (batch, D) -- participation mass
    s_p = torch.einsum("bnd,bn->bd", phi, precision)
    s_mu = torch.einsum("bnd,bn->bd", phi, precision * mu)
    s_mu2 = torch.einsum("bnd,bn->bd", phi, precision * mu**2)
    s_e = torch.einsum("bnd,bn->bd", phi, e)

    a_raw = torch.einsum("bnd,bd->bn", phi, s_0)
    p_raw = torch.einsum("bnd,bd->bn", phi, s_p)
    mu_num_raw = torch.einsum("bnd,bd->bn", phi, s_mu)
    mu2_num_raw = torch.einsum("bnd,bd->bn", phi, s_mu2)
    e_num_raw = torch.einsum("bnd,bd->bn", phi, s_e)

    self_compat = (phi * phi).sum(dim=-1)  # (batch, n) -- exact, K_ii = phi_i . phi_i
    a = (a_raw - self_compat).clamp(min=0.0)
    p = (p_raw - self_compat * precision).clamp(min=0.0)
    mu_num = mu_num_raw - self_compat * precision * mu
    mu2_num = mu2_num_raw - self_compat * precision * mu**2
    e_num = e_num_raw - self_compat * e

    denom_eps = max(eps, _DENOM_FLOOR)
    c = mu_num / (p + denom_eps)
    mu_local = torch.tanh(c + bias)
    evidence_local = e_num / (a + denom_eps)

    p_bar = p / (a + denom_eps)
    m2 = mu2_num / (p + denom_eps)
    disagreement = (m2 - c**2).clamp(min=0.0)
    uncertainty_local = torch.sqrt(1.0 / (p_bar + eps) + disagreement)

    return mu_local, evidence_local, uncertainty_local


class AssociationSemanticUpdateFunction(nn.Module):
    """`F_z` for the association field: how `z` drifts given its own
    current value, the just-computed local content summary (`mu_local`,
    `e_local`, `u_local` -- "what's around me," content-wise), and the
    just-updated belief. Deliberately **not** a kernel-weighted average
    of neighbors' `z` -- the same lesson `field_functions.py::
    FieldSemanticUpdateFunction`'s docstring documents ("z is an identity
    a cell carries and updates, not a density to sit at the mean of");
    that lesson doesn't depend on which kernel produced the neighborhood,
    so it applies here too. Where the field variant used `(r, r_bar)` as
    its "what's around me" signal (this design has no `r`), the local
    association fusion's own content output serves the same role at zero
    extra cost -- already computed for the write-gate blend.
    """

    def __init__(self, association_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = _mlp(in_features=association_dim + 6, hidden_dim=hidden_dim, out_features=association_dim)

    def forward(
        self,
        z: torch.Tensor,
        mu_local: torch.Tensor,
        e_local: torch.Tensor,
        u_local: torch.Tensor,
        mu_next: torch.Tensor,
        evidence_next: torch.Tensor,
        uncertainty_next: torch.Tensor,
    ) -> torch.Tensor:
        """`z`: `(batch, n_cells, association_dim)`. All other inputs:
        `(batch, n_cells)`. Returns the raw `F_z` output (before the
        `beta_z`-gated residual update + normalize, which
        `association_dynamics.py` applies): `(batch, n_cells,
        association_dim)`."""
        scalars = torch.stack(
            [mu_local, e_local, u_local, mu_next, evidence_next, uncertainty_next], dim=-1
        )
        features = torch.cat([z, scalars], dim=-1)
        return self.net(features)
