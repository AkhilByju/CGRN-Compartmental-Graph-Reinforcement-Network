"""`precision_fusion` -- the belief-fusion math CellV1 reuses at every
step (docs/architecture_v1.md Part II, "This is basically CellV0.1
operating over dynamically selected neighbors instead of a fixed layer";
Part IV, "Reuse the thing we've already tested").

Formula-identical to `src.models.architecture_v0.integration`'s
`_scale_stable_precision_fusion` (Method E / "CellV0.1", Experiment 004I) --
reproduced here rather than imported for two reasons: (1) V1 additionally
needs the returned content-weights `alpha` for the semantic-address update
(Part V, `z_bar^L = sum_j alpha^L_ij z_j`), which the V0 function computes
internally but doesn't return; (2) here the per-connection weight `A` is a
dynamically computed association matrix (`routing.py`), not a learned
dense `(w_ij, g_ij)` pair -- the function signature reflects that (`A`
instead of separate content-weight/relevance-gate tensors), even though
the underlying arithmetic is the same.
"""

from __future__ import annotations

import torch


def precision_fusion(
    m: torch.Tensor,
    a: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`m`, `a`, `e_j`, `u_j` broadcast to a common `(..., n)` shape, where
    the last axis `n` is the set of sources being fused (incoming cells
    for local/global fusion; the 3 sources `{self, local, global}` for the
    final per-cell fuse). Returns `(mu, evidence, uncertainty, alpha)`,
    each of shape `(...)` (the last axis reduced away) except `alpha`,
    which keeps shape `(..., n)`.

    Identical arithmetic to Method E (see
    `src.models.architecture_v0.integration._scale_stable_precision_fusion`
    for the full derivation): `a` plays exactly the role `g` (the
    sigmoid-relevance gate) played there, `e_j`/`u_j` the sender-side
    evidence/uncertainty, and `n_eff = (sum(a))^2 / (sum(a^2) + eps)` the
    same effective-source-count normalizer that keeps `evidence`/
    `uncertainty` scale-stable as the number of sources being fused varies
    (here: the number of live edges after sparsification, which is itself
    input-dependent -- exactly the case Method E was built for).

    Degenerate case (a cell with no live edges into it, `a` all zero):
    `evidence -> 0` and `uncertainty -> sqrt(1/eps)` (very large), i.e. "no
    information arrived, total uncertainty" -- falls out of the formula
    without a special case.
    """
    precision = a * e_j / (u_j**2 + eps)
    denom = precision.sum(dim=-1, keepdim=True) + eps
    alpha = precision / denom
    c = (alpha * m).sum(dim=-1)
    mu = torch.tanh(c + bias)

    a_sum = a.sum(dim=-1)
    a_sq_sum = (a**2).sum(dim=-1)
    n_eff = (a_sum**2) / (a_sq_sum + eps)

    evidence = (a * e_j).sum(dim=-1) / (n_eff + eps)
    effective_precision = precision.sum(dim=-1) / (n_eff + eps)
    base_uncertainty_sq = 1.0 / (effective_precision + eps)
    disagreement = (alpha * (m - c.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(base_uncertainty_sq + disagreement)
    return mu, evidence, uncertainty, alpha
