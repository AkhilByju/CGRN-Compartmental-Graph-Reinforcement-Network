"""Sparse-neighbor belief fusion for CellV1.5 (docs/architecture_v1.md
§16.5): the same scale-stable precision-fusion arithmetic every CellV1
variant uses (`fusion.py::precision_fusion`, Method E / CellV0.1),
restricted to each cell's *existing* structural in-neighbors instead of
either the full population (§1, §15) or an LSH/field-retrieved candidate
pool (§11, §13). Realized via `index_add_`-based scatter reduction, not
a padded dense-per-target layout -- CellV1.5's structural graph has no
degree bound (a cell may end up a hub with dozens of in-edges while most
have a handful, §16.8), so padding every target to the population's max
degree could cost `O(n_cells * E)` in the pathological case where most
edges point at one cell -- genuinely quadratic-ish, defeating the point
of a sparse substrate. Scatter reduction costs `O(E)` regardless of the
degree distribution.

Mathematically identical to `fusion.py::precision_fusion` called on a
dense `(n_cells, n_cells)` layout with zero-`a` padding for non-edges
(precision, hence `alpha`, is exactly zero wherever `a` is zero, so a
missing edge contributes nothing to any reduction) -- `tests/
test_structural_fusion.py` checks exactly this equivalence on small
hand-built graphs against `fusion.py::precision_fusion` directly, the
same "sparse must equal dense, not just look similar" bar `tests/
test_sparse_dense_consistency.py` already holds CellV1.1 to.

**Why `edge_message_full` exists, not just `m`/`p` separately (§16.6's
utility signal).** §16.6 defines the pruning signal from `m_ij = w_ij *
a_ij(t) * mu_i` -- the *full* effective transmitted content, folding in
`a_ij` directly. That's a different quantity from this module's own
`m_ij = w_ij * mu_i` (content only, used in the `w`/`a` content/precision
split, §16.5) -- both are called `m_ij` in the user's spec because they
serve different purposes in different subsections; kept as two distinctly
named tensors here to avoid confusion. `q_ij = |m_ij * dL/dm_ij|` needs a
*real* gradient, which requires `edge_message_full` to actually appear in
the computation that produces the fused output, not just be equal to it
algebraically. Since `p_ij * (content m_ij) = (a_ij * precision_no_a_ij)
* m_ij = edge_message_full * precision_no_a_ij` (associativity -- `a_ij`
factored onto `m_ij` instead of onto the precision term), routing
`pm_sum` through `edge_message_full * precision_no_a` instead of
`p * m` directly gives `edge_message_full` a genuine place in the graph,
with `.grad` after backward equal to the true `dL/d(edge_message_full)`
-- while `pm_sum`'s *value* is unchanged (verified in
`tests/test_structural_fusion.py`). Plain `m` (content only, no `a`) is
still needed on its own for the disagreement term (`(m - c)^2`), so both
`m` and `edge_message_full = a * m` coexist as related-but-distinct nodes.
"""

from __future__ import annotations

import torch


def sparse_structural_fusion(
    mu: torch.Tensor,
    evidence: torch.Tensor,
    uncertainty: torch.Tensor,
    edge_index: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    n_cells: int,
    bias: torch.Tensor,
    eps: float,
    edge_message_sink: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mu`/`evidence`/`uncertainty`: `(batch, n_cells)`. `edge_index`:
    `(E, 2)` long, `(source, target)`, shared across the batch (the
    structural topology is a model-level property, not per-example).
    `w`: `(E,)` -- `StructuralAddress.edge_weights`'s output, also shared
    across the batch (addresses don't depend on input). `a`: `(batch,
    E)` -- `a_ij(t)`, the per-edge functional gate, genuinely per-example
    (`structural_dynamics.py`'s `phi_i . phi_j`). `bias`: `(n_cells,)`.
    Returns `(mu_structural, evidence_structural, uncertainty_structural)`,
    each `(batch, n_cells)` -- §16.5's `Structural_j`.

    If `edge_message_sink` is given, the full effective per-edge message
    `w_ij * a_ij(t) * mu_i` (§16.6's `m_ij`, distinct from this
    function's own content-only `m_ij` -- see module docstring) is
    appended to it with `.retain_grad()` already called, so a caller can
    read `.grad` after `loss.backward()` to compute the utility signal
    `q_ij = |m_ij * dL/dm_ij|`.

    A cell with zero incoming structural edges gets the same degenerate
    fallback `fusion.py::precision_fusion` documents for a cell with no
    live edges: `evidence -> 0`, `uncertainty -> sqrt(1/eps)` (very
    large) -- falls out of `eps`-floored division, no special case
    needed. §16.8's safety constraint (every cell keeps at least one
    incoming edge) means this shouldn't arise for `target` once the
    structural graph is bootstrapped, but is exercised and tested
    directly (`tests/test_structural_fusion.py::
    test_isolated_target_gets_precision_fusions_degenerate_fallback`).
    """
    batch = mu.shape[0]
    device = mu.device
    dtype = mu.dtype
    n_edges = edge_index.shape[0]

    if n_edges == 0:
        zeros = torch.zeros(batch, n_cells, device=device, dtype=dtype)
        mu_structural = torch.tanh(zeros + bias)
        evidence_structural = zeros
        uncertainty_structural = torch.sqrt(torch.full_like(zeros, 1.0 / eps))
        return mu_structural, evidence_structural, uncertainty_structural

    source, target = edge_index[:, 0], edge_index[:, 1]

    sender_mu = mu[:, source]  # (batch, E)
    sender_evidence = evidence[:, source]  # (batch, E)
    sender_uncertainty = uncertainty[:, source]  # (batch, E)

    m = w.unsqueeze(0) * sender_mu  # (batch, E) -- content only, §16.5
    precision_no_a = sender_evidence / (sender_uncertainty**2 + eps)  # (batch, E)

    edge_message_full = a * m  # (batch, E) -- = w*a*mu_i, §16.6's utility signal
    if edge_message_sink is not None:
        edge_message_full.retain_grad()
        edge_message_sink.append(edge_message_full)

    p = a * precision_no_a  # (batch, E) -- precision, §16.5

    def scatter_sum(values: torch.Tensor) -> torch.Tensor:
        """`values`: `(batch, E)`. Returns `(batch, n_cells)`, summed per target."""
        out = torch.zeros(batch, n_cells, device=device, dtype=values.dtype)
        out.index_add_(1, target, values)
        return out

    p_sum_raw = scatter_sum(p)
    denom = p_sum_raw + eps
    pm_sum = scatter_sum(edge_message_full * precision_no_a)  # = scatter_sum(p * m), routed through edge_message_full
    c = pm_sum / denom
    mu_structural = torch.tanh(c + bias)

    a_sum = scatter_sum(a)
    a_sq_sum = scatter_sum(a**2)
    n_eff = (a_sum**2) / (a_sq_sum + eps)

    evidence_structural = scatter_sum(a * sender_evidence) / (n_eff + eps)
    effective_precision = p_sum_raw / (n_eff + eps)
    base_uncertainty_sq = 1.0 / (effective_precision + eps)

    c_at_edge = c[:, target]  # (batch, E) -- gather each edge's target's c back
    disagreement_num = scatter_sum(p * (m - c_at_edge) ** 2)
    disagreement = disagreement_num / denom
    uncertainty_structural = torch.sqrt(base_uncertainty_sq + disagreement)

    return mu_structural, evidence_structural, uncertainty_structural
