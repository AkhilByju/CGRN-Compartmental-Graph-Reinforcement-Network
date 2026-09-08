"""BeliefLayer -- the belief-aggregation operator for CellV0.

Combines N incoming `BeliefCell`s (`cell.py`) into one outgoing
`BeliefCell`, per the candidate methods recorded in docs/research_log.md
("CellV0 aggregation candidates") and docs/architecture_v0.md Sec 1:

- "reliability"         -- Method A: reliability-weighted consensus
- "support_conflict"    -- Method B: positive/negative support integration
- "precision"           -- Method C: precision-based belief fusion
- "normalized_precision" -- Method D: precision fusion with evidence/
  uncertainty normalized by incoming relevance (`sum(g)`), added in
  Experiment 004F (docs/research_log.md "Experiment 004F") specifically to
  fix 004D/004E's finding that Method C's `evidence`/`uncertainty` are not
  scale-stable -- see its own docstring below.
- "scale_stable_precision" -- Method E ("CellV0.1"): precision fusion
  normalized by an *effective source count* `N_eff` (a participation-ratio/
  Kish's-effective-sample-size statistic over the relevance gates) rather
  than the raw sum used by Method D, added in Experiment 004I
  (docs/research_log.md "Experiment 004I") -- see its own docstring below
  for why this is a refinement of Method D, not a third independent fix.

All five share the same learned-connection structure: a content weight
`w_ij` (how cell j's content affects cell i) and a relevance gate
`g_ij = sigmoid(a_ij)` (how relevant cell j is to cell i), so switching
`aggregation` is the only thing that changes between them. Comparing
"reliability"/"support_conflict"/"precision" against each other and a plain
MLP baseline was Experiment 002/003 (docs/experiment_protocol.md) --
**none of them is chosen as "the" CellV0 aggregation rule**; "precision" is
Experiment 004's ongoing subject of study, not a conclusion. Methods D and E
are both deliberate, user-specified additions (not agent-invented
aggregation rules -- CLAUDE.md Sec 2), kept alongside Method C rather than
replacing it.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from src.models.architecture_v0.cell import BeliefCell

AggregationMethod = Literal[
    "reliability",
    "support_conflict",
    "precision",
    "normalized_precision",
    "scale_stable_precision",
]

AGGREGATION_METHODS: tuple[AggregationMethod, ...] = (
    "reliability",
    "support_conflict",
    "precision",
    "normalized_precision",
    "scale_stable_precision",
)


def _reliability_consensus(
    m: torch.Tensor,
    g: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method A. `m`, `g`, `e_j`, `u_j` broadcast to `(batch, out_cells,
    in_cells)`; returns `(mu, evidence, uncertainty)` of shape `(batch,
    out_cells)`."""
    r = g * e_j / (1.0 + u_j)
    alpha = r / (r.sum(dim=-1, keepdim=True) + eps)
    c = (alpha * m).sum(dim=-1)
    mu = torch.tanh(c + bias)
    evidence = r.sum(dim=-1)
    disagreement = (alpha * (m - c.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(disagreement + 1.0 / (evidence + eps))
    return mu, evidence, uncertainty


def _support_conflict(
    m: torch.Tensor,
    g: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method B."""
    r = g * e_j / (1.0 + u_j)
    q = torch.tanh(m)
    positive = (r * torch.clamp(q, min=0.0)).sum(dim=-1)
    negative = (r * torch.clamp(-q, min=0.0)).sum(dim=-1)
    evidence = positive + negative
    c = (positive - negative) / (evidence + eps)
    mu = torch.tanh(c + bias)
    conflict = 2.0 * torch.minimum(positive, negative) / (evidence + eps)
    uncertainty = conflict + 1.0 / torch.sqrt(evidence + eps)
    return mu, evidence, uncertainty


def _precision_fusion(
    m: torch.Tensor,
    g: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method C."""
    precision = g * e_j / (u_j**2 + eps)
    alpha = precision / (precision.sum(dim=-1, keepdim=True) + eps)
    c = (alpha * m).sum(dim=-1)
    mu = torch.tanh(c + bias)
    evidence = (g * e_j).sum(dim=-1)
    base_uncertainty_sq = 1.0 / (precision.sum(dim=-1) + eps)
    disagreement = (alpha * (m - c.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(base_uncertainty_sq + disagreement)
    return mu, evidence, uncertainty


def _normalized_precision_fusion(
    m: torch.Tensor,
    g: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method D -- "Normalized Precision" (Experiment 004F,
    docs/research_log.md). Identical to Method C except `evidence` and the
    base-uncertainty term are each divided by `G = sum(g)`, the total
    incoming relevance, instead of left as raw unnormalized sums.

    Motivation (Experiment 004D/004E): Method C treats `hidden_cells`
    incoming cells as `hidden_cells` independent observations, so
    `evidence = sum(g * e_j)` and `precision.sum()` both grow with the
    number of incoming cells even when those cells carry the exact same
    information (004E's duplication test) -- purely a width artifact, not
    a property of the information itself. Method D's conceptual fix: a
    layer's incoming cells are different *representations*, not
    automatically independent *observations*, so evidence/precision are
    averaged (weighted by relevance `g`) rather than summed:

        G_i    = sum_j(g_ij)
        e_i    = sum_j(g_ij * e_j) / G_i        (was: sum_j(g_ij * e_j))
        p_bar_i = sum_j(p_ij) / G_i              (was: sum_j(p_ij))
        u_base_i = sqrt(1 / (p_bar_i + eps))

    `alpha` (the content-weighting used for `mu`) and the `disagreement`
    term are left byte-for-byte identical to Method C -- cells still
    compete for influence over `mu` exactly as before; only the parts that
    made `evidence`/`uncertainty` scale with `hidden_cells` change. A
    duplication test that repeats the *same* content `N` times should now
    leave `evidence`/`uncertainty` approximately unchanged as `N` grows
    (verified in `tests/test_belief_layer.py` and
    `experiments/004_cellv0_scaling/duplication_test.py`), unlike Method C.
    """
    precision = g * e_j / (u_j**2 + eps)
    alpha = precision / (precision.sum(dim=-1, keepdim=True) + eps)
    c = (alpha * m).sum(dim=-1)
    mu = torch.tanh(c + bias)

    total_relevance = g.sum(dim=-1) + eps
    evidence = (g * e_j).sum(dim=-1) / total_relevance
    mean_precision = precision.sum(dim=-1) / total_relevance
    base_uncertainty_sq = 1.0 / (mean_precision + eps)
    disagreement = (alpha * (m - c.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(base_uncertainty_sq + disagreement)
    return mu, evidence, uncertainty


def _scale_stable_precision_fusion(
    m: torch.Tensor,
    g: torch.Tensor,
    e_j: torch.Tensor,
    u_j: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Method E -- "Scale-Stable Precision" / "CellV0.1" (Experiment 004I,
    docs/research_log.md). A refinement of Method D
    (`_normalized_precision_fusion`): identical in every respect except what
    `evidence`/the base-uncertainty term are normalized by.

    Method D divides by `G = sum(g)`, the raw total incoming relevance --
    correct for the duplication test (N identical, equally-relevant copies),
    but `G` still grows with the number of incoming connections even when
    most of them are only weakly relevant (many small-but-nonzero `g`
    values). Method E instead divides by an **effective source count**, the
    participation ratio (Kish's effective sample size) over the relevance
    gates:

        N_eff_i = (sum_j(g_ij))^2 / (sum_j(g_ij^2) + eps)

    Ten equally-relevant inputs (`g_ij` all equal) give `N_eff ~ 10`; one
    dominant input among many near-irrelevant ones gives `N_eff ~ 1`; adding
    connections with `g_ij` genuinely close to 0 barely moves it (though
    many connections with *moderate* weak relevance still add up -- `N_eff`
    tracks the *effective* count, not a literal "ignore small g" cutoff).

    For N identical, equally-weighted duplicates (004E's duplication test),
    `N_eff = (N*g0)^2 / (N*g0^2) = N` for any `g0 > 0` -- so, like Method D,
    Method E's `evidence`/`uncertainty` do not change with `N`. The two
    methods are NOT numerically identical here, though: Method D's
    `G = N*g0` cancels the `N` and the `g0` factor together against the
    numerator, always settling at exactly `evidence = e0`; Method E's `N`
    cancels but `g0` does not, settling at `evidence = g0 * e0` instead --
    the same value at every `N`, but a different (smaller, since
    `g0 = sigmoid(.) < 1`) constant than Method D's. Both are legitimately
    "duplicate-invariant" by 004E's definition (constant in `N`); they only
    coincide numerically in the limit `g0 -> 1`.

    Where the two methods actually produce different *behavior*, not just a
    different constant, is unequal relevance: with a uniform `e_j` across
    incoming cells, Method D's `evidence` is *always* exactly `e0` no matter
    how relevance is distributed (the raw sum cancels identically
    regardless of composition), so it cannot distinguish "one relevant
    source" from "one relevant source plus fifty weak ones" the way
    Method E's `N_eff` does (see `tests/test_scale_stable_precision.py`).

    Content aggregation (`alpha`, `c`, `mu`) and the disagreement term are
    byte-for-byte identical to Methods C/D -- only `evidence` and the base-
    uncertainty term change, exactly as with Method D:

        e_i      = sum_j(g_ij * e_j) / N_eff_i     (was: sum_j(g_ij * e_j))
        p_eff_i  = sum_j(p_ij) / N_eff_i            (was: sum_j(p_ij))
        u_base_i = sqrt(1 / (p_eff_i + eps))
    """
    precision = g * e_j / (u_j**2 + eps)
    alpha = precision / (precision.sum(dim=-1, keepdim=True) + eps)
    c = (alpha * m).sum(dim=-1)
    mu = torch.tanh(c + bias)

    g_sum = g.sum(dim=-1)
    g_sq_sum = (g**2).sum(dim=-1)
    n_eff = (g_sum**2) / (g_sq_sum + eps)

    evidence = (g * e_j).sum(dim=-1) / (n_eff + eps)
    effective_precision = precision.sum(dim=-1) / (n_eff + eps)
    base_uncertainty_sq = 1.0 / (effective_precision + eps)
    disagreement = (alpha * (m - c.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(base_uncertainty_sq + disagreement)
    return mu, evidence, uncertainty


_AGGREGATORS = {
    "reliability": _reliability_consensus,
    "support_conflict": _support_conflict,
    "precision": _precision_fusion,
    "normalized_precision": _normalized_precision_fusion,
    "scale_stable_precision": _scale_stable_precision_fusion,
}


class BeliefLayer(nn.Module):
    """Maps a `BeliefCell` over `in_cells` to a `BeliefCell` over
    `out_cells`. Input/output `mu`/`evidence`/`uncertainty` have shape
    `(batch, in_cells)` / `(batch, out_cells)` respectively.
    """

    def __init__(
        self,
        in_cells: int,
        out_cells: int,
        aggregation: AggregationMethod = "reliability",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown aggregation method '{aggregation}'. Expected one of "
                f"{AGGREGATION_METHODS}."
            )
        self.in_cells = in_cells
        self.out_cells = out_cells
        self.aggregation = aggregation
        self.eps = eps

        self.content_weight = nn.Parameter(torch.empty(out_cells, in_cells))
        self.relevance_logit = nn.Parameter(torch.zeros(out_cells, in_cells))
        self.bias = nn.Parameter(torch.zeros(out_cells))
        nn.init.kaiming_uniform_(self.content_weight, a=math.sqrt(5))

    def forward(
        self,
        incoming: BeliefCell,
        source_participation: torch.Tensor | None = None,
    ) -> BeliefCell:
        """`source_participation`, if given, is a per-example, per-source-cell
        gain of shape `(batch, in_cells)` in `[0, 1]` -- each incoming cell
        `i`'s learned relevance `g[j, i]` is multiplied by its own
        `participation[i]` (shared across every receiver `j`) before the
        aggregation formula runs, unchanged, on the result. This is the one
        hook CellV1.6's `PrecisionRegulatedAssemblyGate`
        (`src/models/architecture_v1/assembly_gate.py`) needs; passing
        `None` (the default) leaves this layer byte-for-byte identical to
        the CellV0/CellV0.1 layer every prior experiment used.
        """
        if incoming.mu.shape[-1] != self.in_cells:
            raise ValueError(
                f"Expected {self.in_cells} incoming cells, got shape {tuple(incoming.mu.shape)}."
            )

        # (batch, out_cells, in_cells)
        m = self.content_weight.unsqueeze(0) * incoming.mu.unsqueeze(1)
        g = torch.sigmoid(self.relevance_logit).unsqueeze(0)
        if source_participation is not None:
            if source_participation.shape[-1] != self.in_cells:
                raise ValueError(
                    f"source_participation must have {self.in_cells} source cells, "
                    f"got shape {tuple(source_participation.shape)}."
                )
            # g_effective[b, j, i] = g[j, i] * participation[b, i]
            g = g * source_participation.unsqueeze(1)
        e_j = incoming.evidence.unsqueeze(1)
        u_j = incoming.uncertainty.unsqueeze(1)

        aggregator = _AGGREGATORS[self.aggregation]
        mu, evidence, uncertainty = aggregator(m, g, e_j, u_j, self.bias, self.eps)

        return BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty)
