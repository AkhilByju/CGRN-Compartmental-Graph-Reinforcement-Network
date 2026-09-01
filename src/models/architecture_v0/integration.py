"""BeliefLayer -- the belief-aggregation operator for CellV0.

Combines N incoming `BeliefCell`s (`cell.py`) into one outgoing
`BeliefCell`, per the three candidate methods recorded in
docs/research_log.md ("CellV0 aggregation candidates") and
docs/architecture_v0.md Sec 1:

- "reliability"      -- Method A: reliability-weighted consensus
- "support_conflict" -- Method B: positive/negative support integration
- "precision"        -- Method C: precision-based belief fusion

All three share the same learned-connection structure: a content weight
`w_ij` (how cell j's content affects cell i) and a relevance gate
`g_ij = sigmoid(a_ij)` (how relevant cell j is to cell i), so switching
`aggregation` is the only thing that changes between them. Comparing the
three against each other and a plain MLP baseline is Experiment 002/003
(docs/experiment_protocol.md) -- **none of the three is chosen yet**; do
not treat any one of them as "the" CellV0 aggregation rule until that
comparison is run.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from src.models.architecture_v0.cell import BeliefCell

AggregationMethod = Literal["reliability", "support_conflict", "precision"]

AGGREGATION_METHODS: tuple[AggregationMethod, ...] = (
    "reliability",
    "support_conflict",
    "precision",
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


_AGGREGATORS = {
    "reliability": _reliability_consensus,
    "support_conflict": _support_conflict,
    "precision": _precision_fusion,
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

    def forward(self, incoming: BeliefCell) -> BeliefCell:
        if incoming.mu.shape[-1] != self.in_cells:
            raise ValueError(
                f"Expected {self.in_cells} incoming cells, got shape {tuple(incoming.mu.shape)}."
            )

        # (batch, out_cells, in_cells)
        m = self.content_weight.unsqueeze(0) * incoming.mu.unsqueeze(1)
        g = torch.sigmoid(self.relevance_logit).unsqueeze(0)
        e_j = incoming.evidence.unsqueeze(1)
        u_j = incoming.uncertainty.unsqueeze(1)

        aggregator = _AGGREGATORS[self.aggregation]
        mu, evidence, uncertainty = aggregator(m, g, e_j, u_j, self.bias, self.eps)

        return BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty)
