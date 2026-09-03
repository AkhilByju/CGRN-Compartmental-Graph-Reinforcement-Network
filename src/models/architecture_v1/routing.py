"""Dynamic graph construction (docs/architecture_v1.md Part I, Part III):
`LocalAssociation` computes the input-dependent, mutual, sparse local
graph `A^L`; `GlobalRouting` computes the input-dependent, directed,
sparse long-range graph `A^G`. Neither has a `cluster_id` or fixed
neighbor count -- both come out of `sparsemax` (`sparsemax.py`), which can
assign a cell anywhere from zero real neighbors (everything routed to the
learned "null" option) up to `n_cells - 1`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.sparsemax import MASK_VALUE, sparse_association_with_null


def _mask_diagonal(scores: torch.Tensor) -> torch.Tensor:
    """Excludes self-association: `scores[..., i, i] = MASK_VALUE`."""
    n = scores.size(-1)
    eye = torch.eye(n, device=scores.device, dtype=torch.bool)
    return scores.masked_fill(eye, MASK_VALUE)


def _safe_sqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`sqrt` with `x == 0` gradient-safe. Plain `sqrt` has an infinite
    derivative at exactly 0 (`d/dx sqrt(x) = 1 / (2 sqrt(x))`) -- and
    `sparsemax` *routinely* produces exact zeros by design, so
    `sqrt(a_ij * a_ji)` in `LocalAssociation` would poison the backward
    pass with `nan`/`inf` gradients almost every step. Shifting by
    `sqrt(eps)` keeps `x == 0 -> 0` exactly (so true sparsity is preserved
    at the value level, unlike `sqrt(x + eps)` alone, which would leave a
    small nonzero floor everywhere) while keeping the gradient finite
    everywhere (`sqrt(x + eps)` is smooth on `x >= 0`).
    """
    return torch.sqrt(torch.clamp(x, min=0.0) + eps) - eps**0.5


class LocalAssociation(nn.Module):
    """`docs/architecture_v1.md` Part I. Semantic distance:

        D_ij = ||L(z_i - z_j)||^2 + lambda * (mu_i - mu_j)^2 / (u_i^2 + u_j^2 + eps)

    `L` is a single learned linear map, shared by every pair (§"Every cell
    uses the same L"). Association score `s_ij = -D_ij / tau`, self-loops
    excluded, then `sparsemax` with a learned null option makes the row
    sparse (§ Steps 2-3). Local edges are then made mutual (§ Step 4):

        A^L_ij = sqrt(a_ij * a_ji)

    so a cell isn't "locally associated" with another just because it
    alone wants the connection.
    """

    def __init__(self, association_dim: int, lambda_: float = 1.0, tau: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.metric = nn.Linear(association_dim, association_dim, bias=False)
        self.lambda_ = lambda_
        self.tau = tau
        self.eps = eps
        self.null_logit = nn.Parameter(torch.zeros(1))

    def forward(self, mu: torch.Tensor, uncertainty: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """`mu`/`uncertainty`: `(batch, n_cells)`. `z`: `(batch, n_cells,
        association_dim)`. Returns `A^L`: `(batch, n_cells, n_cells)`,
        symmetric, non-negative, zero diagonal."""
        z_diff = z.unsqueeze(-2) - z.unsqueeze(-3)  # (batch, n, n, d): [..., i, j, :] = z_i - z_j
        metric_term = self.metric(z_diff).pow(2).sum(dim=-1)

        mu_diff_sq = (mu.unsqueeze(-1) - mu.unsqueeze(-2)) ** 2
        u_sq_sum = uncertainty.unsqueeze(-1) ** 2 + uncertainty.unsqueeze(-2) ** 2
        disagreement_term = self.lambda_ * mu_diff_sq / (u_sq_sum + self.eps)

        distance = metric_term + disagreement_term
        scores = _mask_diagonal(-distance / self.tau)
        a = sparse_association_with_null(scores, self.null_logit)

        mutual = a * a.transpose(-1, -2)
        return _safe_sqrt(mutual, self.eps)


class GlobalRouting(nn.Module):
    """`docs/architecture_v1.md` Part III, Step 3. Directed, content-based
    long-range routing:

        S^G_ij = (q_i . k_j) / sqrt(d) + log(need_i + eps) + log(offer_j + eps) - gamma * A^L_ij

    `q = W_Q z`, `k = W_K z` (separate learned linear maps -- global
    matching is directional query/key compatibility, not the symmetric
    distance local association uses, since two different concepts may need
    to interact despite being dissimilar). The `-gamma * A^L_ij` term
    steers global edges toward pairs that *aren't* already local
    neighbors. `sparsemax` with a (separate) learned null option again
    makes this sparse and lets a cell route to nowhere.
    """

    def __init__(self, association_dim: int, key_dim: int | None = None, gamma: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        key_dim = key_dim or association_dim
        self.key_dim = key_dim
        self.query = nn.Linear(association_dim, key_dim, bias=False)
        self.key = nn.Linear(association_dim, key_dim, bias=False)
        self.gamma = gamma
        self.eps = eps
        self.null_logit = nn.Parameter(torch.zeros(1))

    def forward(
        self, z: torch.Tensor, need: torch.Tensor, offer: torch.Tensor, a_local: torch.Tensor
    ) -> torch.Tensor:
        """`z`: `(batch, n_cells, association_dim)`. `need`/`offer`:
        `(batch, n_cells)`, both in `(0, 1)`. `a_local`: `(batch, n_cells,
        n_cells)` (this step's `A^L`). Returns `A^G`: `(batch, n_cells,
        n_cells)`, directed (not symmetrized), zero diagonal."""
        q = self.query(z)
        k = self.key(z)
        content_term = torch.matmul(q, k.transpose(-1, -2)) / (self.key_dim**0.5)

        need_term = torch.log(need + self.eps).unsqueeze(-1)
        offer_term = torch.log(offer + self.eps).unsqueeze(-2)

        scores = content_term + need_term + offer_term - self.gamma * a_local
        scores = _mask_diagonal(scores)
        return sparse_association_with_null(scores, self.null_logit)
