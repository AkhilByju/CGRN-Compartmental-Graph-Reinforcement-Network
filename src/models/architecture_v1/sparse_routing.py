"""CellV1.1's sparse local/global routing (docs/architecture_v1.md's
"Sparse Self-Organizing Belief Graph"). Same association math as
`routing.py`'s `LocalAssociation`/`GlobalRouting`, but the expensive exact
scoring only ever runs on an `O(pool_size)`-sized LSH-selected candidate
set per cell (`lsh.py::lsh_candidates`), not all `n_cells - 1` other
cells -- `O(n_cells * pool_size)` total, not `O(n_cells^2)`. `routing.py`
is untouched and remains "CellV1 Dense Reference"
(docs/architecture_v1.md): correctness reference, and what
`tests/test_sparse_dense_consistency.py` checks this module against when
the candidate pool is large enough to cover every cell.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.lsh import (
    gather_rows,
    gather_scalar,
    gather_vector,
    lookup_value,
    lsh_candidates,
    random_hyperplanes,
)
from src.models.architecture_v1.routing import _safe_sqrt
from src.models.architecture_v1.sparsemax import MASK_VALUE, sparse_association_with_null


class SparseLocalAssociation(nn.Module):
    """Sparse counterpart to `routing.py::LocalAssociation`. Candidate
    selection: `lsh_candidates(z, z, ...)` (symmetric -- local search asks
    "who seems similar to me," using `z` for both sides). Exact scoring
    (semantic distance, `sparsemax`) then runs only on that pool. Mutual
    compatibility (`A^L_ij = sqrt(a_ij * a_ji)`) is computed via
    `lsh.gather_rows`: for each candidate `j`, look up `j`'s *own*
    candidate row and check whether `i` is in it -- if `j` never
    considered `i` a candidate at all (a real possibility with approximate
    LSH search, unlike the dense version's guaranteed-symmetric `a_ji`),
    the reciprocal weight is `0`, not an error.

    Returns `(a_local, candidate_idx)`: `a_local` is `(batch, n_cells,
    pool_size)`, `candidate_idx` is the same shape and says which cell
    each `a_local[..., p]` refers to -- downstream code (message/fusion/
    z-averaging) needs both, unlike the dense version's `(n_cells,
    n_cells)` `A^L` where position alone suffices.
    """

    def __init__(
        self,
        association_dim: int,
        lambda_: float = 1.0,
        tau: float = 1.0,
        eps: float = 1e-8,
        num_hashes: int = 2,
        bits: int = 6,
        chunk_size: int = 10,
        window: int = 0,
    ) -> None:
        super().__init__()
        self.metric = nn.Linear(association_dim, association_dim, bias=False)
        self.lambda_ = lambda_
        self.tau = tau
        self.eps = eps
        self.chunk_size = chunk_size
        self.window = window
        self.null_logit = nn.Parameter(torch.zeros(1))
        self.register_buffer("hyperplanes", random_hyperplanes(association_dim, num_hashes, bits))

    def forward(
        self, mu: torch.Tensor, uncertainty: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = mu.size(-1)
        candidate_idx = lsh_candidates(z, z, self.hyperplanes, self.chunk_size, self.window)

        mu_cand = gather_scalar(mu, candidate_idx)
        u_cand = gather_scalar(uncertainty, candidate_idx)
        z_cand = gather_vector(z, candidate_idx)

        z_diff = z.unsqueeze(2) - z_cand  # (batch, n, pool, d)
        metric_term = self.metric(z_diff).pow(2).sum(dim=-1)

        mu_diff_sq = (mu.unsqueeze(-1) - mu_cand) ** 2
        u_sq_sum = uncertainty.unsqueeze(-1) ** 2 + u_cand**2
        disagreement_term = self.lambda_ * mu_diff_sq / (u_sq_sum + self.eps)

        distance = metric_term + disagreement_term
        scores = -distance / self.tau

        self_idx = torch.arange(n, device=mu.device).view(1, n, 1)
        scores = scores.masked_fill(candidate_idx == self_idx, MASK_VALUE)

        a = sparse_association_with_null(scores, self.null_logit)  # (batch, n, pool)

        reciprocal_owner = gather_rows(candidate_idx, candidate_idx)  # (batch, n, pool, pool)
        reciprocal_weight_rows = gather_rows(a, candidate_idx)  # (batch, n, pool, pool)
        is_self = reciprocal_owner == self_idx.unsqueeze(-1)
        a_reverse = (reciprocal_weight_rows * is_self.to(a.dtype)).sum(dim=-1)  # (batch, n, pool)

        mutual = a * a_reverse
        a_local = _safe_sqrt(mutual, self.eps)
        return a_local, candidate_idx


class SparseGlobalRouting(nn.Module):
    """Sparse counterpart to `routing.py::GlobalRouting`. Candidate
    selection: `lsh_candidates(k, q, ...)` (asymmetric -- `key_vectors=k`
    gets sorted/indexed, each cell's own `q` looks up where it would
    insert into that sorted order via `torch.searchsorted`). The
    `-gamma * A^L_ij` penalty term is looked up against the *local*
    routing's candidate pool (`lsh.lookup_value`) rather than a dense
    `A^L`; a global candidate that wasn't in the local pool at all is
    treated as `A^L_ij = 0` (not already a local neighbor), which is
    exactly what the dense version's `0` entries meant too.
    """

    def __init__(
        self,
        association_dim: int,
        key_dim: int | None = None,
        gamma: float = 1.0,
        eps: float = 1e-8,
        num_hashes: int = 2,
        bits: int = 6,
        chunk_size: int = 4,
        window: int = 0,
    ) -> None:
        super().__init__()
        key_dim = key_dim or association_dim
        self.key_dim = key_dim
        self.query = nn.Linear(association_dim, key_dim, bias=False)
        self.key = nn.Linear(association_dim, key_dim, bias=False)
        self.gamma = gamma
        self.eps = eps
        self.chunk_size = chunk_size
        self.window = window
        self.null_logit = nn.Parameter(torch.zeros(1))
        self.register_buffer("hyperplanes", random_hyperplanes(key_dim, num_hashes, bits))

    def forward(
        self,
        z: torch.Tensor,
        need: torch.Tensor,
        offer: torch.Tensor,
        local_candidate_idx: torch.Tensor,
        local_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = z.size(-2)
        q = self.query(z)
        k = self.key(z)

        candidate_idx = lsh_candidates(k, q, self.hyperplanes, self.chunk_size, self.window)

        k_cand = gather_vector(k, candidate_idx)  # (batch, n, pool, key_dim)
        content_term = torch.einsum("bnd,bnpd->bnp", q, k_cand) / (self.key_dim**0.5)

        offer_cand = gather_scalar(offer, candidate_idx)
        need_term = torch.log(need + self.eps).unsqueeze(-1)
        offer_term = torch.log(offer_cand + self.eps)

        local_penalty = lookup_value(local_candidate_idx, local_weight, candidate_idx)

        scores = content_term + need_term + offer_term - self.gamma * local_penalty

        self_idx = torch.arange(n, device=z.device).view(1, n, 1)
        scores = scores.masked_fill(candidate_idx == self_idx, MASK_VALUE)

        a_global = sparse_association_with_null(scores, self.null_logit)
        return a_global, candidate_idx
