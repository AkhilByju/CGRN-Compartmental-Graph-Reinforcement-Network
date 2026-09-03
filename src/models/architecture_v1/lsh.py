"""Locality-sensitive-hashing candidate selection for CellV1.1
(docs/architecture_v1.md's "Sparse Self-Organizing Belief Graph"
extension). Finds a small, fixed-size candidate pool per cell in
`O(N log N)` (a sort, not an all-pairs comparison), so the expensive exact
association math (`sparse_routing.py`) only ever runs on `O(N * pool_size)`
pairs, not `O(N^2)`.

**The actual mechanism used here is hash-table lookup via sort +
`searchsorted`, not Reformer's chunk-position trick directly** -- Reformer
sorts a single sequence and uses each token's own position in that sorted
order to find its chunk, which only works because a token is always its
own query *and* key. CellV1.1's global routing is asymmetric (`q_i`
queries a `k_j`-sorted structure `q != k`), so a query generally isn't an
element of the sorted key sequence at all. `torch.searchsorted` finds
where a query's hash code *would* insert into the sorted key-hash array --
still `O(log N)` per query given a sorted array -- and the candidate
window is read off around that insertion point. This works identically
for the symmetric (local, `query_vectors is key_vectors`) case, so one
function serves both `sparse_routing.py::SparseLocalAssociation` and
`SparseGlobalRouting`.

No `N x N` tensor is ever constructed here.
"""

from __future__ import annotations

import torch


def random_hyperplanes(
    dim: int, num_hashes: int, bits: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """`(num_hashes, dim, bits)` -- fixed (not learned) random projection
    vectors, one set per hash round. Not an `nn.Parameter`: the whole
    point is a cheap, fixed index, and gradients can't flow through a
    discrete bucket-membership decision anyway (docs/architecture_v1.md's
    "What about differentiability" section)."""
    return torch.randn(num_hashes, dim, bits, generator=generator)


def bucket_ids(vectors: torch.Tensor, hyperplanes: torch.Tensor) -> torch.Tensor:
    """`vectors`: `(batch, n, dim)`. `hyperplanes`: `(num_hashes, dim,
    bits)`. Returns `(batch, num_hashes, n)` integer bucket ids in `[0,
    2^bits)` -- `sign(vector @ hyperplane)` per bit, packed into an
    integer. Vectors that hash to the same bucket are, by construction,
    on the same side of every one of that round's `bits` random
    hyperplanes -- i.e. likely similar in direction (the standard
    sign-random-projection LSH guarantee, which approximates cosine
    similarity)."""
    proj = torch.einsum("bnd,hdk->bhnk", vectors, hyperplanes.to(vectors.dtype))
    bits_on = (proj > 0).long()
    bits = hyperplanes.shape[-1]
    powers = (2 ** torch.arange(bits, device=vectors.device)).view(1, 1, 1, -1)
    return (bits_on * powers).sum(dim=-1)


def lsh_candidates(
    key_vectors: torch.Tensor,
    query_vectors: torch.Tensor,
    hyperplanes: torch.Tensor,
    chunk_size: int,
    window: int = 0,
) -> torch.Tensor:
    """Returns `candidate_idx`: `(batch, n, num_hashes * (2*window+1) *
    chunk_size)`, long, values in `[0, n)` -- for every cell/query `i`,
    the original indices of its candidate pool, gathered without ever
    forming an `(n, n)` tensor. Duplicates are possible (the same cell can
    appear more than once, across hash rounds or overlapping windows) --
    accepted as a minor approximation artifact, not deduplicated (would
    need scatter-based set logic that reintroduces the cost this is meant
    to avoid); `sparse_routing.py`'s exact scoring still runs correctly
    over a pool with a repeated entry, just very mildly over-weighting it.

    `key_vectors`/`query_vectors`: `(batch, n, dim)`. Pass the same tensor
    for both (local/symmetric routing) or different ones (global routing:
    `key_vectors=k`, `query_vectors=q`).
    """
    batch, n, _ = key_vectors.shape
    num_hashes = hyperplanes.shape[0]
    key_buckets = bucket_ids(key_vectors, hyperplanes)  # (batch, H, n)
    query_buckets = bucket_ids(query_vectors, hyperplanes)  # (batch, H, n)

    num_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    within_chunk = torch.arange(chunk_size, device=key_vectors.device)

    pools = []
    for h in range(num_hashes):
        key_h = key_buckets[:, h].contiguous()
        query_h = query_buckets[:, h].contiguous()
        sort_idx = torch.argsort(key_h, dim=-1)  # (batch, n)
        sorted_key_buckets = torch.gather(key_h, -1, sort_idx).contiguous()  # (batch, n)
        insert_pos = torch.searchsorted(sorted_key_buckets, query_h).clamp(max=n - 1)
        chunk_of = insert_pos // chunk_size  # (batch, n)

        for offset in range(-window, window + 1):
            chunk = (chunk_of + offset).clamp(0, num_chunks - 1)  # (batch, n)
            start = chunk * chunk_size
            positions = (start.unsqueeze(-1) + within_chunk).clamp(max=n - 1)  # (batch, n, chunk_size)
            gathered = torch.gather(
                sort_idx.unsqueeze(1).expand(-1, n, -1), -1, positions
            )  # (batch, n, chunk_size)
            pools.append(gathered)

    return torch.cat(pools, dim=-1)


def gather_scalar(values: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    """`values`: `(batch, n)`. `candidate_idx`: `(batch, n, pool)`.
    Returns `(batch, n, pool)` = `values[b, candidate_idx[b, i, p]]` --
    `O(n * pool)`, no `(n, n)` tensor materialized (the `expand` below is
    a zero-cost stride-0 view, not a copy)."""
    n = values.shape[-1]
    return torch.gather(values.unsqueeze(1).expand(-1, n, -1), -1, candidate_idx)


def gather_vector(values: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    """`values`: `(batch, n, dim)`. `candidate_idx`: `(batch, n, pool)`.
    Returns `(batch, n, pool, dim)`."""
    n, dim = values.shape[-2], values.shape[-1]
    expanded = values.unsqueeze(1).expand(-1, n, -1, -1)  # (batch, n, n, dim) view, zero-cost
    index = candidate_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
    return torch.gather(expanded, 2, index)


def gather_rows(matrix: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """`matrix`: `(batch, n, pool)` (e.g. `candidate_idx` itself, or a
    per-candidate weight `a`). `row_idx`: `(batch, n, pool)`, values in
    `[0, n)` (typically `candidate_idx` again). Returns `(batch, n, pool,
    pool)`: for every `(i, p)`, the *entire* row `matrix[b,
    row_idx[b,i,p], :]` -- i.e. "cell `j`'s own candidate row," where
    `j = row_idx[b,i,p]`. `O(n * pool^2)`, not `O(n^2)`: used by
    `sparse_routing.py::SparseLocalAssociation` to make association mutual
    (`A^L_ij = sqrt(a_ij * a_ji)`) without ever forming an `(n, n)` tensor
    -- `a_ji` is found by checking whether `i` appears anywhere in `j`'s
    own (small) candidate row, not by looking it up in a dense matrix."""
    batch, n, pool = matrix.shape
    expanded = matrix.unsqueeze(1).expand(batch, n, n, pool)  # (batch, n, n, pool) view, zero-cost
    index = row_idx.unsqueeze(-1).expand(batch, n, pool, pool)
    return torch.gather(expanded, 2, index)


def lookup_value(
    reference_idx: torch.Tensor, reference_value: torch.Tensor, query_idx: torch.Tensor
) -> torch.Tensor:
    """`reference_idx`/`reference_value`: `(batch, n, pool_ref)` -- a small
    per-cell "known values at these indices" table (e.g. one cell's local
    candidate indices and the local-association weight it gave each).
    `query_idx`: `(batch, n, pool_q)`. Returns `(batch, n, pool_q)`: for
    each query index, `reference_value` where `reference_idx` equals it,
    or `0` if not found (i.e. the queried cell isn't among the
    `pool_ref`-sized reference set at all). `O(n * pool_q * pool_ref)`,
    both pool sizes small constants -- used by
    `sparse_routing.py::SparseGlobalRouting` to look up "is this global
    candidate already a local neighbor, and with what weight" without a
    dense `(n, n)` lookup table."""
    match = reference_idx.unsqueeze(-2) == query_idx.unsqueeze(-1)  # (batch, n, pool_q, pool_ref)
    return (reference_value.unsqueeze(-2) * match.to(reference_value.dtype)).sum(dim=-1)
