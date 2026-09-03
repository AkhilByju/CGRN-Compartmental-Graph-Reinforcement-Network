"""The correctness check `docs/architecture_v1.md` keeps CellV1 Dense
around for: at a small `n_cells`, with the LSH candidate pool configured
to cover every cell (one hash round, `chunk_size == n_cells`), the sparse
implementation must reduce to the *exact same computation* as the dense
one -- not just "similar," but numerically identical up to float32
rounding. If this test ever fails, the sparse implementation has diverged
from the association math it's supposed to be a scalable restriction of.
"""

from __future__ import annotations

import torch

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.dynamics import DynamicBeliefGraphStep
from src.models.architecture_v1.sparse_dynamics import SparseDynamicBeliefGraphStep


def _random_population(batch: int, n: int, d: int, generator: torch.Generator) -> BeliefCellV1:
    z = torch.randn(batch, n, d, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.rand(batch, n, generator=generator) + 0.5,
        uncertainty=torch.rand(batch, n, generator=generator) + 0.5,
        z=z,
    )


def _build_matched_pair(n: int, d: int, hidden_dim: int, use_global_routing: bool):
    dense = DynamicBeliefGraphStep(
        n_cells=n, association_dim=d, hidden_dim=hidden_dim, use_global_routing=use_global_routing
    )
    sparse = SparseDynamicBeliefGraphStep(
        n_cells=n,
        association_dim=d,
        hidden_dim=hidden_dim,
        use_global_routing=use_global_routing,
        num_hashes_local=1,
        chunk_size_local=n,
        window_local=0,
        num_hashes_global=1,
        chunk_size_global=n,
        window_global=0,
    )
    sparse.local_association.metric.load_state_dict(dense.local_association.metric.state_dict())
    sparse.local_association.null_logit.data.copy_(dense.local_association.null_logit.data)
    sparse.message.load_state_dict(dense.message.state_dict())
    sparse.semantic_update.load_state_dict(dense.semantic_update.state_dict())
    sparse.write_gate.load_state_dict(dense.write_gate.state_dict())
    sparse.local_bias.data.copy_(dense.local_bias.data)
    sparse.fuse_bias.data.copy_(dense.fuse_bias.data)
    sparse.source_gate_logit.data.copy_(dense.source_gate_logit.data)
    if use_global_routing:
        sparse.global_routing.query.load_state_dict(dense.global_routing.query.state_dict())
        sparse.global_routing.key.load_state_dict(dense.global_routing.key.state_dict())
        sparse.global_routing.null_logit.data.copy_(dense.global_routing.null_logit.data)
        sparse.need.load_state_dict(dense.need.state_dict())
        sparse.offer.load_state_dict(dense.offer.state_dict())
        sparse.global_bias.data.copy_(dense.global_bias.data)
    return dense, sparse


def test_sparse_matches_dense_when_pool_covers_everyone_full() -> None:
    torch.manual_seed(0)
    batch, n, d = 2, 12, 4
    dense, sparse = _build_matched_pair(n, d, hidden_dim=8, use_global_routing=True)

    g = torch.Generator().manual_seed(1)
    cells = _random_population(batch, n, d, g)

    out_dense = dense(cells)
    out_sparse = sparse(cells)

    assert torch.allclose(out_dense.mu, out_sparse.mu, atol=1e-4)
    assert torch.allclose(out_dense.evidence, out_sparse.evidence, atol=1e-4)
    assert torch.allclose(out_dense.uncertainty, out_sparse.uncertainty, atol=1e-4)
    assert torch.allclose(out_dense.z, out_sparse.z, atol=1e-4)


def test_sparse_matches_dense_when_pool_covers_everyone_local_only() -> None:
    torch.manual_seed(2)
    batch, n, d = 2, 10, 4
    dense, sparse = _build_matched_pair(n, d, hidden_dim=8, use_global_routing=False)

    g = torch.Generator().manual_seed(3)
    cells = _random_population(batch, n, d, g)

    out_dense = dense(cells)
    out_sparse = sparse(cells)

    assert torch.allclose(out_dense.mu, out_sparse.mu, atol=1e-4)
    assert torch.allclose(out_dense.z, out_sparse.z, atol=1e-4)


def test_sparse_matches_dense_over_multiple_steps() -> None:
    # Confirms the consistency isn't a one-step coincidence -- run both 3
    # steps deep from the same starting population.
    torch.manual_seed(4)
    batch, n, d = 2, 10, 4
    dense, sparse = _build_matched_pair(n, d, hidden_dim=8, use_global_routing=True)

    g = torch.Generator().manual_seed(5)
    cells_dense = _random_population(batch, n, d, g)
    cells_sparse = BeliefCellV1(
        mu=cells_dense.mu.clone(),
        evidence=cells_dense.evidence.clone(),
        uncertainty=cells_dense.uncertainty.clone(),
        z=cells_dense.z.clone(),
    )

    for _ in range(3):
        cells_dense = dense(cells_dense)
        cells_sparse = sparse(cells_sparse)

    assert torch.allclose(cells_dense.mu, cells_sparse.mu, atol=1e-3)
    assert torch.allclose(cells_dense.z, cells_sparse.z, atol=1e-3)
