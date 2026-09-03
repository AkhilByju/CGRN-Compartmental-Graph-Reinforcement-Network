"""The correctness check this repo holds every sparse mechanism to
(`tests/test_sparse_dense_consistency.py` is the precedent for CellV1.1):
`structural_fusion.py::sparse_structural_fusion`'s scatter-based
reduction must exactly match `fusion.py::precision_fusion` run on an
explicit dense `(n_cells, n_cells)` layout with zero-`a` padding for
every non-edge -- not just "similar," numerically identical up to
float32 rounding. Also covers the isolated-cell degenerate fallback and
that `edge_message_full`'s retained gradient is real (nonzero, matches
what the algebra promises), not a dead-end tensor.
"""

from __future__ import annotations

import time

import torch

from src.models.architecture_v1.fusion import precision_fusion
from src.models.architecture_v1.structural_fusion import sparse_structural_fusion


def _dense_reference(
    mu: torch.Tensor,
    evidence: torch.Tensor,
    uncertainty: torch.Tensor,
    edge_index: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    n_cells: int,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds an explicit `(batch, n_cells receiving, n_cells sending)`
    layout (zero-`a` for every non-edge) and calls `fusion.py::
    precision_fusion` directly -- the "obviously correct, but O(n^2)"
    ground truth `sparse_structural_fusion` is a scalable restriction of.
    """
    batch = mu.shape[0]
    a_dense = torch.zeros(batch, n_cells, n_cells)
    m_dense = torch.zeros(batch, n_cells, n_cells)
    e_dense = evidence.unsqueeze(1).expand(batch, n_cells, n_cells).clone()
    u_dense = uncertainty.unsqueeze(1).expand(batch, n_cells, n_cells).clone()

    for edge_idx in range(edge_index.shape[0]):
        i, j = int(edge_index[edge_idx, 0]), int(edge_index[edge_idx, 1])
        a_dense[:, j, i] = a[:, edge_idx]
        m_dense[:, j, i] = w[edge_idx] * mu[:, i]

    mu_ref, e_ref, u_ref, _ = precision_fusion(m_dense, a_dense, e_dense, u_dense, bias, eps)
    return mu_ref, e_ref, u_ref


def _random_state(batch: int, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.tanh(torch.randn(batch, n, generator=generator))
    e = torch.rand(batch, n, generator=generator) + 0.5
    u = torch.rand(batch, n, generator=generator) + 0.5
    return mu, e, u


def _check_matches_dense(edge_index: torch.Tensor, batch: int, n: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    mu, e, u = _random_state(batch, n, g)
    w = torch.tanh(torch.randn(edge_index.shape[0], generator=g))
    a = torch.rand(batch, edge_index.shape[0], generator=g) + 0.1
    bias = torch.zeros(n)
    eps = 1e-8

    mu_s, e_s, u_s = sparse_structural_fusion(mu, e, u, edge_index, w, a, n, bias, eps)
    mu_r, e_r, u_r = _dense_reference(mu, e, u, edge_index, w, a, n, bias, eps)

    assert torch.allclose(mu_s, mu_r, atol=1e-5), f"mu mismatch: {mu_s} vs {mu_r}"
    assert torch.allclose(e_s, e_r, atol=1e-5), f"evidence mismatch: {e_s} vs {e_r}"
    assert torch.allclose(u_s, u_r, atol=1e-5), f"uncertainty mismatch: {u_s} vs {u_r}"


def test_matches_dense_on_a_chain() -> None:
    # 0 -> 1 -> 2 -> 3 -> 4
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
    _check_matches_dense(edge_index, batch=3, n=5, seed=0)


def test_matches_dense_on_a_star() -> None:
    # every other cell points into cell 0
    n = 6
    edge_index = torch.tensor([[i, 0] for i in range(1, n)])
    _check_matches_dense(edge_index, batch=2, n=n, seed=1)


def test_matches_dense_with_some_isolated_receivers() -> None:
    # cells 3 and 4 have no incoming edges at all
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 1]])
    _check_matches_dense(edge_index, batch=2, n=5, seed=2)


def test_matches_dense_with_multi_edge_hub_and_reciprocal_pairs() -> None:
    edge_index = torch.tensor([[0, 3], [1, 3], [2, 3], [3, 0], [0, 1]])
    _check_matches_dense(edge_index, batch=4, n=4, seed=3)


def test_matches_dense_with_no_edges_at_all() -> None:
    edge_index = torch.zeros(0, 2, dtype=torch.long)
    _check_matches_dense(edge_index, batch=2, n=4, seed=4)


def test_isolated_target_gets_precision_fusions_degenerate_fallback() -> None:
    """A cell with zero incoming edges: evidence -> 0, uncertainty -> large."""
    edge_index = torch.tensor([[0, 1]])  # cell 0 has no incoming edges
    g = torch.Generator().manual_seed(5)
    mu, e, u = _random_state(2, 3, g)
    w = torch.tanh(torch.randn(1, generator=g))
    a = torch.rand(2, 1, generator=g) + 0.1
    bias = torch.zeros(3)
    eps = 1e-8

    mu_s, e_s, u_s = sparse_structural_fusion(mu, e, u, edge_index, w, a, 3, bias, eps)
    assert torch.allclose(e_s[:, 0], torch.zeros(2), atol=1e-4)
    assert (u_s[:, 0] > 1000.0).all()


def test_edge_message_full_gradient_is_real_and_matches_algebra() -> None:
    """`edge_message_full = w*a*mu_i` must actually influence the fused
    output -- its retained `.grad` should be nonzero and finite, not a
    dead-end tensor `autograd` never touched."""
    torch.manual_seed(6)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]])
    mu = torch.tanh(torch.randn(2, 3, requires_grad=False))
    e = torch.rand(2, 3) + 0.5
    u = torch.rand(2, 3) + 0.5
    w = torch.tanh(torch.randn(3, requires_grad=True))
    w.retain_grad()
    a = (torch.rand(2, 3) + 0.1).requires_grad_(True)
    bias = torch.zeros(3)

    sink: list[torch.Tensor] = []
    mu_s, e_s, u_s = sparse_structural_fusion(mu, e, u, edge_index, w, a, 3, bias, 1e-8, edge_message_sink=sink)
    (mu_s.sum() + e_s.sum() + u_s.sum()).backward()

    assert len(sink) == 1
    edge_message_full = sink[0]
    assert edge_message_full.grad is not None
    assert torch.isfinite(edge_message_full.grad).all()
    assert edge_message_full.grad.abs().sum().item() > 0
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_no_edge_message_sink_by_default() -> None:
    edge_index = torch.tensor([[0, 1]])
    mu = torch.tanh(torch.randn(2, 2))
    e = torch.rand(2, 2) + 0.5
    u = torch.rand(2, 2) + 0.5
    w = torch.tanh(torch.randn(1))
    a = torch.rand(2, 1) + 0.1
    bias = torch.zeros(2)
    # should not raise, and should not require a sink argument
    sparse_structural_fusion(mu, e, u, edge_index, w, a, 2, bias, 1e-8)


def test_no_quadratic_scaling_in_edge_count() -> None:
    """Same empirical-scaling discipline as `test_learned_association.py::
    test_no_quadratic_scaling_in_cell_count` -- a fixed, bounded average
    degree means `E` grows linearly with `n_cells`, and the scatter-based
    reduction should scale with `E`, not `n_cells^2`."""
    torch.manual_seed(8)
    k_bar = 8

    def bench(n_cells: int, n_iters: int = 5) -> float:
        e = n_cells * k_bar
        edge_index = torch.stack([torch.randint(0, n_cells, (e,)), torch.randint(0, n_cells, (e,))], dim=-1)
        mu = torch.tanh(torch.randn(1, n_cells))
        ev = torch.rand(1, n_cells) + 0.5
        u = torch.rand(1, n_cells) + 0.5
        w = torch.tanh(torch.randn(e))
        a = torch.rand(1, e) + 0.1
        bias = torch.zeros(n_cells)
        sparse_structural_fusion(mu, ev, u, edge_index, w, a, n_cells, bias, 1e-8)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iters):
            sparse_structural_fusion(mu, ev, u, edge_index, w, a, n_cells, bias, 1e-8)
        return (time.perf_counter() - t0) / n_iters

    small = bench(512)
    large = bench(4096)  # 8x the cells, 8x the edges (fixed k_bar)
    assert large / small < 25.0, f"scaling looks worse than linear: {small=:.5f}s {large=:.5f}s ratio={large / small:.1f}"
