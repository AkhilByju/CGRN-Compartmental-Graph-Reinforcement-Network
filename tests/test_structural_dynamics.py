"""Tests for `structural_dynamics.py`: `StructuralRefinementStep`/`Core`
validity, gradient finiteness, the empty-graph degenerate case, and the
`update_edge_utility` post-backward integration contract."""

from __future__ import annotations

import torch

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.structural_dynamics import StructuralRefinementCore, StructuralRefinementStep


def _random_cells(batch: int, n: int, assoc_dim: int, generator: torch.Generator) -> BeliefCellV1:
    z = torch.randn(batch, n, assoc_dim, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.rand(batch, n, generator=generator) + 0.5,
        uncertainty=torch.rand(batch, n, generator=generator) + 0.5,
        z=z,
    )


def _chain_edges(n: int) -> torch.Tensor:
    return torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=-1)


def test_step_preserves_validity_with_edges() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, assoc = 3, 10, 6
    step = StructuralRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8)
    cells = _random_cells(batch, n, assoc, g)
    edge_index = _chain_edges(n)
    w = torch.tanh(torch.randn(edge_index.shape[0], generator=g))

    out = step(cells, edge_index, w)
    assert torch.isfinite(out.mu).all()
    assert (out.evidence >= 0).all() and torch.isfinite(out.evidence).all()
    assert (out.uncertainty >= 0).all() and torch.isfinite(out.uncertainty).all()
    norms = out.z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_step_preserves_validity_with_no_edges() -> None:
    """No structural edges yet (e.g. before bootstrap) -- should not
    crash, and should fall back to precision_fusion's no-live-edges case
    for the structural source."""
    g = torch.Generator().manual_seed(1)
    batch, n, assoc = 2, 6, 4
    step = StructuralRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8)
    cells = _random_cells(batch, n, assoc, g)
    edge_index = torch.zeros(0, 2, dtype=torch.long)
    w = torch.zeros(0)

    out = step(cells, edge_index, w)
    assert torch.isfinite(out.mu).all()
    assert torch.isfinite(out.evidence).all()
    assert torch.isfinite(out.uncertainty).all()


def test_gradients_finite_through_one_step() -> None:
    g = torch.Generator().manual_seed(2)
    batch, n, assoc = 2, 8, 4
    step = StructuralRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8)
    cells = _random_cells(batch, n, assoc, g)
    cells = BeliefCellV1(
        mu=cells.mu.requires_grad_(True), evidence=cells.evidence, uncertainty=cells.uncertainty, z=cells.z
    )
    edge_index = _chain_edges(n)
    w = torch.tanh(torch.randn(edge_index.shape[0], generator=g))

    out = step(cells, edge_index, w)
    (out.mu.sum() + out.evidence.sum() + out.uncertainty.sum() + out.z.sum()).backward()
    assert torch.isfinite(cells.mu.grad).all()
    for p in step.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_core_applies_shared_step_num_steps_times() -> None:
    n, assoc, num_steps = 12, 6, 3
    core = StructuralRefinementCore(n, assoc, num_steps=num_steps, hidden_dim=8)
    assert sum(1 for m in core.modules() if isinstance(m, StructuralRefinementStep)) == 1

    core.edges.add_edges(_chain_edges(n))
    g = torch.Generator().manual_seed(3)
    cells = _random_cells(2, n, assoc, g)
    out = core(cells)
    assert out.mu.shape == (2, n)
    assert torch.isfinite(out.mu).all()


def test_core_w_is_computed_once_not_per_step() -> None:
    """Addresses don't change within a forward pass, so `w` should be
    identical across all `T` steps -- checked indirectly: a core with
    num_steps=1 and num_steps=3 (edges/addresses held fixed, weights
    copied) should use the same edge weights each step, which we verify
    by confirming StructuralAddress.edge_weights is deterministic given
    fixed parameters (already covered in test_structural.py) and that
    the Core doesn't recompute addresses from evolving state (structural
    address has no dependence on `cells` at all -- checked by construction:
    `StructuralAddress.forward` doesn't exist / `edge_weights` takes no
    cell state)."""
    n, assoc = 8, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    core.edges.add_edges(_chain_edges(n))
    w1 = core.structural_address.edge_weights(core.edges.edge_index)
    g = torch.Generator().manual_seed(4)
    cells = _random_cells(2, n, assoc, g)
    core(cells)
    w2 = core.structural_address.edge_weights(core.edges.edge_index)
    assert torch.equal(w1, w2)  # unchanged by a forward pass (no optimizer step happened)


def test_update_edge_utility_after_backward_changes_utility() -> None:
    n, assoc = 8, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    core.edges.add_edges(_chain_edges(n))
    core.edges.utility = torch.zeros(core.edges.n_edges)
    core.train()

    g = torch.Generator().manual_seed(5)
    cells = _random_cells(3, n, assoc, g)
    out = core(cells)
    loss = out.mu.sum() + out.evidence.sum() + out.uncertainty.sum()
    loss.backward()

    core.update_edge_utility()
    assert torch.isfinite(core.edges.utility).all()
    assert (core.edges.utility >= 0).all()
    # at least one edge should have picked up a nonzero utility signal
    assert core.edges.utility.sum().item() > 0


def test_update_edge_utility_without_backward_is_noop() -> None:
    n, assoc = 6, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    core.edges.add_edges(_chain_edges(n))
    core.edges.utility = torch.full((core.edges.n_edges,), 0.5)
    # never ran forward/backward -- update should no-op, not crash
    core.update_edge_utility()
    assert torch.equal(core.edges.utility, torch.full((core.edges.n_edges,), 0.5))


def test_no_edge_grad_tracking_in_eval_mode() -> None:
    n, assoc = 6, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    core.edges.add_edges(_chain_edges(n))
    core.eval()
    g = torch.Generator().manual_seed(6)
    cells = _random_cells(2, n, assoc, g)
    core(cells)
    assert core._last_edge_messages == []


def test_bootstrap_then_forward_produces_valid_output() -> None:
    n, assoc = 10, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    g = torch.Generator().manual_seed(7)
    cells = _random_cells(4, n, assoc, g)

    core.bootstrap_structural_graph(cells, generator=g)
    assert core.edges.n_edges > 0
    assert (core.edges.in_degree() >= 1).all()

    out = core(cells)
    assert torch.isfinite(out.mu).all()


def test_maybe_run_structural_plasticity_respects_schedule() -> None:
    n, assoc = 10, 4
    core = StructuralRefinementCore(n, assoc, num_steps=1, hidden_dim=8)
    core.plasticity_config.warmup_steps = 5
    core.plasticity_config.update_interval = 10
    g = torch.Generator().manual_seed(8)
    cells = _random_cells(2, n, assoc, g)
    core.bootstrap_structural_graph(cells, generator=g)

    ran_early = core.maybe_run_structural_plasticity(cells, step=2, total_steps=1000, generator=g)
    ran_on_time = core.maybe_run_structural_plasticity(cells, step=5, total_steps=1000, generator=g)
    assert ran_early is False
    assert ran_on_time is True
