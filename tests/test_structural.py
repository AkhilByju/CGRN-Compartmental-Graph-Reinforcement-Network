"""Tests for `structural.py`: `StructuralAddress`'s `w_ij` formula,
`EdgeRegistry`'s runtime-resizable buffers, `CellActivity`'s EMA."""

from __future__ import annotations

import torch

from src.models.architecture_v1.structural import CellActivity, EdgeRegistry, StructuralAddress


def test_structural_address_is_input_independent() -> None:
    """§16.2: `s_i` must not depend on the current input -- calling
    `project`/`edge_weights` twice with no state change gives identical
    output."""
    addr = StructuralAddress(n_cells=10, d_s=4)
    edge_index = torch.tensor([[0, 1], [2, 3], [1, 0]])
    w1 = addr.edge_weights(edge_index)
    w2 = addr.edge_weights(edge_index)
    assert torch.equal(w1, w2)


def test_edge_weights_bounded_and_no_grad_leak_into_address_shape() -> None:
    addr = StructuralAddress(n_cells=6, d_s=4)
    edge_index = torch.tensor([[0, 1], [1, 2], [5, 0]])
    w = addr.edge_weights(edge_index)
    assert w.shape == (3,)
    assert torch.isfinite(w).all()
    assert (w.abs() <= 1.0).all()  # tanh


def test_edge_weights_empty_edge_index() -> None:
    addr = StructuralAddress(n_cells=6, d_s=4)
    w = addr.edge_weights(torch.zeros(0, 2, dtype=torch.long))
    assert w.shape == (0,)


def test_edge_weights_gradient_reaches_address_and_projections() -> None:
    addr = StructuralAddress(n_cells=5, d_s=3)
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 0]])
    w = addr.edge_weights(edge_index)
    w.sum().backward()
    assert addr.s.grad is not None and torch.isfinite(addr.s.grad).all()
    assert addr.w_out.weight.grad is not None and torch.isfinite(addr.w_out.weight.grad).all()
    assert addr.w_in.weight.grad is not None and torch.isfinite(addr.w_in.weight.grad).all()


def test_edge_registry_add_and_remove_resize_buffers() -> None:
    reg = EdgeRegistry(n_cells=5)
    assert reg.n_edges == 0

    reg.add_edges(torch.tensor([[0, 1], [1, 2], [2, 3]]))
    assert reg.n_edges == 3
    assert reg.edge_index.shape == (3, 2)
    assert reg.utility.shape == (3,)
    assert reg.age.shape == (3,)
    assert torch.equal(reg.utility, torch.zeros(3))

    reg.utility = torch.tensor([0.1, 5.0, 0.2])
    keep = reg.utility > 0.15
    reg.remove_edges(keep)
    assert reg.n_edges == 2
    assert torch.equal(reg.edge_index, torch.tensor([[1, 2], [2, 3]]))


def test_edge_registry_in_degree() -> None:
    reg = EdgeRegistry(n_cells=4)
    reg.add_edges(torch.tensor([[0, 1], [2, 1], [3, 1], [0, 2]]))
    in_deg = reg.in_degree()
    assert in_deg.tolist() == [0, 3, 1, 0]


def test_edge_registry_add_edges_noop_on_empty() -> None:
    reg = EdgeRegistry(n_cells=3)
    reg.add_edges(torch.zeros(0, 2, dtype=torch.long))
    assert reg.n_edges == 0


def test_edge_registry_increment_age() -> None:
    reg = EdgeRegistry(n_cells=3)
    reg.add_edges(torch.tensor([[0, 1]]))
    reg.increment_age()
    reg.increment_age()
    assert reg.age.tolist() == [2]


def test_cell_activity_ema_moves_toward_new_value() -> None:
    act = CellActivity(n_cells=3, decay=0.5)
    mu = torch.tensor([[1.0, 1.0, 1.0]])
    e = torch.tensor([[1.0, 1.0, 1.0]])
    u = torch.tensor([[0.0, 0.0, 0.0]])
    act.update(mu, e, u)
    # raw = |1|*1/(1+0) = 1; value = 0.5*0 + 0.5*1 = 0.5
    assert torch.allclose(act.value, torch.tensor([0.5, 0.5, 0.5]))
    act.update(mu, e, u)
    assert torch.allclose(act.value, torch.tensor([0.75, 0.75, 0.75]))


def test_cell_activity_update_does_not_track_grad() -> None:
    act = CellActivity(n_cells=2)
    mu = torch.tensor([[1.0, 1.0]], requires_grad=True)
    e = torch.tensor([[1.0, 1.0]])
    u = torch.tensor([[0.0, 0.0]])
    act.update(mu, e, u)
    assert not act.value.requires_grad
