import torch

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.field_dynamics import FieldCellState, FieldRefinementCore, FieldRefinementStep


def _random_state(batch: int, n: int, assoc_dim: int, routing_dim: int, generator: torch.Generator) -> FieldCellState:
    z = torch.randn(batch, n, assoc_dim, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    cells = BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.ones(batch, n),
        uncertainty=torch.ones(batch, n),
        z=z,
    )
    r = torch.randn(batch, n, routing_dim, generator=generator)
    return FieldCellState(cells=cells, r=r)


def test_one_step_preserves_validity() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, assoc, routing = 3, 16, 6, 5
    step = FieldRefinementStep(n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8)
    state = _random_state(batch, n, assoc, routing, g)

    out = step(state)

    assert out.cells.mu.shape == (batch, n)
    assert out.r.shape == (batch, n, routing)
    assert torch.isfinite(out.cells.mu).all()
    assert (out.cells.evidence >= 0).all() and torch.isfinite(out.cells.evidence).all()
    assert (out.cells.uncertainty >= 0).all() and torch.isfinite(out.cells.uncertainty).all()
    assert torch.isfinite(out.r).all()


def test_one_step_keeps_z_unit_norm() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, assoc, routing = 2, 12, 6, 4
    step = FieldRefinementStep(n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8)
    state = _random_state(batch, n, assoc, routing, g)

    out = step(state)

    norms = out.cells.z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gradients_finite_through_one_step() -> None:
    g = torch.Generator().manual_seed(2)
    batch, n, assoc, routing = 2, 16, 6, 5
    step = FieldRefinementStep(n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8)
    state = _random_state(batch, n, assoc, routing, g)
    state = FieldCellState(
        cells=BeliefCellV1(
            mu=state.cells.mu.requires_grad_(True),
            evidence=state.cells.evidence,
            uncertainty=state.cells.uncertainty,
            z=state.cells.z,
        ),
        r=state.r.requires_grad_(True),
    )

    out = step(state)
    (out.cells.mu.sum() + out.cells.evidence.sum() + out.cells.uncertainty.sum() + out.r.sum()).backward()

    assert torch.isfinite(state.cells.mu.grad).all()
    assert torch.isfinite(state.r.grad).all()


def test_routing_gate_has_no_gradient_at_one_step_but_does_at_two() -> None:
    # r's movement only matters for a *future* step -- at T=1 there is no
    # future step, so routing_gate legitimately gets zero gradient; at
    # T=2 it must not.
    g = torch.Generator().manual_seed(3)
    batch, n, assoc, routing = 2, 12, 6, 4

    core1 = FieldRefinementCore(n, assoc, routing, num_steps=1, num_features=64, hidden_dim=8)
    state1 = _random_state(batch, n, assoc, routing, g)
    out1 = core1(state1)
    out1.cells.mu.sum().backward()
    grad1 = core1.step.routing_gate.net[0].weight.grad
    assert grad1 is None or torch.allclose(grad1, torch.zeros_like(grad1), atol=1e-6)

    core2 = FieldRefinementCore(n, assoc, routing, num_steps=2, num_features=64, hidden_dim=8)
    state2 = _random_state(batch, n, assoc, routing, g)
    out2 = core2(state2)
    out2.cells.mu.sum().backward()
    grad2 = core2.step.routing_gate.net[0].weight.grad
    assert grad2 is not None
    assert torch.isfinite(grad2).all()
    assert grad2.abs().sum().item() > 0


def test_core_applies_the_same_step_num_steps_times() -> None:
    n, assoc, routing, num_steps = 10, 6, 4, 3
    core = FieldRefinementCore(n, assoc, routing, num_steps=num_steps, num_features=64, hidden_dim=8)
    assert sum(1 for m in core.modules() if isinstance(m, FieldRefinementStep)) == 1

    g = torch.Generator().manual_seed(4)
    state = _random_state(2, n, assoc, routing, g)
    out = core(state)

    assert out.cells.mu.shape == (2, n)
    assert torch.isfinite(out.cells.mu).all()
