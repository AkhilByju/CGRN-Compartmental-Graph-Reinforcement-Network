import torch

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.sparse_dynamics import SparseDynamicBeliefGraphCore, SparseDynamicBeliefGraphStep


def _random_population(batch: int, n: int, d: int, generator: torch.Generator) -> BeliefCellV1:
    z = torch.randn(batch, n, d, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.ones(batch, n),
        uncertainty=torch.ones(batch, n),
        z=z,
    )


def test_one_step_preserves_belief_cell_validity() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, d = 2, 24, 4
    step = SparseDynamicBeliefGraphStep(
        n_cells=n, association_dim=d, hidden_dim=8, chunk_size_local=8, chunk_size_global=4
    )
    cells = _random_population(batch, n, d, g)

    out = step(cells)

    assert out.mu.shape == (batch, n)
    assert out.z.shape == (batch, n, d)
    assert torch.isfinite(out.mu).all()
    assert (out.evidence >= 0).all() and torch.isfinite(out.evidence).all()
    assert (out.uncertainty >= 0).all() and torch.isfinite(out.uncertainty).all()


def test_one_step_keeps_z_unit_norm() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, d = 2, 24, 4
    step = SparseDynamicBeliefGraphStep(
        n_cells=n, association_dim=d, hidden_dim=8, chunk_size_local=8, chunk_size_global=4
    )
    cells = _random_population(batch, n, d, g)

    out = step(cells)

    norms = out.z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gradients_are_finite_through_one_step() -> None:
    g = torch.Generator().manual_seed(2)
    batch, n, d = 2, 24, 4
    step = SparseDynamicBeliefGraphStep(
        n_cells=n, association_dim=d, hidden_dim=8, chunk_size_local=8, chunk_size_global=4
    )
    cells = _random_population(batch, n, d, g)
    cells = BeliefCellV1(
        mu=cells.mu.requires_grad_(True),
        evidence=cells.evidence,
        uncertainty=cells.uncertainty,
        z=cells.z.requires_grad_(True),
    )

    out = step(cells)
    (out.mu.sum() + out.evidence.sum() + out.uncertainty.sum() + out.z.sum()).backward()

    assert torch.isfinite(cells.mu.grad).all()
    assert torch.isfinite(cells.z.grad).all()
    for p in step.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_local_only_ablation_has_no_global_modules() -> None:
    step = SparseDynamicBeliefGraphStep(
        n_cells=20, association_dim=4, hidden_dim=8, chunk_size_local=8, use_global_routing=False
    )
    assert step.global_routing is None
    assert step.need is None
    assert step.offer is None
    assert step.source_gate_logit.numel() == 2


def test_local_only_returns_none_for_global_graph() -> None:
    g = torch.Generator().manual_seed(3)
    batch, n, d = 2, 20, 4
    step = SparseDynamicBeliefGraphStep(
        n_cells=n, association_dim=d, hidden_dim=8, chunk_size_local=8, use_global_routing=False
    )
    cells = _random_population(batch, n, d, g)

    out, (local_idx, a_local, global_idx, a_global) = step.forward_with_graphs(cells)

    assert global_idx is None
    assert a_global is None
    assert torch.isfinite(out.mu).all()


def test_core_applies_the_same_step_num_steps_times() -> None:
    n, d, num_steps = 16, 4, 4
    core = SparseDynamicBeliefGraphCore(
        n_cells=n, association_dim=d, num_steps=num_steps, hidden_dim=8, chunk_size_local=8, chunk_size_global=4
    )
    assert sum(1 for m in core.modules() if isinstance(m, SparseDynamicBeliefGraphStep)) == 1

    g = torch.Generator().manual_seed(4)
    cells = _random_population(2, n, d, g)
    out, graphs = core.forward_with_graphs(cells)

    assert out.mu.shape == (2, n)
    assert len(graphs) == num_steps
    assert torch.isfinite(out.mu).all()
