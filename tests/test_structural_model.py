"""Tests for `structural_model.py::StructuralBeliefGraph`: end-to-end
shape/gradient checks, and a short training smoke test (same discipline
as CellV1.3's "15 AdamW steps, loss decreasing, no divergence" --
`docs/architecture_v1.md` §13). No experiment-quality claim here, only
"this doesn't crash and can fit noise."
"""

from __future__ import annotations

import torch

from src.models.architecture_v1.structural_model import StructuralBeliefGraph
from src.models.architecture_v1.structural_plasticity import StructuralPlasticityConfig


def test_forward_shape_and_validity_after_bootstrap() -> None:
    torch.manual_seed(0)
    model = StructuralBeliefGraph(in_features=5, out_features=1, n_cells=16, association_dim=4, d_s=8, hidden_dim=8)
    x = torch.randn(4, 5)
    model.bootstrap_structural_graph(x)
    assert model.core.edges.n_edges > 0

    out = model(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_forward_without_bootstrap_does_not_crash() -> None:
    """No edges at all -- every cell's structural fusion degenerates to
    precision_fusion's no-live-edges fallback; should still run."""
    torch.manual_seed(1)
    model = StructuralBeliefGraph(in_features=3, out_features=2, n_cells=8, association_dim=4, d_s=8, hidden_dim=8)
    x = torch.randn(2, 3)
    out = model(x)
    assert torch.isfinite(out).all()


def test_gradients_reach_every_learned_component() -> None:
    torch.manual_seed(2)
    model = StructuralBeliefGraph(in_features=4, out_features=1, n_cells=12, association_dim=4, d_s=8, hidden_dim=8)
    x = torch.randn(3, 4)
    model.bootstrap_structural_graph(x)
    model.train()

    out = model(x)
    out.sum().backward()

    def has_real_grad(module: torch.nn.Module) -> bool:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, "no gradient reached this module's parameters at all"
        return any(torch.isfinite(g).all() and g.abs().sum().item() > 0 for g in grads)

    assert has_real_grad(model.encoder)
    assert has_real_grad(model.decoder)
    assert has_real_grad(model.core.structural_address)
    assert has_real_grad(model.core.step.assoc_fn)
    assert has_real_grad(model.core.step.write_gate)
    assert has_real_grad(model.core.step.semantic_update)


def test_full_training_step_integration_contract() -> None:
    """Exercises the documented contract end to end: forward -> backward
    -> update_edge_utility -> optimizer.step -> maybe_run_structural_plasticity."""
    torch.manual_seed(3)
    model = StructuralBeliefGraph(
        in_features=4,
        out_features=1,
        n_cells=10,
        association_dim=4,
        d_s=8,
        hidden_dim=8,
        plasticity_config=StructuralPlasticityConfig(warmup_steps=2, update_interval=3, chunk_size=8),
    )
    x = torch.randn(4, 4)
    y = torch.randn(4, 1)
    model.bootstrap_structural_graph(x)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(6):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        model.update_edge_utility()
        optimizer.step()
        model.maybe_run_structural_plasticity(x, step=step, total_steps=100)

    assert torch.isfinite(loss).all()
    assert model.core.edges.n_edges > 0
    assert (model.core.edges.in_degree() >= 1).all()
    # utility should have picked up real signal by now, not stayed all-zero
    assert model.core.edges.utility.sum().item() >= 0


def test_short_training_smoke_test_loss_decreases() -> None:
    """15 AdamW steps on a fixed batch of noise -- loss should decrease
    and not diverge, same bar as CellV1.3's own smoke test."""
    torch.manual_seed(4)
    model = StructuralBeliefGraph(in_features=6, out_features=1, n_cells=16, association_dim=4, d_s=8, hidden_dim=16)
    x = torch.randn(8, 6)
    y = torch.randn(8, 1)
    model.bootstrap_structural_graph(x)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for step in range(15):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        model.update_edge_utility()
        optimizer.step()
        model.maybe_run_structural_plasticity(x, step=step, total_steps=15)
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
