"""Tests for the Learned Association Field (`learned_association.py`,
`association_dynamics.py`, `association_model.py`) -- same discipline as
`tests/test_global_field.py`: no NaN/Inf, e/u stay non-negative, self-
contribution is exactly excluded, output changes when other cells
change, permutation equivariance, no `(n_cells, n_cells)` allocation, and
gradients reach every learned component. Plus the field-line's own
validity/z-unit-norm checks (`test_field_dynamics.py`'s style).
"""

from __future__ import annotations

import time

import torch

from src.models.architecture_v1.association_dynamics import AssociationRefinementCore, AssociationRefinementStep
from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.learned_association import AssociationFunction, learned_local_association_fusion


def _random_cells(batch: int, n: int, assoc_dim: int, generator: torch.Generator) -> BeliefCellV1:
    z = torch.randn(batch, n, assoc_dim, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.rand(batch, n, generator=generator) + 0.5,
        uncertainty=torch.rand(batch, n, generator=generator) + 0.5,
        z=z,
    )


# --- validity / z-unit-norm, local-only and local+global ---


def test_one_step_preserves_validity() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, assoc = 3, 16, 6
    for use_global in (False, True):
        step = AssociationRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8, use_global=use_global, global_dim=8)
        cells = _random_cells(batch, n, assoc, g)
        out = step(cells)
        assert torch.isfinite(out.mu).all()
        assert (out.evidence >= 0).all() and torch.isfinite(out.evidence).all()
        assert (out.uncertainty >= 0).all() and torch.isfinite(out.uncertainty).all()
        norms = out.z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gradients_finite_through_one_step() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, assoc = 2, 16, 6
    for use_global in (False, True):
        step = AssociationRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8, use_global=use_global, global_dim=8)
        cells = _random_cells(batch, n, assoc, g)
        cells = BeliefCellV1(
            mu=cells.mu.requires_grad_(True), evidence=cells.evidence, uncertainty=cells.uncertainty, z=cells.z
        )
        out = step(cells)
        (out.mu.sum() + out.evidence.sum() + out.uncertainty.sum()).backward()
        assert torch.isfinite(cells.mu.grad).all()
        for p in step.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


def test_core_applies_the_same_step_num_steps_times() -> None:
    n, assoc, num_steps = 10, 6, 3
    core = AssociationRefinementCore(n, assoc, num_steps=num_steps, hidden_dim=8)
    assert sum(1 for m in core.modules() if isinstance(m, AssociationRefinementStep)) == 1

    g = torch.Generator().manual_seed(2)
    cells = _random_cells(2, n, assoc, g)
    out = core(cells)
    assert out.mu.shape == (2, n)
    assert torch.isfinite(out.mu).all()


# --- self-contribution excluded ---


def test_self_contribution_excluded_for_isolated_cell() -> None:
    torch.manual_seed(3)
    batch, n, d = 2, 1, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    z = torch.randn(batch, n, 6)
    z = z / z.norm(dim=-1, keepdim=True)

    assoc_fn = AssociationFunction(association_dim=6, assoc_dim=d, hidden_dim=8)
    phi = assoc_fn(mu, e, u, z)
    bias = torch.zeros(n)

    mu_local, e_local, u_local = learned_local_association_fusion(mu, e, u, phi, bias, eps=1e-8)

    assert torch.allclose(e_local, torch.zeros_like(e_local), atol=1e-2)
    # mu_local = tanh(c + bias); c -> 0 when isolated, so mu_local -> tanh(bias) = tanh(0) = 0 here
    assert torch.allclose(mu_local, torch.zeros_like(mu_local), atol=1e-2)
    assert (u_local > 5.0).all()


# --- output changes when other cells change ---


def test_output_changes_when_other_cells_change() -> None:
    torch.manual_seed(4)
    batch, n, d = 2, 6, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    z = torch.randn(batch, n, 6)
    z = z / z.norm(dim=-1, keepdim=True)

    assoc_fn = AssociationFunction(association_dim=6, assoc_dim=d, hidden_dim=8)
    phi = assoc_fn(mu, e, u, z)
    bias = torch.zeros(n)

    mu_local1, _, _ = learned_local_association_fusion(mu, e, u, phi, bias, eps=1e-8)

    mu2 = mu.clone()
    mu2[:, 1:] = torch.tanh(torch.randn(batch, n - 1))
    phi2 = assoc_fn(mu2, e, u, z)
    mu_local2, _, _ = learned_local_association_fusion(mu2, e, u, phi2, bias, eps=1e-8)

    assert not torch.allclose(mu_local1[:, 0], mu_local2[:, 0], atol=1e-4)


# --- permutation equivariance ---


def test_permutation_equivariance() -> None:
    torch.manual_seed(5)
    batch, n, d = 2, 7, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    z = torch.randn(batch, n, 6)
    z = z / z.norm(dim=-1, keepdim=True)

    assoc_fn = AssociationFunction(association_dim=6, assoc_dim=d, hidden_dim=8)
    phi = assoc_fn(mu, e, u, z)
    bias = torch.zeros(n)

    perm = torch.randperm(n)
    mu_local, e_local, u_local = learned_local_association_fusion(mu, e, u, phi, bias, eps=1e-8)
    mu_localp, e_localp, u_localp = learned_local_association_fusion(
        mu[:, perm], e[:, perm], u[:, perm], phi[:, perm], bias[perm], eps=1e-8
    )

    assert torch.allclose(mu_local[:, perm], mu_localp, atol=1e-5)
    assert torch.allclose(e_local[:, perm], e_localp, atol=1e-5)
    assert torch.allclose(u_local[:, perm], u_localp, atol=1e-5)


# --- no (n_cells, n_cells) allocation -- empirical scaling check ---


def test_no_quadratic_scaling_in_cell_count() -> None:
    torch.manual_seed(6)
    d = 8
    assoc_fn = AssociationFunction(association_dim=6, assoc_dim=d, hidden_dim=8)

    def bench(n: int, n_iters: int = 5) -> float:
        mu = torch.tanh(torch.randn(1, n))
        e = torch.rand(1, n) + 0.5
        u = torch.rand(1, n) + 0.5
        z = torch.randn(1, n, 6)
        z = z / z.norm(dim=-1, keepdim=True)
        phi = assoc_fn(mu, e, u, z)
        bias = torch.zeros(n)
        learned_local_association_fusion(mu, e, u, phi, bias, eps=1e-8)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iters):
            learned_local_association_fusion(mu, e, u, phi, bias, eps=1e-8)
        return (time.perf_counter() - t0) / n_iters

    small = bench(512)
    large = bench(4096)  # 8x the cells
    assert large / small < 25.0, f"scaling looks quadratic: {small=:.5f}s {large=:.5f}s ratio={large / small:.1f}"


# --- gradients reach every learned component ---


def test_gradient_reaches_assoc_fn_and_global_components() -> None:
    g = torch.Generator().manual_seed(7)
    batch, n, assoc = 2, 16, 6
    step = AssociationRefinementStep(n_cells=n, association_dim=assoc, hidden_dim=8, use_global=True, global_dim=8)
    cells = _random_cells(batch, n, assoc, g)

    out = step(cells)
    # z's update (semantic_update) only affects z, not mu/e/u within this
    # single step -- include it in the loss, or its gradient is
    # legitimately zero here (same reasoning as
    # test_field_dynamics.py::test_routing_gate_has_no_gradient_at_one_step_but_does_at_two).
    (out.mu.sum() + out.z.sum()).backward()

    def has_real_grad(module: torch.nn.Module) -> bool:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, "no gradient reached this module's parameters at all"
        return any(torch.isfinite(gr).all() and gr.abs().sum().item() > 0 for gr in grads)

    assert has_real_grad(step.assoc_fn)
    assert has_real_grad(step.send_fn)
    assert has_real_grad(step.need_fn)
    assert has_real_grad(step.global_query_key.query)
    assert has_real_grad(step.global_query_key.key)
    assert has_real_grad(step.semantic_update)
