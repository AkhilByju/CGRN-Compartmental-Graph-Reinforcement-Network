"""The 8 tests the user's global-communication-field spec asked for
(docs/architecture_v1.md's global field section): no NaN/Inf, e/u stay
non-negative, self-contribution is exactly excluded, the global proposal
changes when other cells change, a need~0 receiver is unaffected by
global content, permutation equivariance, no `(n_cells, n_cells)`
allocation, and gradients reach send/need/Wq/Wk.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.field_dynamics import FieldCellState, FieldRefinementStep
from src.models.architecture_v1.global_field import linear_global_belief_field


def _random_state(batch: int, n: int, assoc_dim: int, routing_dim: int, generator: torch.Generator) -> FieldCellState:
    z = torch.randn(batch, n, assoc_dim, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    cells = BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.rand(batch, n, generator=generator) + 0.5,
        uncertainty=torch.rand(batch, n, generator=generator) + 0.5,
        z=z,
    )
    r = torch.randn(batch, n, routing_dim, generator=generator)
    return FieldCellState(cells=cells, r=r)


# --- 1 & 2: no NaN/Inf, e/u stay non-negative, through a full step ---


def test_one_step_preserves_validity_with_global() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, assoc, routing = 3, 16, 6, 5
    step = FieldRefinementStep(
        n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8,
        use_global=True, global_dim=8,
    )
    state = _random_state(batch, n, assoc, routing, g)

    out = step(state)

    assert torch.isfinite(out.cells.mu).all()
    assert (out.cells.evidence >= 0).all() and torch.isfinite(out.cells.evidence).all()
    assert (out.cells.uncertainty >= 0).all() and torch.isfinite(out.cells.uncertainty).all()
    assert torch.isfinite(out.r).all()


def test_gradients_finite_through_one_step_with_global() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, assoc, routing = 2, 16, 6, 5
    step = FieldRefinementStep(
        n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8,
        use_global=True, global_dim=8,
    )
    state = _random_state(batch, n, assoc, routing, g)
    state = FieldCellState(
        cells=BeliefCellV1(
            mu=state.cells.mu.requires_grad_(True),
            evidence=state.cells.evidence,
            uncertainty=state.cells.uncertainty,
            z=state.cells.z,
        ),
        r=state.r,
    )

    out = step(state)
    (out.cells.mu.sum() + out.cells.evidence.sum() + out.cells.uncertainty.sum() + out.r.sum()).backward()

    assert torch.isfinite(state.cells.mu.grad).all()
    for p in step.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# --- 3: self-contribution is exactly excluded ---


def test_self_contribution_excluded_for_isolated_cell() -> None:
    # A single cell has no one else to hear from -- if self-removal were
    # wrong, it would retrieve (an approximation of) its own state back.
    # Correct exact self-removal degenerates to precision_fusion's own
    # documented "no live edges" case: evidence/content -> 0, uncertainty
    # -> large (dominated by 1/eps).
    torch.manual_seed(2)
    batch, n, d_g = 2, 1, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    send = torch.rand(batch, n)
    q = torch.randn(batch, n, d_g)
    k = torch.randn(batch, n, d_g)

    mu_g, e_g, u_g = linear_global_belief_field(mu, e, u, send, q, k, eps=1e-8)

    # Bounded by (float32 cancellation residual) / _DENOM_FLOOR -- orders
    # of magnitude below what a real (non-excluded) self contribution
    # would give (~e's own O(1) scale -- see global_field.py's
    # _DENOM_FLOOR comment for the bug this floor fixed).
    assert torch.allclose(e_g, torch.zeros_like(e_g), atol=1e-2)
    assert torch.allclose(mu_g, torch.zeros_like(mu_g), atol=1e-2)
    assert (u_g > 50.0).all()


# --- 4: global output changes when other cells change ---


def test_global_output_changes_when_other_cells_change() -> None:
    torch.manual_seed(3)
    batch, n, d_g = 2, 6, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    send = torch.rand(batch, n)
    q = torch.randn(batch, n, d_g)
    k = torch.randn(batch, n, d_g)

    mu_g1, _, _ = linear_global_belief_field(mu, e, u, send, q, k, eps=1e-8)

    mu2 = mu.clone()
    mu2[:, 1:] = torch.tanh(torch.randn(batch, n - 1))  # perturb every cell except cell 0
    mu_g2, _, _ = linear_global_belief_field(mu2, e, u, send, q, k, eps=1e-8)

    assert not torch.allclose(mu_g1[:, 0], mu_g2[:, 0], atol=1e-4)


# --- 5: a need~0 receiver is unaffected by global content ---


def test_need_zero_makes_fuse_independent_of_global_content() -> None:
    from src.models.architecture_v1.fusion import precision_fusion

    torch.manual_seed(4)
    batch, n = 2, 5
    mu = torch.tanh(torch.randn(batch, n))
    mu_field = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.1
    e_field = torch.rand(batch, n) + 0.1
    u = torch.rand(batch, n) + 0.1
    u_field = torch.rand(batch, n) + 0.1
    u_global = torch.rand(batch, n) + 0.1
    need = torch.zeros(batch, n)  # need ~ 0 for every cell
    bias = torch.zeros(n)
    gate = torch.sigmoid(torch.zeros(3)).expand(batch, n, 3)

    def fused_mu(mu_global: torch.Tensor, e_global_raw: torch.Tensor) -> torch.Tensor:
        e_global_gated = need * e_global_raw  # exactly field_dynamics.py's need application
        m_fuse = torch.stack([mu, mu_field, mu_global], dim=-1)
        e_fuse = torch.stack([e, e_field, e_global_gated], dim=-1)
        u_fuse = torch.stack([u, u_field, u_global], dim=-1)
        mu_hat, _, _, _ = precision_fusion(m_fuse, gate, e_fuse, u_fuse, bias, 1e-8)
        return mu_hat

    e_global_raw = torch.rand(batch, n) + 0.5
    out_a = fused_mu(torch.randn(batch, n) * 5.0, e_global_raw)
    out_b = fused_mu(torch.randn(batch, n) * 5.0, e_global_raw)  # different, arbitrary global content

    assert torch.allclose(out_a, out_b, atol=1e-6)


# --- 6: permutation equivariance ---


def test_permutation_equivariance() -> None:
    torch.manual_seed(5)
    batch, n, d_g = 2, 7, 8
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.5
    u = torch.rand(batch, n) + 0.5
    send = torch.rand(batch, n)
    q = torch.randn(batch, n, d_g)
    k = torch.randn(batch, n, d_g)

    perm = torch.randperm(n)
    mu_g, e_g, u_g = linear_global_belief_field(mu, e, u, send, q, k, eps=1e-8)
    mu_gp, e_gp, u_gp = linear_global_belief_field(
        mu[:, perm], e[:, perm], u[:, perm], send[:, perm], q[:, perm], k[:, perm], eps=1e-8
    )

    assert torch.allclose(mu_g[:, perm], mu_gp, atol=1e-5)
    assert torch.allclose(e_g[:, perm], e_gp, atol=1e-5)
    assert torch.allclose(u_g[:, perm], u_gp, atol=1e-5)


# --- 7: no (n_cells, n_cells) allocation -- empirical scaling check ---


def test_no_quadratic_scaling_in_cell_count() -> None:
    # An O(n_cells^2) implementation would take ~64x longer at 8x the
    # cell count; O(n_cells * global_dim) takes ~8x longer. Generous
    # buffer for constant-factor noise -- this only needs to catch a
    # genuine quadratic term, not measure exact linearity.
    torch.manual_seed(6)
    d_g = 8

    def bench(n: int, n_iters: int = 5) -> float:
        mu = torch.tanh(torch.randn(1, n))
        e = torch.rand(1, n) + 0.5
        u = torch.rand(1, n) + 0.5
        send = torch.rand(1, n)
        q = torch.randn(1, n, d_g)
        k = torch.randn(1, n, d_g)
        linear_global_belief_field(mu, e, u, send, q, k, eps=1e-8)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iters):
            linear_global_belief_field(mu, e, u, send, q, k, eps=1e-8)
        return (time.perf_counter() - t0) / n_iters

    small = bench(512)
    large = bench(4096)  # 8x the cells
    assert large / small < 25.0, f"scaling looks quadratic: {small=:.5f}s {large=:.5f}s ratio={large / small:.1f}"


# --- 8: gradient reaches send, need, Wq, Wk ---


def test_gradient_reaches_send_need_query_key() -> None:
    g = torch.Generator().manual_seed(7)
    batch, n, assoc, routing = 2, 16, 6, 5
    step = FieldRefinementStep(
        n_cells=n, association_dim=assoc, routing_dim=routing, num_features=64, hidden_dim=8,
        use_global=True, global_dim=8,
    )
    state = _random_state(batch, n, assoc, routing, g)

    out = step(state)
    out.cells.mu.sum().backward()

    def has_real_grad(module: nn.Module) -> bool:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, "no gradient reached this module's parameters at all"
        return any(torch.isfinite(gr).all() and gr.abs().sum().item() > 0 for gr in grads)

    assert has_real_grad(step.send_fn)
    assert has_real_grad(step.need_fn)
    assert has_real_grad(step.global_query_key.query)
    assert has_real_grad(step.global_query_key.key)
