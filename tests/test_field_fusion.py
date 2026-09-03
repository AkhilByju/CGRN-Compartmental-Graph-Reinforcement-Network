"""`local_field_fusion` (self-anchored orthogonal RFF) vs.
`exact_local_field_fusion` (dense reference) -- docs/architecture_v1.md
§12's diagnostic, encoded as regression tests. Checks the properties that
diagnostic actually established: self-anchoring guarantees `a_sum`/
`p_sum` stay strictly bounded away from zero (no more catastrophic
division blowups), and the approximation is *bounded and reasonably
correlated* with the exact reference at realistic R -- not exact (this is
a statistical approximation, unlike CellV1.1's LSH pooling, which reduces
to dense exactly when the pool covers everyone).
"""

from __future__ import annotations

import math

import torch

from src.models.architecture_v1.field_fusion import exact_local_field_fusion, local_field_fusion
from src.models.architecture_v1.random_features import RandomFourierFeatures


def _problem(batch=3, n=20, dim=6, routing_dim=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.tanh(torch.randn(batch, n, generator=g))
    e = torch.rand(batch, n, generator=g) + 0.5
    u = torch.rand(batch, n, generator=g) + 0.5
    r = torch.randn(batch, n, routing_dim, generator=g)
    x = torch.randn(batch, n, dim, generator=g)
    weight = torch.rand(batch, n, generator=g) + 0.1
    bias = torch.zeros(n)
    return mu, e, u, r, x, weight, bias


def _features(dim, num_features, seed=1):
    g = torch.Generator().manual_seed(seed)
    psi_mod = RandomFourierFeatures(dim=dim, num_features=num_features, sigma=1.0, orthogonal=True, generator=g)
    psi_sq_mod = RandomFourierFeatures(
        dim=dim, num_features=num_features, sigma=1.0 / math.sqrt(2.0), orthogonal=True, generator=g
    )
    return psi_mod, psi_sq_mod


def test_shapes() -> None:
    mu, e, u, r, x, weight, bias = _problem()
    psi_mod, psi_sq_mod = _features(dim=6, num_features=64)
    psi, psi_sq = psi_mod(x), psi_sq_mod(x)

    mu_out, e_out, u_out, r_bar = local_field_fusion(mu, e, u, r, weight, psi, psi_sq, bias, eps=1e-8)

    assert mu_out.shape == mu.shape
    assert e_out.shape == e.shape
    assert u_out.shape == u.shape
    assert r_bar.shape == r.shape


def test_outputs_are_finite_and_valid() -> None:
    mu, e, u, r, x, weight, bias = _problem()
    psi_mod, psi_sq_mod = _features(dim=6, num_features=128)
    psi, psi_sq = psi_mod(x), psi_sq_mod(x)

    mu_out, e_out, u_out, r_bar = local_field_fusion(mu, e, u, r, weight, psi, psi_sq, bias, eps=1e-8)

    assert torch.isfinite(mu_out).all()
    assert torch.isfinite(e_out).all() and (e_out >= 0).all()
    assert torch.isfinite(u_out).all() and (u_out >= 0).all()
    assert torch.isfinite(r_bar).all()


def test_no_catastrophic_blowup_across_many_seeds() -> None:
    # docs/architecture_v1.md §12: before self-anchoring, this regularly
    # produced errors in the thousands-to-millions. Regression test: with
    # self-anchoring, outputs stay in a sane bounded range across many
    # random draws.
    for seed in range(10):
        mu, e, u, r, x, weight, bias = _problem(seed=seed)
        psi_mod, psi_sq_mod = _features(dim=6, num_features=64, seed=seed + 100)
        psi, psi_sq = psi_mod(x), psi_sq_mod(x)
        mu_out, e_out, u_out, r_bar = local_field_fusion(mu, e, u, r, weight, psi, psi_sq, bias, eps=1e-8)
        assert mu_out.abs().max().item() <= 1.0  # tanh-bounded
        assert e_out.max().item() < 100.0, f"seed={seed}: evidence blew up"
        assert u_out.max().item() < 1000.0, f"seed={seed}: uncertainty blew up"
        assert r_bar.abs().max().item() < 1000.0, f"seed={seed}: r_bar blew up"


def test_approximation_correlates_with_exact_reference_at_realistic_r() -> None:
    mu, e, u, r, x, weight, bias = _problem(n=20)
    mu_exact, e_exact, u_exact, rbar_exact = exact_local_field_fusion(mu, e, u, r, weight, x, bias, eps=1e-8)

    psi_mod, psi_sq_mod = _features(dim=6, num_features=256)
    psi, psi_sq = psi_mod(x), psi_sq_mod(x)
    mu_rf, e_rf, u_rf, rbar_rf = local_field_fusion(mu, e, u, r, weight, psi, psi_sq, bias, eps=1e-8)

    def corr(a, b):
        a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
        return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)

    # Loose bounds -- this is a statistical approximation, not exact
    # equality (see module docstring). Regression test for "still roughly
    # tracks the exact math," not a tight numerical-accuracy guarantee.
    assert corr(mu_rf, mu_exact).item() > 0.5
    assert corr(rbar_rf, rbar_exact).item() > 0.5


def test_gradients_are_finite() -> None:
    mu, e, u, r, x, weight, bias = _problem()
    mu.requires_grad_(True)
    r.requires_grad_(True)
    x.requires_grad_(True)
    psi_mod, psi_sq_mod = _features(dim=6, num_features=64)
    psi, psi_sq = psi_mod(x), psi_sq_mod(x)

    mu_out, e_out, u_out, r_bar = local_field_fusion(mu, e, u, r, weight, psi, psi_sq, bias, eps=1e-8)
    (mu_out.sum() + e_out.sum() + u_out.sum() + r_bar.sum()).backward()

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(r.grad).all()
    assert torch.isfinite(x.grad).all()


def test_exact_reference_diagonal_self_weight() -> None:
    # Sanity check on exact_local_field_fusion itself: a single isolated
    # cell (far from everyone else) should fuse to close to its own state,
    # since its own self-contribution (K_ii=1) dominates.
    mu = torch.tensor([[0.6]])
    e = torch.tensor([[1.0]])
    u = torch.tensor([[1.0]])
    r = torch.zeros(1, 1, 3)
    x = torch.zeros(1, 1, 3)
    weight = torch.tensor([[1.0]])
    bias = torch.zeros(1)

    mu_out, e_out, u_out, r_bar = exact_local_field_fusion(mu, e, u, r, weight, x, bias, eps=1e-8)
    assert torch.isclose(mu_out, torch.tanh(mu), atol=1e-4)
