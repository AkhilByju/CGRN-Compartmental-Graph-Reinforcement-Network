"""`precision_fusion` (src/models/architecture_v1/fusion.py) is the same
arithmetic as CellV0's Method E / "CellV0.1"
(`src.models.architecture_v0.integration._scale_stable_precision_fusion`,
covered by `tests/test_scale_stable_precision.py` and
`tests/test_duplication_invariance.py`) -- these tests check the same
qualitative properties survive the V1 signature change (an explicit
association matrix `a` instead of a learned dense `(w, g)` pair, and the
additional `alpha` return value V1 needs for the semantic-address update).
"""

import math

import torch

from src.models.architecture_v1.fusion import precision_fusion

E0, U0, BIAS0 = 1.0, 1.0, 0.0


def _inverse_tanh(y: float) -> float:
    return 0.5 * math.log((1 + y) / (1 - y))


def test_single_source_recovers_that_sources_content() -> None:
    # With exactly one nonzero-weight source, n_eff == 1, so this should
    # reduce to (approximately) that source's own (content, evidence).
    m = torch.tensor([[0.6]])
    a = torch.tensor([[1.0]])
    e_j = torch.tensor([[E0]])
    u_j = torch.tensor([[U0]])
    bias = torch.zeros(1)

    mu, evidence, uncertainty, alpha = precision_fusion(m, a, e_j, u_j, bias, eps=1e-8)

    assert torch.isclose(mu, torch.tensor([math.tanh(0.6)]), atol=1e-5)
    assert torch.isclose(evidence, torch.tensor([E0]), atol=1e-4)
    assert torch.allclose(alpha, torch.ones(1, 1), atol=1e-5)


def test_all_zero_weights_yield_zero_evidence_and_large_uncertainty() -> None:
    # No live edges into a cell -- the degenerate case docs/architecture_v1.md
    # implies must not crash and should read as "no information, total
    # uncertainty," not as a spurious confident belief.
    m = torch.zeros(1, 4)
    a = torch.zeros(1, 4)
    e_j = torch.full((1, 4), E0)
    u_j = torch.full((1, 4), U0)
    bias = torch.zeros(1)

    mu, evidence, uncertainty, alpha = precision_fusion(m, a, e_j, u_j, bias, eps=1e-8)

    assert torch.isclose(evidence, torch.zeros(1), atol=1e-4)
    assert uncertainty.item() > 100.0
    assert torch.isfinite(mu).all()
    assert torch.isfinite(alpha).all()


def test_duplicate_sources_leave_evidence_and_uncertainty_unchanged() -> None:
    # Method E's duplication-invariance property (tests/test_duplication_invariance.py),
    # re-checked for this signature: N identical (m, a, e, u) sources should
    # produce the same (evidence, uncertainty) regardless of N.
    def run(n: int) -> tuple[float, float]:
        m = torch.full((1, n), 0.4)
        a = torch.full((1, n), 0.5)
        e_j = torch.full((1, n), E0)
        u_j = torch.full((1, n), U0)
        bias = torch.zeros(1)
        _, evidence, uncertainty, _ = precision_fusion(m, a, e_j, u_j, bias, eps=1e-8)
        return evidence.item(), uncertainty.item()

    e1, u1 = run(1)
    e4, u4 = run(4)
    e16, u16 = run(16)

    assert math.isclose(e1, e4, rel_tol=1e-3)
    assert math.isclose(e4, e16, rel_tol=1e-3)
    assert math.isclose(u1, u4, rel_tol=1e-3)
    assert math.isclose(u4, u16, rel_tol=1e-3)


def test_alpha_sums_to_one_when_some_weight_present() -> None:
    m = torch.randn(2, 5)
    a = torch.rand(2, 5) + 0.01  # keep strictly positive so denom isn't ~eps
    e_j = torch.rand(2, 5) + 0.1
    u_j = torch.rand(2, 5) + 0.1
    bias = torch.zeros(())

    _, _, _, alpha = precision_fusion(m, a, e_j, u_j, bias, eps=1e-8)

    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_gradients_are_finite() -> None:
    m = torch.randn(2, 6, requires_grad=True)
    a = torch.rand(2, 6, requires_grad=True)
    e_j = (torch.rand(2, 6) + 0.1).requires_grad_(True)
    u_j = (torch.rand(2, 6) + 0.1).requires_grad_(True)
    bias = torch.zeros((), requires_grad=True)

    mu, evidence, uncertainty, alpha = precision_fusion(m, a, e_j, u_j, bias, eps=1e-8)
    (mu.sum() + evidence.sum() + uncertainty.sum() + alpha.sum()).backward()

    for t in (m, a, e_j, u_j, bias):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
