"""Paper A Phase 3 Part B -- model construction: parameter matching to the
CellV0.3 target, the same-width control, CellV0.3's (mu=x, e=c, u=0) contract,
and the external NeuMiss wrapper (NaN preserved, official package pinned).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.paper_a.real_reliability.models import (
    CELLV03,
    CONFIDENCE_MLP,
    CONFIDENCE_MLP_SAME_WIDTH,
    MODEL_FAMILIES,
    NEUMISS,
    NEUMISS_COMMIT,
    NEUMISS_DEPTHS,
    PLAIN_MLP,
    NeuMissWrapper,
    build_model,
    cellv03_target,
)
from experiments.paper_a.reliability.models import ReliabilityCellV03
from src.models.architecture_v0.cell import BeliefCell

# (name, in_features, param_budget)
_SPECS = [("aps", 170, 100_000), ("air_quality", 8, 50_000)]


def test_model_families_are_the_five_declared():
    assert MODEL_FAMILIES == (
        PLAIN_MLP, CONFIDENCE_MLP, CONFIDENCE_MLP_SAME_WIDTH, CELLV03, NEUMISS
    )


@pytest.mark.parametrize("name,in_f,budget", _SPECS)
def test_cellv03_is_the_target_within_budget(name, in_f, budget):
    hc, target = cellv03_target(in_f, budget)
    b = build_model(CELLV03, in_f, budget)
    assert isinstance(b.model, ReliabilityCellV03)
    assert b.parameter_count == target
    assert b.hidden_size == hc
    assert b.parameter_count <= budget


@pytest.mark.parametrize("name,in_f,budget", _SPECS)
def test_plain_and_matched_confidence_are_within_2pct(name, in_f, budget):
    _, target = cellv03_target(in_f, budget)
    for fam in (PLAIN_MLP, CONFIDENCE_MLP):
        b = build_model(fam, in_f, budget)
        rel = abs(b.parameter_count - target) / target
        assert rel <= 0.02, f"{name}/{fam} off by {rel:.3%}"
        assert b.sizing["param_match_within_2pct"] is True


@pytest.mark.parametrize("name,in_f,budget", _SPECS)
def test_same_width_uses_cellv03_width_and_reports_its_ratio(name, in_f, budget):
    hc, target = cellv03_target(in_f, budget)
    b = build_model(CONFIDENCE_MLP_SAME_WIDTH, in_f, budget)
    assert b.hidden_size == hc
    assert b.sizing["same_width_as_cellv03"] is True
    assert b.sizing["param_ratio_vs_cellv03"] == pytest.approx(b.parameter_count / target)
    first_linear = next(m for m in b.model.net.net if isinstance(m, nn.Linear))
    assert first_linear.in_features == 2 * in_f  # concat(x, c)


def _spy_net_input(model) -> dict:
    captured: dict = {}
    real = model.net.forward

    def spy(t):
        captured["in"] = t.detach().clone()
        return real(t)

    model.net.forward = spy
    return captured


def test_confidence_mlp_gets_concat_x_c_and_plain_does_not():
    x = torch.randn(4, 8)
    c = torch.rand(4, 8).clamp_min(1e-3)

    conf = build_model(CONFIDENCE_MLP, 8, 50_000).model
    cap = _spy_net_input(conf)
    conf(x, c)
    assert torch.equal(cap["in"], torch.cat([x, c], dim=-1))

    plain = build_model(PLAIN_MLP, 8, 50_000).model
    cap2 = _spy_net_input(plain)
    plain(x, c)
    assert torch.equal(cap2["in"], x)


def test_cellv03_receives_mu_x_e_c_u_zero():
    m: ReliabilityCellV03 = build_model(CELLV03, 8, 50_000).model
    captured: dict = {}
    real = m.net.layer1.forward

    def spy(belief):
        captured["b"] = belief
        return real(belief)

    m.net.layer1.forward = spy
    x = torch.randn(4, 8)
    c = torch.rand(4, 8).clamp_min(1e-3)
    m(x, c)
    b: BeliefCell = captured["b"]
    assert torch.equal(b.mu, x)
    assert torch.equal(b.evidence, c)
    assert torch.equal(b.uncertainty, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# NeuMiss (official implementation)
# ---------------------------------------------------------------------------


def test_neumiss_uses_the_official_pinned_package():
    import neumiss  # the external dependency
    from neumiss import NeuMissBlock  # noqa: F401

    assert NEUMISS_COMMIT == "7902b8dbe7114e8dc3010b5e8b132c35806a9d74"
    b = build_model(NEUMISS, 8, 50_000, neumiss_depth=3)
    assert isinstance(b.model, NeuMissWrapper)
    assert isinstance(b.model.block, neumiss.NeuMissBlock)
    assert b.model.block.depth == 3
    assert b.sizing["neumiss_commit"] == NEUMISS_COMMIT


@pytest.mark.parametrize("depth", NEUMISS_DEPTHS)
def test_neumiss_wrapper_preserves_nans_and_stays_finite(depth):
    b = build_model(NEUMISS, 8, 50_000, neumiss_depth=depth)
    x = torch.randn(6, 8)
    x[0, :] = float("nan")   # a whole row missing
    x[2, 3] = float("nan")
    out = b.model(x, torch.ones(6, 8))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()
    # gradients flow and stay finite through the Neumann iterations
    out.sum().backward()
    grads = [p.grad for p in b.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_neumiss_requires_an_explicit_depth():
    with pytest.raises(ValueError, match="neumiss_depth"):
        build_model(NEUMISS, 8, 50_000)


def test_neumiss_param_count_is_reported_not_matched():
    b1 = build_model(NEUMISS, 170, 100_000, neumiss_depth=1)
    b5 = build_model(NEUMISS, 170, 100_000, neumiss_depth=5)
    # shared-weight block -> depth does not change the parameter count
    assert b1.parameter_count == b5.parameter_count
    assert "params_within_budget" in b1.sizing
