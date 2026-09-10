"""Paper A Phase 2 -- model construction, the (x, c) contract each family
honours, and parameter matching against the CellV0.3 target (Phase-2 task
Sec 6 / Sec 7 / Sec 16).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.paper_a.reliability.models import (
    BELIEF_DIAG_KEYS,
    CELLV03,
    CONFIDENCE_MLP,
    MODEL_FAMILIES,
    PLAIN_MLP,
    RELIABILITY_GATED_MLP,
    ReliabilityCellV03,
    belief_diagnostics,
    build_model,
    cellv03_target,
)
from src.models.architecture_v0.cell import BeliefCell

# (name, in_features, out_features, param_budget) -- the four frozen Phase-2 sets
_SPECS = [
    ("digits", 64, 10, 25_000),
    ("california_housing", 8, 1, 10_000),
    ("mnist", 784, 10, 150_000),
    ("fashion_mnist", 784, 10, 150_000),
]


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_param_match_within_2pct(name, in_f, out_f, budget, family):
    _, target = cellv03_target(in_f, out_f, budget)
    built = build_model(family, in_f, out_f, budget)
    rel = abs(built.parameter_count - target) / target
    assert rel <= 0.02, f"{name}/{family} off by {rel:.3%}"
    if family != CELLV03:
        assert built.sizing["param_match_within_2pct"] is True


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv03_is_the_target_and_within_budget(name, in_f, out_f, budget):
    built = build_model(CELLV03, in_f, out_f, budget)
    hc, target = cellv03_target(in_f, out_f, budget)
    assert isinstance(built.model, ReliabilityCellV03)
    assert built.parameter_count == target
    assert built.hidden_size == hc
    assert built.parameter_count <= budget


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_confidence_mlp_input_layer_is_twice_the_feature_dim(name, in_f, out_f, budget):
    b = build_model(CONFIDENCE_MLP, in_f, out_f, budget)
    first_linear = next(m for m in b.model.net.net if isinstance(m, nn.Linear))
    assert first_linear.in_features == 2 * in_f


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_plain_and_gated_mlp_input_layer_is_the_feature_dim(name, in_f, out_f, budget):
    for fam in (PLAIN_MLP, RELIABILITY_GATED_MLP):
        b = build_model(fam, in_f, out_f, budget)
        first_linear = next(m for m in b.model.net.net if isinstance(m, nn.Linear))
        assert first_linear.in_features == in_f


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_mlp_families_are_linear_silu_linear(name, in_f, out_f, budget):
    for fam in (PLAIN_MLP, CONFIDENCE_MLP, RELIABILITY_GATED_MLP):
        layers = list(build_model(fam, in_f, out_f, budget).model.net.net)
        assert [type(m) for m in layers] == [nn.Linear, nn.SiLU, nn.Linear]
        assert layers[-1].out_features == out_f


# ---------------------------------------------------------------------------
# The (x, c) contract per family (Sec 16)
# ---------------------------------------------------------------------------


def _spy_net_input(model: nn.Module) -> dict:
    """Capture the tensor actually fed to the underlying MLP's ``net``."""
    captured: dict[str, torch.Tensor] = {}
    real = model.net.forward

    def spy(t):
        captured["input"] = t.detach().clone()
        return real(t)

    model.net.forward = spy
    return captured


def _xc(in_f: int, n: int = 8):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, in_f, generator=g)
    c = torch.rand(n, in_f, generator=g).clamp_min(1e-3)
    return x, c


def test_plain_mlp_never_receives_c():
    b = build_model(PLAIN_MLP, 12, 3, 10_000)
    cap = _spy_net_input(b.model)
    x, c = _xc(12)
    b.model(x, c)
    assert torch.equal(cap["input"], x)


def test_confidence_mlp_receives_x_then_c_concatenated():
    b = build_model(CONFIDENCE_MLP, 12, 3, 10_000)
    cap = _spy_net_input(b.model)
    x, c = _xc(12)
    b.model(x, c)
    assert torch.equal(cap["input"], torch.cat([x, c], dim=-1))
    assert cap["input"].shape[-1] == 24


def test_reliability_gated_mlp_receives_exactly_x_times_c():
    b = build_model(RELIABILITY_GATED_MLP, 12, 3, 10_000)
    cap = _spy_net_input(b.model)
    x, c = _xc(12)
    b.model(x, c)
    assert torch.equal(cap["input"], c * x)


def test_cellv03_receives_mu_x_e_c_u_zero():
    b = build_model(CELLV03, 12, 3, 10_000)
    model: ReliabilityCellV03 = b.model
    captured: dict[str, BeliefCell] = {}
    real = model.net.layer1.forward

    def spy(incoming):
        captured["belief"] = incoming
        return real(incoming)

    model.net.layer1.forward = spy
    x, c = _xc(12)
    model(x, c)
    belief = captured["belief"]
    assert torch.equal(belief.mu, x)
    assert torch.equal(belief.evidence, c)
    assert torch.equal(belief.uncertainty, torch.zeros_like(x))


def test_cellv03_wrapper_does_not_alter_belief_network_v03():
    # The wrapper composes layer1/layer2/readout exactly as BeliefNetworkV03
    # does, only swapping the input belief. With e=1, u=0 it must reproduce
    # the frozen network's own forward.
    torch.manual_seed(0)
    wrap = ReliabilityCellV03(6, 8, 2)
    x = torch.randn(5, 6)
    ones = torch.ones_like(x)
    ref = wrap.net(x)  # BeliefNetworkV03.forward uses initial_belief (e=1, u=0)
    got = wrap(x, ones)
    assert torch.allclose(ref, got, atol=1e-6)


# ---------------------------------------------------------------------------
# Belief diagnostics
# ---------------------------------------------------------------------------


def test_belief_diagnostics_keys_and_finiteness():
    b = build_model(CELLV03, 10, 3, 10_000)
    x, c = _xc(10, n=200)
    diag = belief_diagnostics(b.model, x, c, chunk=64)
    assert set(diag) == set(BELIEF_DIAG_KEYS)
    for k, v in diag.items():
        assert v == v and abs(v) < float("inf"), k
    for layer in ("layer1", "layer2"):
        assert diag[f"diag_{layer}_e_min"] > 0.0
        assert diag[f"diag_{layer}_u_min"] >= 0.0
        assert diag[f"diag_{layer}_precision_cv"] >= 0.0


def test_belief_diagnostics_are_chunk_invariant():
    b = build_model(CELLV03, 10, 3, 10_000)
    x, c = _xc(10, n=300)
    a = belief_diagnostics(b.model, x, c, chunk=300)
    d = belief_diagnostics(b.model, x, c, chunk=41)
    for k in a:
        noisy = k.endswith("_std") or k.endswith("_cv")
        tol = dict(rel=5e-2, abs=1e-4) if noisy else dict(rel=1e-5, abs=1e-6)
        assert a[k] == pytest.approx(d[k], **tol), k


def test_belief_diagnostics_reject_non_cellv03():
    b = build_model(PLAIN_MLP, 10, 3, 10_000)
    with pytest.raises(TypeError):
        belief_diagnostics(b.model, *_xc(10))
