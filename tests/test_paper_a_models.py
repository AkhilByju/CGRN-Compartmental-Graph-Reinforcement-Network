"""Paper A Phase 1 -- model construction: parameter matching, CellV0.1
freezing, and the fixed-confidence ablation's contract (Paper-A task Sec 15).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.paper_a.models import (
    STATE_COUNT_MULTIPLIER,
    FixedConfidenceBeliefNetwork,
    batched_forward,
    belief_diagnostics,
    belief_hidden_cells_for_budget,
    belief_param_count,
    build_cellv01,
    build_fixed_confidence,
    build_matched_mlp,
    build_state_count_mlp,
)
from src.models.architecture_v0.belief_network import BeliefNetwork
from src.models.architecture_v0.cell import BeliefCell

# (in_features, out_features, param_budget) for the seven benchmark datasets.
_SPECS = [
    ("breast_cancer", 30, 2, 10_000),
    ("wine", 13, 3, 10_000),
    ("digits", 64, 10, 25_000),
    ("diabetes", 10, 1, 10_000),
    ("california_housing", 8, 1, 10_000),
    ("mnist", 784, 10, 150_000),
    ("fashion_mnist", 784, 10, 150_000),
]


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv01_hidden_cells_is_largest_within_budget(name, in_f, out_f, budget):
    hc = belief_hidden_cells_for_budget(in_f, out_f, budget)
    assert belief_param_count(hc, in_f, out_f) <= budget
    assert belief_param_count(hc + 1, in_f, out_f) > budget


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv01_is_frozen_scale_stable_precision(name, in_f, out_f, budget):
    built = build_cellv01(in_f, out_f, budget)
    model = built.model
    assert isinstance(model, BeliefNetwork)
    assert model.layer1.aggregation == "scale_stable_precision"
    assert model.layer2.aggregation == "scale_stable_precision"
    assert built.parameter_count == belief_param_count(built.hidden_size, in_f, out_f)


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_matched_mlp_is_within_2pct(name, in_f, out_f, budget):
    cell = build_cellv01(in_f, out_f, budget)
    mlp = build_matched_mlp(in_f, out_f, budget)
    rel = abs(mlp.parameter_count - cell.parameter_count) / cell.parameter_count
    assert rel <= 0.02, f"{name}: matched MLP off by {rel:.3%}"
    assert mlp.sizing["param_match_within_2pct"] is True
    assert mlp.sizing["param_diff"] == mlp.parameter_count - cell.parameter_count


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_matched_mlp_architecture_is_linear_silu_linear(name, in_f, out_f, budget):
    mlp = build_matched_mlp(in_f, out_f, budget).model
    layers = list(mlp.net)
    assert [type(m) for m in layers] == [nn.Linear, nn.SiLU, nn.Linear]
    assert layers[0].in_features == in_f
    assert layers[-1].out_features == out_f


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_state_count_mlp_width_is_3x_hidden_cells(name, in_f, out_f, budget):
    cell = build_cellv01(in_f, out_f, budget)
    sc = build_state_count_mlp(in_f, out_f, budget)
    assert sc.hidden_size == STATE_COUNT_MULTIPLIER * cell.hidden_size
    # not parameter-matched -- just report the (possibly larger, possibly
    # smaller) count faithfully.
    assert sc.sizing["params_more_than_cellv01"] == sc.parameter_count - cell.parameter_count


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_fixed_confidence_has_identical_param_count_to_cellv01(name, in_f, out_f, budget):
    cell = build_cellv01(in_f, out_f, budget)
    fc = build_fixed_confidence(in_f, out_f, budget)
    assert fc.parameter_count == cell.parameter_count
    # same learnable tensors, same shapes
    cell_shapes = {k: tuple(v.shape) for k, v in cell.model.named_parameters()}
    fc_shapes = {k: tuple(v.shape) for k, v in fc.model.named_parameters()}
    assert cell_shapes == fc_shapes


def test_fixed_confidence_forces_incoming_e_and_u_to_one():
    torch.manual_seed(0)
    model = FixedConfidenceBeliefNetwork(in_features=6, hidden_cells=8, out_features=2)

    captured: dict[str, BeliefCell] = {}
    real_layer2_forward = model.layer2.forward

    def spy(incoming, source_participation=None):
        captured["incoming"] = incoming
        return real_layer2_forward(incoming, source_participation)

    model.layer2.forward = spy
    x = torch.randn(16, 6)
    model(x)

    incoming = captured["incoming"]
    assert torch.equal(incoming.evidence, torch.ones_like(incoming.evidence))
    assert torch.equal(incoming.uncertainty, torch.ones_like(incoming.uncertainty))

    # ...and the reset is doing something: layer 1's *native* output carries
    # non-trivial (non-all-ones) evidence/uncertainty before it's overwritten.
    b0 = BeliefCell.from_observed_features(x)
    b1_native = model.layer1(b0)
    assert not torch.allclose(b1_native.evidence, torch.ones_like(b1_native.evidence))


def test_batched_forward_matches_full_forward():
    torch.manual_seed(0)
    model = BeliefNetwork(12, 10, 3, aggregation="scale_stable_precision")
    x = torch.randn(500, 12)
    full = model(x)
    chunked = batched_forward(model, x, chunk=64)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_belief_diagnostics_are_chunk_invariant_and_finite():
    torch.manual_seed(1)
    model = BeliefNetwork(12, 10, 3, aggregation="scale_stable_precision")
    x = torch.randn(400, 12)
    a = belief_diagnostics(model, x, chunk=400)
    b = belief_diagnostics(model, x, chunk=37)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6)
        assert a[k] == a[k]  # not NaN
    assert a["diag_layer1_min_e"] >= 0.0
    assert a["diag_layer1_min_u"] >= 0.0


def test_fixed_confidence_and_cellv01_differ_in_output():
    torch.manual_seed(0)
    fc = FixedConfidenceBeliefNetwork(in_features=6, hidden_cells=8, out_features=1)
    torch.manual_seed(0)
    cell = BeliefNetwork(6, 8, 1, aggregation="scale_stable_precision")
    # identical init (same seed, same construction order) but the fixed-
    # confidence interception changes the forward pass.
    x = torch.randn(32, 6)
    assert not torch.allclose(fc(x), cell(x))
