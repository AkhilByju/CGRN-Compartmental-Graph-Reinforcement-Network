"""Paper A -- CellV0.2 wiring into the frozen Phase-1 protocol: parameter
budgeting, the builder contract, the diagnostics hook, and the guarantee
that CellV0.2 is not a member of the original (frozen) screening grid.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.paper_a.harness import ALL_FAMILIES, run_one
from experiments.paper_a.models import (
    CELLV02_FAMILY,
    MODEL_FAMILIES,
    batched_forward,
    belief_v02_hidden_cells_for_budget,
    belief_v02_param_count,
    build_cellv01,
    build_cellv02,
    build_model,
    precision_gain_diagnostics,
)
from src.models.architecture_v0.precision_gain import BeliefNetworkV02, PrecisionGainLayer

_SPECS = [
    ("breast_cancer", 30, 2, 10_000),
    ("wine", 13, 3, 10_000),
    ("digits", 64, 10, 25_000),
    ("diabetes", 10, 1, 10_000),
    ("california_housing", 8, 1, 10_000),
    ("mnist", 784, 10, 150_000),
    ("fashion_mnist", 784, 10, 150_000),
]


def test_cellv02_is_not_in_the_frozen_phase1_grid() -> None:
    # The original screen is frozen; CellV0.2 must never be swept into it by
    # `run_phase1.py`, only run explicitly by `run_phase1_v02.py`.
    assert CELLV02_FAMILY not in MODEL_FAMILIES
    assert CELLV02_FAMILY in ALL_FAMILIES  # ...but the harness still accepts it


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv02_hidden_cells_is_largest_within_budget(name, in_f, out_f, budget) -> None:
    hc = belief_v02_hidden_cells_for_budget(in_f, out_f, budget)
    assert belief_v02_param_count(hc, in_f, out_f) <= budget
    assert belief_v02_param_count(hc + 1, in_f, out_f) > budget


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_build_cellv02_contract(name, in_f, out_f, budget) -> None:
    built = build_cellv02(in_f, out_f, budget)
    model = built.model
    assert isinstance(model, BeliefNetworkV02)
    assert isinstance(model.layer1, PrecisionGainLayer)
    assert isinstance(model.layer2, PrecisionGainLayer)
    assert isinstance(model.readout, nn.Linear)

    # exactly three trainable parameter objects per hidden layer, + readout
    per_layer = {n for n, _ in model.layer1.named_parameters()}
    assert per_layer == {"V", "gain_raw", "bias"}

    assert built.family == CELLV02_FAMILY
    assert built.parameter_count == belief_v02_param_count(built.hidden_size, in_f, out_f)
    assert built.parameter_count <= budget
    assert built.sizing["params_within_budget"] is True

    # same construction path as build_model
    assert build_model(CELLV02_FAMILY, in_f, out_f, budget).parameter_count == built.parameter_count


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv02_gets_more_hidden_cells_than_cellv01_at_equal_budget(
    name, in_f, out_f, budget
) -> None:
    # No relevance-gate matrix -> ~half the per-connection cost -> a wider
    # hidden population fits the identical budget. This is reported, not
    # equalized (Paper-A task).
    c2 = build_cellv02(in_f, out_f, budget)
    c1 = build_cellv01(in_f, out_f, budget)
    assert c2.hidden_size > c1.hidden_size
    assert c2.sizing["cellv01_hidden_cells"] == c1.hidden_size


def test_cellv02_readout_consumes_confidence() -> None:
    # The readout is applied to relative_gain(final_precision) * final_mu,
    # not final_mu -- so zeroing the readout-input scaling path changes the
    # output. Concretely: a model whose final belief has uniform precision
    # (relative_gain == 1) must match plain readout(mu); perturbing u must
    # then move the prediction.
    torch.manual_seed(0)
    net = BeliefNetworkV02(6, 8, 2)
    x = torch.randn(4, 6)
    pred, belief = net.forward_with_beliefs(x)

    from src.models.architecture_v0.precision_gain import effective_precision, relative_gain

    pi = effective_precision(belief.evidence, belief.uncertainty)
    scaled = relative_gain(pi, net.eps) * belief.mu
    assert torch.allclose(pred, net.readout(scaled), atol=1e-6)
    assert not torch.allclose(pred, net.readout(belief.mu), atol=1e-4)


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS[:4])
def test_precision_gain_diagnostics_shape_and_finiteness(name, in_f, out_f, budget) -> None:
    model = build_cellv02(in_f, out_f, budget).model
    x = torch.randn(300, in_f)
    diag = precision_gain_diagnostics(model, x, chunk=64)

    expected = {
        f"diag_{layer}_{stat}"
        for layer in ("layer1", "layer2")
        for stat in (
            "precision_mean", "precision_std", "precision_min", "precision_max",
            "relative_gain_mean", "relative_gain_std",
        )
    }
    assert set(diag) == expected
    for k, v in diag.items():
        assert v == v and abs(v) < float("inf"), k
    for layer in ("layer1", "layer2"):
        assert 0.0 < diag[f"diag_{layer}_relative_gain_mean"] < 2.0
        assert diag[f"diag_{layer}_precision_min"] >= 0.0


def test_precision_gain_diagnostics_are_chunk_invariant() -> None:
    torch.manual_seed(1)
    model = build_cellv02(10, 3, 10_000).model
    x = torch.randn(400, 10)
    a = precision_gain_diagnostics(model, x, chunk=400)
    b = precision_gain_diagnostics(model, x, chunk=37)
    for k in a:
        # `_std` accumulates E[x^2]-E[x]^2, so the std keys carry a little
        # chunk-order cancellation noise; mean/min/max are order-exact.
        tol = dict(rel=5e-2, abs=1e-4) if k.endswith("_std") else dict(rel=1e-5, abs=1e-6)
        assert a[k] == pytest.approx(b[k], **tol)


def test_precision_gain_diagnostics_reject_non_v02_model() -> None:
    with pytest.raises(TypeError):
        precision_gain_diagnostics(build_cellv01(10, 2, 10_000).model, torch.randn(4, 10))


def test_batched_forward_matches_full_forward_for_cellv02() -> None:
    torch.manual_seed(0)
    model = build_cellv02(12, 3, 10_000).model
    x = torch.randn(500, 12)
    assert torch.allclose(model(x), batched_forward(model, x, chunk=64), atol=1e-5)


def test_cellv02_run_one_end_to_end_writes_record(tmp_path) -> None:
    r = run_one(
        "diabetes", 0.25, CELLV02_FAMILY, 0,
        device=torch.device("cpu"),
        results_dir=tmp_path,
        experiment_id="paper_a_phase1_cellv02_test",
        max_steps=400,
    )
    assert r["family"] == CELLV02_FAMILY
    assert r["headline_metric"] == "r2"
    assert r["parameter_count"] <= 10_000
    assert any(k.startswith("diag_layer1_precision") for k in r)
    written = list(tmp_path.glob("paper_a_phase1_cellv02_test_*.json"))
    assert written
