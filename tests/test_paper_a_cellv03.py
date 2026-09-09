"""Paper A -- CellV0.3 wiring into the frozen Phase-1 protocol: parameter
budgeting, the builder contract, the diagnostics hook (incl. the init
snapshot), and the guarantee that CellV0.3 is not a member of the original
(frozen) screening grid.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.paper_a.harness import ALL_FAMILIES, run_one
from experiments.paper_a.models import (
    CELLV03_FAMILY,
    MODEL_FAMILIES,
    batched_forward,
    belief_v02_hidden_cells_for_budget,
    belief_v03_hidden_cells_for_budget,
    belief_v03_param_count,
    build_cellv01,
    build_cellv02,
    build_cellv03,
    build_model,
    conflict_normalized_diagnostics,
)
from src.models.architecture_v0.conflict_normalized import (
    BeliefNetworkV03,
    ConflictNormalizedLayer,
)

_SPECS = [
    ("breast_cancer", 30, 2, 10_000),
    ("wine", 13, 3, 10_000),
    ("digits", 64, 10, 25_000),
    ("diabetes", 10, 1, 10_000),
    ("california_housing", 8, 1, 10_000),
    ("mnist", 784, 10, 150_000),
    ("fashion_mnist", 784, 10, 150_000),
]

_V03_DIAG_KEYS = {
    f"diag_{layer}_{q}_{stat}"
    for layer in ("layer1", "layer2")
    for q in ("e", "u", "precision", "sqrt_precision")
    for stat in ("mean", "std", "min", "max")
} | {
    f"diag_{layer}_{k}"
    for layer in ("layer1", "layer2")
    for k in ("abs_consensus_mean", "precision_cv")
}


def test_cellv03_is_not_in_the_frozen_phase1_grid() -> None:
    assert CELLV03_FAMILY not in MODEL_FAMILIES
    assert CELLV03_FAMILY in ALL_FAMILIES  # ...but the harness still accepts it


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv03_hidden_cells_is_largest_within_budget(name, in_f, out_f, budget) -> None:
    hc = belief_v03_hidden_cells_for_budget(in_f, out_f, budget)
    assert belief_v03_param_count(hc, in_f, out_f) <= budget
    assert belief_v03_param_count(hc + 1, in_f, out_f) > budget
    # CellV0.3's parameterization is identical to CellV0.2's, so it fits the
    # exact same hidden width at every budget.
    assert hc == belief_v02_hidden_cells_for_budget(in_f, out_f, budget)


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_build_cellv03_contract(name, in_f, out_f, budget) -> None:
    built = build_cellv03(in_f, out_f, budget)
    model = built.model
    assert isinstance(model, BeliefNetworkV03)
    assert isinstance(model.layer1, ConflictNormalizedLayer)
    assert isinstance(model.layer2, ConflictNormalizedLayer)
    assert isinstance(model.readout, nn.Linear)

    per_layer = {n for n, _ in model.layer1.named_parameters()}
    assert per_layer == {"V", "gain_raw", "bias"}

    assert built.family == CELLV03_FAMILY
    assert built.parameter_count == belief_v03_param_count(built.hidden_size, in_f, out_f)
    assert built.parameter_count <= budget
    assert built.sizing["params_within_budget"] is True

    assert build_model(CELLV03_FAMILY, in_f, out_f, budget).parameter_count == built.parameter_count


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_cellv03_matches_cellv02_sizing_and_beats_cellv01_at_equal_budget(
    name, in_f, out_f, budget
) -> None:
    c3 = build_cellv03(in_f, out_f, budget)
    c2 = build_cellv02(in_f, out_f, budget)
    c1 = build_cellv01(in_f, out_f, budget)
    assert c3.hidden_size == c2.hidden_size
    assert c3.parameter_count == c2.parameter_count
    assert c3.hidden_size > c1.hidden_size
    assert c3.sizing["cellv01_hidden_cells"] == c1.hidden_size
    assert c3.sizing["cellv02_hidden_cells"] == c2.hidden_size


def test_cellv03_readout_consumes_final_mu_directly() -> None:
    # Unlike CellV0.2, the readout is applied to final_mu with no precision
    # re-scaling -- every CellV0.3 cell already folded sqrt(pi) into its own
    # activation.
    torch.manual_seed(0)
    net = BeliefNetworkV03(6, 8, 2)
    x = torch.randn(4, 6)
    (b1, _), (b2, _) = net.layer_states_verbose(x)
    assert torch.allclose(net(x), net.readout(b2.mu), atol=1e-6)


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS[:4])
def test_conflict_normalized_diagnostics_shape_and_finiteness(name, in_f, out_f, budget) -> None:
    model = build_cellv03(in_f, out_f, budget).model
    x = torch.randn(300, in_f)
    diag = conflict_normalized_diagnostics(model, x, chunk=64)

    assert set(diag) == _V03_DIAG_KEYS
    for k, v in diag.items():
        assert v == v and abs(v) < float("inf"), k
    for layer in ("layer1", "layer2"):
        assert diag[f"diag_{layer}_e_min"] > 0.0
        assert diag[f"diag_{layer}_u_min"] >= 0.0
        assert diag[f"diag_{layer}_precision_min"] > 0.0
        assert diag[f"diag_{layer}_precision_cv"] >= 0.0


def test_conflict_normalized_diagnostics_are_chunk_invariant() -> None:
    torch.manual_seed(1)
    model = build_cellv03(10, 3, 10_000).model
    x = torch.randn(400, 10)
    a = conflict_normalized_diagnostics(model, x, chunk=400)
    b = conflict_normalized_diagnostics(model, x, chunk=37)
    for k in a:
        # E[x^2]-E[x]^2 accumulation -> std / cv carry a little chunk-order
        # cancellation noise; mean/min/max/abs_consensus are order-exact.
        noisy = k.endswith("_std") or k.endswith("_cv")
        tol = dict(rel=5e-2, abs=1e-4) if noisy else dict(rel=1e-5, abs=1e-6)
        assert a[k] == pytest.approx(b[k], **tol), k


def test_conflict_normalized_diagnostics_reject_non_v03_model() -> None:
    with pytest.raises(TypeError):
        conflict_normalized_diagnostics(build_cellv02(10, 2, 10_000).model, torch.randn(4, 10))


def test_batched_forward_matches_full_forward_for_cellv03() -> None:
    torch.manual_seed(0)
    model = build_cellv03(12, 3, 10_000).model
    x = torch.randn(500, 12)
    assert torch.allclose(model(x), batched_forward(model, x, chunk=64), atol=1e-5)


def test_cellv03_run_one_end_to_end_writes_record_with_init_and_final_diagnostics(tmp_path) -> None:
    r = run_one(
        "diabetes", 0.25, CELLV03_FAMILY, 0,
        device=torch.device("cpu"),
        results_dir=tmp_path,
        experiment_id="paper_a_phase1_cellv03_test",
        max_steps=400,
    )
    assert r["family"] == CELLV03_FAMILY
    assert r["headline_metric"] == "r2"
    assert r["parameter_count"] <= 10_000
    assert any(k.startswith("diag_layer1_precision") for k in r)
    # the init snapshot is recorded alongside the best-checkpoint diagnostics
    assert set(r["diagnostics_init"]) == _V03_DIAG_KEYS

    import json

    written = [
        p for p in tmp_path.glob("paper_a_phase1_cellv03_test_*.json")
        if not p.name.endswith("_history.json")
    ]
    assert written
    rec = json.loads(written[0].read_text())
    assert set(rec["config"]["extra"]["diagnostics_init"]) == _V03_DIAG_KEYS
