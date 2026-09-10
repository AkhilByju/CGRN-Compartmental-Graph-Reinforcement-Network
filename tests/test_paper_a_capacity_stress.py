"""Paper A Phase 3 Part A -- the same-width Confidence MLP capacity control
(Phase-3 task Part A). It must use CellV0.3's exact hidden width, keep the
Phase-2 Confidence MLP architecture, and be allowed to exceed the parameter
budget. It must NOT be a member of the frozen Phase-2 grid.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from experiments.paper_a.capacity_stress import CAPACITY_DATASETS, CAPACITY_EXPERIMENT_ID
from experiments.paper_a.reliability.harness import run_one
from experiments.paper_a.reliability.models import (
    ALL_FAMILIES,
    CONFIDENCE_MLP,
    CONFIDENCE_MLP_SAME_WIDTH,
    MODEL_FAMILIES,
    build_model,
    cellv03_target,
)

_SPECS = [
    ("mnist", 784, 10, 150_000),
    ("fashion_mnist", 784, 10, 150_000),
    ("digits", 64, 10, 25_000),
]


def test_same_width_is_not_in_the_frozen_phase2_grid():
    assert CONFIDENCE_MLP_SAME_WIDTH not in MODEL_FAMILIES
    assert CONFIDENCE_MLP_SAME_WIDTH in ALL_FAMILIES
    assert set(CAPACITY_DATASETS) == {"mnist", "fashion_mnist", "digits"}


@pytest.mark.parametrize("name,in_f,out_f,budget", _SPECS)
def test_same_width_uses_cellv03_hidden_width(name, in_f, out_f, budget):
    hc, target = cellv03_target(in_f, out_f, budget)
    sw = build_model(CONFIDENCE_MLP_SAME_WIDTH, in_f, out_f, budget)
    assert sw.hidden_size == hc
    assert sw.sizing["cellv03_hidden_cells"] == hc
    assert sw.sizing["same_width_as_cellv03"] is True
    # its underlying net takes concat(x, c) -> 2 * in_features
    first_linear = next(m for m in sw.model.net.net if isinstance(m, nn.Linear))
    assert first_linear.in_features == 2 * in_f
    # ...and the ratio vs CellV0.3 is reported, whatever it is
    assert sw.sizing["param_ratio_vs_cellv03"] == pytest.approx(sw.parameter_count / target)


def test_same_width_architecture_matches_the_phase2_confidence_mlp():
    sw = build_model(CONFIDENCE_MLP_SAME_WIDTH, 64, 10, 25_000).model
    cm = build_model(CONFIDENCE_MLP, 64, 10, 25_000).model
    assert [type(m) for m in sw.net.net] == [type(m) for m in cm.net.net]
    assert isinstance(sw.net.net[1], nn.SiLU)


def test_same_width_is_allowed_to_exceed_the_budget_on_the_image_sets():
    for name, in_f, out_f, budget in _SPECS[:2]:  # mnist, fashion
        sw = build_model(CONFIDENCE_MLP_SAME_WIDTH, in_f, out_f, budget)
        assert sw.parameter_count > budget  # over-powered capacity control
        assert sw.sizing["param_ratio_vs_cellv03"] > 1.0


def test_capacity_stress_run_one_end_to_end(tmp_path):
    r = run_one(
        "digits", "missing", CONFIDENCE_MLP_SAME_WIDTH, 0,
        device=torch.device("cpu"), results_dir=tmp_path,
        experiment_id=CAPACITY_EXPERIMENT_ID, max_steps=150,
    )
    assert r["family"] == CONFIDENCE_MLP_SAME_WIDTH
    assert r["interventions"] == []  # not CellV0.3
    assert all(s["belief_diag"] is None for s in r["sweep"]["severities"])
    rec = json.loads(
        next(p for p in tmp_path.glob("*.json") if not p.name.endswith("_history.json")).read_text()
    )
    assert rec["experiment_id"] == CAPACITY_EXPERIMENT_ID
    assert rec["config"]["extra"]["sizing"]["same_width_as_cellv03"] is True
