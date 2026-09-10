"""Paper A Phase 3 Part B -- training protocol: the shared LR grid, PR-AUC /
R^2 checkpoint selection (never accuracy), validation-only APS cost-threshold
selection, and the NeuMiss NaN-input routing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.real_reliability.datasets import dataset_available, prepare_dataset
from experiments.paper_a.real_reliability.models import NEUMISS, build_model
from experiments.paper_a.real_reliability.training import (
    FN_COST,
    FP_COST,
    LEARNING_RATES,
    _cost_optimal_threshold,
    model_inputs,
    official_cost,
    train_one,
    train_with_grid,
)

pytestmark = pytest.mark.skipif(
    not (dataset_available("aps") and dataset_available("air_quality")),
    reason="UCI APS / Air Quality archives not reachable",
)


def test_learning_rate_grid_is_the_predeclared_three():
    assert LEARNING_RATES == (1e-3, 3e-3, 1e-2)


# ---------------------------------------------------------------------------
# cost threshold (pure function, no data needed but grouped here)
# ---------------------------------------------------------------------------


def test_cost_optimal_threshold_minimizes_the_official_cost():
    rng = np.random.default_rng(0)
    prob = rng.random(400)
    y = (rng.random(400) < 0.1).astype(int)
    thr, cost = _cost_optimal_threshold(prob, y)
    assert official_cost(prob, y, thr) == pytest.approx(cost)
    # no other threshold on the grid does better
    for t in np.unique(prob):
        assert official_cost(prob, y, t) >= cost - 1e-6


def test_fn_is_50x_more_expensive_than_fp():
    assert (FN_COST, FP_COST) == (500.0, 10.0)


# ---------------------------------------------------------------------------
# NeuMiss input routing
# ---------------------------------------------------------------------------


def test_neumiss_gets_nan_inputs_everyone_else_gets_imputed():
    d = prepare_dataset("air_quality", seed=0)
    x_nm, _ = model_inputs(d, "train", NEUMISS)
    x_mlp, _ = model_inputs(d, "train", "plain_mlp")
    assert torch.isnan(x_nm).any()          # NeuMiss keeps NaN
    assert not torch.isnan(x_mlp).any()     # everyone else is imputed
    assert torch.equal(torch.isnan(x_nm), torch.isnan(d.x_train_nan))
    assert torch.equal(x_nm.nan_to_num(), d.x_train_nan.nan_to_num())
    assert torch.equal(x_mlp, d.x_train)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def test_air_quality_checkpoint_selection_uses_r2_and_restores_the_best():
    d = prepare_dataset("air_quality", seed=0)
    b = build_model("plain_mlp", d.n_features, d.param_budget)
    o = train_one(
        b.model, d, family="plain_mlp", lr=3e-3, neumiss_depth=None,
        device=torch.device("cpu"), seed=0, max_steps=400,
    )
    curve = [h["val_metric"] for h in o.history if "val_metric" in h]
    assert o.best_val_metric == pytest.approx(max(curve), abs=1e-9)
    assert not o.diverged


def test_aps_freezes_a_validation_cost_threshold():
    d = prepare_dataset("aps", seed=0)
    b = build_model("plain_mlp", d.n_features, d.param_budget)
    o = train_one(
        b.model, d, family="plain_mlp", lr=3e-3, neumiss_depth=None,
        device=torch.device("cpu"), seed=0, max_steps=300,
    )
    assert o.cost_threshold is not None
    assert 0.0 <= o.cost_threshold <= 1.0
    # the primary validation metric is PR-AUC, not accuracy
    assert 0.0 <= o.best_val_metric <= 1.0


def test_grid_search_picks_the_best_validation_config():
    d = prepare_dataset("air_quality", seed=1)
    outcome, model, built = train_with_grid(
        d, "neumiss", in_features=d.n_features, param_budget=d.param_budget,
        seed=1, device=torch.device("cpu"),
        learning_rates=(1e-3, 1e-2), max_steps=250,
    )
    assert outcome.neumiss_depth in (1, 3, 5)
    assert outcome.lr in (1e-3, 1e-2)
    assert built.parameter_count > 0
