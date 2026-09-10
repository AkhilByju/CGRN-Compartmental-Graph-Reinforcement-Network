"""Paper A Phase 2 -- confidence-intervention test contract (Sec 15): eval
only, deterministic, CellV0.3-only, true/all-ones/shuffled with deltas vs
true.
"""

from __future__ import annotations

import pytest
import torch

from experiments.paper_a.datasets import prepare_dataset
from experiments.paper_a.reliability.corruption import MAX_OOD_SEVERITY, MAX_TRAIN_SEVERITY, MISSING
from experiments.paper_a.reliability.interventions import (
    CONDITIONS,
    REGIMES,
    run_all_regimes,
    run_intervention,
)
from experiments.paper_a.reliability.models import CELLV03, PLAIN_MLP, build_model
from experiments.paper_a.reliability.training import train_model


@pytest.fixture(scope="module")
def trained():
    prep = prepare_dataset("digits", seed=0, train_fraction=1.0)  # stands in for the real sets
    torch.manual_seed(0)
    model = build_model(CELLV03, prep.n_features, prep.out_features, prep.param_budget).model
    train_model(
        model, prep, corruption_family=MISSING,
        device=torch.device("cpu"), seed=0, max_steps=200, val_every=50,
    )
    return prep, model


def test_intervention_reports_three_conditions_and_two_deltas(trained):
    prep, model = trained
    r = run_intervention(model, prep, MISSING, 0, "in_dist_max", device=torch.device("cpu"))
    assert set(r.condition_metric) == set(CONDITIONS)
    assert set(r.delta_vs_true) == {"all_ones", "shuffled"}
    assert r.delta_vs_true["all_ones"] == pytest.approx(
        r.condition_metric["all_ones"] - r.condition_metric["true"], abs=1e-9
    )
    assert r.severity == MAX_TRAIN_SEVERITY[MISSING]


def test_intervention_is_deterministic(trained):
    prep, model = trained
    a = run_intervention(model, prep, MISSING, 0, "ood_max", device=torch.device("cpu"))
    b = run_intervention(model, prep, MISSING, 0, "ood_max", device=torch.device("cpu"))
    assert a.condition_metric == pytest.approx(b.condition_metric, abs=1e-9)
    assert a.severity == MAX_OOD_SEVERITY[MISSING]


def test_all_regimes_covers_both_severity_regimes(trained):
    prep, model = trained
    results = run_all_regimes(model, prep, MISSING, 0, device=torch.device("cpu"))
    assert [r.regime for r in results] == list(REGIMES)


def test_true_condition_matches_the_plain_forward(trained):
    prep, model = trained
    from experiments.paper_a.reliability.corruption import corrupt
    from experiments.paper_a.reliability.evaluate import _score
    from experiments.paper_a.reliability.training import forward_xc

    cor = corrupt(
        MISSING, prep.x_test,
        experiment_seed=0, split="test", epoch=0, replica=0, severity=0.3,
    )
    pred = forward_xc(model, cor.x, cor.c)
    acc = _score("classification", pred, prep)["accuracy"]
    # a single-replica 'true' run must reproduce the hand-computed accuracy
    r = run_intervention(
        model, prep, MISSING, 0, "in_dist_max",
        device=torch.device("cpu"), n_replicas=1,
    )
    assert r.condition_metric["true"] == pytest.approx(acc, abs=1e-9)


def test_intervention_rejects_non_cellv03(trained):
    prep, _ = trained
    torch.manual_seed(0)
    mlp = build_model(PLAIN_MLP, prep.n_features, prep.out_features, prep.param_budget).model
    with pytest.raises(TypeError):
        run_intervention(mlp, prep, MISSING, 0, "in_dist_max", device=torch.device("cpu"))
