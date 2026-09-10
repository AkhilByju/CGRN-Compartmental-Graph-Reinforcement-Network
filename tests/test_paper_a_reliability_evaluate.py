"""Paper A Phase 2 -- per-severity evaluation, replica averaging, corruption
AUC and OOD drop (Phase-2 task Sec 10 / Sec 11 / Sec 16).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.datasets import prepare_dataset
from experiments.paper_a.reliability.corruption import (
    GAUSSIAN,
    GAUSSIAN_S_TEST,
    MISSING,
    MISSING_P_TEST,
    corrupt,
)
from experiments.paper_a.reliability.evaluate import (
    CLASSIFICATION_METRICS,
    corruption_auc,
    evaluate_severity,
    sweep,
)
from experiments.paper_a.reliability.models import CELLV03, PLAIN_MLP, build_model
from experiments.paper_a.reliability.training import train_model


@pytest.fixture(scope="module")
def trained_cellv03():
    prep = prepare_dataset("digits", seed=0, train_fraction=1.0)
    torch.manual_seed(0)
    model = build_model(CELLV03, prep.n_features, prep.out_features, prep.param_budget).model
    train_model(
        model, prep, corruption_family=MISSING,
        device=torch.device("cpu"), seed=0, max_steps=200, val_every=50,
    )
    return prep, model


def test_corruption_auc_is_trapezoid_over_the_grid():
    pts = [(0.0, 1.0), (0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    assert corruption_auc(pts) == pytest.approx(float(np.trapezoid(ys, xs)))
    # order-independent
    assert corruption_auc(list(reversed(pts))) == pytest.approx(corruption_auc(pts))


def test_evaluate_severity_averages_three_deterministic_replicas(trained_cellv03):
    prep, model = trained_cellv03
    a = evaluate_severity(model, CELLV03, prep, MISSING, 0, 0.3, device=torch.device("cpu"))
    b = evaluate_severity(model, CELLV03, prep, MISSING, 0, 0.3, device=torch.device("cpu"))
    assert len(a.replica_metrics) == 3
    assert a.metrics["accuracy"] == pytest.approx(b.metrics["accuracy"], abs=1e-9)
    assert a.metrics["accuracy"] == pytest.approx(
        float(np.mean([rm["accuracy"] for rm in a.replica_metrics]))
    )
    assert a.in_distribution is True
    assert a.belief_diag is not None and "diag_layer2_precision_mean" in a.belief_diag


def test_two_models_evaluated_at_a_severity_see_the_same_corruption(trained_cellv03):
    prep, cellv03 = trained_cellv03
    torch.manual_seed(1)
    mlp = build_model(PLAIN_MLP, prep.n_features, prep.out_features, prep.param_budget).model
    kw = dict(experiment_seed=0, split="test", epoch=0, replica=2, severity=0.5)
    a = corrupt(MISSING, prep.x_test, **kw)
    b = corrupt(MISSING, prep.x_test, **kw)
    assert torch.equal(a.x, b.x) and torch.equal(a.c, b.c)
    # both checkpoints score against that identical corruption
    assert cellv03 is not mlp


def test_sweep_covers_the_full_grid_and_reports_auc_and_ood(trained_cellv03):
    prep, model = trained_cellv03
    sw = sweep(model, CELLV03, prep, MISSING, 0, device=torch.device("cpu"))
    assert [s.severity for s in sw.severities] == list(MISSING_P_TEST)
    assert set(sw.corruption_auc) == set(CLASSIFICATION_METRICS)
    assert set(sw.ood_drop) == set(CLASSIFICATION_METRICS)
    # OOD drop = metric(p=.3) - metric(p=.7)
    expected = sw.by_severity(0.3).metrics["accuracy"] - sw.by_severity(0.7).metrics["accuracy"]
    assert sw.ood_drop["accuracy"] == pytest.approx(expected, abs=1e-9)
    # reliability response has one (l1, l2) pair per severity for CellV0.3
    resp = sw.reliability_response()
    assert set(resp) == set(MISSING_P_TEST)


def test_sweep_gaussian_grid_and_serialization(trained_cellv03):
    prep, model = trained_cellv03
    sw = sweep(model, CELLV03, prep, GAUSSIAN, 0, device=torch.device("cpu"))
    assert [s.severity for s in sw.severities] == list(GAUSSIAN_S_TEST)
    d = sw.to_dict()
    import json

    json.dumps(d)  # must be JSON-serializable for the RunRecord
    assert d["severities"][0]["severity"] == 0.0
