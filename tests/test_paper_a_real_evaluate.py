"""Paper A Phase 3 Part B -- test-set evaluation, missingness stratification,
CellV0.3 belief-by-bin diagnostics, and the confidence intervention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.real_reliability.datasets import (
    MISSINGNESS_BINS,
    dataset_available,
    prepare_dataset,
)
from experiments.paper_a.real_reliability.evaluate import (
    confidence_intervention,
    evaluate_test,
    primary_metric_name,
)
from experiments.paper_a.real_reliability.models import build_model
from experiments.paper_a.real_reliability.training import train_one

pytestmark = pytest.mark.skipif(
    not (dataset_available("aps") and dataset_available("air_quality")),
    reason="UCI APS / Air Quality archives not reachable",
)


def _train(name, family, max_steps=250, depth=None):
    d = prepare_dataset(name, seed=0)
    kw = {"neumiss_depth": depth} if depth else {}
    b = build_model(family, d.n_features, d.param_budget, **kw)
    o = train_one(
        b.model, d, family=family, lr=3e-3, neumiss_depth=depth,
        device=torch.device("cpu"), seed=0, max_steps=max_steps,
    )
    return d, b.model, o


def test_primary_metrics_are_pr_auc_and_r2():
    assert primary_metric_name("classification") == "pr_auc"
    assert primary_metric_name("regression") == "r2"


def test_aps_evaluation_reports_the_full_imbalanced_metric_suite():
    d, model, o = _train("aps", "cellv0.3")
    ev = evaluate_test(model, d, "cellv0.3", device=torch.device("cpu"), threshold=o.cost_threshold)
    for k in ("pr_auc", "roc_auc", "balanced_accuracy", "f1", "precision",
              "recall", "official_cost", "cost_per_1000"):
        assert k in ev.overall
    assert ev.primary_metric == "pr_auc"
    # cost matches an independent recomputation at the frozen threshold
    from experiments.paper_a.real_reliability.evaluate import _test_predictions
    from experiments.paper_a.real_reliability.training import official_cost
    prob = torch.sigmoid(_test_predictions(model, d, "cellv0.3", torch.device("cpu"))).numpy()
    y = d.y_test.numpy()
    assert ev.overall["official_cost"] == pytest.approx(
        official_cost(prob, y, o.cost_threshold)
    )
    assert ev.overall["cost_per_1000"] == pytest.approx(ev.overall["official_cost"] / 16.0)


def test_air_quality_evaluation_is_on_the_original_target_scale():
    d, model, _ = _train("air_quality", "plain_mlp")
    ev = evaluate_test(model, d, "plain_mlp", device=torch.device("cpu"))
    assert set(ev.overall) >= {"r2", "rmse", "mae"}
    # a sane CO(GT) RMSE is in single digits (ppm-ish), not ~1 (standardized)
    assert 0.1 < ev.overall["rmse"] < 5.0


def test_missingness_strata_partition_the_test_set_and_skip_empty_bins():
    d, model, o = _train("aps", "cellv0.3")
    ev = evaluate_test(model, d, "cellv0.3", device=torch.device("cpu"), threshold=o.cost_threshold)
    labels = [r["bin"] for r in ev.by_stratum]
    assert set(labels).issubset({b[0] for b in MISSINGNESS_BINS})
    assert sum(r["n"] for r in ev.by_stratum) == d.x_test.shape[0]  # every example counted once


def test_cellv03_belief_by_stratum_present_only_for_cellv03():
    d, model, o = _train("aps", "cellv0.3")
    ev = evaluate_test(model, d, "cellv0.3", device=torch.device("cpu"), threshold=o.cost_threshold)
    assert ev.belief_by_stratum
    for r in ev.belief_by_stratum:
        assert {"l1_pi_mean", "l2_pi_mean", "l2_u_mean", "l2_pi_cv"} <= set(r)

    d2, model2, _ = _train("air_quality", "plain_mlp")
    ev2 = evaluate_test(model2, d2, "plain_mlp", device=torch.device("cpu"))
    assert ev2.belief_by_stratum == []


def test_confidence_intervention_scores_three_conditions_deterministically():
    d, model, _ = _train("air_quality", "cellv0.3")
    a = confidence_intervention(model, d, device=torch.device("cpu"), seed=0)
    b = confidence_intervention(model, d, device=torch.device("cpu"), seed=0)
    assert a.to_dict() == b.to_dict()
    assert a.delta_all_ones == pytest.approx(a.true - a.all_ones)
    assert a.delta_shuffled == pytest.approx(a.true - a.shuffled)


def test_confidence_intervention_rejects_non_cellv03():
    d, model, _ = _train("air_quality", "plain_mlp")
    with pytest.raises(TypeError):
        confidence_intervention(model, d, device=torch.device("cpu"))


def test_all_ones_intervention_actually_changes_the_input_where_features_are_missing():
    # sanity: on APS (mixed observed/missing per row) all-ones c differs from true c
    d = prepare_dataset("aps", seed=0)
    assert not torch.equal(d.c_test, torch.ones_like(d.c_test))
    assert (d.c_test == 1e-3).any()
    missing_rate = float((d.c_test == 1e-3).float().mean())
    assert np.isclose(missing_rate, d.meta["overall_missing_frac"], atol=0.05)
