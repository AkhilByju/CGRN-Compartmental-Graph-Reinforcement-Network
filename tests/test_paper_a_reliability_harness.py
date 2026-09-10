"""Paper A Phase 2 -- end-to-end run harness: one (dataset, corruption, model,
seed) cell writes a complete RunRecord with the severity sweep, AUC/OOD, and
(for CellV0.3) belief diagnostics + confidence interventions. Fast: bundled
digits, hard step cap, CPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from experiments.paper_a.reliability.corruption import (
    GAUSSIAN,
    GAUSSIAN_S_TEST,
    MISSING,
    MISSING_P_TEST,
)
from experiments.paper_a.reliability.harness import EXPERIMENT_ID, RELIABILITY_DATASETS, run_one
from experiments.paper_a.reliability.models import MODEL_FAMILIES


def test_reliability_datasets_are_the_four_frozen_sets():
    assert set(RELIABILITY_DATASETS) == {"mnist", "fashion_mnist", "digits", "california_housing"}


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_run_one_writes_a_complete_record(tmp_path, family):
    r = run_one(
        "digits", MISSING, family, 0,
        device=torch.device("cpu"), results_dir=tmp_path, max_steps=200,
    )
    assert r["family"] == family
    assert r["primary_metric"] == "accuracy"
    assert r["parameter_count"] > 0
    # full severity grid present
    sev = [s["severity"] for s in r["sweep"]["severities"]]
    assert sev == list(MISSING_P_TEST)
    assert "corruption_auc" in r["sweep"] and "ood_drop" in r["sweep"]

    records = [p for p in tmp_path.glob("*.json") if not p.name.endswith("_history.json")]
    assert len(records) == 1
    rec = json.loads(records[0].read_text())
    assert rec["experiment_id"] == EXPERIMENT_ID
    assert rec["config"]["extra"]["corruption_family"] == MISSING
    assert rec["config"]["extra"]["sweep"]["family"] == family
    for key in ("git_commit", "parameter_count", "seed", "dataset", "architecture"):
        assert key in rec
    history = json.loads(next(tmp_path.glob("*_history.json")).read_text())
    assert history  # validation curve retained


def test_cellv03_run_records_diagnostics_and_no_interventions_off_the_two_sets(tmp_path):
    r = run_one(
        "digits", GAUSSIAN, "cellv0.3", 0,
        device=torch.device("cpu"), results_dir=tmp_path, max_steps=200,
    )
    # belief diagnostics at every severity
    for s in r["sweep"]["severities"]:
        assert s["belief_diag"] is not None
        assert "diag_layer2_precision_mean" in s["belief_diag"]
        assert "diag_layer2_precision_cv" in s["belief_diag"]
    assert [s["severity"] for s in r["sweep"]["severities"]] == list(GAUSSIAN_S_TEST)
    # digits is not one of the two intervention datasets
    assert r["interventions"] == []
    assert set(r["diagnostics_init"]) == {"clean", "max_train"}


def test_non_cellv03_has_no_belief_diagnostics(tmp_path):
    r = run_one(
        "digits", MISSING, "plain_mlp", 0,
        device=torch.device("cpu"), results_dir=tmp_path, max_steps=150,
    )
    assert all(s["belief_diag"] is None for s in r["sweep"]["severities"])
    assert r["diagnostics_init"] == {}


def test_regression_run_uses_r2_primary(tmp_path):
    r = run_one(
        "california_housing", MISSING, "plain_mlp", 0,
        device=torch.device("cpu"), results_dir=tmp_path, max_steps=120,
    )
    assert r["primary_metric"] == "r2"
    assert r["sweep"]["severities"][0]["metrics"].keys() == {"r2", "rmse"}
