"""Paper A Phase 3 Part B -- end-to-end run harness + HGB reference. Fast:
tiny LR grid, hard step cap, CPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from experiments.paper_a.real_reliability.datasets import dataset_available
from experiments.paper_a.real_reliability.harness import (
    EXPERIMENT_ID,
    HGB_EXPERIMENT_ID,
    run_hgb_reference,
    run_one,
)
from experiments.paper_a.real_reliability.models import MODEL_FAMILIES

pytestmark = pytest.mark.skipif(
    not (dataset_available("aps") and dataset_available("air_quality")),
    reason="UCI APS / Air Quality archives not reachable",
)


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_run_one_air_quality_writes_a_complete_record(tmp_path, family):
    r = run_one(
        "air_quality", family, 0,
        device=torch.device("cpu"), results_dir=tmp_path,
        learning_rates=(1e-3, 1e-2), max_steps=120,
    )
    assert r["family"] == family
    assert r["primary_metric"] == "r2"
    assert r["parameter_count"] > 0
    assert r["selected_lr"] in (1e-3, 1e-2)
    if family == "neumiss":
        assert r["neumiss_depth"] in (1, 3, 5)
        assert r["evaluation"]["belief_by_stratum"] == []
    if family == "cellv0.3":
        assert r["intervention"] is not None
        assert r["evaluation"]["belief_by_stratum"]
    else:
        assert r["intervention"] is None

    rec = json.loads(
        next(p for p in tmp_path.glob("*.json") if not p.name.endswith("_history.json")).read_text()
    )
    assert rec["experiment_id"] == EXPERIMENT_ID
    assert rec["config"]["extra"]["selected_lr"] == r["selected_lr"]


def test_run_one_aps_uses_pr_auc_and_freezes_a_cost_threshold(tmp_path):
    r = run_one(
        "aps", "plain_mlp", 0,
        device=torch.device("cpu"), results_dir=tmp_path,
        learning_rates=(3e-3,), max_steps=150,
    )
    assert r["primary_metric"] == "pr_auc"
    assert r["efficiency"]["cost_threshold"] is not None
    ov = r["evaluation"]["overall"]
    assert {"pr_auc", "roc_auc", "official_cost", "cost_per_1000", "recall"} <= set(ov)
    assert sum(s["n"] for s in r["evaluation"]["by_stratum"]) == 16_000


def test_hgb_reference_runs_and_is_flagged_reference_only(tmp_path):
    for name, key in (("air_quality", "r2"), ("aps", "pr_auc")):
        h = run_hgb_reference(name, 0, results_dir=tmp_path)
        assert h["reference_only"] is True
        assert key in h["evaluation"]["overall"]
    recs = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert any(rec["experiment_id"] == HGB_EXPERIMENT_ID for rec in recs)
    assert all(
        rec["config"]["extra"].get("reference_only")
        for rec in recs if rec["experiment_id"] == HGB_EXPERIMENT_ID
    )
