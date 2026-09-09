"""Paper A Phase 1 -- harness / training-protocol smoke tests. Fast: a
bundled tiny dataset, a hard step cap, CPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from experiments.paper_a.datasets import prepare_dataset
from experiments.paper_a.harness import run_one
from experiments.paper_a.models import build_cellv01
from experiments.paper_a.training import train_model


@pytest.mark.parametrize("family", ["cellv0.1", "mlp_matched", "mlp_state_count"])
def test_run_one_writes_a_complete_record(tmp_path, family):
    r = run_one(
        "wine",
        0.25,
        family,
        0,
        device=torch.device("cpu"),
        results_dir=tmp_path,
        max_steps=300,
    )
    assert r["family"] == family
    assert r["headline_metric"] == "accuracy"
    assert 0.0 <= r["headline_value"] <= 1.0
    assert r["parameter_count"] > 0

    files = list(tmp_path.glob("*.json"))
    record_files = [f for f in files if not f.name.endswith("_history.json")]
    history_files = [f for f in files if f.name.endswith("_history.json")]
    assert len(record_files) == 1 and len(history_files) == 1

    rec = json.loads(record_files[0].read_text())
    for key in ("git_commit", "parameter_count", "seed", "dataset", "architecture"):
        assert key in rec
    assert rec["config"]["extra"]["train_fraction"] == 0.25
    assert rec["config"]["extra"]["device"] == "cpu"
    assert rec["validation_metrics"]["best_val_step"] >= 0
    history = json.loads(history_files[0].read_text())
    assert len(history) >= 1  # validation curve retained


def test_best_validation_checkpoint_is_restored():
    prepared = prepare_dataset("wine", seed=0, train_fraction=1.0)
    model = build_cellv01(prepared.n_features, prepared.out_features, prepared.param_budget).model
    outcome = train_model(
        model,
        prepared,
        device=torch.device("cpu"),
        seed=0,
        max_steps=600,
        val_every=25,
        patience_checks=4,
    )
    # the reported best-val loss must be the minimum over the whole curve
    val_losses = [h["val_loss"] for h in outcome.history if "val_loss" in h]
    assert outcome.best_val_loss == pytest.approx(min(val_losses), abs=1e-6)
    assert not outcome.diverged

    # re-evaluating the (restored) model reproduces the recorded best metric
    model.eval()
    with torch.no_grad():
        pred = model(prepared.x_val)
    acc = (pred.argmax(-1) == prepared.y_val).float().mean().item()
    assert acc == pytest.approx(outcome.best_val_metric, abs=1e-6)


def test_shared_protocol_matches_the_repo_cellv01_convention():
    from experiments.paper_a.training import DEFAULT_LR, DEFAULT_WEIGHT_DECAY

    assert DEFAULT_LR == 1e-2  # repo's CellV0.1 protocol (Paper-A task Sec 7)
    assert DEFAULT_WEIGHT_DECAY == 0.0


def test_regression_run_reports_rmse_on_original_scale(tmp_path):
    r = run_one(
        "diabetes",
        1.0,
        "mlp_matched",
        0,
        device=torch.device("cpu"),
        results_dir=tmp_path,
        max_steps=400,
    )
    assert r["headline_metric"] == "r2"
    # diabetes targets are ~25-350; a sane RMSE is in the tens, not ~1
    # (which is what you'd get if RMSE were computed on standardized targets).
    assert 20.0 < r["test_rmse"] < 120.0
