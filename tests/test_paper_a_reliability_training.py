"""Paper A Phase 2 -- training-loop protocol checks: corrupted-validation
checkpoint selection, per-epoch corruption determinism, best-checkpoint
restore (Phase-2 task Sec 9). Fast: bundled digits, hard step cap, CPU.
"""

from __future__ import annotations

import pytest
import torch

from experiments.paper_a.datasets import prepare_dataset
from experiments.paper_a.reliability.corruption import GAUSSIAN, MISSING, in_dist_severities
from experiments.paper_a.reliability.models import CELLV03, PLAIN_MLP, build_model
from experiments.paper_a.reliability.training import train_model


@pytest.mark.parametrize("family", [PLAIN_MLP, CELLV03])
@pytest.mark.parametrize("corruption", [MISSING, GAUSSIAN])
def test_train_runs_and_selects_a_finite_checkpoint(family, corruption):
    prep = prepare_dataset("digits", seed=0, train_fraction=1.0)
    model = build_model(family, prep.n_features, prep.out_features, prep.param_budget).model
    out = train_model(
        model, prep, corruption_family=corruption,
        device=torch.device("cpu"), seed=0, max_steps=300, val_every=50,
    )
    assert not out.diverged
    assert 0.0 <= out.best_val_metric <= 1.0
    assert out.best_val_step >= 0
    assert out.in_dist_severities == in_dist_severities(corruption)
    assert out.history and all("val_metric" in h for h in out.history if h.get("val_loss"))


def test_training_is_reproducible_from_seed():
    prep = prepare_dataset("digits", seed=1, train_fraction=1.0)
    outs = []
    for _ in range(2):
        torch.manual_seed(1)  # the harness set_seed()s before build_model, likewise
        model = build_model(PLAIN_MLP, prep.n_features, prep.out_features, prep.param_budget).model
        outs.append(
            train_model(
                model, prep, corruption_family=MISSING,
                device=torch.device("cpu"), seed=1, max_steps=250, val_every=50,
            )
        )
    assert outs[0].best_val_metric == pytest.approx(outs[1].best_val_metric, abs=1e-9)
    assert outs[0].best_val_step == outs[1].best_val_step
    assert [h["val_metric"] for h in outs[0].history if "val_metric" in h] == pytest.approx(
        [h["val_metric"] for h in outs[1].history if "val_metric" in h], abs=1e-9
    )


def test_best_checkpoint_metric_is_the_curve_maximum():
    prep = prepare_dataset("digits", seed=0, train_fraction=1.0)
    model = build_model(PLAIN_MLP, prep.n_features, prep.out_features, prep.param_budget).model
    out = train_model(
        model, prep, corruption_family=GAUSSIAN,
        device=torch.device("cpu"), seed=0, max_steps=400, val_every=25, patience_checks=6,
    )
    curve = [h["val_metric"] for h in out.history if "val_metric" in h]
    assert out.best_val_metric == pytest.approx(max(curve), abs=1e-9)
