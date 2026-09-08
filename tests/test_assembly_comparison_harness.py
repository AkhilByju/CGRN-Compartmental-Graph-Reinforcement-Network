"""The two harness additions CellV1.6's first validation experiment
(`experiments/v1_001_dynamic_groups/run_assembly_comparison.py`) needs:

1. `_train_until_convergence`'s opt-in `capture_final_state` must not
   change the training trajectory -- it only additionally returns the
   end-of-training parameters (before the best-checkpoint restore).
2. `_build_assembly_comparison_models` must give `cellv0.1` and
   `cellv1_6` the SAME backbone width, with `cellv1_6` exactly 34
   parameters larger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HARNESS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "v1_001_dynamic_groups"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

harness = pytest.importorskip("harness")


def _toy_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 5, generator=g)
    y = (x[:, :1] * x[:, 1:2]).detach()
    return x[:150], y[:150], x[150:], y[150:]


def test_capture_final_state_does_not_change_training() -> None:
    x_tr, y_tr, x_val, y_val = _toy_problem()
    common = dict(
        loss_fn=torch.nn.MSELoss(),
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_val,
        y_val=y_val,
        device=torch.device("cpu"),
        seed=0,
        regression=True,
        batch_size=32,
        lr=1e-2,
        val_every=25,
        patience_steps=50,
        max_steps=200,
    )

    torch.manual_seed(0)
    model_a = torch.nn.Linear(5, 1)
    torch.manual_seed(0)
    model_b = torch.nn.Linear(5, 1)

    out_a = harness._train_until_convergence(model_a, **common, capture_final_state=False)
    out_b = harness._train_until_convergence(model_b, **common, capture_final_state=True)

    for key in ("steps_to_convergence", "total_steps_run", "best_val_metric"):
        assert out_a[key] == pytest.approx(out_b[key]), key
    assert "final_state" not in out_a
    assert set(out_b["final_state"]) == set(model_b.state_dict())
    # best-checkpoint state (restored into both models) must match
    for k, v in model_a.state_dict().items():
        assert torch.allclose(v, model_b.state_dict()[k])


def test_assembly_comparison_models_are_matched_plus_34() -> None:
    for level, in_features in [("hard", 288), ("very_hard", 576)]:
        models, sizing = harness._build_assembly_comparison_models(level, in_features, 1, seed=0)
        assert set(models) == {"cellv0.1", "cellv1_6"}
        assert sizing["cellv1_6__param_increase"] == 34
        assert sizing["cellv0.1__params"] == sizing["reference_param_budget"] or abs(
            sizing["cellv0.1__params"] - sizing["reference_param_budget"]
        ) < 300  # closest achievable belief-network width to the budget
        # same backbone width in both arms
        assert models["cellv0.1"].layer1.out_cells == sizing["hidden_cells"]
        assert models["cellv1_6"].layer1.out_cells == sizing["hidden_cells"]
