import numpy as np
import pytest
import torch

from src.evaluation.classification import accuracy, f1
from src.evaluation.efficiency import count_parameters, wall_clock
from src.evaluation.regression import mae, r_squared, rmse


def test_regression_metrics_on_perfect_predictions() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0])
    assert mae(targets, targets) == 0.0
    assert rmse(targets, targets) == 0.0
    assert r_squared(targets, targets) == 1.0


def test_regression_metrics_on_known_error() -> None:
    predictions = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 3.0])
    assert mae(predictions, targets) == 2.0
    # float32 tensor round-trip vs. numpy float64 -- compare approximately.
    assert rmse(predictions, targets) == pytest.approx(np.sqrt(5.0), rel=1e-6)


def test_classification_accuracy_with_logits() -> None:
    logits = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([0, 1])
    assert accuracy(logits, targets) == 1.0


def test_classification_f1_binary() -> None:
    predictions = torch.tensor([1, 0, 1, 1])
    targets = torch.tensor([1, 0, 0, 1])
    assert 0.0 <= f1(predictions, targets) <= 1.0


def test_count_parameters_counts_trainable_only_by_default() -> None:
    model = torch.nn.Linear(4, 2)
    total = count_parameters(model)
    assert total == 4 * 2 + 2


def test_wall_clock_measures_nonnegative_time() -> None:
    with wall_clock() as timer:
        pass
    assert timer.elapsed_seconds >= 0.0
