import pytest
import torch

from src.evaluation.calibration import (
    calibration_mse,
    pearson_correlation,
    spearman_correlation,
)


def test_pearson_perfect_positive_correlation() -> None:
    t = torch.arange(20, dtype=torch.float32)
    p = 2.0 * t + 1.0
    assert pearson_correlation(p, t) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation() -> None:
    t = torch.arange(20, dtype=torch.float32)
    p = -t
    assert pearson_correlation(p, t) == pytest.approx(-1.0)


def test_pearson_nan_when_predicted_is_constant() -> None:
    t = torch.arange(20, dtype=torch.float32)
    p = torch.ones(20)
    result = pearson_correlation(p, t)
    assert result != result  # NaN != NaN


def test_spearman_perfect_for_monotonic_nonlinear_transform() -> None:
    t = torch.arange(1, 21, dtype=torch.float32)
    p = t**3  # monotonic but nonlinear -- Pearson would be < 1, Spearman should be 1
    assert spearman_correlation(p, t) == pytest.approx(1.0)
    assert pearson_correlation(p, t) < 0.99


def test_calibration_mse_zero_for_identical_tensors() -> None:
    t = torch.rand(20)
    assert calibration_mse(t, t) == pytest.approx(0.0, abs=1e-6)


def test_calibration_mse_matches_manual_computation() -> None:
    p = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([1.5, 2.0, 2.5])
    expected = ((0.5**2) + 0.0 + (0.5**2)) / 3
    assert calibration_mse(p, t) == pytest.approx(expected)
