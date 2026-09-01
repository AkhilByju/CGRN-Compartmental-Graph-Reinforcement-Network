import pytest
import torch

from src.models.architecture_v0.cell import BeliefCell


def test_valid_belief_cell_construction() -> None:
    mu = torch.tensor([0.1, -0.2])
    evidence = torch.tensor([1.0, 0.0])
    uncertainty = torch.tensor([0.5, 2.0])

    cell = BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty)

    assert cell.shape == mu.shape
    assert cell.dtype == mu.dtype
    assert cell.device == mu.device


def test_scalar_shape_is_allowed() -> None:
    cell = BeliefCell(
        mu=torch.tensor(0.5), evidence=torch.tensor(1.0), uncertainty=torch.tensor(0.1)
    )
    assert cell.shape == torch.Size([])


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError):
        BeliefCell(mu=torch.zeros(2), evidence=torch.zeros(3), uncertainty=torch.zeros(2))


def test_mismatched_dtypes_raise() -> None:
    with pytest.raises(ValueError):
        BeliefCell(
            mu=torch.zeros(2, dtype=torch.float32),
            evidence=torch.zeros(2, dtype=torch.float64),
            uncertainty=torch.zeros(2, dtype=torch.float32),
        )


def test_negative_evidence_raises() -> None:
    with pytest.raises(ValueError):
        BeliefCell(
            mu=torch.zeros(2), evidence=torch.tensor([-1.0, 0.0]), uncertainty=torch.zeros(2)
        )


def test_negative_uncertainty_raises() -> None:
    with pytest.raises(ValueError):
        BeliefCell(
            mu=torch.zeros(2), evidence=torch.zeros(2), uncertainty=torch.tensor([-0.1, 0.0])
        )


def test_zero_evidence_and_uncertainty_are_allowed() -> None:
    # Non-negative, not strictly positive -- zero evidence/uncertainty are
    # legitimate limiting cases (e.g. a belief with no support yet).
    BeliefCell(mu=torch.zeros(2), evidence=torch.zeros(2), uncertainty=torch.zeros(2))


def test_to_moves_all_fields_dtype() -> None:
    cell = BeliefCell(mu=torch.zeros(2), evidence=torch.zeros(2), uncertainty=torch.zeros(2))
    moved = cell.to(torch.float64)
    assert moved.dtype == torch.float64
    assert moved.mu.dtype == torch.float64
    assert moved.evidence.dtype == torch.float64
    assert moved.uncertainty.dtype == torch.float64


def test_detach_removes_grad() -> None:
    mu = torch.zeros(2, requires_grad=True)
    cell = BeliefCell(mu=mu, evidence=torch.zeros(2), uncertainty=torch.zeros(2))
    detached = cell.detach()
    assert not detached.mu.requires_grad


def test_batched_layer_shape() -> None:
    batch, num_cells = 4, 8
    cell = BeliefCell(
        mu=torch.randn(batch, num_cells),
        evidence=torch.rand(batch, num_cells),
        uncertainty=torch.rand(batch, num_cells),
    )
    assert cell.shape == (batch, num_cells)


def test_from_observed_features_sets_unit_evidence_and_uncertainty() -> None:
    x = torch.tensor([1.0, -2.0, 0.5])
    cell = BeliefCell.from_observed_features(x)
    assert torch.equal(cell.mu, x)
    assert torch.equal(cell.evidence, torch.ones_like(x))
    assert torch.equal(cell.uncertainty, torch.ones_like(x))
