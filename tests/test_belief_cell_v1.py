import pytest
import torch

from src.models.architecture_v1.cell import BeliefCellV1


def _cell(batch=2, n=5, d=3) -> BeliefCellV1:
    return BeliefCellV1(
        mu=torch.randn(batch, n),
        evidence=torch.rand(batch, n),
        uncertainty=torch.rand(batch, n) + 0.1,
        z=torch.randn(batch, n, d),
    )


def test_valid_construction_exposes_shape_properties() -> None:
    cell = _cell(batch=2, n=5, d=3)
    assert cell.shape == (2, 5)
    assert cell.n_cells == 5
    assert cell.association_dim == 3


def test_z_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        BeliefCellV1(
            mu=torch.randn(2, 5),
            evidence=torch.rand(2, 5),
            uncertainty=torch.rand(2, 5),
            z=torch.randn(2, 4, 3),  # wrong n_cells axis
        )


def test_mismatched_scalar_shapes_raise() -> None:
    with pytest.raises(ValueError):
        BeliefCellV1(
            mu=torch.randn(2, 5),
            evidence=torch.rand(2, 4),
            uncertainty=torch.rand(2, 5),
            z=torch.randn(2, 5, 3),
        )


def test_negative_evidence_raises() -> None:
    with pytest.raises(ValueError):
        BeliefCellV1(
            mu=torch.randn(2, 5),
            evidence=-torch.rand(2, 5),
            uncertainty=torch.rand(2, 5),
            z=torch.randn(2, 5, 3),
        )


def test_negative_uncertainty_raises() -> None:
    with pytest.raises(ValueError):
        BeliefCellV1(
            mu=torch.randn(2, 5),
            evidence=torch.rand(2, 5),
            uncertainty=-torch.rand(2, 5),
            z=torch.randn(2, 5, 3),
        )


def test_to_moves_all_four_fields() -> None:
    cell = _cell().to(dtype=torch.float64)
    assert cell.mu.dtype == torch.float64
    assert cell.evidence.dtype == torch.float64
    assert cell.uncertainty.dtype == torch.float64
    assert cell.z.dtype == torch.float64


def test_detach_removes_grad() -> None:
    mu = torch.randn(2, 5, requires_grad=True)
    cell = BeliefCellV1(
        mu=mu,
        evidence=torch.rand(2, 5),
        uncertainty=torch.rand(2, 5) + 0.1,
        z=torch.randn(2, 5, 3),
    )
    detached = cell.detach()
    assert not detached.mu.requires_grad
