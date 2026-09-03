import torch

from src.data.synthetic.dynamic_groups import FEATURES_PER_OBJECT, N_OBJECTS, dynamic_groups
from src.models.architecture_v1.object_encoder import ObjectSeededEncoder


def test_seeded_cells_carry_object_values_as_mu() -> None:
    n_cells, d = 40, 6
    encoder = ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=n_cells, association_dim=d)
    x, _, _ = dynamic_groups(4, seed=0)

    cells = encoder(x)

    objects = x.view(4, N_OBJECTS, FEATURES_PER_OBJECT)
    values = objects[..., 2]
    assert torch.allclose(cells.mu[:, :N_OBJECTS], values, atol=1e-6)


def test_filler_cells_start_neutral_with_low_evidence_high_uncertainty() -> None:
    n_cells, d = 40, 6
    encoder = ObjectSeededEncoder(
        n_objects=N_OBJECTS, n_cells=n_cells, association_dim=d, filler_evidence=0.1, filler_uncertainty=5.0
    )
    x, _, _ = dynamic_groups(4, seed=0)

    cells = encoder(x)

    filler_mu = cells.mu[:, N_OBJECTS:]
    filler_e = cells.evidence[:, N_OBJECTS:]
    filler_u = cells.uncertainty[:, N_OBJECTS:]
    assert torch.allclose(filler_mu, torch.zeros_like(filler_mu))
    assert torch.allclose(filler_e, torch.full_like(filler_e, 0.1))
    assert torch.allclose(filler_u, torch.full_like(filler_u, 5.0))
    # objects get full evidence/unit uncertainty, unlike filler cells
    assert torch.allclose(cells.evidence[:, :N_OBJECTS], torch.ones(4, N_OBJECTS))
    assert torch.allclose(cells.uncertainty[:, :N_OBJECTS], torch.ones(4, N_OBJECTS))


def test_z_is_unit_norm_for_both_seeded_and_filler_cells() -> None:
    n_cells, d = 40, 6
    encoder = ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=n_cells, association_dim=d)
    x, _, _ = dynamic_groups(4, seed=0)

    cells = encoder(x)

    norms = cells.z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_output_shapes() -> None:
    n_cells, d = 50, 8
    encoder = ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=n_cells, association_dim=d)
    x, _, _ = dynamic_groups(3, seed=0)

    cells = encoder(x)

    assert cells.mu.shape == (3, n_cells)
    assert cells.z.shape == (3, n_cells, d)


def test_rejects_n_cells_smaller_than_n_objects() -> None:
    import pytest

    with pytest.raises(ValueError):
        ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=N_OBJECTS - 1, association_dim=4)


def test_gradients_are_finite() -> None:
    n_cells, d = 32, 6
    encoder = ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=n_cells, association_dim=d)
    x, _, _ = dynamic_groups(4, seed=0)
    x.requires_grad_(True)

    cells = encoder(x)
    (cells.mu.sum() + cells.z.sum()).backward()

    assert torch.isfinite(x.grad).all()
    for p in encoder.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
