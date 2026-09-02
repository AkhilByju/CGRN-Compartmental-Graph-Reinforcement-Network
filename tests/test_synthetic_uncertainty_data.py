import pytest
import torch

from src.data.synthetic.uncertainty import U_REGRESSION_DATASETS, make_uncertainty_splits


@pytest.mark.parametrize("name", U_REGRESSION_DATASETS)
def test_splits_shapes_and_determinism(name: str) -> None:
    a = make_uncertainty_splits(name, n_train=100, n_val=20, n_test=20, seed=0)
    b = make_uncertainty_splits(name, n_train=100, n_val=20, n_test=20, seed=0)
    for split in ("train", "val", "test"):
        xa, ya, stda = a[split]
        xb, yb, stdb = b[split]
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)
        assert torch.equal(stda, stdb)

    x_train, y_train, std_train = a["train"]
    assert x_train.shape[0] == 100
    assert y_train.shape == (100, 1)
    assert std_train.shape == (100, 1)


def test_different_seeds_produce_different_data() -> None:
    x0, _, _ = make_uncertainty_splits("u1_heteroscedastic_1d", 50, 0, 0, seed=0)["train"]
    x1, _, _ = make_uncertainty_splits("u1_heteroscedastic_1d", 50, 0, 0, seed=1)["train"]
    assert not torch.equal(x0, x1)


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        make_uncertainty_splits("bogus", 10, 0, 0, seed=0)


@pytest.mark.parametrize("name", U_REGRESSION_DATASETS)
def test_true_std_is_positive(name: str) -> None:
    _, _, true_std = make_uncertainty_splits(name, 500, 0, 0, seed=0)["train"]
    assert bool((true_std > 0).all())


def test_u1_true_std_grows_with_distance_from_origin() -> None:
    x, _, true_std = make_uncertainty_splits("u1_heteroscedastic_1d", 1000, 0, 0, seed=0)["train"]
    order = torch.argsort(x.abs().squeeze(-1))
    near = true_std[order[: len(order) // 10]]
    far = true_std[order[-len(order) // 10 :]]
    assert near.mean().item() < far.mean().item()


def test_u2_true_std_depends_on_feature_one_only() -> None:
    x, _, true_std = make_uncertainty_splits(
        "u2_heteroscedastic_interaction", 1000, 0, 0, seed=0
    )["train"]
    order = torch.argsort(x[:, 0].abs())
    near = true_std[order[: len(order) // 10]]
    far = true_std[order[-len(order) // 10 :]]
    assert near.mean().item() < far.mean().item()
