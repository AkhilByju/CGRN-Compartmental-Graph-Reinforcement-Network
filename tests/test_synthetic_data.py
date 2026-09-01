import pytest
import torch

from src.data.synthetic.classification import (
    CLASSIFICATION_DATASETS,
    MARGIN_FUNCTIONS,
    make_classification_splits,
)
from src.data.synthetic.regression import (
    REGRESSION_DATASETS,
    make_regression_splits,
    regression_ood_inputs,
)
from src.data.synthetic.utils import standardize


@pytest.mark.parametrize("name", REGRESSION_DATASETS)
def test_regression_splits_shapes_and_determinism(name: str) -> None:
    a = make_regression_splits(name, n_train=100, n_val=20, n_test=20, seed=0)
    b = make_regression_splits(name, n_train=100, n_val=20, n_test=20, seed=0)
    for split in ("train", "val", "test"):
        xa, ya = a[split]
        xb, yb = b[split]
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)

    x_train, y_train = a["train"]
    assert x_train.shape[0] == 100
    assert y_train.shape == (100, 1)


def test_regression_different_seeds_produce_different_data() -> None:
    x0, _ = make_regression_splits("r0_linear", 50, 0, 0, seed=0)["train"]
    x1, _ = make_regression_splits("r0_linear", 50, 0, 0, seed=1)["train"]
    assert not torch.equal(x0, x1)


def test_unknown_regression_dataset_raises() -> None:
    with pytest.raises(ValueError):
        make_regression_splits("bogus", 10, 0, 0, seed=0)


def test_regression_ood_inputs_are_outside_training_domain() -> None:
    x = regression_ood_inputs("r0_linear", n_samples=200, seed=0)
    assert (x.abs() > 2.0).all()  # r0_linear's training domain is [-2, 2]


@pytest.mark.parametrize("name", CLASSIFICATION_DATASETS)
def test_classification_splits_shapes_and_label_balance(name: str) -> None:
    splits = make_classification_splits(name, n_train=1000, n_val=100, n_test=100, seed=0)
    x_train, y_train = splits["train"]
    assert x_train.shape[0] == 1000
    assert y_train.shape == (1000,)
    assert set(y_train.unique().tolist()) <= {0, 1}
    positive_fraction = y_train.float().mean().item()
    assert 0.2 < positive_fraction < 0.8


def test_unknown_classification_dataset_raises() -> None:
    with pytest.raises(ValueError):
        make_classification_splits("bogus", 10, 0, 0, seed=0)


@pytest.mark.parametrize("name", CLASSIFICATION_DATASETS)
def test_margin_function_is_lower_near_boundary_than_far(name: str) -> None:
    x, _ = make_classification_splits(name, 500, 0, 0, seed=0)["train"]
    margin = MARGIN_FUNCTIONS[name](x).abs()
    order = torch.argsort(margin)
    near = margin[order[: len(order) // 10]]
    far = margin[order[-len(order) // 10 :]]
    assert near.mean().item() < far.mean().item()


def test_standardize_zero_mean_unit_std_on_train() -> None:
    torch.manual_seed(0)
    x = torch.randn(1000, 3) * 5 + 10
    (x_std,) = standardize(x)
    assert x_std.mean(dim=0).abs().max().item() < 1e-4
    assert (x_std.std(dim=0) - 1.0).abs().max().item() < 1e-2


def test_standardize_applies_train_stats_to_other_splits() -> None:
    train = torch.tensor([[0.0], [2.0], [4.0]])
    other = torch.tensor([[2.0]])
    _, other_std = standardize(train, other)
    assert other_std.item() == pytest.approx(0.0, abs=1e-6)
