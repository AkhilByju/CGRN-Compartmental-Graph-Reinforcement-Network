import pytest
import torch

from src.evaluation.efficiency import count_parameters
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim, param_count_for_mlp


def test_forward_shape() -> None:
    model = MLPBaseline(in_features=4, hidden_dim=16, out_features=1, num_hidden_layers=2)
    x = torch.randn(10, 4)
    out = model(x)
    assert out.shape == (10, 1)


def test_param_count_matches_actual_module() -> None:
    model = MLPBaseline(in_features=4, hidden_dim=16, out_features=1, num_hidden_layers=2)
    assert count_parameters(model) == param_count_for_mlp(4, 16, 1, 2)


def test_match_hidden_dim_finds_exact_match_when_available() -> None:
    target = param_count_for_mlp(in_features=4, hidden_dim=20, out_features=1, num_hidden_layers=2)
    found_dim = match_hidden_dim(target, in_features=4, out_features=1, num_hidden_layers=2)
    assert found_dim == 20


def test_invalid_num_hidden_layers_raises() -> None:
    with pytest.raises(ValueError):
        MLPBaseline(in_features=2, hidden_dim=4, out_features=1, num_hidden_layers=0)
