import pytest
import torch

from src.evaluation.efficiency import count_parameters
from src.models.architecture_v0.belief_network import BeliefNetwork
from src.models.architecture_v0.integration import AGGREGATION_METHODS


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_forward_shape(aggregation: str) -> None:
    net = BeliefNetwork(in_features=4, hidden_cells=8, out_features=1, aggregation=aggregation)
    x = torch.randn(16, 4)
    out = net(x)
    assert out.shape == (16, 1)


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_forward_with_beliefs_returns_final_layer_belief(aggregation: str) -> None:
    net = BeliefNetwork(in_features=3, hidden_cells=5, out_features=2, aggregation=aggregation)
    x = torch.randn(4, 3)
    pred, belief = net.forward_with_beliefs(x)
    assert pred.shape == (4, 2)
    assert belief.shape == (4, 5)
    assert torch.isfinite(belief.evidence).all()
    assert torch.isfinite(belief.uncertainty).all()


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_gradients_flow_to_all_parameters(aggregation: str) -> None:
    net = BeliefNetwork(in_features=3, hidden_cells=4, out_features=1, aggregation=aggregation)
    x = torch.randn(8, 3)
    out = net(x)
    out.sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_param_count_matches_manual_formula() -> None:
    net = BeliefNetwork(in_features=4, hidden_cells=8, out_features=1)
    expected = (
        8 * (2 * 4 + 1)  # layer1: BeliefLayer(4 -> 8)
        + 8 * (2 * 8 + 1)  # layer2: BeliefLayer(8 -> 8)
        + (8 * 1 + 1)  # readout: Linear(8 -> 1)
    )
    assert count_parameters(net) == expected
