import math

import pytest
import torch

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.integration import AGGREGATION_METHODS, BeliefLayer


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_forward_shape(aggregation: str) -> None:
    batch, in_cells, out_cells = 4, 6, 3
    layer = BeliefLayer(in_cells, out_cells, aggregation=aggregation)
    incoming = BeliefCell(
        mu=torch.randn(batch, in_cells),
        evidence=torch.rand(batch, in_cells) * 5,
        uncertainty=torch.rand(batch, in_cells) + 0.1,
    )

    out = layer(incoming)

    assert out.shape == (batch, out_cells)


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_forward_produces_valid_finite_belief_cell(aggregation: str) -> None:
    # BeliefCell's own validation (non-negative evidence/uncertainty) is
    # itself a correctness check on the aggregation formulas: construction
    # would raise if a formula ever produced a negative value.
    batch, in_cells, out_cells = 8, 10, 4
    layer = BeliefLayer(in_cells, out_cells, aggregation=aggregation)
    incoming = BeliefCell(
        mu=torch.randn(batch, in_cells),
        evidence=torch.rand(batch, in_cells) * 5,
        uncertainty=torch.rand(batch, in_cells) + 0.1,
    )

    out = layer(incoming)  # must not raise

    assert torch.isfinite(out.mu).all()
    assert torch.isfinite(out.evidence).all()
    assert torch.isfinite(out.uncertainty).all()


@pytest.mark.parametrize("aggregation", AGGREGATION_METHODS)
def test_gradients_flow_from_mu_through_evidence_and_uncertainty(aggregation: str) -> None:
    # The design's "critical gradient property": mu_i must depend on
    # (mu_j, evidence_j, uncertainty_j), so backprop through the final mu
    # must reach all three incoming tensors, not just mu_j.
    layer = BeliefLayer(in_cells=5, out_cells=1, aggregation=aggregation)
    mu = torch.randn(2, 5, requires_grad=True)
    evidence = (torch.rand(2, 5) * 5 + 0.5).requires_grad_()
    uncertainty = (torch.rand(2, 5) + 0.1).requires_grad_()

    out = layer(BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty))
    out.mu.sum().backward()

    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert evidence.grad is not None and torch.isfinite(evidence.grad).all()
    assert (evidence.grad != 0).any()
    assert uncertainty.grad is not None and torch.isfinite(uncertainty.grad).all()
    assert (uncertainty.grad != 0).any()


def test_reliability_consensus_single_input_cell_matches_hand_computation() -> None:
    """With exactly one incoming cell, Method A simplifies enough to check
    by hand: alpha collapses to ~1 (float32 rounds 1+1e-8 to 1.0), so
    c == m, evidence == r exactly, and disagreement collapses to ~0."""
    layer = BeliefLayer(in_cells=1, out_cells=1, aggregation="reliability")
    with torch.no_grad():
        layer.content_weight.fill_(1.0)
        layer.relevance_logit.fill_(0.0)  # g = sigmoid(0) = 0.5
        layer.bias.fill_(0.0)

    incoming = BeliefCell(
        mu=torch.tensor([[1.0]]),
        evidence=torch.tensor([[4.0]]),
        uncertainty=torch.tensor([[1.0]]),
    )

    out = layer(incoming)

    # r = g * e / (1 + u) = 0.5 * 4 / 2 = 1.0
    assert out.evidence.item() == pytest.approx(1.0, abs=1e-5)
    # c = m = w * mu = 1.0 -> mu = tanh(1.0)
    assert out.mu.item() == pytest.approx(math.tanh(1.0), abs=1e-5)
    # disagreement ~ 0 with a single input -> uncertainty = sqrt(1 / evidence)
    assert out.uncertainty.item() == pytest.approx(1.0, abs=1e-4)


def test_unknown_aggregation_raises() -> None:
    with pytest.raises(ValueError):
        BeliefLayer(in_cells=2, out_cells=2, aggregation="bogus")


def test_wrong_input_cell_count_raises() -> None:
    layer = BeliefLayer(in_cells=3, out_cells=2)
    incoming = BeliefCell(
        mu=torch.zeros(1, 4), evidence=torch.zeros(1, 4), uncertainty=torch.ones(1, 4)
    )
    with pytest.raises(ValueError):
        layer(incoming)
