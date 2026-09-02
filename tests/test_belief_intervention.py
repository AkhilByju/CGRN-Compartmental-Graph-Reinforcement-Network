import pytest
import torch

from src.evaluation.intervention import PERTURBATION_MODES, perturb_belief
from src.models.architecture_v0.cell import BeliefCell


def _sample_belief(batch: int = 5, cells: int = 6) -> BeliefCell:
    return BeliefCell(
        mu=torch.randn(batch, cells),
        evidence=torch.rand(batch, cells) * 5 + 0.1,
        uncertainty=torch.rand(batch, cells) * 3 + 0.1,
    )


def test_baseline_is_identity() -> None:
    belief = _sample_belief()
    out = perturb_belief(belief, "baseline", seed=0)
    assert torch.equal(out.mu, belief.mu)
    assert torch.equal(out.evidence, belief.evidence)
    assert torch.equal(out.uncertainty, belief.uncertainty)


def test_evidence_ones_sets_evidence_and_preserves_rest() -> None:
    belief = _sample_belief()
    out = perturb_belief(belief, "evidence_ones", seed=0)
    assert torch.equal(out.mu, belief.mu)
    assert torch.equal(out.evidence, torch.ones_like(belief.evidence))
    assert torch.equal(out.uncertainty, belief.uncertainty)


def test_uncertainty_ones_sets_uncertainty_and_preserves_rest() -> None:
    belief = _sample_belief()
    out = perturb_belief(belief, "uncertainty_ones", seed=0)
    assert torch.equal(out.mu, belief.mu)
    assert torch.equal(out.evidence, belief.evidence)
    assert torch.equal(out.uncertainty, torch.ones_like(belief.uncertainty))


def test_shuffle_preserves_mu_and_per_example_multiset() -> None:
    belief = _sample_belief()
    out = perturb_belief(belief, "shuffle_evidence_uncertainty", seed=0)
    assert torch.equal(out.mu, belief.mu)
    # Shuffling permutes which cell holds which (evidence, uncertainty) pair,
    # so each example's set of values is unchanged, only their assignment to
    # cells is.
    for i in range(belief.shape[0]):
        assert torch.equal(out.evidence[i].sort().values, belief.evidence[i].sort().values)
        assert torch.equal(out.uncertainty[i].sort().values, belief.uncertainty[i].sort().values)


def test_shuffle_is_reproducible_given_same_seed() -> None:
    belief = _sample_belief()
    out_a = perturb_belief(belief, "shuffle_evidence_uncertainty", seed=42)
    out_b = perturb_belief(belief, "shuffle_evidence_uncertainty", seed=42)
    assert torch.equal(out_a.evidence, out_b.evidence)
    assert torch.equal(out_a.uncertainty, out_b.uncertainty)


def test_shuffle_with_many_cells_actually_changes_assignment() -> None:
    # With enough cells, an identity permutation on every one of many seeds
    # would be a suspicious coincidence -- a loose sanity check that
    # shuffling isn't silently a no-op.
    belief = _sample_belief(batch=1, cells=32)
    out = perturb_belief(belief, "shuffle_evidence_uncertainty", seed=0)
    assert not torch.equal(out.evidence, belief.evidence)


def test_uncertainty_random_stays_within_observed_range_and_nonnegative() -> None:
    belief = _sample_belief()
    out = perturb_belief(belief, "uncertainty_random", seed=0)
    assert torch.equal(out.mu, belief.mu)
    assert torch.equal(out.evidence, belief.evidence)
    assert (out.uncertainty >= 0).all()
    assert out.uncertainty.min() >= belief.uncertainty.min() - 1e-6
    assert out.uncertainty.max() <= belief.uncertainty.max() + 1e-6


def test_uncertainty_random_is_reproducible_given_same_seed() -> None:
    belief = _sample_belief()
    out_a = perturb_belief(belief, "uncertainty_random", seed=7)
    out_b = perturb_belief(belief, "uncertainty_random", seed=7)
    assert torch.equal(out_a.uncertainty, out_b.uncertainty)


@pytest.mark.parametrize("mode", PERTURBATION_MODES)
def test_all_modes_produce_a_valid_belief_cell(mode: str) -> None:
    # BeliefCell's own constructor validates non-negativity/shape/dtype/
    # device agreement -- a correctness check on the perturbations
    # themselves, mirroring tests/test_belief_layer.py's approach.
    belief = _sample_belief()
    out = perturb_belief(belief, mode, seed=0)  # must not raise
    assert out.shape == belief.shape


def test_unknown_mode_raises() -> None:
    belief = _sample_belief()
    with pytest.raises(ValueError):
        perturb_belief(belief, "bogus", seed=0)
