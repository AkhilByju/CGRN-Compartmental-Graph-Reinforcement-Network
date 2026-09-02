"""Experiment 004E -- the duplication test, as an automated regression check.

Duplicating the exact same belief content N times carries no new
information, so a principled aggregation rule should leave `evidence`/
`uncertainty` approximately unchanged as N grows (`mu` -- the content
estimate -- should also stay put, since N identical opinions still agree on
the same content). `"precision"` (Method C) is known to fail this
(Experiment 004D/004E, docs/research_log.md): `evidence` is an unnormalized
sum over incoming cells, so it grows with N even when every incoming cell
says the same thing. `"normalized_precision"` (Method D, Experiment 004F)
and `"scale_stable_precision"` (Method E, Experiment 004I) were both added
to fix this -- for N *equal-relevance* duplicates specifically, both
methods' `evidence`/`uncertainty` stop changing with N (Method E's
effective-source-count `N_eff` reduces to exactly `N`, same as Method D's
raw-sum `G = N*g0` cancels `N` against the numerator), but they settle at
different absolute levels unless the shared relevance `g0` happens to be 1
(see `tests/test_scale_stable_precision.py` for where the two methods
actually diverge in *behavior*, not just a constant: unequal relevance
across incoming cells). These tests codify all of this so a future change
to any of the three formulas can't silently reintroduce/fix this without
notice.
"""

from __future__ import annotations

import pytest
import torch

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.integration import BeliefLayer

MU0, EVIDENCE0, UNCERTAINTY0 = 0.5, 1.0, 1.0
CONTENT_WEIGHT0, RELEVANCE_LOGIT0, BIAS0 = 1.0, 0.0, 0.0


def _duplicated_belief(n: int) -> BeliefCell:
    """`n` identical copies of one belief -- same content, same evidence,
    same uncertainty -- as a single-example batch."""
    return BeliefCell(
        mu=torch.full((1, n), MU0),
        evidence=torch.full((1, n), EVIDENCE0),
        uncertainty=torch.full((1, n), UNCERTAINTY0),
    )


def _equivalent_layer(n: int, aggregation: str) -> BeliefLayer:
    """A `BeliefLayer(in_cells=n, out_cells=1)` where every one of the `n`
    incoming connections has the identical weight -- "equivalent
    connections" for however many copies are fed in, isolating the effect
    of `n` alone from any weight variation."""
    layer = BeliefLayer(in_cells=n, out_cells=1, aggregation=aggregation)
    with torch.no_grad():
        layer.content_weight.fill_(CONTENT_WEIGHT0)
        layer.relevance_logit.fill_(RELEVANCE_LOGIT0)
        layer.bias.fill_(BIAS0)
    return layer


def _run(n: int, aggregation: str) -> BeliefCell:
    layer = _equivalent_layer(n, aggregation)
    with torch.no_grad():
        return layer(_duplicated_belief(n))


def test_precision_evidence_grows_with_duplicate_count() -> None:
    # Documents the known bug (Experiment 004D/004E): evidence = sum(g*e_j)
    # is unnormalized, so duplicating the same content 16x should inflate
    # evidence by roughly 16x.
    out1 = _run(1, "precision")
    out16 = _run(16, "precision")
    ratio = (out16.evidence / out1.evidence).item()
    assert ratio == pytest.approx(16.0, rel=0.05)


def test_precision_uncertainty_shrinks_with_duplicate_count() -> None:
    # base_uncertainty_sq = 1/sum(precision) -> shrinks as 1/n, so
    # uncertainty shrinks as 1/sqrt(n).
    out1 = _run(1, "precision")
    out16 = _run(16, "precision")
    ratio = (out16.uncertainty / out1.uncertainty).item()
    assert ratio == pytest.approx(1.0 / (16**0.5), rel=0.05)


FIXED_NORMALIZER_METHODS = ["normalized_precision", "scale_stable_precision"]


@pytest.mark.parametrize("aggregation", FIXED_NORMALIZER_METHODS)
def test_evidence_is_duplicate_invariant(aggregation: str) -> None:
    out1 = _run(1, aggregation)
    out16 = _run(16, aggregation)
    assert out16.evidence.item() == pytest.approx(out1.evidence.item(), rel=0.02)


@pytest.mark.parametrize("aggregation", FIXED_NORMALIZER_METHODS)
def test_uncertainty_is_duplicate_invariant(aggregation: str) -> None:
    out1 = _run(1, aggregation)
    out16 = _run(16, aggregation)
    assert out16.uncertainty.item() == pytest.approx(out1.uncertainty.item(), rel=0.02)


@pytest.mark.parametrize("aggregation", ["precision", *FIXED_NORMALIZER_METHODS])
def test_mu_is_duplicate_invariant_for_all(aggregation: str) -> None:
    # Duplicating identical opinions shouldn't change the content estimate
    # itself for any of the three formulas -- alpha-weighted averaging
    # already handles this correctly in all of them; only evidence/
    # uncertainty differ.
    out1 = _run(1, aggregation)
    out16 = _run(16, aggregation)
    assert out16.mu.item() == pytest.approx(out1.mu.item(), abs=1e-4)


@pytest.mark.parametrize("aggregation", FIXED_NORMALIZER_METHODS)
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_evidence_stays_near_single_copy_value(n: int, aggregation: str) -> None:
    # Stronger, per-N check (not just endpoints 1 vs 16): evidence should
    # sit near the single-copy value at every duplicate count, not just
    # coincidentally match at the extremes.
    baseline = _run(1, aggregation).evidence.item()
    out = _run(n, aggregation)
    assert out.evidence.item() == pytest.approx(baseline, rel=0.02)


def test_normalized_and_scale_stable_settle_at_a_fixed_ratio_for_equal_relevance() -> None:
    # Corrected analytical claim (integration.py::_scale_stable_precision_
    # fusion's docstring -- NOT "the two methods produce identical output"):
    # for N equal-g duplicates, N_eff == N for any g0 > 0, while Method D's
    # G == N*g0. Both cancel N cleanly, so both are individually duplicate-
    # invariant, but Method D settles at evidence=e0 while Method E settles
    # at evidence=g0*e0 -- a fixed ratio of g0 between them at every N, not
    # numerically identical. They only coincide as g0 -> 1.
    g0 = torch.sigmoid(torch.tensor(RELEVANCE_LOGIT0)).item()
    for n in (1, 2, 4, 8, 16):
        d = _run(n, "normalized_precision")
        e = _run(n, "scale_stable_precision")
        assert e.evidence.item() == pytest.approx(g0 * d.evidence.item(), rel=1e-3)
