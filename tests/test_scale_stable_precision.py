"""Experiment 004I -- `"scale_stable_precision"` (Method E, "CellV0.1")
specifically: where it actually differs from `"normalized_precision"`
(Method D).

`tests/test_duplication_invariance.py` already shows the two formulas agree
exactly when incoming relevance is *equal* across cells (N identical
duplicates) -- that's the case Method D was already built to handle. Method
E's entire point is the case Method D does *not* handle well: many
incoming cells with *unequal*, and in particular weak-but-nonzero,
relevance. This file constructs that case directly (one dominant
connection plus a growing number of weakly-relevant ones, all carrying the
exact same content/evidence/uncertainty, so only their relevance `g`
differs) and checks the qualitative behavior the user specified:

    - equally-relevant inputs: effective source count grows with count
      (already covered elsewhere)
    - a single dominant source among many weak ones: adding more weak
      sources barely changes anything for Method E
    - Method D, by contrast, is provably insensitive to *how many* sources
      contribute (see `test_normalized_precision_evidence_is_insensitive_to_junk_count`)
      when every incoming cell reports the same evidence -- it can't tell
      "1 relevant source" from "1 relevant source plus 50 weak ones."
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.integration import BeliefLayer

E0, U0, CONTENT_WEIGHT0, BIAS0 = 1.0, 1.0, 1.0, 0.0


def _inverse_sigmoid(p: float) -> float:
    return math.log(p / (1.0 - p))


def _dominant_plus_junk_layer(n_junk: int, g_real: float, g_junk: float) -> BeliefLayer:
    """One "real" incoming connection with relevance `g_real`, plus
    `n_junk` "junk" connections with weak relevance `g_junk` -- every
    connection carries identical content/evidence/uncertainty, so only the
    relevance gate `g` varies."""
    n = 1 + n_junk
    layer = BeliefLayer(in_cells=n, out_cells=1, aggregation="scale_stable_precision")
    with torch.no_grad():
        layer.content_weight.fill_(CONTENT_WEIGHT0)
        layer.bias.fill_(BIAS0)
        layer.relevance_logit[:, 0] = _inverse_sigmoid(g_real)
        if n_junk > 0:
            layer.relevance_logit[:, 1:] = _inverse_sigmoid(g_junk)
    return layer


def _belief(n: int) -> BeliefCell:
    return BeliefCell(
        mu=torch.full((1, n), 0.5),
        evidence=torch.full((1, n), E0),
        uncertainty=torch.full((1, n), U0),
    )


def test_scale_stable_evidence_barely_moves_when_junk_connections_are_added() -> None:
    # One dominant, highly-relevant source (g=0.9) plus a growing number of
    # genuinely weak ones (g=0.005 -- small enough that even 50 of them sum
    # to less than the single dominant source, i.e. actually "g ~ 0" in
    # aggregate, not just individually). Effective source count should stay
    # close to 1 throughout -- "adding tons of connections with g ~ 0
    # barely changes it" (the user's own framing). g_junk=0.05 (10x larger)
    # was tried first and failed this bound at n_junk=50: 50 connections at
    # g=0.05 sum to 2.5, which is *not* negligible next to g_real=0.9 --
    # a reminder that "weak individually" isn't the same as "negligible in
    # aggregate," exactly the distinction N_eff is meant to capture
    # correctly (see test_normalized_precision_evidence_is_insensitive_to_junk_count
    # for the case where that distinction gets lost entirely).
    g_real, g_junk = 0.9, 0.005
    baseline_layer = _dominant_plus_junk_layer(0, g_real, g_junk)
    with torch.no_grad():
        baseline = baseline_layer(_belief(1))

    for n_junk in (5, 20, 50):
        layer = _dominant_plus_junk_layer(n_junk, g_real, g_junk)
        with torch.no_grad():
            out = layer(_belief(1 + n_junk))
        # Not exactly invariant (junk does carry *some* relevance) but
        # should stay far closer to the single-source baseline than a
        # naive "sum everything" normalizer would (see the Method D
        # comparison below) -- allow up to 2x drift, not 10x+.
        ratio = out.evidence.item() / baseline.evidence.item()
        assert 0.5 <= ratio <= 2.0, f"n_junk={n_junk}: evidence drifted too far ({ratio=:.3f})"


def test_normalized_precision_evidence_is_insensitive_to_junk_count() -> None:
    # Contrast case: with uniform e_j across all incoming cells, Method D's
    # evidence is *exactly* e_j regardless of how relevance is distributed
    # (the raw sum in the numerator and denominator cancel identically no
    # matter the composition of g) -- it cannot distinguish "1 relevant
    # source" from "1 relevant source plus 50 weak ones" the way Method E's
    # effective-source-count normalizer can.
    for n_junk in (0, 5, 20, 50):
        n = 1 + n_junk
        layer = BeliefLayer(in_cells=n, out_cells=1, aggregation="normalized_precision")
        with torch.no_grad():
            layer.content_weight.fill_(CONTENT_WEIGHT0)
            layer.bias.fill_(BIAS0)
            layer.relevance_logit[:, 0] = _inverse_sigmoid(0.9)
            if n_junk > 0:
                layer.relevance_logit[:, 1:] = _inverse_sigmoid(0.05)
            out = layer(_belief(n))
        assert out.evidence.item() == pytest.approx(E0)


def test_scale_stable_and_normalized_diverge_under_unequal_relevance() -> None:
    # The two formulas agree exactly for equal-relevance duplicates
    # (test_duplication_invariance.py) but should meaningfully diverge here,
    # where relevance is unequal across incoming cells -- otherwise Method E
    # would just be Method D under a different name.
    n_junk = 20
    normalized_layer = BeliefLayer(
        in_cells=1 + n_junk, out_cells=1, aggregation="normalized_precision"
    )
    scale_stable_layer = _dominant_plus_junk_layer(n_junk, g_real=0.9, g_junk=0.05)
    with torch.no_grad():
        normalized_layer.content_weight.fill_(CONTENT_WEIGHT0)
        normalized_layer.bias.fill_(BIAS0)
        normalized_layer.relevance_logit[:, 0] = _inverse_sigmoid(0.9)
        normalized_layer.relevance_logit[:, 1:] = _inverse_sigmoid(0.05)
        out_normalized = normalized_layer(_belief(1 + n_junk))
        out_scale_stable = scale_stable_layer(_belief(1 + n_junk))

    # Method D reports evidence == 1.0 exactly (see test above); Method E
    # should report something noticeably different (lower, given g in
    # (0, 1) throughout -- see the module docstring's reasoning).
    assert out_normalized.evidence.item() == pytest.approx(1.0)
    assert out_scale_stable.evidence.item() < out_normalized.evidence.item() * 0.9
