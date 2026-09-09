"""Paper A Sec 10 -- the controlled duplication-invariance experiment, as an
automated check (Paper-A task Sec 15).

This complements `tests/test_duplication_invariance.py` (which feeds one
belief value through equal connections): here a *heterogeneous* fixed set of
6 distinct source beliefs -- distinct content, evidence, uncertainty, AND
relevance -- is duplicated as a whole multiset, which is the exact scenario
Paper A's report Table F is built from.
"""

from __future__ import annotations

import pytest

from experiments.paper_a.duplication_experiment import (
    MULTIPLICITIES,
    run_experiment,
    run_for_aggregation,
)


def test_scale_stable_precision_is_duplication_invariant():
    payload = run_experiment()
    v = payload["verdicts"]["scale_stable_precision"]
    assert v["mu_invariant"], v
    assert v["e_invariant"], v
    assert v["u_invariant"], v
    # tight: float64 accumulation noise only
    assert v["max_abs_de"] < 1e-6
    assert v["max_abs_du"] < 1e-6


def test_normalized_precision_is_also_duplication_invariant():
    v = run_experiment()["verdicts"]["normalized_precision"]
    assert v["mu_invariant"] and v["e_invariant"] and v["u_invariant"]


def test_old_precision_rule_inflates_confidence_with_duplication():
    # The historical control: naive accumulated confidence. Evidence grows
    # ~linearly with the multiset multiplicity; uncertainty shrinks.
    rows = run_for_aggregation("precision")
    by_m = {r["multiplicity"]: r for r in rows}
    assert by_m[16]["e_ratio_vs_x1"] == pytest.approx(16.0, rel=1e-6)
    assert by_m[8]["e_ratio_vs_x1"] == pytest.approx(8.0, rel=1e-6)
    assert by_m[16]["out_u"] < by_m[1]["out_u"]

    v = run_experiment()["verdicts"]["precision"]
    assert not v["e_invariant"]
    assert not v["u_invariant"]


def test_old_precision_rule_keeps_content_estimate_invariant():
    # mu (the content estimate) is alpha-weighted averaging in all three
    # rules, so duplicating identical opinions must not move it even for the
    # non-scale-stable rule.
    v = run_experiment()["verdicts"]["precision"]
    assert v["mu_invariant"], v


def test_multiplicities_are_the_documented_set():
    assert MULTIPLICITIES == (1, 2, 4, 8, 16)
