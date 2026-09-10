"""Paper A Phase 2 -- report summarizer: it must build every table from a
small real run set without crashing, and its headline arithmetic must match a
hand computation.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from experiments.paper_a.reliability.harness import run_one
from experiments.paper_a.reliability.summarize import (
    _auc_per_seed,
    build_report,
    by_cell,
    load_records,
)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("reliability_raw")
    for cf in ("missing", "gaussian"):
        for fam in ("plain_mlp", "confidence_mlp", "reliability_gated_mlp", "cellv0.3"):
            for seed in (0, 1):
                run_one(
                    "digits", cf, fam, seed,
                    device=torch.device("cpu"), results_dir=d, max_steps=200,
                )
    return d


def test_build_report_produces_every_table(raw_dir):
    report = build_report(raw_dir)
    for marker in (
        "## A. Missingness",
        "## B. Heterogeneous Gaussian noise",
        "## C. CellV0.3 vs the baselines",
        "## D. OOD degradation",
        "## E. Efficiency",
        "## F. CellV0.3 belief diagnostics",
        "## G. Confidence interventions",
        "## Predeclared go/no-go read",
        "## Failures and caveats",
    ):
        assert marker in report, marker
    # the four families are all named in the severity table
    for label in ("Plain MLP", "Confidence MLP", "Reliability-Gated MLP", "CellV0.3"):
        assert label in report


def test_records_load_and_group_by_cell(raw_dir):
    rows = load_records(raw_dir)
    assert len(rows) == 16  # 2 corruption x 4 model x 2 seed
    cells = by_cell(rows)
    assert len(cells[("digits", "missing", "cellv0.3")]) == 2


def test_compact_auc_delta_matches_hand_computation(raw_dir):
    rows = load_records(raw_dir)
    cells = by_cell(rows)
    v3 = cells[("digits", "gaussian", "cellv0.3")]
    gate = cells[("digits", "gaussian", "reliability_gated_mlp")]
    hand = statistics.mean(_auc_per_seed(v3, "accuracy")) - statistics.mean(
        _auc_per_seed(gate, "accuracy")
    )
    report = build_report(raw_dir)
    # the compact table row for digits/gaussian must carry this delta
    line = next(
        ln for ln in report.splitlines()
        if ln.startswith("| digits | gaussian |") and "±" not in ln
    )
    assert f"{hand:+.4f}" in line
