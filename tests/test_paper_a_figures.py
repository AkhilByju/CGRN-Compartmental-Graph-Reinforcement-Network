"""Validation for the Paper-A TMLR figure-generation pipeline
(`paper/figures/`). These are data-integrity checks on the frozen,
already-written Phase-2/Phase-3 processed run records -- nothing here
trains a model or touches a result file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "paper" / "figures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib import validate as V  # noqa: E402


def test_no_duplicate_run_records():
    V.check_no_duplicate_records()


def test_seed_counts_are_three_for_every_plotted_cell():
    V.check_seed_counts()


def test_severity_grids_sorted_and_unique():
    V.check_severities_sorted()


def test_phase3_same_width_mlp_joins_phase2_cells():
    V.check_phase3_joins_phase2()


def test_corruption_replicas_averaged_within_seed():
    V.check_replica_averaging()


def test_aps_missingness_bins_match_frozen_definition():
    V.check_aps_bins_match_frozen_definition()


def test_fig2_regenerates_deterministically(tmp_path):
    V.check_deterministic_regeneration(
        _REPO_ROOT / "paper" / "figures" / "make_fig2_controlled_corruption.py",
        _REPO_ROOT / "paper" / "figures" / "fig2_controlled_corruption_data.csv",
    )


def test_fig3_regenerates_deterministically(tmp_path):
    V.check_deterministic_regeneration(
        _REPO_ROOT / "paper" / "figures" / "make_fig3_aps_missingness_mechanism.py",
        _REPO_ROOT / "paper" / "figures" / "fig3_aps_missingness_mechanism_data.csv",
    )
