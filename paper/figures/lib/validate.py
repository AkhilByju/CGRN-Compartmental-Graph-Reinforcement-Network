"""Validation checks for the Paper-A figure data pipeline.

Run directly (``python -m paper.figures.lib.validate``) or imported from
``tests/test_paper_a_figures.py``. Every check raises ``AssertionError``
with a specific message on failure; ``run_all()`` collects and reports
every failure rather than stopping at the first one.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from lib import data as D  # noqa: E402

FIG2_DATASETS = ("mnist", "fashion_mnist")
FIG2_CORRUPTIONS = (D.MISSING, D.GAUSSIAN)
FIG2_PHASE2_FAMILIES = ("cellv0.3", "plain_mlp", "confidence_mlp")
FIG2_PHASE3_FAMILY = "confidence_mlp_same_width"


def check_seed_counts() -> None:
    rel = D.group_by(
        D.load_reliability_runs(),
        lambda r: (r["dataset"], r["corruption_family"], r["family"]),
    )
    for ds in FIG2_DATASETS:
        for cf in FIG2_CORRUPTIONS:
            for fam in FIG2_PHASE2_FAMILIES:
                rows = rel.get((ds, cf, fam), [])
                D.assert_seed_count(rows, D.EXPECTED_N_SEEDS, f"reliability {ds}/{cf}/{fam}")

    cap = D.group_by(
        D.load_capacity_stress_runs(),
        lambda r: (r["dataset"], r["corruption_family"], r["family"]),
    )
    for ds in FIG2_DATASETS:
        for cf in FIG2_CORRUPTIONS:
            rows = cap.get((ds, cf, FIG2_PHASE3_FAMILY), [])
            D.assert_seed_count(rows, D.EXPECTED_N_SEEDS, f"capacity_stress {ds}/{cf}")

    real = D.group_by(D.load_real_reliability_runs(), lambda r: (r["dataset"], r["family"]))
    for fam in ("cellv0.3", "neumiss", "confidence_mlp", "plain_mlp"):
        rows = real.get(("aps", fam), [])
        D.assert_seed_count(rows, D.EXPECTED_N_SEEDS, f"real_reliability aps/{fam}")


def check_severities_sorted() -> None:
    for cf in FIG2_CORRUPTIONS:
        sevs = D.severities_for(cf)
        assert list(sevs) == sorted(sevs), f"{cf} severity grid not sorted: {sevs}"
        assert len(set(sevs)) == len(sevs), f"{cf} severity grid has duplicates: {sevs}"


def check_phase3_joins_phase2() -> None:
    """Same-width Confidence MLP (Phase 3 Part A) must share dataset /
    corruption-family / severity-grid identity with the Phase-2 records it
    is plotted alongside -- otherwise the join in Fig 2 would silently
    compare mismatched conditions."""
    rel_rows = D.load_reliability_runs()
    cap_rows = D.load_capacity_stress_runs()
    rel_keys = {(r["dataset"], r["corruption_family"]) for r in rel_rows}
    cap_keys = {(r["dataset"], r["corruption_family"]) for r in cap_rows}
    for ds in FIG2_DATASETS:
        for cf in FIG2_CORRUPTIONS:
            assert (ds, cf) in rel_keys, f"missing Phase-2 cell {ds}/{cf}"
            assert (ds, cf) in cap_keys, f"missing Phase-3 same-width cell {ds}/{cf}"
    # severity grids must match exactly between the two sources for every
    # shared cell (both call the same frozen harness, but verify anyway).
    by_cell_rel = D.group_by(rel_rows, lambda r: (r["dataset"], r["corruption_family"]))
    by_cell_cap = D.group_by(cap_rows, lambda r: (r["dataset"], r["corruption_family"]))
    for ds in FIG2_DATASETS:
        for cf in FIG2_CORRUPTIONS:
            rel_sevs = {
                sv["severity"] for r in by_cell_rel[(ds, cf)] for sv in r["sweep"]["severities"]
            }
            cap_sevs = {
                sv["severity"] for r in by_cell_cap[(ds, cf)] for sv in r["sweep"]["severities"]
            }
            assert rel_sevs == cap_sevs, (
                f"{ds}/{cf}: Phase-2 severities {sorted(rel_sevs)} != "
                f"Phase-3 severities {sorted(cap_sevs)}"
            )


def check_replica_averaging() -> None:
    for r in D.load_reliability_runs() + D.load_capacity_stress_runs():
        primary = r["primary_metric"]
        for sv in r["sweep"]["severities"]:
            D.check_replica_average(sv, primary, r["run_id"])


def check_aps_bins_match_frozen_definition() -> None:
    # Restrict to the neural families actually plotted (Fig 3, left panel) --
    # the non-neural HistGradientBoosting reference has no `by_stratum`
    # breakdown recorded and is out of scope for this check.
    rows = [
        r for r in D.load_real_reliability_runs()
        if r["dataset"] == "aps" and r["family"] in FIG2_PHASE2_FAMILIES + ("neumiss",)
    ]
    assert rows, "no APS records found"
    for fam in ("cellv0.3", "neumiss", "confidence_mlp", "plain_mlp"):
        assert any(r["family"] == fam for r in rows), f"missing aps/{fam} records"
    for r in rows:
        bins = [s["bin"] for s in r["evaluation"]["by_stratum"]]
        assert bins == list(D.MISSINGNESS_BIN_LABELS), (
            f"{r['run_id']}: by_stratum bin order {bins} != frozen definition "
            f"{D.MISSINGNESS_BIN_LABELS}"
        )
        if r["family"] == "cellv0.3":
            belief_bins = [s["bin"] for s in r["evaluation"].get("belief_by_stratum", [])]
            assert belief_bins == list(D.MISSINGNESS_BIN_LABELS), (
                f"{r['run_id']}: belief_by_stratum bin order {belief_bins} != frozen "
                f"definition {D.MISSINGNESS_BIN_LABELS}"
            )


def check_no_duplicate_records() -> None:
    # load_* already raises on duplicates internally; calling them here
    # makes the check explicit and re-runs it as its own reported item.
    D.load_reliability_runs()
    D.load_capacity_stress_runs()
    D.load_real_reliability_runs()


def check_deterministic_regeneration(make_script: Path, out_csv: Path) -> None:
    """Run a `make_fig*.py` script twice and confirm the exported CSV's
    content hash is identical both times -- the figures are a pure
    function of the frozen JSON, no RNG, no wall-clock-dependent output."""
    hashes = []
    for _ in range(2):
        subprocess.run([sys.executable, str(make_script)], check=True, cwd=str(D.REPO_ROOT))
        hashes.append(hashlib.sha256(out_csv.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1], f"{make_script.name} is not deterministic: {hashes}"


CHECKS = [
    ("no duplicate run records", check_no_duplicate_records),
    ("seed counts == 3 for every plotted cell", check_seed_counts),
    ("severity grids sorted / unique", check_severities_sorted),
    ("Phase-3 same-width MLP joins Phase-2 cells correctly", check_phase3_joins_phase2),
    ("corruption replicas averaged within-seed before cross-seed stats", check_replica_averaging),
    ("APS missingness bins match frozen definition", check_aps_bins_match_frozen_definition),
]


def run_all() -> bool:
    ok = True
    for name, fn in CHECKS:
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            ok = False
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {e}")
            ok = False
        else:
            print(f"[PASS] {name}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
