"""Loading + aggregation for the Paper-A figures.

Everything here reads frozen, already-written processed run records
(``experiments/paper_a/**/results/processed/*.json``, each produced by the
experiment's own ``--summary-out`` at run time -- not reconstructed from
Markdown). No experiment is re-run and no result file is modified.

All aggregation follows the same two-step rule used by the committed report
generators (``experiments/paper_a/reliability/summarize.py``,
``publication_validation.py``): corruption *replicas* are already averaged
per (model, seed, severity) inside the processed record's ``metrics`` field
(``replica_metrics`` are the raw per-replica values kept for audit); we only
ever take mean/std *across seeds* on top of that.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.paper_a.real_reliability.datasets import MISSINGNESS_BINS  # noqa: E402
from experiments.paper_a.reliability.corruption import (  # noqa: E402
    GAUSSIAN,
    GAUSSIAN_S_TEST,
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
    MISSING,
    MISSING_P_TEST,
)

CORRUPTION_FAMILIES = (MISSING, GAUSSIAN)

# ---------------------------------------------------------------------------
# Frozen source files (Paper-A Phase 2 / Phase 3). Paths only -- provenance
# is printed by each `make_fig*.py` script before it plots anything.
# ---------------------------------------------------------------------------

PAPER_A = REPO_ROOT / "experiments" / "paper_a"

_RELIABILITY_PROCESSED = PAPER_A / "reliability" / "results" / "processed"
_REAL_RELIABILITY_PROCESSED = PAPER_A / "real_reliability" / "results" / "processed"

RELIABILITY_RUNS = _RELIABILITY_PROCESSED / "reliability_runs.json"
CAPACITY_STRESS_RUNS = _RELIABILITY_PROCESSED / "capacity_stress_runs.json"
REAL_RELIABILITY_RUNS = _REAL_RELIABILITY_PROCESSED / "real_reliability_runs.json"

# Phase-1 clean 7-dataset screen (for the optional supplementary figure).
PHASE1_CELLV03_RUNS = PAPER_A / "results" / "processed" / "phase1_cellv03_runs.json"
PHASE1_MLP_SOURCES = {
    "breast_cancer": PAPER_A / "results" / "processed" / "phase1_runs_small.json",
    "wine": PAPER_A / "results" / "processed" / "phase1_runs_small.json",
    "diabetes": PAPER_A / "results" / "processed" / "phase1_runs_small.json",
    "california_housing": PAPER_A / "results" / "processed" / "phase1_runs_mid.json",
    "digits": PAPER_A / "results" / "processed" / "phase1_runs_mid.json",
    "mnist": PAPER_A / "results" / "processed" / "phase1_runs_mnist_1.0.json",
    "fashion_mnist": PAPER_A / "results" / "processed" / "phase1_runs_fashion_mnist_1.0.json",
}

EXPECTED_N_SEEDS = 3


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected frozen Paper-A record file not found: {path}")
    return json.loads(path.read_text())


def mean_std(values: Iterable[float | None]) -> tuple[float, float, int]:
    """Mean/sample-std over the non-None values; matches
    `reliability/summarize.py`'s `agg` (ddof=1 std, 0.0 for n==1)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def group_by(rows: list[dict], key_fn: Callable[[dict], Any]) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out


def assert_unique_keys(rows: list[dict], key_fn: Callable[[dict], Any], source_name: str) -> None:
    """Raise if two rows in a processed record file share the same
    (dataset, family, seed, ...) key -- i.e. a duplicated/rerun record that
    would silently double-count a seed."""
    seen: dict[Any, int] = {}
    for r in rows:
        k = key_fn(r)
        seen[k] = seen.get(k, 0) + 1
    dupes = {k: n for k, n in seen.items() if n > 1}
    if dupes:
        raise ValueError(f"Duplicate run records in {source_name}: {dupes}")


def assert_seed_count(rows: list[dict], expected: int, context: str) -> None:
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) != expected:
        raise ValueError(
            f"{context}: expected {expected} seeds, found {len(seeds)} ({seeds})"
        )


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_reliability_runs() -> list[dict]:
    rows = load_json(RELIABILITY_RUNS)
    assert_unique_keys(
        rows,
        lambda r: (r["dataset"], r["corruption_family"], r["family"], r["seed"]),
        RELIABILITY_RUNS.name,
    )
    return rows


def load_capacity_stress_runs() -> list[dict]:
    rows = load_json(CAPACITY_STRESS_RUNS)
    assert_unique_keys(
        rows,
        lambda r: (r["dataset"], r["corruption_family"], r["family"], r["seed"]),
        CAPACITY_STRESS_RUNS.name,
    )
    return rows


def load_real_reliability_runs() -> list[dict]:
    rows = load_json(REAL_RELIABILITY_RUNS)
    assert_unique_keys(
        rows,
        lambda r: (r["dataset"], r["family"], r["seed"]),
        REAL_RELIABILITY_RUNS.name,
    )
    return rows


def load_phase1_cellv03_runs() -> list[dict]:
    rows = load_json(PHASE1_CELLV03_RUNS)
    assert_unique_keys(
        rows,
        lambda r: (r["dataset"], r["train_fraction"], r["family"], r["seed"]),
        PHASE1_CELLV03_RUNS.name,
    )
    return rows


PHASE1_CLEAN_DATASETS: tuple[str, ...] = (
    "breast_cancer", "wine", "digits", "diabetes",
    "california_housing", "mnist", "fashion_mnist",
)


def phase1_paired_delta(dataset: str, train_fraction: float = 1.0) -> tuple[float, float, int, str]:
    """BVU - MLP(matched) headline metric, paired by seed, at one
    train_fraction. Returns (mean, std, n_seeds, metric_name)."""
    v03 = [
        r for r in load_phase1_cellv03_runs()
        if r["dataset"] == dataset and abs(r["train_fraction"] - train_fraction) < 1e-9
    ]
    mlp = [
        r for r in load_phase1_mlp_runs(dataset)
        if r["family"] == "mlp_matched" and abs(r["train_fraction"] - train_fraction) < 1e-9
    ]
    v03_by_seed = {r["seed"]: r for r in v03}
    mlp_by_seed = {r["seed"]: r for r in mlp}
    seeds = sorted(set(v03_by_seed) & set(mlp_by_seed))
    if not seeds:
        raise ValueError(f"no paired seeds for {dataset} @ train_fraction={train_fraction}")
    metric = v03_by_seed[seeds[0]]["headline_metric"]
    deltas = [v03_by_seed[s]["headline_value"] - mlp_by_seed[s]["headline_value"] for s in seeds]
    m, sd, n = mean_std(deltas)
    return m, sd, n, metric


def load_phase1_mlp_runs(dataset: str) -> list[dict]:
    path = PHASE1_MLP_SOURCES[dataset]
    rows = [r for r in load_json(path) if r["dataset"] == dataset]
    assert_unique_keys(
        rows,
        lambda r: (r["dataset"], r["train_fraction"], r["family"], r["seed"]),
        f"{path.name} [{dataset}]",
    )
    return rows


# ---------------------------------------------------------------------------
# Corruption-severity extraction (Fig 2)
# ---------------------------------------------------------------------------


def severities_for(corruption_family: str) -> tuple[float, ...]:
    return MISSING_P_TEST if corruption_family == MISSING else GAUSSIAN_S_TEST


def metric_at_severity(row: dict, severity: float, metric_key: str) -> float | None:
    """Pull the already replica-averaged metric at one test severity from a
    single (dataset, corruption_family, family, seed) processed record."""
    hits = [
        sv for sv in row["sweep"]["severities"]
        if abs(sv["severity"] - severity) < 1e-9
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected exactly one severity={severity} entry in run {row.get('run_id')}, "
            f"found {len(hits)}"
        )
    sv = hits[0]
    check_replica_average(sv, metric_key, row.get("run_id", "?"))
    return sv["metrics"].get(metric_key)


def check_replica_average(sv: dict, metric_key: str, run_id: str, tol: float = 1e-6) -> None:
    """Confirm `metrics[metric_key]` really is the mean of `replica_metrics`
    (i.e. replicas were averaged within-seed, before this code averages
    across seeds) -- guards against silently treating replicas as extra
    seeds."""
    replicas = sv.get("replica_metrics") or []
    if not replicas:
        return
    vals = [rp.get(metric_key) for rp in replicas if rp.get(metric_key) is not None]
    if not vals:
        return
    expected = statistics.mean(vals)
    actual = sv["metrics"].get(metric_key)
    if actual is None:
        return
    if abs(expected - actual) > tol * max(1.0, abs(expected)):
        raise ValueError(
            f"{run_id} severity={sv['severity']}: metrics[{metric_key}]={actual} does not "
            f"match mean(replica_metrics)={expected} -- replica averaging invariant broken"
        )


def series_for(
    rows: list[dict], corruption_family: str, metric_key: str
) -> tuple[list[float], list[float], list[float], list[int]]:
    """rows = the (<=EXPECTED_N_SEEDS) processed records for one
    (dataset, corruption_family, family). Returns (severities, means, stds,
    n_seeds_per_point), one entry per severity in the frozen test grid, in
    ascending numeric order."""
    sevs = severities_for(corruption_family)
    assert list(sevs) == sorted(sevs), "severity grid must be sorted numerically"
    means, stds, ns = [], [], []
    for s in sevs:
        vals = [metric_at_severity(r, s, metric_key) for r in rows]
        m, sd, n = mean_std(vals)
        means.append(m)
        stds.append(sd)
        ns.append(n)
    return list(sevs), means, stds, ns


def train_boundary(corruption_family: str) -> float:
    return MAX_TRAIN_SEVERITY[corruption_family]


def ood_boundary(corruption_family: str) -> float:
    return MAX_OOD_SEVERITY[corruption_family]


# ---------------------------------------------------------------------------
# APS missingness-stratified extraction (Fig 3)
# ---------------------------------------------------------------------------

MISSINGNESS_BIN_LABELS: tuple[str, ...] = tuple(label for label, _lo, _hi in MISSINGNESS_BINS)

# Concise axis tick labels for the frozen bin definitions above (display
# only -- `MISSINGNESS_BIN_LABELS` above is what's actually matched against
# the saved records).
COMPACT_BIN_LABEL: dict[str, str] = {
    "0": "0",
    "(0, 0.10]": "0-10%",
    "(0.10, 0.25]": "10-25%",
    "(0.25, 0.50]": "25-50%",
    ">0.50": ">50%",
}


def stratum_metric_series(
    rows: list[dict], metric_key: str
) -> tuple[list[str], list[float], list[float], list[int]]:
    """rows = the per-seed processed records for one (dataset, family).
    Returns (bin_labels, means, stds, n) across `evaluation.by_stratum`,
    in the frozen bin order. A bin with zero valid seed values is dropped
    (no point plotted) rather than interpolated."""
    labels, means, stds, ns = [], [], [], []
    for label in MISSINGNESS_BIN_LABELS:
        vals = []
        for r in rows:
            strata = {s["bin"]: s for s in r["evaluation"]["by_stratum"]}
            if label in strata and strata[label].get(metric_key) is not None:
                vals.append(strata[label][metric_key])
        if not vals:
            continue
        m, sd, n = mean_std(vals)
        labels.append(label)
        means.append(m)
        stds.append(sd)
        ns.append(n)
    return labels, means, stds, ns


def belief_stratum_series(
    rows: list[dict], layer_key: str
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Same as `stratum_metric_series` but over `evaluation.belief_by_stratum`
    (BVU-only internal diagnostics), e.g. layer_key='l1_pi_mean'."""
    labels, means, stds, ns = [], [], [], []
    for label in MISSINGNESS_BIN_LABELS:
        vals = []
        for r in rows:
            strata = {s["bin"]: s for s in r["evaluation"].get("belief_by_stratum", [])}
            if label in strata and strata[label].get(layer_key) is not None:
                vals.append(strata[label][layer_key])
        if not vals:
            continue
        m, sd, n = mean_std(vals)
        labels.append(label)
        means.append(m)
        stds.append(sd)
        ns.append(n)
    return labels, means, stds, ns
