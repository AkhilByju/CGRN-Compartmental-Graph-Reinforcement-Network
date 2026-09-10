#!/usr/bin/env python3
"""Aggregate the Paper-A Phase-2 reliability runs (`paper_a_reliability`) into
the committed report `experiments/paper_a/reliability/reliability_results.md`
(Tables A-G + failures/caveats, Phase-2 task Sec 17).

No significance testing -- with n=3 seeds the paired differences are reported
per seed and as a mean, nothing more (Sec 8). The scientifically load-bearing
comparisons are CellV0.3 vs the two *reliability-aware* baselines
(Confidence-Augmented MLP, Reliability-Gated MLP), not vs the plain MLP
(Sec 18).

Usage:
    python experiments/paper_a/reliability/summarize.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.paper_a.reliability.corruption import (  # noqa: E402
    GAUSSIAN,
    GAUSSIAN_S_TEST,
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
    MISSING,
    MISSING_P_TEST,
)
from experiments.paper_a.reliability.harness import (  # noqa: E402
    EXPERIMENT_ID,
    RELIABILITY_DATASETS,
)
from experiments.paper_a.reliability.interventions import (  # noqa: E402
    CONDITIONS,
    INTERVENTION_DATASETS,
    REGIMES,
)
from experiments.paper_a.reliability.models import (  # noqa: E402
    CELLV03,
    CONFIDENCE_MLP,
    PLAIN_MLP,
    RELIABILITY_GATED_MLP,
)

_FAMILY_LABEL = {
    PLAIN_MLP: "Plain MLP",
    CONFIDENCE_MLP: "Confidence MLP",
    RELIABILITY_GATED_MLP: "Reliability-Gated MLP",
    CELLV03: "CellV0.3",
}
_FAMILY_ORDER = (PLAIN_MLP, CONFIDENCE_MLP, RELIABILITY_GATED_MLP, CELLV03)
_CORRUPTIONS = (MISSING, GAUSSIAN)
_TINY_PI = 1e-6
_RESPONSE_TOL = 5e-3  # mean-pi change below this across the grid -> "flat / no response"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(raw_dir: Path) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith("_history.json"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("experiment_id") != EXPERIMENT_ID:
            continue
        extra = rec["config"].get("extra", {})
        key = (rec["dataset"], extra.get("corruption_family"), rec["architecture"], rec["seed"])
        latest[key] = {
            "dataset": rec["dataset"],
            "corruption_family": extra.get("corruption_family"),
            "family": rec["architecture"],
            "seed": rec["seed"],
            "task_type": extra.get("task_type"),
            "primary_metric": extra.get("primary_metric"),
            "params": rec["parameter_count"],
            "sizing": extra.get("sizing", {}),
            "sweep": extra.get("sweep", {}),
            "interventions": extra.get("interventions", []),
            "diagnostics_init": extra.get("diagnostics_init", {}),
            "efficiency": extra.get("efficiency", {}),
            "diverged": bool(extra.get("diverged", False)),
            "cap_hit": bool(extra.get("cap_hit", False)),
            "best_val_metric": rec.get("validation_metrics", {}).get("best_val_metric"),
        }
    return list(latest.values())


def by_cell(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["corruption_family"], r["family"])].append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["seed"])
    return cells


# ---------------------------------------------------------------------------
# Small stats / formatting helpers
# ---------------------------------------------------------------------------


def agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _sev_metric_per_seed(runs: list[dict], severity: float, metric: str) -> list[float]:
    out = []
    for r in runs:
        for sv in r["sweep"].get("severities", []):
            if abs(sv["severity"] - severity) < 1e-9:
                out.append(sv["metrics"].get(metric))
    return out


def _auc_per_seed(runs: list[dict], metric: str) -> list[float]:
    return [r["sweep"].get("corruption_auc", {}).get(metric) for r in runs]


def _diag_per_seed(runs: list[dict], severity: float, key: str) -> list[float]:
    out = []
    for r in runs:
        for sv in r["sweep"].get("severities", []):
            if abs(sv["severity"] - severity) < 1e-9 and sv.get("belief_diag"):
                out.append(sv["belief_diag"].get(key))
    return [v for v in out if v is not None]


def _severities(cf: str) -> tuple[float, ...]:
    return MISSING_P_TEST if cf == MISSING else GAUSSIAN_S_TEST


def _sev_label(cf: str, sev: float) -> str:
    return f"{int(round(sev * 100))}%" if cf == MISSING else f"{sev:g}"


def _present_datasets(rows: list[dict]) -> list[str]:
    return [d for d in RELIABILITY_DATASETS if any(r["dataset"] == d for r in rows)]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_header(rows: list[dict], present: list[str]) -> list[str]:
    missing = [d for d in RELIABILITY_DATASETS if d not in present]
    out = [
        "# Paper A — Phase 2: reliability / corruption benchmark for CellV0.3",
        "",
        f"Runs loaded: {len(rows)} (experiment `{EXPERIMENT_ID}`). "
        f"Datasets present: {', '.join(present) or 'none'}.",
    ]
    if missing:
        out.append(f"Datasets **absent** from the loaded records: {', '.join(missing)}.")
    out += [
        "",
        "**Question.** When input information has heterogeneous and *known* "
        "reliability, does explicitly propagating reliability/conflict through "
        "CellV0.3 provide a useful inductive bias beyond conventional networks "
        "given the same corrupted observations and the same reliability "
        "information?",
        "",
        "**Four families, all at the CellV0.3 parameter count** (fitted to the "
        "Phase-1 per-dataset budget):",
        "",
        "| Family | Input | Reliability seen |",
        "|---|---|---|",
        "| Plain MLP (A) | `x_corrupted` | none |",
        "| Confidence MLP (B) | `concat(x_corrupted, c)` | exact `c` |",
        "| Reliability-Gated MLP (C) | `c * x_corrupted` | exploited directly |",
        "| CellV0.3 (D) | belief `(mu=x_corrupted, e=c, u=0)` | exact `c` |",
        "",
        "**Two corruption families**, applied after Phase-1 preprocessing: "
        "missing-feature (drop-to-zero, `c = 1.0`/`1e-3`) and heterogeneous "
        "Gaussian (`sigma_j ~ U(0, s)` per feature, `c_j = 1/(1+sigma_j^2)`). "
        "Training severity is drawn per example from the discrete training "
        "regime; evaluation uses the fixed grids "
        f"`p ∈ {{{', '.join(f'{p:g}' for p in MISSING_P_TEST)}}}` and "
        f"`s ∈ {{{', '.join(f'{s:g}' for s in GAUSSIAN_S_TEST)}}}` "
        "(3 deterministic corruption replicas per severity, identical across "
        "models, averaged before any across-seed statistic).",
        "",
        "Shared frozen protocol: AdamW, lr=1e-2, weight_decay=0, batch 128, "
        "best-checkpoint restore on the averaged in-distribution "
        "corrupted-validation primary metric, ~1500-step early-stop patience, "
        "15000-step cap. Nothing tuned per model or dataset. CellV0.3's "
        "equations are frozen; only its *input belief* carries the reliability "
        "signal (Phase-2 task Sec 1). n=3 seeds; no significance test.",
        "",
        "**Interpretation boundary (Sec 18).** A CellV0.3 win over the plain "
        "MLP is *not* sufficient — the plain MLP has no reliability signal. The "
        "meaningful comparisons are vs the Confidence MLP and the "
        "Reliability-Gated MLP. `e`/`u`/`pi` are internal computational "
        "reliability variables, not calibrated predictive probabilities.",
        "",
    ]
    return out


def section_compact(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## Compact summary — CellV0.3 corruption-AUC Δ vs each baseline "
        "(primary metric, mean of 3 seeds)",
        "",
        "| Dataset | Corruption | Δ vs Plain | Δ vs Confidence MLP | Δ vs Reliability-Gated |",
        "|---|---|---|---|---|",
    ]
    tally = {"conf_win": 0, "conf_loss": 0, "gate_win": 0, "gate_loss": 0, "n": 0}
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            v3 = cells.get((ds, cf, CELLV03))
            if not v3:
                continue
            a3 = agg(_auc_per_seed(v3, primary))[0]
            parts = []
            for fam in (PLAIN_MLP, CONFIDENCE_MLP, RELIABILITY_GATED_MLP):
                other = cells.get((ds, cf, fam))
                if not other:
                    parts.append("—")
                    continue
                d = a3 - agg(_auc_per_seed(other, primary))[0]
                parts.append(f"{d:+.4f}")
                if fam == CONFIDENCE_MLP:
                    tally["n"] += 1
                    tally["conf_win" if d > 0 else "conf_loss"] += 1
                if fam == RELIABILITY_GATED_MLP:
                    tally["gate_win" if d > 0 else "gate_loss"] += 1
            out.append(f"| {ds} | {cf} | {parts[0]} | {parts[1]} | {parts[2]} |")
    out += [
        "",
        f"_Positive = CellV0.3 has the larger area under the corruption/metric "
        f"curve. Across {tally['n']} (dataset, corruption) cells: vs Confidence "
        f"MLP CellV0.3 ahead in {tally['conf_win']}, behind in "
        f"{tally['conf_loss']}; vs Reliability-Gated MLP ahead in "
        f"{tally['gate_win']}, behind in {tally['gate_loss']}. RMSE-based AUC "
        f"(regression) is 'lower is better' and is negated here so + always "
        f"means CellV0.3 better._",
        "",
    ]
    return out


def _primary_from_cells(cells: dict, ds: str) -> str:
    for (d, _cf, _fam), runs in cells.items():
        if d == ds and runs:
            return runs[0]["primary_metric"]
    return "accuracy"


def _severity_table(cells: dict, present: list[str], cf: str, letter: str, title: str) -> list[str]:
    sevs = _severities(cf)
    header_cells = " | ".join(_sev_label(cf, s) for s in sevs)
    out = [
        f"## {letter}. {title}",
        "",
        "Primary metric (accuracy / R²), mean ± std over 3 seeds, at each "
        "severity, plus trapezoidal corruption-AUC over the grid.",
        "",
        f"| Dataset | Model | {header_cells} | corruption AUC |",
        "|---|---|" + "---|" * (len(sevs) + 1),
    ]
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for fam in _FAMILY_ORDER:
            runs = cells.get((ds, cf, fam))
            if not runs:
                continue
            cols = []
            for s in sevs:
                m, sd = agg(_sev_metric_per_seed(runs, s, primary))
                cols.append(f"{m:.3f}±{sd:.3f}")
            auc_m, auc_s = agg(_auc_per_seed(runs, primary))
            out.append(
                f"| {ds} | {_FAMILY_LABEL[fam]} | {' | '.join(cols)} | "
                f"{auc_m:.4f}±{auc_s:.4f} |"
            )
    out.append("")
    return out


def section_paired(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## C. CellV0.3 vs the baselines — paired difference at every severity",
        "",
        "Per (dataset, corruption, severity): mean over 3 seeds of "
        "`primary(CellV0.3, seed) − primary(baseline, seed)`. The last two "
        "columns are the scientifically meaningful ones (Sec 18).",
        "",
        "| Dataset | Corruption | Severity | V03 − Plain | V03 − Confidence MLP "
        "| V03 − Reliability-Gated |",
        "|---|---|---|---|---|---|",
    ]
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            v3 = cells.get((ds, cf, CELLV03))
            if not v3:
                continue
            v3_by_seed = {r["seed"]: r for r in v3}
            for s in _severities(cf):
                row = [ds, cf, _sev_label(cf, s)]
                for fam in (PLAIN_MLP, CONFIDENCE_MLP, RELIABILITY_GATED_MLP):
                    other = {r["seed"]: r for r in cells.get((ds, cf, fam), [])}
                    seeds = sorted(set(v3_by_seed) & set(other))
                    if not seeds:
                        row.append("—")
                        continue
                    deltas = []
                    for sd in seeds:
                        a = _sev_metric_per_seed([v3_by_seed[sd]], s, primary)
                        b = _sev_metric_per_seed([other[sd]], s, primary)
                        if a and b and a[0] is not None and b[0] is not None:
                            deltas.append(a[0] - b[0])
                    row.append(f"{statistics.mean(deltas):+.4f}" if deltas else "—")
                out.append("| " + " | ".join(row) + " |")
    out += [
        "",
        "_Positive = CellV0.3 better. n=3 seeds; no significance test._",
        "",
    ]
    return out


def section_ood(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## D. OOD degradation — max training severity → most severe test",
        "",
        "`OOD_drop = primary(max-train-severity) − primary(most-severe-test)` "
        "(missingness `p: .3→.7`, Gaussian `s: .75→1.5`). Smaller is better.",
        "",
        "| Dataset | Corruption | Model | at max-train | at max-OOD | OOD_drop (mean ± std) |",
        "|---|---|---|---|---|---|",
    ]
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            lo, hi = MAX_TRAIN_SEVERITY[cf], MAX_OOD_SEVERITY[cf]
            for fam in _FAMILY_ORDER:
                runs = cells.get((ds, cf, fam))
                if not runs:
                    continue
                lo_m, _ = agg(_sev_metric_per_seed(runs, lo, primary))
                hi_m, _ = agg(_sev_metric_per_seed(runs, hi, primary))
                drops = [
                    r["sweep"]["ood_drop"][primary]
                    for r in runs
                    if r["sweep"].get("ood_drop")
                ]
                dm, dsd = agg(drops)
                out.append(
                    f"| {ds} | {cf} | {_FAMILY_LABEL[fam]} | {lo_m:.3f} | "
                    f"{hi_m:.3f} | {dm:+.4f} ± {dsd:.4f} |"
                )
    out.append("")
    return out


def section_efficiency(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## E. Efficiency",
        "",
        "Per (dataset, model): parameter count, hidden width, % off the "
        "CellV0.3 target, best-validation step, total steps, train wall-clock, "
        "time/step. Mean over the runs per (dataset, model) — both corruption "
        "families × all seeds.",
        "",
        "| Dataset | Model | Params | Hidden | Δ% vs target | Best-val step | "
        "Total steps | Train wall (s) | Time/step (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in present:
        target = None
        for fam in _FAMILY_ORDER:
            runs = [
                r for cf in _CORRUPTIONS for r in cells.get((ds, cf, fam), [])
            ]
            if not runs:
                continue
            params = runs[0]["params"]
            sizing = runs[0]["sizing"]
            hidden = sizing.get("hidden_cells", sizing.get("hidden_dim", "?"))
            if fam == CELLV03:
                target = params
            reldiff = sizing.get("param_rel_diff")
            pct = f"{reldiff * 100:+.2f}%" if reldiff is not None else "0.00%"
            bv_m, _ = agg([r["efficiency"].get("best_val_step") for r in runs])
            ts_m, _ = agg([r["efficiency"].get("total_steps") for r in runs])
            wc_m, _ = agg([r["efficiency"].get("train_wall_clock_seconds") for r in runs])
            tps_m, _ = agg([r["efficiency"].get("time_per_step_seconds") for r in runs])
            out.append(
                f"| {ds} | {_FAMILY_LABEL[fam]} | {params:,} | {hidden} | {pct} | "
                f"{bv_m:.0f} | {ts_m:.0f} | {wc_m:.1f} | {tps_m * 1e3:.2f} |"
            )
        if target is not None:
            out.append(f"| {ds} | _(CellV0.3 target param count: {target:,})_ | | | | | | | |")
    out.append("")
    return out


def section_belief(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## F. CellV0.3 belief diagnostics across corruption severity (Sec 13/14)",
        "",
        "Trained CellV0.3, replica- and seed-averaged. `pi = e/(1+e·u)`; "
        "`CoV(pi) = std/mean` over the test set × cells. Sec 14 asks whether "
        "mean hidden precision responds to increasing corruption — the curve is "
        "reported as-is, not repaired.",
        "",
        "| Dataset | Corruption | Severity | L1 π mean | L2 π mean | L2 CoV(π) | "
        "L2 √π mean | L2 u mean | L2 |consensus| |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    response_rows: list[str] = []
    for ds in present:
        for cf in _CORRUPTIONS:
            runs = cells.get((ds, cf, CELLV03))
            if not runs:
                continue
            l1_series, l2_series = [], []
            for s in _severities(cf):
                l1 = agg(_diag_per_seed(runs, s, "diag_layer1_precision_mean"))[0]
                l2 = agg(_diag_per_seed(runs, s, "diag_layer2_precision_mean"))[0]
                cov = agg(_diag_per_seed(runs, s, "diag_layer2_precision_cv"))[0]
                sq = agg(_diag_per_seed(runs, s, "diag_layer2_sqrt_precision_mean"))[0]
                u = agg(_diag_per_seed(runs, s, "diag_layer2_u_mean"))[0]
                con = agg(_diag_per_seed(runs, s, "diag_layer2_abs_consensus_mean"))[0]
                l1_series.append((s, l1))
                l2_series.append((s, l2))
                out.append(
                    f"| {ds} | {cf} | {_sev_label(cf, s)} | {l1:.3f} | {l2:.3f} | "
                    f"{cov:.3f} | {sq:.3f} | {u:.3f} | {con:.3f} |"
                )
            response_rows.append(_response_line(ds, cf, l1_series, l2_series))
    out += [
        "",
        "### Reliability-response (Sec 14) — does mean hidden π fall as corruption rises?",
        "",
        "| Dataset | Corruption | L1 π (clean → worst) | L2 π (clean → worst) | Read |",
        "|---|---|---|---|---|",
    ]
    out += response_rows
    out += [
        "",
        "_A sensible mechanism generally shows lower effective precision as "
        "observation reliability deteriorates; where it does not, the row is "
        "flagged and left unrepaired (Sec 14)._",
        "",
    ]
    return out


def _response_line(ds: str, cf: str, l1: list[tuple], l2: list[tuple]) -> str:
    l1_lo, l1_hi = l1[0][1], l1[-1][1]
    l2_lo, l2_hi = l2[0][1], l2[-1][1]
    d1, d2 = l1_hi - l1_lo, l2_hi - l2_lo
    if d1 < -_RESPONSE_TOL or d2 < -_RESPONSE_TOL:
        read = "π falls with severity"
    elif abs(d1) <= _RESPONSE_TOL and abs(d2) <= _RESPONSE_TOL:
        read = "**flat — no response**"
    else:
        read = "**π rises with severity**"
    return (
        f"| {ds} | {cf} | {l1_lo:.3f} → {l1_hi:.3f} | {l2_lo:.3f} → {l2_hi:.3f} | {read} |"
    )


def section_interventions(cells: dict) -> list[str]:
    out = [
        "## G. Confidence interventions (Sec 15) — evaluation only, no retraining",
        "",
        "Trained CellV0.3 on Fashion-MNIST and California Housing, corrupted "
        "observations held fixed, reliability map swapped: `true` (actual `c`), "
        "`all_ones` (`c := 1`), `shuffled` (`c` permuted across features per "
        "example). Primary metric, mean over 3 seeds × 3 replicas.",
        "",
        "| Dataset | Corruption | Regime | true | all-ones (Δ) | shuffled (Δ) |",
        "|---|---|---|---|---|---|",
    ]
    any_row = False
    for ds in INTERVENTION_DATASETS:
        for cf in _CORRUPTIONS:
            runs = cells.get((ds, cf, CELLV03))
            if not runs:
                continue
            for regime in REGIMES:
                per_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
                deltas: dict[str, list[float]] = {"all_ones": [], "shuffled": []}
                sev = None
                for r in runs:
                    for iv in r.get("interventions", []):
                        if iv["corruption_family"] == cf and iv["regime"] == regime:
                            sev = iv["severity"]
                            for c in CONDITIONS:
                                per_cond[c].append(iv["condition_metric"][c])
                            for c in deltas:
                                deltas[c].append(iv["delta_vs_true"][c])
                if sev is None or not per_cond["true"]:
                    continue
                any_row = True
                t = statistics.mean(per_cond["true"])
                ao = statistics.mean(deltas["all_ones"])
                sh = statistics.mean(deltas["shuffled"])
                out.append(
                    f"| {ds} | {cf} | {regime} (sev {sev:g}) | {t:.4f} | "
                    f"{ao:+.4f} | {sh:+.4f} |"
                )
    if not any_row:
        out.append("| _no CellV0.3 intervention records found_ | | | | | |")
    out += [
        "",
        "_A negative `shuffled` Δ that is larger in magnitude than the "
        "`all_ones` Δ means correct observation↔reliability alignment (not just "
        "the marginal `c` distribution) matters to CellV0.3._",
        "",
    ]
    return out


def section_failures(rows: list[dict], cells: dict, present: list[str]) -> list[str]:
    out = ["## Failures and caveats", ""]

    diverged = [r for r in rows if r["diverged"]]
    out.append(
        "**Divergence / NaNs:** "
        + ("none."
           if not diverged
           else ", ".join(
               f"{r['dataset']}/{r['corruption_family']}/{r['family']}/seed{r['seed']}"
               for r in diverged
           ))
    )
    out.append("")

    cap = sorted({
        (r["dataset"], r["corruption_family"], r["family"]) for r in rows if r["cap_hit"]
    })
    if cap:
        out.append("**Step-cap hits (early stopping did not terminate):**")
        out.append("")
        for ds, cf, fam in cap:
            cell = cells.get((ds, cf, fam), [])
            n = sum(1 for r in cell if r["cap_hit"])
            out.append(
                f"- {ds} / {cf} / {_FAMILY_LABEL.get(fam, fam)} ({n}/{len(cell)} seeds)"
            )
    else:
        out.append("**Step-cap hits:** none.")
    out.append("")

    # unstable seeds: CellV0.3 primary-metric seed-std at the clean severity
    # more than 3x the plain MLP's
    unstable = []
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            v3 = cells.get((ds, cf, CELLV03))
            mm = cells.get((ds, cf, PLAIN_MLP))
            if not (v3 and mm):
                continue
            _, s3 = agg(_sev_metric_per_seed(v3, 0.0, primary))
            _, sm = agg(_sev_metric_per_seed(mm, 0.0, primary))
            if sm > 1e-6 and s3 > 3 * sm:
                unstable.append(f"{ds}/{cf} (CellV0.3 std {s3:.4f} vs Plain {sm:.4f})")
    out.append("**High CellV0.3 seed variance (clean severity, >3× Plain MLP):** "
               + ("; ".join(unstable) if unstable else "none."))
    out.append("")

    # parameter-match failures
    pm_fail = []
    for r in rows:
        if r["family"] == CELLV03:
            continue
        w = r["sizing"].get("param_match_within_2pct")
        if w is False:
            pct = r["sizing"].get("param_rel_diff", 0) * 100
            pm_fail.append(f"{r['dataset']}/{r['family']} ({pct:.2f}%)")
    out.append("**Parameter-match failures (>2% off the CellV0.3 target):** "
               + ("; ".join(sorted(set(pm_fail))) if pm_fail else "none."))
    out.append("")

    # datasets/corruptions where CellV0.3 clearly loses to a reliability-aware baseline
    losses = []
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            v3 = cells.get((ds, cf, CELLV03))
            if not v3:
                continue
            a3 = agg(_auc_per_seed(v3, primary))[0]
            for fam in (CONFIDENCE_MLP, RELIABILITY_GATED_MLP):
                other = cells.get((ds, cf, fam))
                if not other:
                    continue
                d = a3 - agg(_auc_per_seed(other, primary))[0]
                if d < -0.005:
                    losses.append(f"{ds}/{cf}: {d:+.4f} vs {_FAMILY_LABEL[fam]}")
    out.append("**Cells where CellV0.3's corruption-AUC clearly loses (< −0.005) "
               "to a reliability-aware baseline:** "
               + ("; ".join(losses) if losses else "none."))
    out.append("")

    # hidden precision fails to respond
    flat = []
    tiny = []
    for ds in present:
        for cf in _CORRUPTIONS:
            runs = cells.get((ds, cf, CELLV03))
            if not runs:
                continue
            sevs = _severities(cf)
            l2 = [agg(_diag_per_seed(runs, s, "diag_layer2_precision_mean"))[0] for s in sevs]
            l1 = [agg(_diag_per_seed(runs, s, "diag_layer1_precision_mean"))[0] for s in sevs]
            if abs(l2[-1] - l2[0]) <= _RESPONSE_TOL and abs(l1[-1] - l1[0]) <= _RESPONSE_TOL:
                flat.append(f"{ds}/{cf}")
            for s in sevs:
                for layer in ("layer1", "layer2"):
                    lo = agg(_diag_per_seed(runs, s, f"diag_{layer}_precision_min"))[0]
                    if lo is not None and lo < _TINY_PI:
                        tiny.append(f"{ds}/{cf} s={_sev_label(cf, s)} {layer} (min π={lo:.1e})")
    out.append("**CellV0.3 hidden precision does not respond to corruption "
               f"(|Δ mean π| ≤ {_RESPONSE_TOL} across the grid, both layers):** "
               + ("; ".join(flat) if flat else "none."))
    out.append("")
    if tiny:
        out.append("**Numeric-range flag** (a single test example × cell with output "
                   "precision `< 1e-6` — large local conflict; all finite, no NaN/"
                   "divergence, no protocol impact):")
        out.append("")
        out += [f"- {t}" for t in tiny]
    else:
        out.append("**Numeric-range flag:** none (min output precision ≥ 1e-6).")
    out.append("")
    return out


def section_verdict(cells: dict, present: list[str]) -> list[str]:
    """The predeclared go/no-go read (Sec 19) — report the evidence, do not
    redesign anything."""
    conf_deltas, gate_deltas, ood_better_conf, ood_better_gate = [], [], 0, 0
    n = 0
    for ds in present:
        primary = _primary_from_cells(cells, ds)
        for cf in _CORRUPTIONS:
            v3 = cells.get((ds, cf, CELLV03))
            if not v3:
                continue
            a3 = agg(_auc_per_seed(v3, primary))[0]
            d3 = agg([r["sweep"]["ood_drop"][primary] for r in v3 if r["sweep"].get("ood_drop")])[0]
            for fam, bucket, ood_ctr in (
                (CONFIDENCE_MLP, conf_deltas, "conf"),
                (RELIABILITY_GATED_MLP, gate_deltas, "gate"),
            ):
                other = cells.get((ds, cf, fam))
                if not other:
                    continue
                bucket.append(a3 - agg(_auc_per_seed(other, primary))[0])
                do = agg(
                    [r["sweep"]["ood_drop"][primary] for r in other if r["sweep"].get("ood_drop")]
                )[0]
                if d3 < do:  # smaller degradation
                    if ood_ctr == "conf":
                        ood_better_conf += 1
                    else:
                        ood_better_gate += 1
                if fam == CONFIDENCE_MLP:
                    n += 1

    conf_win = sum(1 for d in conf_deltas if d > 0)
    gate_win = sum(1 for d in gate_deltas if d > 0)
    # median, not mean: one regression cell where a baseline is unstable can put
    # a ±10-R² outlier into the mean. Win-counts drive the classification.
    conf_med = statistics.median(conf_deltas) if conf_deltas else float("nan")
    gate_med = statistics.median(gate_deltas) if gate_deltas else float("nan")

    strong = (
        conf_win >= max(1, int(0.8 * n)) and gate_win >= max(1, int(0.8 * n))
        and conf_med > 0 and gate_med > 0
    )
    negative = conf_win <= n / 2 and gate_win <= n / 2
    if strong:
        read = (
            "**Strong positive** — CellV0.3 shows a consistent corruption-AUC "
            "advantage over BOTH reliability-aware baselines."
        )
    elif negative:
        read = (
            "**Negative** — the confidence-aware / gated MLPs match or beat "
            "CellV0.3 across the reliability benchmark. Per the predeclared "
            "criteria: do NOT create CellV0.4 or tune V0.3; record the result."
        )
    else:
        read = (
            "**Weak / neutral** — CellV0.3 is not consistently ahead of BOTH "
            "reliability-aware baselines. Reliability information helps, but the "
            "special cell is not clearly necessary."
        )

    return [
        "## Predeclared go/no-go read (Sec 19)",
        "",
        f"Across {n} (dataset, corruption) cells, CellV0.3 corruption-AUC vs the "
        f"reliability-aware baselines: vs Confidence MLP median Δ {conf_med:+.4f} "
        f"(CellV0.3 ahead in {conf_win}/{n}); vs Reliability-Gated MLP median Δ "
        f"{gate_med:+.4f} (ahead in {gate_win}/{n}). CellV0.3 has the smaller "
        f"OOD degradation vs Confidence MLP in {ood_better_conf}/{n} cells and "
        f"vs Reliability-Gated MLP in {ood_better_gate}/{n}. (Per-cell deltas in "
        "the compact summary and Table C; a large regression Δ where a baseline "
        "is unstable is a real finding, not a CellV0.3 advantage — see "
        "Failures.)",
        "",
        read,
        "",
        "_This is the evidence as recorded. Do not write the paper yet; do not "
        "alter V0.3 after these results (Sec 18)._",
        "",
    ]


def build_report(raw_dir: Path) -> str:
    rows = load_records(raw_dir)
    present = _present_datasets(rows)
    lines = section_header(rows, present)
    if not rows:
        lines.append(
            "_No `paper_a_reliability` records found — run "
            "`python experiments/paper_a/reliability/run_reliability.py`._"
        )
        return "\n".join(lines) + "\n"
    cells = by_cell(rows)
    lines += section_compact(cells, present)
    lines += _severity_table(cells, present, MISSING, "A", "Missingness")
    lines += _severity_table(cells, present, GAUSSIAN, "B", "Heterogeneous Gaussian noise")
    lines += section_paired(cells, present)
    lines += section_ood(cells, present)
    lines += section_efficiency(cells, present)
    lines += section_belief(cells, present)
    lines += section_interventions(cells)
    lines += section_verdict(cells, present)
    lines += section_failures(rows, cells, present)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_HERE / "results" / "raw"))
    parser.add_argument("--out", default=str(_HERE / "reliability_results.md"))
    args = parser.parse_args()
    report = build_report(Path(args.raw_dir))
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
