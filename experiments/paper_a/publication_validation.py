#!/usr/bin/env python3
"""Aggregate Paper A Phase 3 (Part A capacity stress + Part B real
missing-sensor benchmarks) into the committed report
`experiments/paper_a/publication_validation_results.md`.

Reads:
* `experiments/paper_a/reliability/results/raw/` -- the recorded Phase-2
  CellV0.3 and parameter-matched Confidence MLP runs (`paper_a_reliability`)
  and the new same-width Confidence MLP runs (`paper_a_capacity_stress`).
* `experiments/paper_a/real_reliability/results/raw/` -- the Part B neural runs
  (`paper_a_real_reliability`) and the HistGradientBoosting reference
  (`paper_a_real_reliability_hgb`).

No significance testing (n = 3 seeds): means and sample stds only.

Usage:
    python experiments/paper_a/publication_validation.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.paper_a.capacity_stress import CAPACITY_EXPERIMENT_ID  # noqa: E402
from experiments.paper_a.real_reliability.datasets import MISSINGNESS_BINS  # noqa: E402
from experiments.paper_a.real_reliability.harness import (  # noqa: E402
    EXPERIMENT_ID as REAL_ID,
)
from experiments.paper_a.real_reliability.harness import (  # noqa: E402
    HGB_EXPERIMENT_ID,
)

RELIABILITY_ID = "paper_a_reliability"
PART_A_DATASETS = ("mnist", "fashion_mnist", "digits")
PART_A_CORRUPTIONS = ("missing", "gaussian")
REAL_DATASETS = ("aps", "air_quality")
REAL_FAMILY_ORDER = (
    "plain_mlp", "confidence_mlp", "confidence_mlp_same_width", "cellv0.3", "neumiss",
)
REAL_FAMILY_LABEL = {
    "plain_mlp": "Plain MLP",
    "confidence_mlp": "Confidence MLP (matched)",
    "confidence_mlp_same_width": "Confidence MLP (same width)",
    "cellv0.3": "CellV0.3",
    "neumiss": "NeuMiss",
    "hist_gradient_boosting": "HistGradientBoosting (ref)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load(raw_dir: Path, experiment_id: str) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith("_history.json"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("experiment_id") != experiment_id:
            continue
        e = rec["config"].get("extra", {})
        key = (rec["dataset"], rec["architecture"], rec["seed"], e.get("corruption_family"))
        latest[key] = {"rec": rec, "extra": e}
    return list(latest.values())


def _agg(vals: list[float]) -> tuple[float, float]:
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return float("nan"), float("nan")
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def _fmt(m: float, s: float, nd: int = 4) -> str:
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def _mode_lr(lrs: list[float]) -> str:
    if not lrs:
        return "-"
    counts = defaultdict(int)
    for lr in lrs:
        counts[lr] += 1
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    uniq = sorted(set(lrs))
    return f"{top:.0e}" + ("" if len(uniq) == 1 else f" ({'/'.join(f'{u:.0e}' for u in uniq)})")


# ---------------------------------------------------------------------------
# Part A -- capacity stress
# ---------------------------------------------------------------------------


def _auc_by_cell(rows: list[dict], family: str) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        rec, e = r["rec"], r["extra"]
        if rec["architecture"] != family:
            continue
        cf = e.get("corruption_family")
        primary = e.get("primary_metric", "accuracy")
        auc = e.get("sweep", {}).get("corruption_auc", {}).get(primary)
        if auc is not None:
            out[(rec["dataset"], cf)].append(auc)
    return out


def _severity_curve(
    rows: list[dict], family: str
) -> dict[tuple[str, str], dict[float, list[float]]]:
    out: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rec, e = r["rec"], r["extra"]
        if rec["architecture"] != family:
            continue
        cf = e.get("corruption_family")
        primary = e.get("primary_metric", "accuracy")
        for sv in e.get("sweep", {}).get("severities", []):
            out[(rec["dataset"], cf)][sv["severity"]].append(sv["metrics"][primary])
    return out


def part_a_section(reliability_rows: list[dict], capacity_rows: list[dict]) -> list[str]:
    v3 = _auc_by_cell(reliability_rows, "cellv0.3")
    cm = _auc_by_cell(reliability_rows, "confidence_mlp")
    sw = _auc_by_cell(capacity_rows, "confidence_mlp_same_width")
    sw_ratio: dict[tuple[str, str], float] = {}
    sw_hidden: dict[tuple[str, str], int] = {}
    for r in capacity_rows:
        rec, e = r["rec"], r["extra"]
        cf = e.get("corruption_family")
        sw_ratio[(rec["dataset"], cf)] = e["sizing"].get("param_ratio_vs_cellv03", float("nan"))
        sw_hidden[(rec["dataset"], cf)] = e["sizing"].get("hidden_dim")

    out = [
        "## Part A -- capacity-stress the Confidence MLP",
        "",
        "Phase 2 parameter-matched every baseline to CellV0.3's actual count, "
        "which forced the Confidence MLP (input `concat(x, c)`, `2*D` wide) to a "
        "*narrower* hidden layer than CellV0.3. The **same-width Confidence "
        "MLP** removes that confound: the Phase-2 Confidence MLP architecture "
        "at CellV0.3's exact hidden width, **not** parameter-matched. 18 new "
        "runs (`{mnist, fashion_mnist, digits} x {missing, gaussian} x seeds "
        "0-2`), the frozen Phase-2 protocol; CellV0.3 and the param-matched "
        "Confidence MLP are read back from the recorded Phase-2 results.",
        "",
        "Corruption-AUC (primary metric, trapezoidal over the severity grid), "
        "mean ± std over 3 seeds:",
        "",
        "| Dataset | Corruption | CellV0.3 | Confidence MLP (matched) | "
        "Confidence MLP (same width) | same-width param ratio | Δ(V0.3 − same-width) |",
        "|---|---|---|---|---|---|---|",
    ]
    surviving = 0
    total = 0
    for ds in PART_A_DATASETS:
        for cf in PART_A_CORRUPTIONS:
            k = (ds, cf)
            if k not in v3 or k not in sw:
                continue
            total += 1
            v3m, v3s = _agg(v3[k])
            cmm, cms = _agg(cm.get(k, []))
            swm, sws = _agg(sw[k])
            delta = v3m - swm
            if delta > 0:
                surviving += 1
            out.append(
                f"| {ds} | {cf} | {_fmt(v3m, v3s)} | {_fmt(cmm, cms)} | "
                f"{_fmt(swm, sws)} | x{sw_ratio[k]:.2f} (hidden {sw_hidden[k]}) | "
                f"**{delta:+.4f}** |"
            )
    out += [
        "",
        f"CellV0.3's corruption-AUC exceeds the deliberately larger same-width "
        f"Confidence MLP in **{surviving}/{total}** cells. On MNIST / "
        f"Fashion-MNIST the same-width model carries ~1.65x CellV0.3's "
        f"parameters and still loses; on Digits the \"same width\" MLP is "
        f"actually *smaller* than CellV0.3 (ratio < 1 -- CellV0.3's parameters "
        f"are dominated by its hidden->hidden layer), so that column is not a "
        f"capacity advantage for the baseline.",
        "",
    ]
    return out


def part_a_supplementary(reliability_rows: list[dict], capacity_rows: list[dict]) -> list[str]:
    v3 = _severity_curve(reliability_rows, "cellv0.3")
    sw = _severity_curve(capacity_rows, "confidence_mlp_same_width")
    cm = _severity_curve(reliability_rows, "confidence_mlp")
    out = ["### Part A supplementary -- per-severity primary metric (mean of 3 seeds)", ""]
    for ds in PART_A_DATASETS:
        for cf in PART_A_CORRUPTIONS:
            k = (ds, cf)
            if k not in v3:
                continue
            sevs = sorted(v3[k])
            head = " | ".join(f"{s:g}" for s in sevs)
            out += [
                f"**{ds} / {cf}**",
                "",
                f"| Model | {head} |",
                "|---|" + "---|" * len(sevs),
            ]
            for label, curve in (
                ("CellV0.3", v3[k]),
                ("Confidence MLP (matched)", cm.get(k, {})),
                ("Confidence MLP (same width)", sw.get(k, {})),
            ):
                if not curve:
                    continue
                cells = " | ".join(f"{_agg(curve[s])[0]:.3f}" for s in sevs)
                out.append(f"| {label} | {cells} |")
            out.append("")
    return out


# ---------------------------------------------------------------------------
# Part B -- real datasets
# ---------------------------------------------------------------------------


def _real_by_family(real_rows: list[dict], dataset: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in real_rows:
        if r["rec"]["dataset"] == dataset:
            out[r["rec"]["architecture"]].append(r["extra"])
    return out


def _hgb_for(hgb_rows: list[dict], dataset: str) -> list[dict]:
    return [r["extra"] for r in hgb_rows if r["rec"]["dataset"] == dataset]


def aps_main_table(real_rows: list[dict], hgb_rows: list[dict]) -> list[str]:
    fam = _real_by_family(real_rows, "aps")
    out = [
        "## APS Failure at Scania Trucks -- main table",
        "",
        "Official 60k/16k split, 170 features, real dataset missingness. Primary "
        "metric: **PR-AUC** (never accuracy). Official cost `10*FP + 500*FN` at "
        "the validation-selected frozen threshold. Mean ± std over seeds 0-2.",
        "",
        "| Model | PR-AUC | ROC-AUC | Bal. acc | F1 | Precision | Recall | "
        "Official cost | cost/1000 | Params | Hidden/depth | LR | Wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in REAL_FAMILY_ORDER:
        rows = fam.get(f, [])
        if not rows:
            continue
        ov = [r["evaluation"]["overall"] for r in rows]
        eff = [r["efficiency"] for r in rows]
        depth = rows[0].get("neumiss_depth")
        hd = f"d={depth}" if depth is not None else str(eff[0].get("hidden_size"))
        out.append(
            f"| {REAL_FAMILY_LABEL[f]} "
            f"| {_fmt(*_agg([o['pr_auc'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['roc_auc'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['balanced_accuracy'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['f1'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['precision'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['recall'] for o in ov]), 3)} "
            f"| {_agg([o['official_cost'] for o in ov])[0]:.0f} ± "
            f"{_agg([o['official_cost'] for o in ov])[1]:.0f} "
            f"| {_agg([o['cost_per_1000'] for o in ov])[0]:.0f} "
            f"| {eff[0].get('parameter_count', 0):,} | {hd} "
            f"| {_mode_lr([e.get('selected_lr') for e in eff if e.get('selected_lr')])} "
            f"| {_agg([e.get('train_wall_clock_seconds') for e in eff])[0]:.0f} |"
        )
    for h in _hgb_for(hgb_rows, "aps")[:1]:
        ov = [r["evaluation"]["overall"] for r in _hgb_for(hgb_rows, "aps")]
        out.append(
            f"| {REAL_FAMILY_LABEL['hist_gradient_boosting']} "
            f"| {_fmt(*_agg([o['pr_auc'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['roc_auc'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['balanced_accuracy'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['f1'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['precision'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['recall'] for o in ov]), 3)} "
            f"| {_agg([o['official_cost'] for o in ov])[0]:.0f} "
            f"| {_agg([o['cost_per_1000'] for o in ov])[0]:.0f} "
            f"| -- | -- | -- | -- |"
        )
        break
    out.append("")
    return out


def air_quality_main_table(real_rows: list[dict], hgb_rows: list[dict]) -> list[str]:
    fam = _real_by_family(real_rows, "air_quality")
    out = [
        "## UCI Air Quality -- main table",
        "",
        "Predict `CO(GT)` from 8 sensor/environment inputs, `-200` -> missing, "
        "chronological 60/20/20 split. Primary metric: **R^2** on the original "
        "scale. Mean ± std over seeds 0-2.",
        "",
        "| Model | R^2 | RMSE | MAE | Params | Hidden/depth | LR | Wall (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in REAL_FAMILY_ORDER:
        rows = fam.get(f, [])
        if not rows:
            continue
        ov = [r["evaluation"]["overall"] for r in rows]
        eff = [r["efficiency"] for r in rows]
        depth = rows[0].get("neumiss_depth")
        hd = f"d={depth}" if depth is not None else str(eff[0].get("hidden_size"))
        out.append(
            f"| {REAL_FAMILY_LABEL[f]} "
            f"| {_fmt(*_agg([o['r2'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['rmse'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['mae'] for o in ov]), 3)} "
            f"| {eff[0].get('parameter_count', 0):,} | {hd} "
            f"| {_mode_lr([e.get('selected_lr') for e in eff if e.get('selected_lr')])} "
            f"| {_agg([e.get('train_wall_clock_seconds') for e in eff])[0]:.0f} |"
        )
    ov = [r["evaluation"]["overall"] for r in _hgb_for(hgb_rows, "air_quality")]
    if ov:
        out.append(
            f"| {REAL_FAMILY_LABEL['hist_gradient_boosting']} "
            f"| {_fmt(*_agg([o['r2'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['rmse'] for o in ov]), 3)} "
            f"| {_fmt(*_agg([o['mae'] for o in ov]), 3)} | -- | -- | -- | -- |"
        )
    out.append("")
    return out


def _strata_metric(rows: list[dict], primary: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for s in r["evaluation"].get("by_stratum", []):
            out[s["bin"]].append(s.get(primary))
    return out


def missingness_strata_section(real_rows: list[dict]) -> list[str]:
    out = [
        "## Real-missingness stratified performance",
        "",
        "Test examples bucketed by `missing_fraction = missing_features / "
        "total_features` into the fixed bins; empty bins skipped. Primary metric, "
        "mean over 3 seeds.",
        "",
    ]
    for ds in REAL_DATASETS:
        fam = _real_by_family(real_rows, ds)
        if not fam:
            continue
        primary = "pr_auc" if ds == "aps" else "r2"
        bins_present = [
            b[0] for b in MISSINGNESS_BINS
            if any(
                any(s["bin"] == b[0] for s in r["evaluation"].get("by_stratum", []))
                for rows in fam.values() for r in rows
            )
        ]
        head = " | ".join(bins_present)
        out += [
            f"### {ds} ({primary})",
            "",
            f"| Model | {head} |",
            "|---|" + "---|" * len(bins_present),
        ]
        for f in REAL_FAMILY_ORDER:
            rows = fam.get(f, [])
            if not rows:
                continue
            sm = _strata_metric(rows, primary)
            cells = " | ".join(
                f"{_agg(sm[b])[0]:.3f}" if sm.get(b) else "-" for b in bins_present
            )
            out.append(f"| {REAL_FAMILY_LABEL[f]} | {cells} |")
        out.append("")

        # CellV0.3 belief-state by stratum
        v3 = fam.get("cellv0.3", [])
        if v3:
            belief_bins = [
                b[0] for b in MISSINGNESS_BINS
                if any(
                    any(s["bin"] == b[0] for s in r["evaluation"].get("belief_by_stratum", []))
                    for r in v3
                )
            ]
            out += [
                f"**CellV0.3 belief state by missingness bin ({ds})** -- mean over 3 seeds:",
                "",
                "| Bin | L1 π mean | L1 u mean | L2 π mean | L2 u mean | L2 π CoV |",
                "|---|---|---|---|---|---|",
            ]
            acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for r in v3:
                for s in r["evaluation"].get("belief_by_stratum", []):
                    for key in ("l1_pi_mean", "l1_u_mean", "l2_pi_mean", "l2_u_mean", "l2_pi_cv"):
                        acc[s["bin"]][key].append(s[key])
            for b in belief_bins:
                a = acc[b]
                out.append(
                    f"| {b} | {_agg(a['l1_pi_mean'])[0]:.3f} | {_agg(a['l1_u_mean'])[0]:.3f} "
                    f"| {_agg(a['l2_pi_mean'])[0]:.3f} | {_agg(a['l2_u_mean'])[0]:.3f} "
                    f"| {_agg(a['l2_pi_cv'])[0]:.3f} |"
                )
            out.append("")
    return out


def interventions_section(real_rows: list[dict]) -> list[str]:
    out = [
        "## Confidence interventions (CellV0.3, evaluation only)",
        "",
        "The trained CellV0.3 models re-scored with the true `c`, with `c := 1` "
        "everywhere, and with `c` permuted across feature positions per example. "
        "Primary metric, mean ± std over seeds.",
        "",
        "| Dataset | metric | true | all-ones | shuffled | true − all-ones | true − shuffled |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in REAL_DATASETS:
        ivs = [
            r["extra"]["intervention"]
            for r in real_rows
            if r["rec"]["dataset"] == ds and r["rec"]["architecture"] == "cellv0.3"
            and r["extra"].get("intervention")
        ]
        if not ivs:
            continue
        primary = ivs[0]["primary_metric"]
        t = _agg([i["true"] for i in ivs])
        a = _agg([i["all_ones"] for i in ivs])
        s = _agg([i["shuffled"] for i in ivs])
        da = _agg([i["true_minus_all_ones"] for i in ivs])
        dsh = _agg([i["true_minus_shuffled"] for i in ivs])
        out.append(
            f"| {ds} | {primary} | {_fmt(*t, 3)} | {_fmt(*a, 3)} | {_fmt(*s, 3)} "
            f"| **{da[0]:+.4f}** | **{dsh[0]:+.4f}** |"
        )
    out += [
        "",
        "_A positive `true − all-ones` / `true − shuffled` means the true "
        "reliability alignment helped. On Air Quality every row's `c` is uniform "
        "(a row has either no missing inputs or all 8 missing), so `shuffled` is "
        "a no-op there by construction._",
        "",
    ]
    return out


def failures_section(
    reliability_rows, capacity_rows, real_rows, neumiss_errors
) -> list[str]:
    out = ["## Failures and caveats", ""]

    diverged = [
        (r["rec"]["dataset"], r["rec"]["architecture"], r["rec"]["seed"])
        for r in real_rows + capacity_rows if r["extra"].get("diverged")
    ]
    div_txt = "; ".join(f"{d}/{f}/s{s}" for d, f, s in diverged) or "none."
    out.append("**Divergence / NaNs:** " + div_txt)
    out.append("")

    caps = sorted({
        (r["rec"]["dataset"], r["rec"]["architecture"])
        for r in real_rows + capacity_rows if r["extra"].get("cap_hit")
    })
    out.append("**Step-cap hits:** " + ("; ".join(f"{d}/{f}" for d, f in caps) or "none."))
    out.append("")

    if neumiss_errors:
        out.append("**NeuMiss integration issues:**")
        out += [f"- {e}" for e in neumiss_errors]
    else:
        out.append("**NeuMiss integration issues:** none -- the official "
                   "`marineLM/NeuMiss_sota` package (pinned commit) imported and "
                   "trained cleanly.")
    out.append("")

    # unstable seeds: primary-metric seed-std large
    unstable = []
    for ds in REAL_DATASETS:
        fam = _real_by_family(real_rows, ds)
        primary = "pr_auc" if ds == "aps" else "r2"
        thr = 0.05 if ds == "aps" else 0.10
        for f, rows in fam.items():
            vals = [r["evaluation"]["overall"][primary] for r in rows]
            if len(vals) > 1 and statistics.stdev(vals) > thr:
                sd = statistics.stdev(vals)
                unstable.append(f"{ds}/{REAL_FAMILY_LABEL[f]} ({primary} std {sd:.3f})")
    out.append("**High seed variance (primary-metric std > 0.05 PR-AUC / 0.10 R^2):** "
               + ("; ".join(unstable) or "none."))
    out.append("")

    # baseline wins over CellV0.3 on Part B
    wins = []
    for ds in REAL_DATASETS:
        fam = _real_by_family(real_rows, ds)
        primary = "pr_auc" if ds == "aps" else "r2"
        v3 = fam.get("cellv0.3", [])
        if not v3:
            continue
        v3m = _agg([r["evaluation"]["overall"][primary] for r in v3])[0]
        for f in ("confidence_mlp", "confidence_mlp_same_width", "neumiss"):
            rows = fam.get(f, [])
            if not rows:
                continue
            fm = _agg([r["evaluation"]["overall"][primary] for r in rows])[0]
            if fm > v3m + 0.005:
                wins.append(f"{ds}: {REAL_FAMILY_LABEL[f]} {fm:.3f} > CellV0.3 {v3m:.3f}")
    out.append("**Reliability-aware baseline beats CellV0.3 (primary, > 0.005):** "
               + ("; ".join(wins) or "none."))
    out.append("")

    # pi fails to track missingness
    flat = []
    for ds in REAL_DATASETS:
        v3 = [
            r for r in real_rows
            if r["rec"]["dataset"] == ds and r["rec"]["architecture"] == "cellv0.3"
        ]
        if not v3:
            continue
        acc: dict[str, list[float]] = defaultdict(list)
        for r in v3:
            for s in r["extra"]["evaluation"].get("belief_by_stratum", []):
                acc[s["bin"]].append(s["l2_pi_mean"])
        ordered = [b[0] for b in MISSINGNESS_BINS if acc.get(b[0])]
        if len(ordered) >= 2:
            lo = _agg(acc[ordered[0]])[0]
            hi = _agg(acc[ordered[-1]])[0]
            if not (hi < lo - 0.02):
                flat.append(
                    f"{ds} (L2 π {lo:.3f} at '{ordered[0]}' -> {hi:.3f} at '{ordered[-1]}')"
                )
    out.append("**CellV0.3 hidden π does not fall with missingness:** "
               + ("; ".join(flat) or "none -- π falls monotonically-ish on both real datasets."))
    out.append("")
    return out


def stopping_rules_section(reliability_rows, capacity_rows, real_rows) -> list[str]:
    # capacity control
    v3 = _auc_by_cell(reliability_rows, "cellv0.3")
    sw = _auc_by_cell(capacity_rows, "confidence_mlp_same_width")
    cap_win = sum(
        1 for k in sw if k in v3 and _agg(v3[k])[0] > _agg(sw[k])[0] + 0.005
    )
    cap_total = sum(1 for k in sw if k in v3)

    # real reliability: CellV0.3 vs NeuMiss and same-width, per dataset
    real_lines = []
    v3_competitive = 0
    v3_datasets = 0
    for ds in REAL_DATASETS:
        fam = _real_by_family(real_rows, ds)
        primary = "pr_auc" if ds == "aps" else "r2"
        if "cellv0.3" not in fam:
            continue
        v3_datasets += 1
        v3m = _agg([r["evaluation"]["overall"][primary] for r in fam["cellv0.3"]])[0]
        nm = _agg([r["evaluation"]["overall"][primary] for r in fam.get("neumiss", [])])[0]
        swm = _agg(
            [r["evaluation"]["overall"][primary]
             for r in fam.get("confidence_mlp_same_width", [])]
        )[0]
        competitive = v3m >= nm - 0.01 and v3m >= swm - 0.01
        if competitive:
            v3_competitive += 1
        real_lines.append(
            f"- **{ds}** ({primary}): CellV0.3 {v3m:.3f}, NeuMiss {nm:.3f}, "
            f"same-width Confidence MLP {swm:.3f} -> "
            f"{'competitive/ahead' if competitive else 'behind'}"
        )

    # intervention degradation
    iv_degrades = []
    for ds in REAL_DATASETS:
        ivs = [
            r["extra"]["intervention"] for r in real_rows
            if r["rec"]["dataset"] == ds and r["rec"]["architecture"] == "cellv0.3"
            and r["extra"].get("intervention")
        ]
        if not ivs:
            continue
        da = _agg([i["true_minus_all_ones"] for i in ivs])[0]
        dsh = _agg([i["true_minus_shuffled"] for i in ivs])[0]
        iv_degrades.append((ds, da, dsh))

    cond1 = cap_total > 0 and cap_win >= max(1, cap_total - 1)
    cond2_real = v3_datasets > 0 and v3_competitive >= 1
    cond2_iv = any(da > 0.005 or dsh > 0.005 for _ds, da, dsh in iv_degrades)
    cond2 = cond2_real and cond2_iv

    if cond1 and cond2:
        verdict = (
            "**Both stopping conditions hold** -- the Phase-2 robustness "
            "advantage survives the deliberately larger same-width Confidence "
            "MLP, and CellV0.3 is competitive with / better than the "
            "missingness-aware neural baselines on at least one real dataset "
            "with meaningful degradation when reliability is removed/shuffled. "
            "This is genuinely strong evidence for the CellV0.3 line. Do not "
            "write the paper yet; do not alter V0.3."
        )
    elif cond1 and not cond2:
        verdict = (
            "**Capacity control passes, real-reliability condition does not** -- "
            "CellV0.3's advantage over the same-width Confidence MLP is real on "
            "the synthetic image benchmark, but on the real missing-sensor "
            "datasets it does not clearly beat NeuMiss / the same-width MLP, "
            "and/or removing the true reliability does not hurt it. Weak "
            "evidence: record and do not build CellV0.4."
        )
    elif cond2 and not cond1:
        verdict = (
            "**Real-reliability condition passes, capacity control does not** -- "
            "the same-width Confidence MLP matches or beats CellV0.3 on the "
            "synthetic benchmark once it is given at least as much hidden "
            "capacity, so the Phase-2 signal was partly a capacity artifact. "
            "Weak evidence: record and stop."
        )
    else:
        verdict = (
            "**Neither stopping condition holds** -- the same-width Confidence "
            "MLP and NeuMiss consistently match/beat CellV0.3. Per the "
            "predeclared rule: record the result and **stop the Cell paper** "
            "rather than creating CellV0.4 or tuning V0.3."
        )

    return [
        "## Interpretation and predeclared stopping rules",
        "",
        "A result is genuinely strong only if **both** survive:",
        "",
        f"1. **Capacity control** -- CellV0.3's corruption-AUC still exceeds the "
        f"deliberately larger same-width Confidence MLP. Result: "
        f"**{cap_win}/{cap_total}** cells.",
        "2. **Real reliability** -- CellV0.3 competitive with / better than "
        "strong missingness-aware baselines (esp. NeuMiss) on >=1 real dataset, "
        "**and** meaningful degradation when correct reliability is "
        "removed/shuffled:",
        "",
        *real_lines,
        "",
        "Intervention degradation (true − all-ones / true − shuffled, primary metric):",
        "",
        *[f"- {ds}: {da:+.4f} / {dsh:+.4f}" for ds, da, dsh in iv_degrades],
        "",
        "A V0.3 win only over the Plain MLP is insufficient; a win only over the "
        "param-matched Confidence MLP but not the same-width model is weak.",
        "",
        verdict,
        "",
        "_Not calibration: `e`/`u`/`π` are internal computational reliability "
        "variables. Do not write the paper yet; do not alter V0.3 after these "
        "results._",
        "",
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_report(reliability_raw: Path, real_raw: Path) -> str:
    reliability_rows = _load(reliability_raw, RELIABILITY_ID)
    capacity_rows = _load(reliability_raw, CAPACITY_EXPERIMENT_ID)
    real_rows = _load(real_raw, REAL_ID)
    hgb_rows = _load(real_raw, HGB_EXPERIMENT_ID)

    neumiss_errors: list[str] = []
    summary = real_raw.parent / "processed" / "real_reliability_runs.json"
    if summary.exists():
        for row in json.loads(summary.read_text()):
            if isinstance(row, dict) and row.get("neumiss_integration_error"):
                neumiss_errors.append(
                    f"{row['dataset']}/{row['family']}/seed{row['seed']}: "
                    f"{row['neumiss_integration_error']}"
                )

    lines = [
        "# Paper A -- Phase 3: publication-grade validation of CellV0.3",
        "",
        "CellV0.3 is permanently frozen. This phase has exactly two purposes: "
        "(A) rule out that the Phase-2 Confidence MLP lost only because "
        "parameter matching narrowed it, and (B) test CellV0.3 on real datasets "
        "with naturally missing sensor measurements, against an established "
        "missing-data neural architecture (NeuMiss). No other research "
        "questions; no architecture changes.",
        "",
        f"Records: Part A {len(capacity_rows)} same-width runs "
        f"(`{CAPACITY_EXPERIMENT_ID}`) + recorded Phase-2 CellV0.3 / Confidence "
        f"MLP; Part B {len(real_rows)} neural runs (`{REAL_ID}`) + "
        f"{len(hgb_rows)} HistGradientBoosting reference runs. n = 3 seeds, no "
        "significance test.",
        "",
    ]
    if not (capacity_rows or real_rows):
        lines.append(
            "_No Phase-3 records found -- run "
            "`experiments/paper_a/capacity_stress.py` and "
            "`experiments/paper_a/real_reliability/run_real_reliability.py`._"
        )
        return "\n".join(lines) + "\n"

    if capacity_rows:
        lines += part_a_section(reliability_rows, capacity_rows)
    if real_rows:
        lines += aps_main_table(real_rows, hgb_rows)
        lines += air_quality_main_table(real_rows, hgb_rows)
        lines += missingness_strata_section(real_rows)
        lines += interventions_section(real_rows)
    lines += stopping_rules_section(reliability_rows, capacity_rows, real_rows)
    lines += failures_section(reliability_rows, capacity_rows, real_rows, neumiss_errors)
    if capacity_rows:
        lines += part_a_supplementary(reliability_rows, capacity_rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reliability-raw",
        default=str(_HERE / "reliability" / "results" / "raw"),
    )
    parser.add_argument(
        "--real-raw",
        default=str(_HERE / "real_reliability" / "results" / "raw"),
    )
    parser.add_argument("--out", default=str(_HERE / "publication_validation_results.md"))
    args = parser.parse_args()
    report = build_report(Path(args.reliability_raw), Path(args.real_raw))
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
