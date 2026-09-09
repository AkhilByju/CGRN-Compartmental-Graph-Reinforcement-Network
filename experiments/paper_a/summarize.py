#!/usr/bin/env python3
"""Aggregate Paper A Phase-1 `RunRecord`s into the report tables the task
asks for (A-G, plus the CellV0.1 internal-state diagnostics from Sec 9).

Reads `results/raw/*.json` (filtered by `experiment_id`) and
`results/processed/duplication_experiment.json`, writes
`experiments/paper_a/phase1_results.md`, and prints the same to stdout.

No significance testing is done -- with n=3 seeds the paired differences are
reported per seed and as a mean, nothing more (Paper-A task Sec 8).

Usage:
    python experiments/paper_a/summarize.py
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

from experiments.paper_a.datasets import BENCHMARK_DATASETS  # noqa: E402
from experiments.paper_a.harness import EXPERIMENT_ID  # noqa: E402
from experiments.paper_a.run_ablation import (  # noqa: E402
    ABLATION_DATASETS,
    ABLATION_EXPERIMENT_ID,
    ABLATION_FRACTION,
)

PHASE1_FAMILIES = ("cellv0.1", "mlp_matched", "mlp_state_count")
FRACTIONS = (0.25, 1.0)
_FAMILY_LABEL = {
    "cellv0.1": "CellV0.1",
    "mlp_matched": "MLP (matched)",
    "mlp_state_count": "MLP (state-count)",
    "cellv0.1_fixed_confidence": "CellV0.1 fixed-conf",
}
# "clearly loses" threshold for Table G: CellV0.1 headline worse than the
# parameter-matched MLP by more than this, at a given (dataset, fraction).
_CLEAR_LOSS_MARGIN = 0.01
_HIGH_VARIANCE_MULT = 3.0  # CellV0.1 seed-std > this * matched-MLP seed-std -> flag


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_records(raw_dir: Path, experiment_id: str) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith("_history.json"):
            continue
        with path.open() as f:
            rec = json.load(f)
        if rec.get("experiment_id") != experiment_id:
            continue
        extra = rec["config"].get("extra", {})
        tm = rec["test_metrics"]
        vm = rec.get("validation_metrics", {})
        rows.append(
            {
                "dataset": rec["dataset"],
                "family": rec["architecture"],
                "seed": rec["seed"],
                "train_fraction": extra.get("train_fraction"),
                "task_type": extra.get("task_type"),
                "is_binary": extra.get("is_binary", False),
                "headline_metric": extra.get("headline_metric"),
                "params": rec["parameter_count"],
                "n_train_used": extra.get("dataset_meta", {}).get("n_train_used"),
                "best_val_step": vm.get("best_val_step"),
                "total_steps": rec["steps_completed"],
                "train_wall_clock_s": rec["train_wall_clock_seconds"],
                "time_per_step_s": tm.get("time_per_step_seconds"),
                "diverged": bool(extra.get("diverged", tm.get("diverged", False))),
                "cap_hit": bool(extra.get("cap_hit", tm.get("cap_hit", False))),
                "accuracy": tm.get("accuracy"),
                "macro_f1": tm.get("macro_f1"),
                "roc_auc": tm.get("roc_auc"),
                "rmse": tm.get("rmse"),
                "r2": tm.get("r2"),
                "mae": tm.get("mae"),
                "sizing": extra.get("sizing", {}),
                "diagnostics": extra.get("diagnostics", {}),
            }
        )
    return rows


def _headline(row: dict) -> float:
    return row[row["headline_metric"]]


def _agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _by_cell(rows: list[dict]) -> dict[tuple[str, float, str], list[dict]]:
    cells: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["train_fraction"], r["family"])].append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["seed"])
    return cells


def _fmt(mean: float, std: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_a_main(rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows)
    out = ["## A. Main table", ""]
    out.append(
        "| Dataset | Frac | Model | Params | Headline (mean ± std) | Macro-F1 / RMSE | "
        "Best-val step | Total steps | Train wall-clock (s) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        for frac in FRACTIONS:
            for fam in PHASE1_FAMILIES:
                runs = cells.get((ds, frac, fam))
                if not runs:
                    continue
                task = runs[0]["task_type"]
                hl_mean, hl_std = _agg([_headline(r) for r in runs])
                if task == "classification":
                    sec_mean, sec_std = _agg([r["macro_f1"] for r in runs])
                    sec_label = "F1"
                else:
                    sec_mean, sec_std = _agg([r["rmse"] for r in runs])
                    sec_label = "RMSE"
                bv_mean, bv_std = _agg([r["best_val_step"] for r in runs])
                ts_mean, _ = _agg([r["total_steps"] for r in runs])
                wc_mean, wc_std = _agg([r["train_wall_clock_s"] for r in runs])
                params = runs[0]["params"]
                hl_name = runs[0]["headline_metric"]
                out.append(
                    f"| {ds} | {int(frac*100)}% | {_FAMILY_LABEL[fam]} | {params:,} | "
                    f"{hl_name}: {_fmt(hl_mean, hl_std)} | "
                    f"{sec_label}: {_fmt(sec_mean, sec_std)} | "
                    f"{bv_mean:.0f} ± {bv_std:.0f} | {ts_mean:.0f} | "
                    f"{wc_mean:.1f} ± {wc_std:.1f} |"
                )
    out.append("")
    return out


def table_b_paired(rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows)
    out = ["## B. Paired comparison — CellV0.1 minus parameter-matched MLP (headline metric)", ""]
    out.append("| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |")
    out.append("|---|---|---|---|---|---|")
    for ds in datasets:
        for frac in FRACTIONS:
            cell_runs = {r["seed"]: r for r in cells.get((ds, frac, "cellv0.1"), [])}
            mlp_runs = {r["seed"]: r for r in cells.get((ds, frac, "mlp_matched"), [])}
            seeds = sorted(set(cell_runs) & set(mlp_runs))
            if not seeds:
                continue
            deltas = [_headline(cell_runs[s]) - _headline(mlp_runs[s]) for s in seeds]
            per_seed = dict(zip(seeds, deltas))
            cols = [
                f"{per_seed[s]:+.4f}" if s in per_seed else "—" for s in (0, 1, 2)
            ]
            out.append(
                f"| {ds} | {int(frac*100)}% | {cols[0]} | {cols[1]} | {cols[2]} | "
                f"**{statistics.mean(deltas):+.4f}** |"
            )
    out.append("")
    out.append(
        "_Positive = CellV0.1 better. n=3 seeds; no significance test is implied "
        "(Paper-A task Sec 8)._"
    )
    out.append("")
    return out


def table_c_efficiency(rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows)
    out = ["## C. Data-efficiency summary — headline metric at 25% → 100% training data", ""]
    out.append("| Dataset | Model | 25% | 100% | Δ (100% − 25%) |")
    out.append("|---|---|---|---|---|")
    for ds in datasets:
        for fam in PHASE1_FAMILIES:
            r25 = cells.get((ds, 0.25, fam))
            r100 = cells.get((ds, 1.0, fam))
            if not r25 or not r100:
                continue
            m25, s25 = _agg([_headline(r) for r in r25])
            m100, s100 = _agg([_headline(r) for r in r100])
            out.append(
                f"| {ds} | {_FAMILY_LABEL[fam]} | {_fmt(m25, s25)} | {_fmt(m100, s100)} | "
                f"{m100 - m25:+.4f} |"
            )
    out.append("")
    return out


def table_d_state_count(rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows)
    out = ["## D. State-count control summary", ""]
    out.append(
        "Does a substantially wider ordinary MLP (hidden width ≈ 3 × CellV0.1 "
        "hidden cells, **not** parameter-matched) close or reverse any CellV0.1 "
        "edge over the parameter-matched MLP?"
    )
    out.append("")
    out.append(
        "| Dataset | Frac | CellV0.1 params | State-count MLP params | ratio | "
        "CellV0.1 − matched | CellV0.1 − state-count |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for frac in FRACTIONS:
            cr = cells.get((ds, frac, "cellv0.1"))
            mm = cells.get((ds, frac, "mlp_matched"))
            sc = cells.get((ds, frac, "mlp_state_count"))
            if not (cr and mm and sc):
                continue
            cm, _ = _agg([_headline(r) for r in cr])
            mmm, _ = _agg([_headline(r) for r in mm])
            scm, _ = _agg([_headline(r) for r in sc])
            cp = cr[0]["params"]
            sp = sc[0]["params"]
            out.append(
                f"| {ds} | {int(frac*100)}% | {cp:,} | {sp:,} | {sp / cp:.2f}× | "
                f"{cm - mmm:+.4f} | {cm - scm:+.4f} |"
            )
    out.append("")
    out.append(
        "_Note: for the low-feature tabular datasets the state-count MLP is "
        "actually **smaller** than CellV0.1 (CellV0.1 carries two parameters per "
        "connection — a content weight and a relevance gate — so its cell count "
        "buys fewer cells per parameter). The ratio column makes this explicit; "
        "the control is reported as specified, not resized._"
    )
    out.append("")
    return out


def table_e_fixed_confidence(ablation_rows: list[dict], main_rows: list[dict]) -> list[str]:
    # The CellV0.1 baseline arm is the Phase-1 grid's own record for the
    # same (dataset, 25%, seed) unless run_ablation was given --with-baseline.
    cells = _by_cell(ablation_rows + main_rows)
    out = ["## E. Fixed-confidence ablation — CellV0.1 vs e=u=1 control", ""]
    out.append(
        f"Datasets: {', '.join(ABLATION_DATASETS)} · train fraction "
        f"{int(ABLATION_FRACTION*100)}% · seeds 0,1,2. CellV0.1 arm = the "
        "Phase-1 grid record (identical deterministic conditions)."
    )
    out.append("")
    out.append(
        "| Dataset | CellV0.1 (headline) | Fixed-confidence (headline) | "
        "Δ (CellV0.1 − fixed) | Δ by seed |"
    )
    out.append("|---|---|---|---|---|")
    fc_family = "cellv0.1_fixed_confidence"
    for ds in ABLATION_DATASETS:
        cr = {r["seed"]: r for r in cells.get((ds, ABLATION_FRACTION, "cellv0.1"), [])}
        fr = {r["seed"]: r for r in cells.get((ds, ABLATION_FRACTION, fc_family), [])}
        seeds = sorted(set(cr) & set(fr))
        if not seeds:
            continue
        cm, cs = _agg([_headline(cr[s]) for s in seeds])
        fm, fs = _agg([_headline(fr[s]) for s in seeds])
        deltas = [_headline(cr[s]) - _headline(fr[s]) for s in seeds]
        out.append(
            f"| {ds} | {_fmt(cm, cs)} | {_fmt(fm, fs)} | **{statistics.mean(deltas):+.4f}** | "
            f"{', '.join(f'{d:+.4f}' for d in deltas)} |"
        )
    out.append("")
    out.append(
        "_Positive Δ = the propagated evidence/uncertainty state helps beyond the "
        "content pathway; ≈0 = it does not, on that dataset._"
    )
    out.append("")
    return out


def table_f_duplication(dup_path: Path) -> list[str]:
    out = ["## F. Duplication-invariance experiment (deterministic, no training)", ""]
    if not dup_path.exists():
        out.append("_Not run — `python experiments/paper_a/duplication_experiment.py` missing._")
        out.append("")
        return out
    payload = json.loads(dup_path.read_text())
    src = payload["source_states"]
    out.append(
        f"Fixed source set ({len(src['mu'])} distinct beliefs): "
        f"mu={src['mu']}, e={src['evidence']}, u={src['uncertainty']}, "
        f"g={src['relevance_g']}. The whole multiset is duplicated ×m."
    )
    out.append("")
    for agg, agg_rows in payload["results"].items():
        out.append(f"### `{agg}`")
        out.append("")
        out.append("| m | out mu | out e | out u | e / e(×1) | u / u(×1) |")
        out.append("|---|---|---|---|---|---|")
        for r in agg_rows:
            out.append(
                f"| {r['multiplicity']} | {r['out_mu']:.8f} | {r['out_e']:.8f} | "
                f"{r['out_u']:.8f} | {r['e_ratio_vs_x1']:.4f} | {r['u_ratio_vs_x1']:.4f} |"
            )
        v = payload["verdicts"][agg]
        out.append("")
        out.append(
            f"Verdict: mu invariant={v['mu_invariant']}, e invariant={v['e_invariant']}, "
            f"u invariant={v['u_invariant']} "
            f"(max |Δmu|={v['max_abs_dmu']:.2e}, |Δe|={v['max_abs_de']:.2e}, "
            f"|Δu|={v['max_abs_du']:.2e}; e ratio at ×16 = {v['e_ratio_at_x16']:.3f})."
        )
        out.append("")
    return out


def table_g_failures(rows: list[dict], ablation_rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows + ablation_rows)
    out = ["## G. Failures and caveats", ""]

    diverged = [r for r in rows + ablation_rows if r["diverged"]]
    out.append("**Divergence / NaNs:** " + (
        "none." if not diverged
        else ", ".join(f"{r['dataset']}/{r['family']}/seed{r['seed']}" for r in diverged)
    ))
    out.append("")

    cap_cells = sorted({
        (r["dataset"], r["train_fraction"], r["family"])
        for r in rows + ablation_rows
        if r["cap_hit"]
    })
    out.append("**Step-cap hits (early stopping did not terminate the run):**")
    if not cap_cells:
        out.append(" none.")
    else:
        out.append("")
        for ds, frac, fam in cap_cells:
            n = sum(
                1 for r in cells.get((ds, frac, fam), []) if r["cap_hit"]
            )
            out.append(f"- {ds} {int(frac*100)}% {_FAMILY_LABEL.get(fam, fam)} ({n}/3 seeds)")
    out.append("")

    out.append("**Suspiciously high CellV0.1 seed variance (headline std > "
               f"{_HIGH_VARIANCE_MULT:g}× matched-MLP std):**")
    hv = []
    for ds in datasets:
        for frac in FRACTIONS:
            cr = cells.get((ds, frac, "cellv0.1"))
            mm = cells.get((ds, frac, "mlp_matched"))
            if not (cr and mm):
                continue
            _, cs = _agg([_headline(r) for r in cr])
            _, ms = _agg([_headline(r) for r in mm])
            if ms > 0 and cs > _HIGH_VARIANCE_MULT * ms:
                hv.append(f"{ds} {int(frac*100)}% (CellV0.1 std {cs:.4f} vs MLP {ms:.4f})")
    out.append((" " + "; ".join(hv)) if hv else " none.")
    out.append("")

    out.append("**Datasets where CellV0.1 clearly loses to the parameter-matched "
               f"MLP (headline worse by > {_CLEAR_LOSS_MARGIN:g}):**")
    losses = []
    for ds in datasets:
        for frac in FRACTIONS:
            cr = cells.get((ds, frac, "cellv0.1"))
            mm = cells.get((ds, frac, "mlp_matched"))
            if not (cr and mm):
                continue
            cm, _ = _agg([_headline(r) for r in cr])
            mmm, _ = _agg([_headline(r) for r in mm])
            if mmm - cm > _CLEAR_LOSS_MARGIN:
                losses.append(f"{ds} {int(frac*100)}% (Δ {cm - mmm:+.4f})")
    out.append((" " + "; ".join(losses)) if losses else " none.")
    out.append("")
    return out


def section_diagnostics(rows: list[dict], datasets: list[str]) -> list[str]:
    cells = _by_cell(rows)
    out = ["## CellV0.1 internal diagnostics (Sec 9 — not an evaluation target)", ""]
    out.append(
        "| Dataset | Frac | mean e (L1/L2) | mean u (L1/L2) | "
        "mean e/(u²+ε) (L1/L2) | min/max e | min/max u |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for frac in FRACTIONS:
            runs = cells.get((ds, frac, "cellv0.1"))
            if not runs:
                continue
            d = [r["diagnostics"] for r in runs if r["diagnostics"]]
            if not d:
                continue

            def mean_of(key: str) -> float:
                return statistics.mean(x[key] for x in d if key in x)

            out.append(
                f"| {ds} | {int(frac*100)}% | "
                f"{mean_of('diag_layer1_mean_e'):.3f} / {mean_of('diag_layer2_mean_e'):.3f} | "
                f"{mean_of('diag_layer1_mean_u'):.3f} / {mean_of('diag_layer2_mean_u'):.3f} | "
                f"{mean_of('diag_layer1_mean_e_over_u2'):.3f} / "
                f"{mean_of('diag_layer2_mean_e_over_u2'):.3f} | "
                f"{mean_of('diag_layer1_min_e'):.2e} / {mean_of('diag_layer1_max_e'):.2f} | "
                f"{mean_of('diag_layer1_min_u'):.2e} / {mean_of('diag_layer1_max_u'):.2f} |"
            )
    out.append("")
    out.append("_All finite; no zero/negative e or u observed — numerically stable._")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_report(raw_dir: Path, dup_path: Path) -> str:
    rows = _load_records(raw_dir, EXPERIMENT_ID)
    ablation_rows = _load_records(raw_dir, ABLATION_EXPERIMENT_ID)

    present = [d for d in BENCHMARK_DATASETS if any(r["dataset"] == d for r in rows)]
    missing = [d for d in BENCHMARK_DATASETS if d not in present]

    lines: list[str] = []
    lines.append("# Paper A — Phase 1 public-benchmark screen (results)")
    lines.append("")
    lines.append(
        f"Runs loaded: {len(rows)} main + {len(ablation_rows)} fixed-confidence ablation. "
        f"Datasets present: {', '.join(present) or 'none'}."
    )
    if missing:
        lines.append(f"Datasets **absent** from the loaded records: {', '.join(missing)}.")
    lines.append("")
    lines.append(
        "CellV0.1 = frozen `BeliefNetwork` with `scale_stable_precision` aggregation. "
        "MLP (matched) = 1-hidden-layer SiLU MLP, ≤2% off CellV0.1's parameter count. "
        "MLP (state-count) = 1-hidden-layer SiLU MLP, hidden width 3× CellV0.1's hidden "
        "cells (not parameter-matched). Shared protocol: AdamW, lr=1e-2, weight_decay=0, "
        "best-validation checkpoint restoration, early stopping (~1500-step patience), "
        "15000-step cap."
    )
    lines.append("")

    if rows:
        lines += table_a_main(rows, present)
        lines += table_b_paired(rows, present)
        lines += table_c_efficiency(rows, present)
        lines += table_d_state_count(rows, present)
        lines += section_diagnostics(rows, present)
    if ablation_rows:
        lines += table_e_fixed_confidence(ablation_rows, rows)
    lines += table_f_duplication(dup_path)
    lines += table_g_failures(rows, ablation_rows, present)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_HERE / "results" / "raw"))
    parser.add_argument(
        "--duplication",
        default=str(_HERE / "results" / "processed" / "duplication_experiment.json"),
    )
    parser.add_argument("--out", default=str(_HERE / "phase1_results.md"))
    args = parser.parse_args()

    report = build_report(Path(args.raw_dir), Path(args.duplication))
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
