#!/usr/bin/env python3
"""Aggregate the CellV0.2 Phase-1 runs (`paper_a_phase1_cellv02`) against the
already-recorded CellV0.1 and parameter-matched-MLP arms (`paper_a_phase1`).

CellV0.2 is the only architecture re-run; the comparison arms are read
verbatim from the frozen Phase-1 records. No significance testing -- with
n=3 seeds the paired differences are reported per seed and as a mean, nothing
more (Paper-A task Sec 8).

Writes `experiments/paper_a/phase1_v02_results.md` and prints it.

Usage:
    python experiments/paper_a/summarize_v02.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.paper_a.datasets import BENCHMARK_DATASETS  # noqa: E402
from experiments.paper_a.harness import EXPERIMENT_ID  # noqa: E402
from experiments.paper_a.run_phase1_v02 import CELLV02_EXPERIMENT_ID  # noqa: E402
from experiments.paper_a.summarize import (  # noqa: E402
    _agg,
    _by_cell,
    _fmt,
    _headline,
    _load_records,
)

FRACTIONS = (0.25, 1.0)
_MARGIN = 0.01  # reporting threshold only -- not a significance claim


def _cell(cells: dict, ds: str, frac: float, fam: str) -> list[dict]:
    return sorted(cells.get((ds, frac, fam), []), key=lambda r: r["seed"])


def _paired(a: list[dict], b: list[dict]) -> tuple[dict[int, float], list[float]]:
    am = {r["seed"]: r for r in a}
    bm = {r["seed"]: r for r in b}
    seeds = sorted(set(am) & set(bm))
    deltas = [_headline(am[s]) - _headline(bm[s]) for s in seeds]
    return dict(zip(seeds, deltas)), deltas


def _delta_cols(per_seed: dict[int, float]) -> list[str]:
    return [f"{per_seed[s]:+.4f}" if s in per_seed else "—" for s in (0, 1, 2)]


def section_header(v02: list[dict], phase1: list[dict], present: list[str]) -> list[str]:
    out = [
        "# Paper A — CellV0.2 (Conservative Precision-Gain Cell) on the frozen "
        "Phase-1 protocol",
        "",
    ]
    out.append(
        f"CellV0.2 runs loaded: {len(v02)} (experiment `{CELLV02_EXPERIMENT_ID}`). "
        f"Comparison arms read from the frozen `{EXPERIMENT_ID}` records: "
        f"{sum(1 for r in phase1 if r['family'] == 'cellv0.1')} CellV0.1, "
        f"{sum(1 for r in phase1 if r['family'] == 'mlp_matched')} matched MLP."
    )
    out.append(f"Datasets present (CellV0.2): {', '.join(present) or 'none'}.")
    missing = [d for d in BENCHMARK_DATASETS if d not in present]
    if missing:
        out.append(f"Datasets **absent** from the CellV0.2 records: {', '.join(missing)}.")
    out.append("")
    out.append(
        "CellV0.2 = `BeliefNetworkV02` — two `PrecisionGainLayer`s (one signed "
        "connection matrix `V` + per-output `gain_raw` + `bias`, **no** relevance "
        "gate) + a confidence-scaled linear readout. Input belief `e=1, u=0`. "
        "Hidden width fitted to the **same** per-dataset parameter budget as "
        "CellV0.1; the resulting (larger) hidden-cell count is reported. Identical "
        "shared protocol: AdamW, lr=1e-2, weight_decay=0, best-validation restore, "
        "~1500-step early-stop patience, 15000-step cap. Nothing tuned; equations "
        "frozen before the run (docs/architecture_v0.md Sec 10)."
    )
    out.append("")
    return out


def section_signal(cells: dict, present: list[str]) -> list[str]:
    out = ["## Headline signal (per dataset) — CellV0.2 vs each recorded arm", ""]
    out.append(
        "| Dataset | Δ vs matched MLP (25% / 100%) | Δ vs CellV0.1 (25% / 100%) "
        "| Read (vs matched) |"
    )
    out.append("|---|---|---|---|")
    tallies = {"win": 0, "loss": 0, "tie": 0}
    for ds in present:
        dm, dc, reads = [], [], []
        for frac in FRACTIONS:
            c2 = _cell(cells, ds, frac, "cellv0.2")
            mm = _cell(cells, ds, frac, "mlp_matched")
            c1 = _cell(cells, ds, frac, "cellv0.1")
            if not (c2 and mm and c1):
                dm.append(float("nan"))
                dc.append(float("nan"))
                continue
            m2 = _agg([_headline(r) for r in c2])[0]
            d_m = m2 - _agg([_headline(r) for r in mm])[0]
            d_c = m2 - _agg([_headline(r) for r in c1])[0]
            dm.append(d_m)
            dc.append(d_c)
            reads.append("win" if d_m > _MARGIN else "loss" if d_m < -_MARGIN else "tie")
        for r in reads:
            tallies[r] += 1
        read = "—"
        if reads:
            if all(x == "win" for x in reads):
                read = "win (both fractions)"
            elif all(x == "loss" for x in reads):
                read = "loss (both fractions)"
            elif "win" in reads and "loss" not in reads:
                read = "win (one fraction)"
            elif "loss" in reads and "win" not in reads:
                read = "loss (one fraction)"
            else:
                read = "mixed"
        out.append(
            f"| {ds} | {dm[0]:+.4f} / {dm[1]:+.4f} | {dc[0]:+.4f} / {dc[1]:+.4f} | {read} |"
        )
    out.append("")
    out.append(
        f"Across all {sum(tallies.values())} (dataset, fraction) cells vs the "
        f"parameter-matched MLP: CellV0.2 ahead by >{_MARGIN} in {tallies['win']}, "
        f"behind by >{_MARGIN} in {tallies['loss']}, within ±{_MARGIN} in "
        f"{tallies['tie']}. Reporting threshold only; no significance test (n=3)."
    )
    out.append("")
    return out


def table_main(cells: dict, present: list[str]) -> list[str]:
    out = ["## A. Main table — CellV0.2", ""]
    out.append(
        "| Dataset | Frac | Hidden cells | Params | Headline (mean ± std) | "
        "2nd metric | Best-val step | Total steps | Train wall-clock (s) | Time/step (ms) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in present:
        for frac in FRACTIONS:
            runs = _cell(cells, ds, frac, "cellv0.2")
            if not runs:
                continue
            task = runs[0]["task_type"]
            hl = runs[0]["headline_metric"]
            hl_m, hl_s = _agg([_headline(r) for r in runs])
            if task == "classification":
                sec_m, sec_s = _agg([r["macro_f1"] for r in runs])
                sec = f"F1: {_fmt(sec_m, sec_s)}"
            else:
                sec_m, sec_s = _agg([r["rmse"] for r in runs])
                sec = f"RMSE: {_fmt(sec_m, sec_s)}"
            bv_m, bv_s = _agg([r["best_val_step"] for r in runs])
            ts_m, _ = _agg([r["total_steps"] for r in runs])
            wc_m, wc_s = _agg([r["train_wall_clock_s"] for r in runs])
            tps_m, _ = _agg([r["time_per_step_s"] for r in runs])
            out.append(
                f"| {ds} | {int(frac*100)}% | {runs[0]['sizing'].get('hidden_cells', '?')} | "
                f"{runs[0]['params']:,} | {hl}: {_fmt(hl_m, hl_s)} | {sec} | "
                f"{bv_m:.0f} ± {bv_s:.0f} | {ts_m:.0f} | {_fmt(wc_m, wc_s, 1)} | "
                f"{tps_m*1e3:.2f} |"
            )
    out.append("")
    return out


def table_paired(cells: dict, present: list[str], other: str, label: str) -> list[str]:
    out = [f"## Paired comparison — CellV0.2 minus {label} (headline metric)", ""]
    out.append("| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |")
    out.append("|---|---|---|---|---|---|")
    for ds in present:
        for frac in FRACTIONS:
            c2 = _cell(cells, ds, frac, "cellv0.2")
            ot = _cell(cells, ds, frac, other)
            if not (c2 and ot):
                continue
            per_seed, deltas = _paired(c2, ot)
            if not deltas:
                continue
            cols = _delta_cols(per_seed)
            out.append(
                f"| {ds} | {int(frac*100)}% | {cols[0]} | {cols[1]} | {cols[2]} | "
                f"**{statistics.mean(deltas):+.4f}** |"
            )
    out.append("")
    out.append(f"_Positive = CellV0.2 better than {label}. n=3 seeds; no significance test._")
    out.append("")
    return out


def table_params(cells: dict, present: list[str]) -> list[str]:
    out = ["## B. Parameter count & hidden width — CellV0.2 vs CellV0.1 (same budget)", ""]
    out.append(
        "| Dataset | Budget | CellV0.2 hidden cells | CellV0.2 params | "
        "CellV0.1 hidden cells | CellV0.1 params | cell ratio |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for ds in present:
        runs = _cell(cells, ds, 1.0, "cellv0.2") or _cell(cells, ds, 0.25, "cellv0.2")
        c1 = _cell(cells, ds, 1.0, "cellv0.1") or _cell(cells, ds, 0.25, "cellv0.1")
        if not (runs and c1):
            continue
        s = runs[0]["sizing"]
        hc2 = s.get("hidden_cells")
        hc1 = s.get("cellv01_hidden_cells")
        out.append(
            f"| {ds} | {s.get('param_budget'):,} | {hc2} | {runs[0]['params']:,} | "
            f"{hc1} | {c1[0]['params']:,} | {hc2 / hc1:.2f}× |"
        )
    out.append("")
    out.append(
        "_CellV0.2 formula: `hc·(in+2) + hc·(hc+2) + hc·out + out`. CellV0.1 "
        "formula: `hc·(2·in+1) + hc·(2·hc+1) + hc·out + out`. Same budget, "
        "roughly 1.5× the hidden cells — its lower per-connection cost is part "
        "of the architecture, not equalized away._"
    )
    out.append("")
    return out


def section_diagnostics(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## CellV0.2 internal diagnostics (observational only — not an "
        "evaluation target)",
        "",
    ]
    out.append(
        "| Dataset | Frac | eff. precision L1 (mean/std, min–max) | "
        "eff. precision L2 (mean/std, min–max) | rel. gain L1 (mean/std) "
        "| rel. gain L2 (mean/std) |"
    )
    out.append("|---|---|---|---|---|---|")
    for ds in present:
        for frac in FRACTIONS:
            runs = [r for r in _cell(cells, ds, frac, "cellv0.2") if r["diagnostics"]]
            if not runs:
                continue
            d = [r["diagnostics"] for r in runs]

            def m(key: str) -> float:
                return statistics.mean(x[key] for x in d if key in x)

            def precision_cell(layer: str) -> str:
                mean = m(f"diag_{layer}_precision_mean")
                std = m(f"diag_{layer}_precision_std")
                lo = m(f"diag_{layer}_precision_min")
                hi = m(f"diag_{layer}_precision_max")
                return f"{mean:.3f} / {std:.3f}, {lo:.2e}–{hi:.2f}"

            def gain_cell(layer: str) -> str:
                return (
                    f"{m(f'diag_{layer}_relative_gain_mean'):.3f} / "
                    f"{m(f'diag_{layer}_relative_gain_std'):.3f}"
                )

            out.append(
                f"| {ds} | {int(frac*100)}% | {precision_cell('layer1')} | "
                f"{precision_cell('layer2')} | {gain_cell('layer1')} | "
                f"{gain_cell('layer2')} |"
            )
    out.append("")
    out.append(
        "_Effective precision `e/(1+e·u)` and relative gain `2π/(π+mean π)` of "
        "each hidden layer's output belief, on the test split. Logged per the "
        "task; nothing is tuned on these._"
    )
    out.append("")
    return out


def table_failures(cells: dict, present: list[str]) -> list[str]:
    out = ["## C. Failures and caveats (CellV0.2)", ""]
    v02 = [
        r
        for ds in present
        for frac in FRACTIONS
        for r in _cell(cells, ds, frac, "cellv0.2")
    ]
    diverged = [r for r in v02 if r["diverged"]]
    if diverged:
        tags = ", ".join(
            f"{r['dataset']}/seed{r['seed']}/{int(r['train_fraction']*100)}%"
            for r in diverged
        )
        out.append(f"**Divergence / NaNs:** {tags}")
    else:
        out.append("**Divergence / NaNs:** none.")
    out.append("")
    cap = sorted({(r["dataset"], r["train_fraction"]) for r in v02 if r["cap_hit"]})
    if cap:
        out.append("**Step-cap hits (early stopping did not terminate):**")
        out.append("")
        for ds, frac in cap:
            n = sum(
                1
                for r in v02
                if r["dataset"] == ds and r["train_fraction"] == frac and r["cap_hit"]
            )
            out.append(f"- {ds} {int(frac*100)}% ({n}/3 seeds)")
    else:
        out.append("**Step-cap hits:** none.")
    out.append("")
    return out


def build_report(raw_dir: Path) -> str:
    v02 = _load_records(raw_dir, CELLV02_EXPERIMENT_ID)
    phase1 = _load_records(raw_dir, EXPERIMENT_ID)
    phase1 = [r for r in phase1 if r["family"] in ("cellv0.1", "mlp_matched")]

    present = [d for d in BENCHMARK_DATASETS if any(r["dataset"] == d for r in v02)]
    cells = _by_cell(v02 + phase1)

    lines: list[str] = []
    lines += section_header(v02, phase1, present)
    if not v02:
        lines.append(
            "_No CellV0.2 records found — run "
            "`python experiments/paper_a/run_phase1_v02.py`._"
        )
        return "\n".join(lines) + "\n"
    lines += section_signal(cells, present)
    lines += table_main(cells, present)
    lines += table_paired(cells, present, "mlp_matched", "the parameter-matched MLP")
    lines += table_paired(cells, present, "cellv0.1", "CellV0.1 (recorded)")
    lines += table_params(cells, present)
    lines += section_diagnostics(cells, present)
    lines += table_failures(cells, present)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_HERE / "results" / "raw"))
    parser.add_argument("--out", default=str(_HERE / "phase1_v02_results.md"))
    args = parser.parse_args()

    report = build_report(Path(args.raw_dir))
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
