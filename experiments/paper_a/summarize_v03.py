#!/usr/bin/env python3
"""Aggregate the CellV0.3 Phase-1 runs (`paper_a_phase1_cellv03`) against the
already-recorded CellV0.1, parameter-matched-MLP (`paper_a_phase1`) and
CellV0.2 (`paper_a_phase1_cellv02`) arms.

CellV0.3 is the only architecture re-run; the comparison arms are read
verbatim from the frozen records. No significance testing -- with n=3 seeds
the paired differences are reported per seed and as a mean, nothing more
(Paper-A task Sec 8).

Also builds the precision-mechanism report (Sec 19): per dataset, the
typical output precision, its coefficient of variation, the typical
sqrt(precision) confidence scale and the typical conflict `u_out` -- compared
between the untrained snapshot and the best-checkpoint network. The question
that answers: does CellV0.3 learn an example/cell-dependent computational
reliability, or does precision again collapse to an effectively constant
value?

Writes `experiments/paper_a/phase1_v03_results.md` and prints it.

Usage:
    python experiments/paper_a/summarize_v03.py
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
from experiments.paper_a.run_phase1_v03 import CELLV03_EXPERIMENT_ID  # noqa: E402
from experiments.paper_a.summarize import (  # noqa: E402
    _agg,
    _by_cell,
    _fmt,
    _headline,
    _load_records,
)

FRACTIONS = (0.25, 1.0)
_MARGIN = 0.01  # reporting threshold only -- not a significance claim
_CV_VARIES = 0.05  # CoV(pi_out) above this -> "varies"; below -> "~constant"


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


def _mean_diag(runs: list[dict], key: str, *, init: bool = False) -> float | None:
    field = "diagnostics_init" if init else "diagnostics"
    vals = [r[field][key] for r in runs if r.get(field) and key in r[field]]
    return statistics.mean(vals) if vals else None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_header(v03: list[dict], present: list[str]) -> list[str]:
    out = [
        "# Paper A — CellV0.3 (Conflict-Normalized Belief Cell) on the frozen "
        "Phase-1 protocol",
        "",
        f"CellV0.3 runs loaded: {len(v03)} (experiment `{CELLV03_EXPERIMENT_ID}`). "
        f"Comparison arms read verbatim from the frozen records: "
        f"`{EXPERIMENT_ID}` (CellV0.1, matched MLP) and "
        f"`{CELLV02_EXPERIMENT_ID}` (CellV0.2).",
        f"Datasets present (CellV0.3): {', '.join(present) or 'none'}.",
    ]
    missing = [d for d in BENCHMARK_DATASETS if d not in present]
    if missing:
        out.append(f"Datasets **absent** from the CellV0.3 records: {', '.join(missing)}.")
    out += [
        "",
        "CellV0.3 = `BeliefNetworkV03` — two `ConflictNormalizedLayer`s (one "
        "signed connection matrix `V` + per-output `gain_raw` + `bias`, **no** "
        "relevance gate; CellV0.2's population-relative gain `2π/(π+mean π)` "
        "removed entirely) + a plain linear readout on `final_mu`. Each cell "
        "forms a precision-weighted signed consensus, takes the conflict "
        "(A-weighted variance of the signed messages) as `u`, derives "
        "`pi_out = e_out/(1+e_out·u_out)` and folds `sqrt(pi_out)` into its "
        "**own** `tanh` activation. Input belief `e=1, u=0`. Hidden width "
        "fitted to the **same** per-dataset parameter budget as CellV0.1 "
        "(identical to CellV0.2's). Identical shared protocol: AdamW, lr=1e-2, "
        "weight_decay=0, best-validation restore, ~1500-step early-stop "
        "patience, 15000-step cap. Nothing tuned; equations frozen before the "
        "run (docs/architecture_v0.md Sec 10).",
        "",
    ]
    return out


def section_compact(cells: dict, present: list[str]) -> list[str]:
    """The one compact summary table the task asks for (Sec 18)."""
    out = [
        "## Compact summary — CellV0.3 headline Δ vs each recorded arm "
        "(mean of 3 seeds, 25% / 100%)",
        "",
        "| Dataset | Δ vs MLP | Δ vs CellV0.1 | Δ vs CellV0.2 |",
        "|---|---|---|---|",
    ]
    for ds in present:
        parts: dict[str, list[str]] = {"mlp_matched": [], "cellv0.1": [], "cellv0.2": []}
        for frac in FRACTIONS:
            c3 = _cell(cells, ds, frac, "cellv0.3")
            m3 = _agg([_headline(r) for r in c3])[0] if c3 else float("nan")
            for fam in parts:
                other = _cell(cells, ds, frac, fam)
                if c3 and other:
                    parts[fam].append(f"{m3 - _agg([_headline(r) for r in other])[0]:+.4f}")
                else:
                    parts[fam].append("—")
        out.append(
            f"| {ds} | {' / '.join(parts['mlp_matched'])} | "
            f"{' / '.join(parts['cellv0.1'])} | {' / '.join(parts['cellv0.2'])} |"
        )
    out += ["", "_Positive = CellV0.3 better. n=3 seeds; no significance test._", ""]
    return out


def section_signal(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## Headline signal — CellV0.3 vs the parameter-matched MLP",
        "",
        "| Dataset | Δ (25% / 100%) | Read |",
        "|---|---|---|",
    ]
    tallies = {"win": 0, "loss": 0, "tie": 0}
    for ds in present:
        dm, reads = [], []
        for frac in FRACTIONS:
            c3 = _cell(cells, ds, frac, "cellv0.3")
            mm = _cell(cells, ds, frac, "mlp_matched")
            if not (c3 and mm):
                dm.append(float("nan"))
                continue
            d = _agg([_headline(r) for r in c3])[0] - _agg([_headline(r) for r in mm])[0]
            dm.append(d)
            reads.append("win" if d > _MARGIN else "loss" if d < -_MARGIN else "tie")
        for r in reads:
            tallies[r] += 1
        read = "—"
        if reads:
            if all(x == "win" for x in reads):
                read = "win (both fractions)"
            elif all(x == "loss" for x in reads):
                read = "loss (both fractions)"
            elif all(x == "tie" for x in reads):
                read = "tie"
            elif "loss" not in reads:
                read = "win (one fraction)"
            elif "win" not in reads:
                read = "loss (one fraction)"
            else:
                read = "mixed"
        out.append(f"| {ds} | {dm[0]:+.4f} / {dm[1]:+.4f} | {read} |")
    out += [
        "",
        f"Across all {sum(tallies.values())} (dataset, fraction) cells vs the "
        f"parameter-matched MLP: CellV0.3 ahead by >{_MARGIN} in {tallies['win']}, "
        f"behind by >{_MARGIN} in {tallies['loss']}, within ±{_MARGIN} in "
        f"{tallies['tie']}. Reporting threshold only; no significance test (n=3).",
        "",
    ]
    return out


def table_main(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## A. Main table — CellV0.3",
        "",
        "| Dataset | Frac | Hidden cells | Params | Headline (mean ± std) | "
        "2nd metric | Best-val step | Total steps | Train wall-clock (s) | Time/step (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in present:
        for frac in FRACTIONS:
            runs = _cell(cells, ds, frac, "cellv0.3")
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
    out = [
        f"## Paired comparison — CellV0.3 minus {label} (headline metric)",
        "",
        "| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |",
        "|---|---|---|---|---|---|",
    ]
    for ds in present:
        for frac in FRACTIONS:
            c3 = _cell(cells, ds, frac, "cellv0.3")
            ot = _cell(cells, ds, frac, other)
            if not (c3 and ot):
                continue
            per_seed, deltas = _paired(c3, ot)
            if not deltas:
                continue
            cols = _delta_cols(per_seed)
            out.append(
                f"| {ds} | {int(frac*100)}% | {cols[0]} | {cols[1]} | {cols[2]} | "
                f"**{statistics.mean(deltas):+.4f}** |"
            )
    out += ["", f"_Positive = CellV0.3 better than {label}. n=3 seeds; no significance test._", ""]
    return out


def table_efficiency(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## Data-efficiency — headline at 25% → 100% training data",
        "",
        "| Dataset | Model | 25% | 100% | Δ (100% − 25%) |",
        "|---|---|---|---|---|",
    ]
    for ds in present:
        for fam, name in (
            ("cellv0.3", "CellV0.3"),
            ("cellv0.2", "CellV0.2 (recorded)"),
            ("cellv0.1", "CellV0.1 (recorded)"),
            ("mlp_matched", "MLP matched (recorded)"),
        ):
            r25 = _cell(cells, ds, 0.25, fam)
            r100 = _cell(cells, ds, 1.0, fam)
            if not (r25 and r100):
                continue
            m25, s25 = _agg([_headline(r) for r in r25])
            m100, s100 = _agg([_headline(r) for r in r100])
            out.append(
                f"| {ds} | {name} | {_fmt(m25, s25)} | {_fmt(m100, s100)} | {m100 - m25:+.4f} |"
            )
    out.append("")
    return out


def table_params(cells: dict, present: list[str]) -> list[str]:
    out = [
        "## B. Parameter count & hidden width — CellV0.3 vs CellV0.2 vs CellV0.1 (same budget)",
        "",
        "| Dataset | Budget | V0.3 hidden | V0.3 params | V0.2 hidden | V0.1 hidden | "
        "V0.1 params | V0.3/V0.1 cell ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds in present:
        runs = _cell(cells, ds, 1.0, "cellv0.3") or _cell(cells, ds, 0.25, "cellv0.3")
        c1 = _cell(cells, ds, 1.0, "cellv0.1") or _cell(cells, ds, 0.25, "cellv0.1")
        if not (runs and c1):
            continue
        s = runs[0]["sizing"]
        hc3 = s.get("hidden_cells")
        hc2 = s.get("cellv02_hidden_cells")
        hc1 = s.get("cellv01_hidden_cells")
        out.append(
            f"| {ds} | {s.get('param_budget'):,} | {hc3} | {runs[0]['params']:,} | "
            f"{hc2} | {hc1} | {c1[0]['params']:,} | {hc3 / hc1:.2f}× |"
        )
    out += [
        "",
        "_CellV0.3's parameterization is identical to CellV0.2's "
        "(`hc·(in+2) + hc·(hc+2) + hc·out + out`), so its hidden width matches "
        "CellV0.2's exactly and is ~1.5× CellV0.1's at the same budget — its "
        "lower per-connection cost is part of the architecture, not equalized._",
        "",
    ]
    return out


def section_mechanism(cells: dict, present: list[str]) -> list[str]:
    """Sec 19 -- does CellV0.3 actually use its belief state?"""
    out = [
        "## Most important — does CellV0.3 use its belief state? (Sec 19)",
        "",
        "Per (dataset, fraction), mean over 3 seeds, **untrained snapshot → "
        "best checkpoint**. `pi_out = e_out/(1+e_out·u_out)`; CoV = "
        "`std(pi_out)/(mean(pi_out)+ε)` across the whole test set × cells; "
        "`k = sqrt(pi_out)` is the factor folded into each cell's activation.",
        "",
        "| Dataset | Frac | L2 pi_out mean | L2 CoV(pi_out) | L2 k=√pi_out mean | "
        "L2 conflict u_out mean | L1 pi_out mean | Read (L2 CoV) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def arrow(runs: list[dict], key: str) -> str:
        i = _mean_diag(runs, key, init=True)
        b = _mean_diag(runs, key, init=False)
        if i is None or b is None:
            return "—"
        return f"{i:.3f} → {b:.3f}"

    varies = 0
    constant = 0
    for ds in present:
        for frac in FRACTIONS:
            runs = _cell(cells, ds, frac, "cellv0.3")
            if not runs:
                continue
            cv_best = _mean_diag(runs, "diag_layer2_precision_cv", init=False)
            read = "—"
            if cv_best is not None:
                if cv_best >= _CV_VARIES:
                    read = f"varies ({cv_best:.2f})"
                    varies += 1
                else:
                    read = f"~constant ({cv_best:.2f})"
                    constant += 1
            out.append(
                f"| {ds} | {int(frac*100)}% | "
                f"{arrow(runs, 'diag_layer2_precision_mean')} | "
                f"{arrow(runs, 'diag_layer2_precision_cv')} | "
                f"{arrow(runs, 'diag_layer2_sqrt_precision_mean')} | "
                f"{arrow(runs, 'diag_layer2_u_mean')} | "
                f"{arrow(runs, 'diag_layer1_precision_mean')} | {read} |"
            )
    out += [
        "",
        f"Of {varies + constant} (dataset, fraction) cells, layer-2 output "
        f"precision **varies** (CoV ≥ {_CV_VARIES}) in {varies} and is "
        f"**effectively constant** (CoV < {_CV_VARIES}) in {constant} at the "
        "best checkpoint. A near-constant precision pathway is not a success "
        "even where accuracy is good — see the verdict in "
        "`docs/research_log.md`.",
        "",
        "_Layer 1's inherited support `e` is exactly 1 by construction (every "
        "input feature starts at precision 1 and `e_out` is a convex "
        "combination), so layer 1 only ever varies `pi_out` through its "
        "conflict `u`._",
        "",
    ]
    return out


def section_failures(cells: dict, present: list[str]) -> list[str]:
    out = ["## C. Failures and caveats (CellV0.3)", ""]
    v03 = [
        r
        for ds in present
        for frac in FRACTIONS
        for r in _cell(cells, ds, frac, "cellv0.3")
    ]
    diverged = [r for r in v03 if r["diverged"]]
    if diverged:
        tags = ", ".join(
            f"{r['dataset']}/seed{r['seed']}/{int(r['train_fraction']*100)}%" for r in diverged
        )
        out.append(f"**Divergence / NaNs:** {tags}")
    else:
        out.append("**Divergence / NaNs:** none.")
    out.append("")

    cap = sorted({(r["dataset"], r["train_fraction"]) for r in v03 if r["cap_hit"]})
    if cap:
        out.append("**Step-cap hits (early stopping did not terminate):**")
        out.append("")
        for ds, frac in cap:
            n = sum(
                1
                for r in v03
                if r["dataset"] == ds and r["train_fraction"] == frac and r["cap_hit"]
            )
            out.append(f"- {ds} {int(frac*100)}% ({n}/3 seeds)")
    else:
        out.append("**Step-cap hits:** none.")
    out.append("")

    tiny = []
    for r in v03:
        d = r["diagnostics"]
        for layer in ("layer1", "layer2"):
            lo = d.get(f"diag_{layer}_precision_min")
            if lo is not None and lo < 1e-6:
                tiny.append(
                    f"{r['dataset']} {int(r['train_fraction']*100)}% "
                    f"seed{r['seed']} {layer} (min π = {lo:.1e})"
                )
    if tiny:
        out.append(
            "**Numeric-range flag** (a single test example × cell with output "
            "precision `< 1e-6` — large local conflict `u`; all values finite, "
            "no NaN/divergence, no protocol impact):"
        )
        out.append("")
        out += [f"- {t}" for t in tiny]
    else:
        out.append("**Numeric-range flag:** none (min output precision ≥ 1e-6).")
    out.append("")
    return out


def build_report(raw_dir: Path) -> str:
    v03 = _load_records(raw_dir, CELLV03_EXPERIMENT_ID)
    phase1 = [
        r
        for r in _load_records(raw_dir, EXPERIMENT_ID)
        if r["family"] in ("cellv0.1", "mlp_matched")
    ]
    v02 = _load_records(raw_dir, CELLV02_EXPERIMENT_ID)

    present = [d for d in BENCHMARK_DATASETS if any(r["dataset"] == d for r in v03)]
    cells = _by_cell(v03 + phase1 + v02)

    lines: list[str] = section_header(v03, present)
    if not v03:
        lines.append(
            "_No CellV0.3 records found — run "
            "`python experiments/paper_a/run_phase1_v03.py`._"
        )
        return "\n".join(lines) + "\n"
    lines += section_compact(cells, present)
    lines += section_signal(cells, present)
    lines += table_main(cells, present)
    lines += table_paired(cells, present, "mlp_matched", "the parameter-matched MLP")
    lines += table_paired(cells, present, "cellv0.1", "CellV0.1 (recorded)")
    lines += table_paired(cells, present, "cellv0.2", "CellV0.2 (recorded)")
    lines += table_efficiency(cells, present)
    lines += table_params(cells, present)
    lines += section_mechanism(cells, present)
    lines += section_failures(cells, present)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_HERE / "results" / "raw"))
    parser.add_argument("--out", default=str(_HERE / "phase1_v03_results.md"))
    args = parser.parse_args()

    report = build_report(Path(args.raw_dir))
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
