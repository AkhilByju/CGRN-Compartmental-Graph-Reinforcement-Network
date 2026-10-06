"""Combine `results/raw/*.json` RunRecords into the Sec P/Q report: per-cell
accuracy tables (clean / max-train-severity / max-OOD-severity /
corruption-AUC), and direct answers to Sec P's five comparison questions.

Usage: python -m experiments.belief_dendrite.summarize
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.belief_dendrite.corruption import (  # noqa: E402
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
)
from experiments.belief_dendrite.harness import DATASETS, SEEDS  # noqa: E402
from experiments.belief_dendrite.models import MODEL_FAMILIES  # noqa: E402

CORRUPTION_FAMILIES = ("missing_patch", "noisy_patch")


def load_records(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith("_history.json"):
            continue
        with p.open() as f:
            rows.append(json.load(f))
    return rows


def by_cell(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        cfg = r["config"]
        key = (cfg["dataset"], cfg["extra"]["corruption_family"], cfg["architecture"])
        cells.setdefault(key, []).append(r)
    return cells


def agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _sev_acc(runs: list[dict], severity: float) -> list[float]:
    return [
        r["test_metrics"][f"accuracy@{severity}"]
        for r in runs
        if f"accuracy@{severity}" in r["test_metrics"]
    ]


def _auc(runs: list[dict]) -> list[float]:
    return [r["test_metrics"]["corruption_auc_primary"] for r in runs]


def _drop(runs: list[dict]) -> list[float]:
    return [r["test_metrics"]["ood_drop_primary"] for r in runs]


def cell_table(cells: dict, present_datasets: list[str]) -> list[str]:
    lines = ["## Accuracy table (mean ± population-std over seeds)\n"]
    for ds in present_datasets:
        for cf in CORRUPTION_FAMILIES:
            fams = [f for f in MODEL_FAMILIES if (ds, cf, f) in cells]
            if not fams:
                continue
            lines.append(f"### {ds} / {cf}\n")
            lo, hi = MAX_TRAIN_SEVERITY[cf], MAX_OOD_SEVERITY[cf]
            lines.append(
                f"| family | n_seeds | acc@clean | acc@max_train({lo}) | "
                f"acc@max_ood({hi}) | corruption_AUC | OOD_drop |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for fam in fams:
                runs = cells[(ds, cf, fam)]
                clean_m, clean_s = agg(_sev_acc(runs, 0.0))
                lo_m, lo_s = agg(_sev_acc(runs, lo))
                hi_m, hi_s = agg(_sev_acc(runs, hi))
                auc_m, auc_s = agg(_auc(runs))
                drop_m, drop_s = agg(_drop(runs))
                lines.append(
                    f"| {fam} | {len(runs)} | {clean_m:.4f}±{clean_s:.4f} | "
                    f"{lo_m:.4f}±{lo_s:.4f} | {hi_m:.4f}±{hi_s:.4f} | "
                    f"{auc_m:.3f}±{auc_s:.3f} | {drop_m:.4f}±{drop_s:.4f} |"
                )
            lines.append("")
    return lines


def _mean_metric_over(
    cells: dict, ds: str, cf: str, fam: str, metric_key: str
) -> tuple[float, float]:
    runs = cells.get((ds, cf, fam), [])
    return agg([r["test_metrics"][metric_key] for r in runs if metric_key in r["test_metrics"]])


def comparison_questions(cells: dict, present_datasets: list[str]) -> list[str]:
    lines = ["## Sec P comparison questions\n"]
    pairs = [
        ("Q1: does dendritic structure itself help?", "scalar_dendrite", "plain_mlp"),
        (
            "Q2: is input reliability gating sufficient?",
            "scalar_dendrite_reliability_gated",
            "belief_dendrite",
        ),
        (
            "Q3: does hierarchical belief propagation beat dense belief propagation?",
            "belief_dendrite",
            "cellv0.3",
        ),
        (
            "Q4: is any gain just because the conventional model lacks reliability?",
            "belief_dendrite",
            "confidence_mlp",
        ),
    ]
    for title, fam_a, fam_b in pairs:
        lines.append(
            f"### {title}\n`{fam_a}` vs `{fam_b}` (corruption_AUC, accuracy primary metric)\n"
        )
        for ds in present_datasets:
            for cf in CORRUPTION_FAMILIES:
                a_m, a_s = _mean_metric_over(cells, ds, cf, fam_a, "corruption_auc_primary")
                b_m, b_s = _mean_metric_over(cells, ds, cf, fam_b, "corruption_auc_primary")
                if a_m != a_m or b_m != b_m:
                    continue
                delta = a_m - b_m
                lines.append(
                    f"- {ds}/{cf}: {fam_a}={a_m:.3f}±{a_s:.3f}  "
                    f"{fam_b}={b_m:.3f}±{b_s:.3f}  delta={delta:+.3f}"
                )
        lines.append("")

    lines.append("### Q5: does the mechanism localize corruption? (belief_dendrite only)\n")
    lines.append(
        "`localization_corr_overlap_vs_branch_pi` at the most severe evaluated severity per cell "
        "(expected direction: negative -- more corruption overlap -> lower branch pi).\n"
    )
    for ds in present_datasets:
        for cf in CORRUPTION_FAMILIES:
            runs = cells.get((ds, cf, "belief_dendrite"), [])
            if not runs:
                continue
            hi = MAX_OOD_SEVERITY[cf]
            corrs = []
            for r in runs:
                sev_list = r["config"]["extra"]["sweep"]["severities"]
                match = next((s for s in sev_list if abs(s["severity"] - hi) < 1e-9), None)
                if match and match.get("diagnostics"):
                    v = match["diagnostics"].get("localization_corr_overlap_vs_branch_pi")
                    if v == v:
                        corrs.append(v)
            m, s = agg(corrs)
            lines.append(
                f"- {ds}/{cf} @ severity={hi}: corr={m:.4f}±{s:.4f} (n_seeds={len(corrs)})"
            )
    lines.append("")
    return lines


def build_report(raw_dir: Path) -> str:
    rows = load_records(raw_dir)
    cells = by_cell(rows)
    present_datasets = [d for d in DATASETS if any(k[0] == d for k in cells)]

    out: list[str] = [
        "# Architecture V2 -- Belief Dendritic Network frozen benchmark report\n",
        f"{len(rows)} run records loaded from `{raw_dir}`.\n",
        f"Datasets present: {present_datasets}. Seeds: {SEEDS}.\n",
    ]
    out += cell_table(cells, present_datasets)
    out += comparison_questions(cells, present_datasets)
    return "\n".join(out)


def main() -> None:
    raw_dir = _REPO_ROOT / "experiments" / "belief_dendrite" / "results" / "raw"
    processed_dir = _REPO_ROOT / "experiments" / "belief_dendrite" / "results" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(raw_dir)
    (processed_dir / "belief_dendrite_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
