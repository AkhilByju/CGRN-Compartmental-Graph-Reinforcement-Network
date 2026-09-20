#!/usr/bin/env python3
"""Paper A -- Figure S1 (supplementary/appendix): clean 7-dataset benchmark.

Visualization only, low priority (build only after Fig 2 / Fig 3 are
correct -- Paper-A figures task). Reads the frozen Phase-1 BVU screen
(`phase1_cellv03_runs.json`) and its recorded parameter-matched-MLP arm
(`phase1_runs_*.json`); writes nothing back to any result file.

A compact forest-style dot/error-bar plot: for each of the 7 frozen
Phase-1 datasets (at the 100% train fraction), the paired per-seed delta
`headline(BVU) - headline(MLP matched)` (mean +/- 1 sample-std over 3
seeds). Datasets mix accuracy and R^2 headline metrics, so the delta -- not
the raw value -- is what is comparable across the row; the metric name is
annotated per row instead of implied by a shared y-axis.

Usage:
    python paper/figures/supplementary/make_figS1_clean_benchmarks.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

from lib import data as D  # noqa: E402
from lib import style as S  # noqa: E402

OUT_DIR = _HERE
DATASET_LABEL = {
    "breast_cancer": "Breast Cancer",
    "wine": "Wine",
    "digits": "Digits",
    "diabetes": "Diabetes",
    "california_housing": "California Housing",
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
}


def main() -> None:
    S.apply_rcparams()

    print("Figure S1 -- clean 7-dataset benchmark (supplementary)")
    print(f"  source: {D.PHASE1_CELLV03_RUNS.relative_to(D.REPO_ROOT)}")
    for ds, path in D.PHASE1_MLP_SOURCES.items():
        print(f"  source: {path.relative_to(D.REPO_ROOT)}  [{ds}]")

    rows = []
    for ds in D.PHASE1_CLEAN_DATASETS:
        m, sd, n, metric = D.phase1_paired_delta(ds, train_fraction=1.0)
        rows.append((ds, m, sd, n, metric))
        print(f"    {ds}: delta={m:+.4f} +/- {sd:.4f} ({metric}, n={n} seeds)")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    y = np.arange(len(rows))[::-1]
    sty = S.style_of(S.CELLV03)
    for yi, (ds, m, sd, n, metric) in zip(y, rows):
        ax.errorbar(
            m, yi, xerr=sd, fmt=sty["marker"], color=sty["color"],
            markersize=sty["markersize"] + 1, elinewidth=1.3, capsize=3, zorder=3,
        )

    ax.axvline(0.0, color="#888888", linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{DATASET_LABEL[ds]}  ({metric})" for ds, _m, _sd, _n, metric in rows
    ])
    ax.set_xlabel(r"$\Delta$ headline metric (BVU $-$ MLP, matched params)")
    ax.set_title(
        "Clean 7-dataset benchmark: BVU vs parameter-matched MLP",
        loc="left", fontsize=9, fontweight="bold",
    )
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()

    pdf_path = OUT_DIR / "figS1_clean_benchmarks.pdf"
    png_path = OUT_DIR / "figS1_clean_benchmarks.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    csv_path = OUT_DIR / "figS1_clean_benchmarks_data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "metric", "delta_mean", "delta_std", "n_seeds"])
        for ds, m, sd, n, metric in rows:
            w.writerow([ds, metric, m, sd, n])

    print(f"Wrote {pdf_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {png_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {csv_path.relative_to(D.REPO_ROOT)}")


if __name__ == "__main__":
    main()
