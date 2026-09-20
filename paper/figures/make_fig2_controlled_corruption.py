#!/usr/bin/env python3
"""Paper A -- Figure 2: controlled corruption robustness (2x2 panel).

Visualization only. Reads frozen Phase-2 (`reliability_runs.json`) and
Phase-3 Part A (`capacity_stress_runs.json`) processed run records; writes
nothing back to any result file.

Panels: {MNIST, Fashion-MNIST} x {missing-feature, heterogeneous Gaussian}.
Curves: BVU, Plain MLP, Confidence MLP (Phase 2) + Same-width
Confidence MLP (Phase 3 Part A). Mean +/- 1 sample-std over 3 seeds; a
severity's replicas are already averaged within-seed inside the processed
record (see `lib/data.py::check_replica_average`).

Usage:
    python paper/figures/make_fig2_controlled_corruption.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib import data as D  # noqa: E402
from lib import style as S  # noqa: E402

OUT_DIR = _HERE
PANELS = [
    ("mnist", D.MISSING, "MNIST -- missing features"),
    ("mnist", D.GAUSSIAN, "MNIST -- heterogeneous Gaussian noise"),
    ("fashion_mnist", D.MISSING, "Fashion-MNIST -- missing features"),
    ("fashion_mnist", D.GAUSSIAN, "Fashion-MNIST -- heterogeneous Gaussian noise"),
]
PHASE2_FAMILIES = (S.CELLV03, S.PLAIN_MLP, S.CONFIDENCE_MLP)
PHASE3_FAMILY = S.CONFIDENCE_MLP_SAME_WIDTH
LEGEND_ORDER = (S.CELLV03, S.PLAIN_MLP, S.CONFIDENCE_MLP, S.CONFIDENCE_MLP_SAME_WIDTH)

X_LABEL = {D.MISSING: "Missing-feature probability", D.GAUSSIAN: "Maximum noise scale $s$"}


def _print_provenance(rel_rows: list[dict], cap_rows: list[dict]) -> None:
    print("Figure 2 -- controlled corruption robustness")
    print(f"  source: {D.RELIABILITY_RUNS.relative_to(D.REPO_ROOT)}  ({len(rel_rows)} rows)")
    print(f"  source: {D.CAPACITY_STRESS_RUNS.relative_to(D.REPO_ROOT)}  ({len(cap_rows)} rows)")
    def _n(rows, ds, cf, fam):
        return sum(
            1 for r in rows
            if r["dataset"] == ds and r["corruption_family"] == cf and r["family"] == fam
        )

    for ds, cf, _title in PANELS:
        for fam in PHASE2_FAMILIES:
            print(f"    {ds}/{cf}/{fam}: {_n(rel_rows, ds, cf, fam)} seed-rows")
        print(f"    {ds}/{cf}/{PHASE3_FAMILY}: {_n(cap_rows, ds, cf, PHASE3_FAMILY)} seed-rows")


def _panel_data(rel_rows, cap_rows, dataset, cf):
    rel_by_fam = D.group_by(
        [r for r in rel_rows if r["dataset"] == dataset and r["corruption_family"] == cf],
        lambda r: r["family"],
    )
    cap_by_fam = D.group_by(
        [r for r in cap_rows if r["dataset"] == dataset and r["corruption_family"] == cf],
        lambda r: r["family"],
    )
    primary = rel_by_fam[S.CELLV03][0]["primary_metric"]
    out = {}
    for fam in PHASE2_FAMILIES:
        sevs, means, stds, ns = D.series_for(rel_by_fam[fam], cf, primary)
        out[fam] = (sevs, means, stds, ns, primary)
    sevs, means, stds, ns = D.series_for(cap_by_fam[PHASE3_FAMILY], cf, primary)
    out[PHASE3_FAMILY] = (sevs, means, stds, ns, primary)
    return out


def _shared_ylim(panel_data_by_dataset: dict[str, list[dict]]) -> dict[str, tuple[float, float]]:
    ylims = {}
    for ds, panels in panel_data_by_dataset.items():
        lo, hi = 1.0, 0.0
        for panel in panels:
            for fam, (sevs, means, stds, ns, primary) in panel.items():
                for m, sd in zip(means, stds):
                    lo = min(lo, m - sd)
                    hi = max(hi, m + sd)
        pad = 0.03 * (hi - lo) if hi > lo else 0.02
        ylims[ds] = (max(0.0, lo - pad), min(1.02, hi + pad))
    return ylims


def main() -> None:
    S.apply_rcparams()

    rel_rows = D.load_reliability_runs()
    cap_rows = D.load_capacity_stress_runs()
    _print_provenance(rel_rows, cap_rows)

    panel_data = {(ds, cf): _panel_data(rel_rows, cap_rows, ds, cf) for ds, cf, _ in PANELS}
    by_dataset = {}
    for ds, cf, _ in PANELS:
        by_dataset.setdefault(ds, []).append(panel_data[(ds, cf)])
    ylims = _shared_ylim(by_dataset)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3))
    csv_rows = []

    for ax, (ds, cf, title) in zip(axes.flat, PANELS):
        pdata = panel_data[(ds, cf)]
        for fam in LEGEND_ORDER:
            sevs, means, stds, ns, primary = pdata[fam]
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            sty = S.style_of(fam)
            ax.plot(sevs, means_arr, label=S.label_of(fam), **sty)
            ax.fill_between(
                sevs, means_arr - stds_arr, means_arr + stds_arr,
                color=sty["color"], alpha=S.BAND_ALPHA, linewidth=0, zorder=sty["zorder"] - 1,
            )
            for sev, m, sd, n in zip(sevs, means, stds, ns):
                csv_rows.append({
                    "panel": title, "dataset": ds, "corruption_family": cf,
                    "model": S.label_of(fam), "severity": sev, "mean": m, "std": sd, "n_seeds": n,
                })

        boundary = D.train_boundary(cf)
        ax.axvline(boundary, **S.TRAIN_BOUNDARY_STYLE)
        x_span = max(sevs) - min(sevs)
        ax.text(
            boundary + 0.015 * x_span, ylims[ds][0] + 0.06 * (ylims[ds][1] - ylims[ds][0]),
            "OOD", ha="left", va="bottom", **S.OOD_LABEL_STYLE,
        )

        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.set_xlabel(X_LABEL[cf])
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(*ylims[ds])
        ax.set_xlim(min(sevs) - 0.02 * x_span, max(sevs) + 0.02 * x_span)
        ax.grid(True, axis="both")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
        frameon=False, columnspacing=1.6, handlelength=2.4,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    pdf_path = OUT_DIR / "fig2_controlled_corruption.pdf"
    png_path = OUT_DIR / "fig2_controlled_corruption.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    csv_path = OUT_DIR / "fig2_controlled_corruption_data.csv"
    fieldnames = [
        "panel", "dataset", "corruption_family", "model", "severity", "mean", "std", "n_seeds",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"Wrote {pdf_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {png_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {csv_path.relative_to(D.REPO_ROOT)}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
