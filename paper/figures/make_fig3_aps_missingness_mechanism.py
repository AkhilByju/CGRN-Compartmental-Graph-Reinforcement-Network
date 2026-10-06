#!/usr/bin/env python3
"""Paper A -- Figure 3: real missingness and mechanism (1x2 panel, APS).

Visualization only. Reads the frozen Phase-3 Part B processed run record
(`real_reliability_runs.json`); writes nothing back to any result file.

Left panel: APS PR-AUC by real missingness stratum, for BVU, NeuMiss,
Confidence MLP, Plain MLP. Right panel: BVU's own internal hidden
precision (pi = e/(1+e*u)) by the same strata, layer 1 and layer 2 shown
separately. Mean +/- 1 sample-std over 3 seeds; no trend line fit; a bin
with no valid seed value is dropped rather than interpolated.

Usage:
    python paper/figures/make_fig3_aps_missingness_mechanism.py
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
DATASET = "aps"
LEFT_FAMILIES = (S.CELLV03, S.NEUMISS, S.CONFIDENCE_MLP, S.PLAIN_MLP)
LAYERS = (("layer1", "l1_pi_mean"), ("layer2", "l2_pi_mean"))


def _print_provenance(rows: list[dict]) -> None:
    print("Figure 3 -- real missingness and mechanism (APS)")
    print(f"  source: {D.REAL_RELIABILITY_RUNS.relative_to(D.REPO_ROOT)}  ({len(rows)} rows total)")
    aps_rows = [r for r in rows if r["dataset"] == DATASET]
    print(f"    dataset=aps rows used: {len(aps_rows)}")
    for fam in LEFT_FAMILIES:
        n = sum(1 for r in aps_rows if r["family"] == fam)
        print(f"    aps/{fam}: {n} seed-rows")


def main() -> None:
    S.apply_rcparams()

    all_rows = D.load_real_reliability_runs()
    _print_provenance(all_rows)
    aps_by_fam = D.group_by([r for r in all_rows if r["dataset"] == DATASET], lambda r: r["family"])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.1, 3.2))
    csv_rows = []

    # ---- Left panel: PR-AUC by missingness stratum, 4 models ----
    bin_labels_ref = None
    for fam in LEFT_FAMILIES:
        rows = aps_by_fam[fam]
        primary = rows[0]["primary_metric"]
        labels, means, stds, ns = D.stratum_metric_series(rows, primary)
        if bin_labels_ref is None:
            bin_labels_ref = labels
        assert labels == bin_labels_ref, f"{fam}: bin set {labels} != {bin_labels_ref}"
        x = np.arange(len(labels))
        means_arr, stds_arr = np.array(means), np.array(stds)
        sty = S.style_of(fam)
        ax_left.plot(x, means_arr, label=S.label_of(fam), **sty)
        ax_left.errorbar(
            x, means_arr, yerr=stds_arr, fmt="none",
            ecolor=sty["color"], elinewidth=1.1, capsize=2.5, alpha=0.7, zorder=sty["zorder"] - 1,
        )
        for lbl, m, sd, n in zip(labels, means, stds, ns):
            csv_rows.append({
                "panel": "performance", "dataset": DATASET, "model": S.label_of(fam),
                "bin": lbl, "layer": "", "metric": primary, "mean": m, "std": sd, "n_seeds": n,
            })

    ax_left.set_xticks(np.arange(len(bin_labels_ref)))
    ax_left.set_xticklabels([D.COMPACT_BIN_LABEL[b] for b in bin_labels_ref])
    ax_left.set_xlabel("Missingness stratum")
    ax_left.set_ylabel("PR-AUC")
    ax_left.set_title(
        "APS -- predictive performance by missingness", loc="left", fontsize=9.5, fontweight="bold",
    )
    ax_left.legend(loc="lower left", frameon=False, fontsize=7.8, handlelength=2.2)
    ax_left.grid(True, axis="both")

    # ---- Right panel: BVU internal precision by stratum, 2 layers ----
    cellv03_rows = aps_by_fam[S.CELLV03]
    for layer_key, field in LAYERS:
        labels, means, stds, ns = D.belief_stratum_series(cellv03_rows, field)
        assert labels == bin_labels_ref, f"{layer_key}: bin set {labels} != {bin_labels_ref}"
        x = np.arange(len(labels))
        means_arr, stds_arr = np.array(means), np.array(stds)
        sty = S.LAYER_STYLE[layer_key]
        ax_right.plot(x, means_arr, label=S.LAYER_LABELS[layer_key], **sty)
        ax_right.errorbar(
            x, means_arr, yerr=stds_arr, fmt="none",
            ecolor=sty["color"], elinewidth=1.1, capsize=2.5, alpha=0.7, zorder=sty["zorder"] - 1,
        )
        for lbl, m, sd, n in zip(labels, means, stds, ns):
            csv_rows.append({
                "panel": "precision", "dataset": DATASET, "model": S.label_of(S.CELLV03),
                "bin": lbl, "layer": layer_key, "metric": "hidden_precision_pi",
                "mean": m, "std": sd, "n_seeds": n,
            })

    ax_right.set_xticks(np.arange(len(bin_labels_ref)))
    ax_right.set_xticklabels([D.COMPACT_BIN_LABEL[b] for b in bin_labels_ref])
    ax_right.set_xlabel("Missingness stratum")
    ax_right.set_ylabel(r"Mean hidden precision $\pi$")
    ax_right.set_title(
        "BVU -- internal precision by missingness",
        loc="left", fontsize=9.5, fontweight="bold",
    )
    ax_right.legend(loc="upper right", frameon=False, fontsize=7.8, handlelength=2.2)
    ax_right.grid(True, axis="both")

    fig.tight_layout()

    pdf_path = OUT_DIR / "fig3_aps_missingness_mechanism.pdf"
    png_path = OUT_DIR / "fig3_aps_missingness_mechanism.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    csv_path = OUT_DIR / "fig3_aps_missingness_mechanism_data.csv"
    fieldnames = ["panel", "dataset", "model", "bin", "layer", "metric", "mean", "std", "n_seeds"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"Wrote {pdf_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {png_path.relative_to(D.REPO_ROOT)}")
    print(f"Wrote {csv_path.relative_to(D.REPO_ROOT)}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
