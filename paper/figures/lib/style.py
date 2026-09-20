"""One fixed visual identity per model, reused across every Paper-A figure.

Colorblind-safe (Okabe-Ito) palette. Every model also gets a distinct
marker + linestyle so figures stay legible in grayscale printing.
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Canonical model keys (match the `family` strings in the processed run
# records) -> a human-readable legend label.
# ---------------------------------------------------------------------------

CELLV03 = "cellv0.3"
PLAIN_MLP = "plain_mlp"
CONFIDENCE_MLP = "confidence_mlp"
CONFIDENCE_MLP_SAME_WIDTH = "confidence_mlp_same_width"
RELIABILITY_GATED_MLP = "reliability_gated_mlp"
NEUMISS = "neumiss"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"

LABELS: dict[str, str] = {
    CELLV03: "BVU",
    PLAIN_MLP: "Plain MLP",
    CONFIDENCE_MLP: "Confidence MLP",
    CONFIDENCE_MLP_SAME_WIDTH: "Same-width Confidence MLP",
    RELIABILITY_GATED_MLP: "Reliability-gated MLP",
    NEUMISS: "NeuMiss",
    HIST_GRADIENT_BOOSTING: "HistGradientBoosting",
}

# Okabe-Ito colorblind-safe palette. BVU gets black (strongest,
# darkest emphasis, per the manuscript spec) and a solid line; every
# baseline gets a lighter color plus a non-solid linestyle and a marker
# that remains distinguishable from the others without color.
_STYLE: dict[str, dict] = {
    CELLV03: dict(
        color="#000000", marker="o", linestyle="-",
        linewidth=2.2, markersize=6.5, zorder=5,
    ),
    PLAIN_MLP: dict(
        color="#0072B2", marker="s", linestyle="--",
        linewidth=1.9, markersize=5.5, zorder=3,
    ),
    CONFIDENCE_MLP: dict(
        color="#D55E00", marker="^", linestyle="-.",
        linewidth=1.9, markersize=6, zorder=3,
    ),
    CONFIDENCE_MLP_SAME_WIDTH: dict(
        color="#56B4E9", marker="D", linestyle=(0, (1, 1)),
        linewidth=1.9, markersize=5, zorder=2,
    ),
    RELIABILITY_GATED_MLP: dict(
        color="#009E73", marker="v", linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.9, markersize=5.5, zorder=2,
    ),
    NEUMISS: dict(
        color="#CC79A7", marker="P", linestyle="--",
        linewidth=1.9, markersize=6.5, zorder=3,
    ),
    HIST_GRADIENT_BOOSTING: dict(
        color="#999999", marker="X", linestyle=(0, (1, 1)),
        linewidth=1.8, markersize=6, zorder=1,
    ),
}


def style_of(family: str) -> dict:
    """Return the fixed `plot(**kwargs)`-ready style for a model family key."""
    if family not in _STYLE:
        raise KeyError(f"No visual identity assigned for model family {family!r}")
    return dict(_STYLE[family])


def label_of(family: str) -> str:
    if family not in LABELS:
        raise KeyError(f"No legend label assigned for model family {family!r}")
    return LABELS[family]


def apply_rcparams() -> None:
    """Publication rcParams: vector text, serif academic type, clean axes."""
    mpl.rcParams.update({
        # Vector / editable text in the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Serif type to match standard LaTeX article/JMLR-style body text.
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman No9 L", "Times New Roman", "Times",
            "Liberation Serif", "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        # Clean, white, minimal-clutter axes.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# Shared visual convention for "training range ends here" boundary lines.
TRAIN_BOUNDARY_STYLE = dict(color="#666666", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
OOD_LABEL_STYLE = dict(fontsize=7.5, color="#666666", style="italic")

# Shared band alpha for +/- 1 std shaded regions.
BAND_ALPHA = 0.16

# BVU internal-diagnostic layers (Fig 3, right panel) -- both derived
# from the BVU black identity, distinguished by shade/marker/linestyle
# rather than a second hue (they are the same model's two hidden layers).
LAYER_STYLE: dict[str, dict] = {
    "layer1": dict(
        color="#999999", marker="o", linestyle="--", linewidth=1.9, markersize=5.5, zorder=3,
    ),
    "layer2": dict(
        color="#000000", marker="s", linestyle="-", linewidth=2.0, markersize=5.5, zorder=4,
    ),
}
LAYER_LABELS: dict[str, str] = {
    "layer1": "Layer 1 precision $\\pi$",
    "layer2": "Layer 2 precision $\\pi$",
}
