"""Builds the task Sec 19 results doc from `results/processed/sweep_summary.json`
and `results/processed/mps_speed_preflight.json`: the main comparison table,
learning-efficiency and training-loss plots, belief-diagnostics and
neutralization tables, generation-diversity table, and a Sec 18
interpretation bucket -- computed mechanically from the rubric's own
stated criteria, not asserted by hand.

Training-loss-vs-tokens (Sec 11's "also report" list) is reconstructed from
`run_sweep.py`'s own per-step console log lines
(`[<model> seed=<s> lr=<lr>] step .../... tokens=... loss=...`,
emitted every `log_every_steps`) rather than logged separately during
training, so no extra instrumentation or re-run was needed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

_STEP_LOG_RE = re.compile(
    r"\[(?P<model>[\w.]+) seed=(?P<seed>\d+) lr=(?P<lr>[\d.eE+-]+)\] "
    r"step (?P<step>\d+)/(?P<total>\d+) tokens=(?P<tokens>\d+) loss=(?P<loss>[\d.]+)"
)


def load_sweep_summary(path: Path | None = None) -> dict[str, Any]:
    path = path or (RESULTS_DIR / "processed" / "sweep_summary.json")
    return json.loads(path.read_text())


def load_preflight(path: Path | None = None) -> dict[str, Any]:
    path = path or (RESULTS_DIR / "processed" / "mps_speed_preflight.json")
    return json.loads(path.read_text())


def parse_training_log(log_path: Path) -> dict[tuple[str, int, float], list[tuple[int, float]]]:
    """`{(model_kind, seed, lr): [(tokens_seen, loss), ...]}`, one entry per
    logged step, in the order they were printed."""
    curves: dict[tuple[str, int, float], list[tuple[int, float]]] = {}
    for line in log_path.read_text().splitlines():
        m = _STEP_LOG_RE.search(line)
        if not m:
            continue
        key = (m.group("model"), int(m.group("seed")), float(m.group("lr")))
        curves.setdefault(key, []).append((int(m.group("tokens")), float(m.group("loss"))))
    return curves


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def aggregate_over_seeds(summary: dict[str, Any], model_kind: str) -> dict[str, Any]:
    """Mean +/- std across seeds of the *selected* run's val/test metrics."""
    seed_results = summary["models"][model_kind]["seeds"].values()
    val_nlls = [
        # validation NLL of the selected (lower-val-NLL) LR run
        min(r["all_lr_val_nll"].values())
        for r in seed_results
    ]
    test_nlls = [r["test_metrics"]["nll"] for r in seed_results]
    test_ppls = [r["test_metrics"]["perplexity"] for r in seed_results]
    test_bpt = [r["test_metrics"]["bits_per_token"] for r in seed_results]
    selected_lrs = [r["selected_lr"] for r in seed_results]
    return {
        "val_nll_mean": _mean(val_nlls),
        "val_nll_std": _std(val_nlls),
        "test_nll_mean": _mean(test_nlls),
        "test_nll_std": _std(test_nlls),
        "test_ppl_mean": _mean(test_ppls),
        "test_bpt_mean": _mean(test_bpt),
        "selected_lrs": selected_lrs,
    }


def build_main_table(summary: dict[str, Any], preflight: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_kind in summary["models"]:
        agg = aggregate_over_seeds(summary, model_kind)
        one_seed = next(iter(summary["models"][model_kind]["seeds"].values()))
        pf = preflight["per_model"][model_kind]
        rows.append(
            {
                "model": model_kind,
                "params": one_seed["param_report"]["total"],
                "hidden_width": one_seed["param_report"]["ffn_hidden_width"],
                "val_nll": agg["val_nll_mean"],
                "test_nll": agg["test_nll_mean"],
                "test_ppl": agg["test_ppl_mean"],
                "bits_per_token": agg["test_bpt_mean"],
                "tokens_per_sec": pf["tokens_per_sec"],
                "ms_per_step": pf["ms_per_step"],
            }
        )
    return rows


def render_main_table_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| model | params | hidden width | val NLL | test NLL | PPL | bits/token "
        "| tokens/sec | ms/step |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params']:,} | {r['hidden_width']} | {r['val_nll']:.4f} "
            f"| {r['test_nll']:.4f} | {r['test_ppl']:.2f} | {r['bits_per_token']:.3f} "
            f"| {r['tokens_per_sec']:,.0f} | {r['ms_per_step']:.2f} |"
        )
    return "\n".join(lines)


def plot_learning_curves(summary: dict[str, Any], out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_kind, model_data in summary["models"].items():
        seeds = list(model_data["seeds"].values())
        curve = seeds[0]["learning_curve"]
        tokens = [pt["tokens_seen"] for pt in curve]
        means, stds = [], []
        for i in range(len(tokens)):
            nlls = [s["learning_curve"][i]["nll"] for s in seeds]
            means.append(_mean(nlls))
            stds.append(_std(nlls))
        means_arr = means
        ax.plot(tokens, means_arr, marker="o", label=model_kind)
        ax.fill_between(
            tokens,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            alpha=0.2,
        )
    ax.set_xlabel("processed training tokens")
    ax.set_ylabel("validation NLL (nats/token)")
    ax.set_title("Learning efficiency: validation NLL vs. processed tokens")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_loss(
    curves: dict[tuple[str, int, float], list[tuple[int, float]]],
    summary: dict[str, Any],
    out_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_kind, model_data in summary["models"].items():
        for seed_str, seed_result in model_data["seeds"].items():
            seed = int(seed_str)
            lr = seed_result["selected_lr"]
            key = (model_kind, seed, lr)
            if key not in curves:
                continue
            points = curves[key]
            tokens = [p[0] for p in points]
            losses = [p[1] for p in points]
            ax.plot(tokens, losses, alpha=0.6, label=f"{model_kind} (seed {seed})")
    ax.set_xlabel("processed training tokens")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_title("Training loss vs. processed tokens (selected-LR runs)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_diagnostics_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """cellv0.3 only: per training_fraction, per layer, mean over seeds."""
    if "cellv0.3" not in summary["models"]:
        return []
    seeds = list(summary["models"]["cellv0.3"]["seeds"].values())
    if not seeds or not seeds[0]["diagnostics"]:
        return []

    rows = []
    n_snapshots = len(seeds[0]["diagnostics"])
    for snap_idx in range(n_snapshots):
        fraction = seeds[0]["diagnostics"][snap_idx]["training_fraction"]
        layer_keys = seeds[0]["diagnostics"][snap_idx]["layers"].keys()
        for layer in layer_keys:
            for metric in ("pi_mean", "pi_std", "pi_cv", "u_mean", "u_std"):
                values = [
                    s["diagnostics"][snap_idx]["layers"][layer][metric] for s in seeds
                ]
                rows.append(
                    {
                        "training_fraction": fraction,
                        "layer": layer,
                        "metric": metric,
                        "mean": _mean(values),
                    }
                )
    return rows


def build_mechanism_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if "cellv0.3" not in summary["models"]:
        return []
    seeds = list(summary["models"]["cellv0.3"]["seeds"].values())
    rows = []
    for key in (
        "precision_neutralized_delta_nll",
        "precision_neutralized_delta_ppl",
        "conflict_neutralized_delta_nll",
        "conflict_neutralized_delta_ppl",
    ):
        values = [s["mechanism_test"]["deltas"][key] for s in seeds if s.get("mechanism_test")]
        if values:
            rows.append({"metric": key, "mean": _mean(values), "std": _std(values)})
    return rows


def build_generation_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_kind, model_data in summary["models"].items():
        seeds = list(model_data["seeds"].values())
        agg: dict[str, float] = {}
        for metric in ("distinct_1", "distinct_2", "repeated_bigram_rate", "repeated_trigram_rate"):
            values = [s["generation_aggregate_diversity"][metric] for s in seeds]
            agg[metric] = _mean(values)
        rows.append({"model": model_kind, **agg})
    return rows


def _sample_efficiency_ok(summary: dict[str, Any], tolerance: float = 1.02) -> bool:
    """cellv0.3's learning-curve val NLL is <= swiglu's at every checkpoint
    (within `tolerance`), averaged over seeds."""
    cellv03_seeds = list(summary["models"]["cellv0.3"]["seeds"].values())
    swiglu_seeds = list(summary["models"]["swiglu"]["seeds"].values())
    n_points = len(cellv03_seeds[0]["learning_curve"])
    for i in range(n_points):
        cellv03_nll = _mean([s["learning_curve"][i]["nll"] for s in cellv03_seeds])
        swiglu_nll = _mean([s["learning_curve"][i]["nll"] for s in swiglu_seeds])
        if cellv03_nll > swiglu_nll * tolerance:
            return False
    return True


def classify_interpretation(
    summary: dict[str, Any],
    main_rows: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    diagnostics_rows: list[dict[str, Any]],
) -> str:
    """Sec 18's four buckets, applied mechanically to the collected numbers.
    "Matches" is defined as within 2% test NLL; "nontrivial belief" as a
    mean hidden `u` clearly above the eps floor; "neutralization hurts" as
    either ablation raising test NLL by more than 0.005 nats/token;
    "reasonable compute" as cellv0.3 within 3x ModernTransformer's ms/step
    (Sec 15's own threshold)."""
    by_model = {r["model"]: r for r in main_rows}
    if "cellv0.3" not in by_model or "swiglu" not in by_model:
        return "insufficient_data"

    cellv03, swiglu = by_model["cellv0.3"], by_model["swiglu"]
    matches_or_beats = cellv03["test_nll"] <= swiglu["test_nll"] * 1.02
    sample_efficiency_ok = _sample_efficiency_ok(summary)

    u_means = [r["mean"] for r in diagnostics_rows if r["metric"] == "u_mean"]
    nontrivial_belief = bool(u_means) and _mean(u_means) > 1e-3

    deltas = {r["metric"]: r["mean"] for r in mechanism_rows}
    neutralization_hurts = (
        deltas.get("precision_neutralized_delta_nll", 0.0) > 0.005
        or deltas.get("conflict_neutralized_delta_nll", 0.0) > 0.005
    )

    reasonable_compute = cellv03["ms_per_step"] <= 3.0 * swiglu["ms_per_step"]

    if (
        matches_or_beats
        and sample_efficiency_ok
        and nontrivial_belief
        and neutralization_hurts
        and reasonable_compute
    ):
        return "strong_positive"
    if matches_or_beats and nontrivial_belief and neutralization_hurts:
        return "mechanism_positive_performance_neutral"
    if matches_or_beats:
        return "interesting_neutral"
    if not matches_or_beats and not reasonable_compute:
        return "negative"
    if nontrivial_belief and neutralization_hurts:
        return "mechanism_positive_performance_neutral"
    return "negative"
