"""Combines `results/raw/*.json` RunRecords (+ `results/processed/
preflight.json`) into `experiments/belief_dendrite/strong_baselines_
results.md` (spec Sec 16): the main comparison table, the reliability-
intervention table, the BeliefDendrite/Transformer diagnostics tables, the
MPS speed table, and direct answers to Sec 13's six comparison questions.

Usage: python -m experiments.belief_dendrite.strong_baselines.report
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.belief_dendrite.strong_baselines.corruption import (  # noqa: E402
    EVAL_PATCH_SIDES,
    MAX_OOD_SEVERITY,
)
from experiments.belief_dendrite.strong_baselines.interventions import (  # noqa: E402
    CONDITIONS,
    INTERVENTION_FAMILIES,
    INTERVENTION_SEVERITIES,
)
from experiments.belief_dendrite.strong_baselines.models import (  # noqa: E402
    BELIEF_DENDRITE,
    CONFIDENCE_CNN,
    CONFIDENCE_TINY_VIT,
    MODEL_FAMILIES,
    RELIABILITY_GATED_TINY_VIT,
    SCALAR_DENDRITE,
    SMALL_CNN,
    TINY_VIT,
)
from experiments.belief_dendrite.strong_baselines.param_budget import (  # noqa: E402
    TARGET_PARAM_BUDGET,
)

_BASE_DIR = _REPO_ROOT / "experiments" / "belief_dendrite" / "strong_baselines"
DEFAULT_RAW_DIR = _BASE_DIR / "results" / "raw"
DEFAULT_PROCESSED_DIR = _BASE_DIR / "results" / "processed"
DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT / "experiments" / "belief_dendrite" / "strong_baselines_results.md"
)

TABLE_SEVERITIES: tuple[float, ...] = (0.0, 12.0, 16.0, 20.0, 24.0)


def load_records(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith("_history.json"):
            continue
        with p.open() as f:
            rows.append(json.load(f))
    return rows


def by_family(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["config"]["architecture"], []).append(r)
    return out


def agg(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v == v]
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


def _ms_per_step(runs: list[dict]) -> list[float]:
    out = []
    for r in runs:
        v = r["test_metrics"].get("time_per_step_seconds")
        if v == v:  # not NaN
            out.append(v * 1000.0)
    return out


def main_table(cells: dict[str, list[dict]]) -> list[str]:
    lines = [
        "## Main table -- CIFAR-10 primary benchmark\n",
        "Mean +/- population-std over seeds. `params` is the sized parameter "
        f"count against a {TARGET_PARAM_BUDGET:,}-parameter target (spec Sec 6).\n",
        "| model | params | clean | patch12 | patch16 | patch20 | patch24 | "
        "corruption_AUC | OOD_drop | ms/step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam in MODEL_FAMILIES:
        runs = cells.get(fam, [])
        if not runs:
            continue
        params = runs[0]["parameter_count"]
        sev_cols = []
        for sev in TABLE_SEVERITIES:
            m, s = agg(_sev_acc(runs, sev))
            sev_cols.append(f"{m:.4f}+/-{s:.4f}")
        auc_m, auc_s = agg(_auc(runs))
        drop_m, drop_s = agg(_drop(runs))
        ms_m, ms_s = agg(_ms_per_step(runs))
        row = (
            f"| {fam} | {params:,} | "
            + " | ".join(sev_cols)
            + f" | {auc_m:.3f}+/-{auc_s:.3f} | {drop_m:.4f}+/-{drop_s:.4f} "
            + f"| {ms_m:.2f}+/-{ms_s:.2f} |"
        )
        lines.append(row)
    lines.append("")
    return lines


def preflight_table(processed_dir: Path) -> list[str]:
    path = processed_dir / "preflight.json"
    lines = ["## MPS speed preflight\n"]
    if not path.exists():
        lines.append("(no `preflight.json` found -- run `run.py` without `--skip-preflight`.)\n")
        return lines
    data = json.loads(path.read_text())
    lines.append(
        f"Device: `{data['device']}`. Batch size used for the sweep: `{data['batch_size']}`.\n"
    )
    lines.append("| family | params | ms/step | examples/sec | MPS current alloc (MB) |")
    lines.append("|---|---|---|---|---|")
    for r in data["results"]:
        mem = r["mps_memory_mb"]["current_allocated_mb"] if r.get("mps_memory_mb") else float("nan")
        lines.append(
            f"| {r['family']} | {r['parameter_count']:,} | {r['ms_per_step']:.2f} | "
            f"{r['examples_per_sec']:.1f} | {mem:.2f} |"
        )
    lines.append("")
    if data["slowdown_flags"]:
        lines.append("**Slowdown flags:**\n")
        for f in data["slowdown_flags"]:
            lines.append(f"- {f}")
        lines.append("")
    return lines


def _diag_series(runs: list[dict], severity: float, key: str) -> list[float]:
    vals = []
    for r in runs:
        sev_list = r["config"]["extra"]["sweep"]["severities"]
        match = next((s for s in sev_list if abs(s["severity"] - severity) < 1e-9), None)
        if match and match.get("diagnostics") and key in match["diagnostics"]:
            v = match["diagnostics"][key]
            if v == v:
                vals.append(v)
    return vals


def belief_dendrite_diagnostics_table(cells: dict[str, list[dict]]) -> list[str]:
    runs = cells.get(BELIEF_DENDRITE, [])
    lines = ["## V2 mechanism diagnostics (BeliefDendrite, spec Sec 10)\n"]
    if not runs:
        lines.append("(no BeliefDendrite runs found.)\n")
        return lines
    lines.append(
        "Per-severity mean over seeds (layer 1 = the local_2d receptive-field layer). "
        "`corr(overlap, branch_pi)` and `corr(overlap, somatic_contribution)` test the "
        "causal chain overlap -> branch precision -> somatic routing (Sec 13 Q6).\n"
    )
    lines.append(
        "| severity | branch_pi (L1) | soma_pi (L1) | soma_conflict u (L1) | "
        "eff._branches (L1) | corr(overlap,branch_pi) | corr(overlap,somatic_contrib) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for sev in EVAL_PATCH_SIDES:
        branch_pi, _ = agg(_diag_series(runs, sev, "diag_layer1_pi_branch_mean"))
        soma_pi, _ = agg(_diag_series(runs, sev, "diag_layer1_pi_soma_mean"))
        soma_u, _ = agg(_diag_series(runs, sev, "diag_layer1_u_soma_mean"))
        eff_b, _ = agg(_diag_series(runs, sev, "diag_layer1_effective_branches_mean"))
        corr_pi, _ = agg(_diag_series(runs, sev, "localization_corr_overlap_vs_branch_pi"))
        corr_rb, _ = agg(
            _diag_series(runs, sev, "localization_corr_overlap_vs_somatic_contribution")
        )
        lines.append(
            f"| {sev} | {branch_pi:.4f} | {soma_pi:.4f} | {soma_u:.4f} | {eff_b:.3f} | "
            f"{corr_pi:.4f} | {corr_rb:.4f} |"
        )
    lines.append("")
    return lines


def transformer_diagnostics_table(cells: dict[str, list[dict]]) -> list[str]:
    lines = ["## Transformer diagnostics (spec Sec 11)\n"]
    conf_runs = cells.get(CONFIDENCE_TINY_VIT, [])
    gated_runs = cells.get(RELIABILITY_GATED_TINY_VIT, [])
    lines.append(
        "`ConfidenceTinyViT`: `||reliability_embedding|| / ||image_patch_embedding||`, mean "
        "over tokens/examples. `ReliabilityGatedTinyViT`: mean patch reliability. Both per "
        "severity, mean over seeds.\n"
    )
    lines.append(
        "| severity | ConfidenceTinyViT norm ratio | "
        "ReliabilityGatedTinyViT mean patch reliability |"
    )
    lines.append("|---|---|---|")
    for sev in EVAL_PATCH_SIDES:
        ratio, _ = agg(
            _diag_series(conf_runs, sev, "reliability_to_image_embedding_norm_ratio_mean")
        )
        mean_rel, _ = agg(_diag_series(gated_runs, sev, "mean_patch_reliability_mean"))
        lines.append(f"| {sev} | {ratio:.4f} | {mean_rel:.4f} |")
    lines.append("")
    return lines


def interventions_table(cells: dict[str, list[dict]]) -> list[str]:
    lines = ["## Reliability interventions (spec Sec 12)\n"]
    lines.append(
        "`delta_X = accuracy(correct reliability) - accuracy(X)`. A large positive delta "
        "means the model's use of reliability depends on correct spatial alignment, not "
        "merely its presence.\n"
    )
    header = (
        "| family | severity | "
        + " | ".join(f"acc_{c}" for c in CONDITIONS)
        + " | delta_all_ones | delta_shuffled |"
    )
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(CONDITIONS) + "---|---|")
    for fam in INTERVENTION_FAMILIES:
        runs = cells.get(fam, [])
        if not runs:
            continue
        for sev in INTERVENTION_SEVERITIES:
            per_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS}
            deltas_ones: list[float] = []
            deltas_shuf: list[float] = []
            for r in runs:
                entries = r["config"]["extra"].get("interventions") or []
                match = next((e for e in entries if abs(e["severity"] - sev) < 1e-9), None)
                if not match:
                    continue
                for c in CONDITIONS:
                    per_condition[c].append(match["accuracy_by_condition"][c])
                deltas_ones.append(match["delta_all_ones"])
                deltas_shuf.append(match["delta_shuffled"])
            if not deltas_ones:
                continue
            acc_cols = [f"{agg(per_condition[c])[0]:.4f}" for c in CONDITIONS]
            row = (
                f"| {fam} | {sev} | "
                + " | ".join(acc_cols)
                + f" | {agg(deltas_ones)[0]:+.4f} | {agg(deltas_shuf)[0]:+.4f} |"
            )
            lines.append(row)
    lines.append("")
    return lines


def _cell_metric(cells: dict, fam: str, key: str) -> tuple[float, float]:
    return agg([r["test_metrics"][key] for r in cells.get(fam, []) if key in r["test_metrics"]])


_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("clean accuracy", "accuracy@0.0"),
    ("corruption_AUC", "corruption_auc_primary"),
    ("acc@patch24 (max OOD)", "accuracy@24.0"),
)


def comparison_questions(cells: dict[str, list[dict]]) -> list[str]:
    lines = ["## Sec 13 comparison questions\n"]
    pairs = [
        ("Q1: is V2 merely rediscovering convolution?", BELIEF_DENDRITE, SMALL_CNN),
        (
            "Q2: does a CNN with the exact reliability mask eliminate the advantage?",
            BELIEF_DENDRITE,
            CONFIDENCE_CNN,
        ),
        ("Q3: can a Transformer learn the same behavior?", BELIEF_DENDRITE, TINY_VIT),
        (
            "Q4a: does explicit reliability eliminate the advantage (additive)?",
            BELIEF_DENDRITE,
            CONFIDENCE_TINY_VIT,
        ),
        (
            "Q4b: does explicit reliability eliminate the advantage (gated)?",
            BELIEF_DENDRITE,
            RELIABILITY_GATED_TINY_VIT,
        ),
        (
            "Q5: is the gain dendritic locality rather than belief propagation?",
            BELIEF_DENDRITE,
            SCALAR_DENDRITE,
        ),
    ]
    for title, fam_a, fam_b in pairs:
        lines.append(f"### {title}\n`{fam_a}` vs `{fam_b}`\n")
        for label, key in _COMPARISON_METRICS:
            a_m, a_s = _cell_metric(cells, fam_a, key)
            b_m, b_s = _cell_metric(cells, fam_b, key)
            if a_m != a_m or b_m != b_m:
                continue
            lines.append(
                f"- {label}: {fam_a}={a_m:.4f}+/-{a_s:.4f}  {fam_b}={b_m:.4f}+/-{b_s:.4f}  "
                f"delta={a_m - b_m:+.4f}"
            )
        lines.append("")

    lines.append(
        "### Q6: does V2 continue to show the intended mechanism on a harder visual dataset?\n"
        "`patch overlap -> branch pi -> somatic contribution` -- see the BeliefDendrite "
        f"diagnostics table above; both correlations at the max-OOD severity "
        f"({MAX_OOD_SEVERITY}) are the headline numbers (expected direction for both: "
        "negative).\n"
    )
    runs = cells.get(BELIEF_DENDRITE, [])
    corr_pi, corr_pi_s = agg(
        _diag_series(runs, MAX_OOD_SEVERITY, "localization_corr_overlap_vs_branch_pi")
    )
    corr_rb, corr_rb_s = agg(
        _diag_series(runs, MAX_OOD_SEVERITY, "localization_corr_overlap_vs_somatic_contribution")
    )
    lines.append(
        f"- corr(overlap, branch_pi) @ patch={MAX_OOD_SEVERITY}: {corr_pi:.4f}+/-{corr_pi_s:.4f}"
    )
    lines.append(
        f"- corr(overlap, somatic_contribution) @ patch={MAX_OOD_SEVERITY}: "
        f"{corr_rb:.4f}+/-{corr_rb_s:.4f}"
    )
    lines.append("")
    return lines


def build_report(raw_dir: Path, processed_dir: Path) -> str:
    rows = load_records(raw_dir)
    cells = by_family(rows)

    out: list[str] = [
        "# Architecture V2 -- Strong CNN + Transformer Baseline Benchmark (CIFAR-10)\n",
        f"{len(rows)} run records loaded from `{raw_dir}`. Families present: "
        f"{sorted(cells.keys())}.\n",
        "BeliefDendriteNetwork and its equations are frozen and reused unmodified from "
        "`src/models/architecture_v2/belief_dendrite.py` -- see `docs/architecture_v2.md`. "
        "MNIST/Fashion-MNIST were not rerun for this new model roster (spec Sec 2 permits "
        "skipping them when not cheap; CIFAR-10 is the primary and, here, only result). "
        "The optional dense CellV0.3 reference (spec Sec 5) was not run, to avoid delaying "
        "the seven required arms, per that section's own instruction.\n",
    ]
    out += main_table(cells)
    out += preflight_table(processed_dir)
    out += interventions_table(cells)
    out += belief_dendrite_diagnostics_table(cells)
    out += transformer_diagnostics_table(cells)
    out += comparison_questions(cells)
    return "\n".join(out)


def main() -> None:
    report = build_report(DEFAULT_RAW_DIR, DEFAULT_PROCESSED_DIR)
    DEFAULT_OUTPUT_PATH.write_text(report)
    print(f"wrote {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
