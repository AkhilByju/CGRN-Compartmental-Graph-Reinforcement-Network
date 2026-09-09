#!/usr/bin/env python3
"""Paper A Sec 10 -- controlled duplication-invariance experiment.

Purely a forward-pass diagnostic, independent of any training. Builds a fixed
set of distinct source belief states (distinct content, evidence, uncertainty,
and relevance), then feeds a `BeliefLayer` that *same multiset repeated*
`m in {1, 2, 4, 8, 16}` times -- every duplicate an exact copy
(content/evidence/uncertainty/relevance/content-weight all identical). Since
the duplicates carry no new information, a scale-stable fusion rule's output
`(mu, e, u)` must not change with `m`.

Runs three of the frozen `BeliefLayer` aggregation rules unmodified:

* ``scale_stable_precision`` -- CellV0.1. Expected: invariant to float tolerance.
* ``normalized_precision``   -- Method D. Expected: also invariant (context).
* ``precision``              -- the pre-scale-stable rule, kept in the repo as a
  historical control. Expected: `evidence` grows ~linearly with `m` and
  `uncertainty` shrinks ~`1/sqrt(m)` -- naive accumulated confidence.

Usage:
    python experiments/paper_a/duplication_experiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.integration import BeliefLayer  # noqa: E402

MULTIPLICITIES: tuple[int, ...] = (1, 2, 4, 8, 16)
AGGREGATIONS: tuple[str, ...] = ("scale_stable_precision", "normalized_precision", "precision")
_DTYPE = torch.float64

# A fixed, deliberately heterogeneous set of source belief states: distinct
# content, a 6x spread in evidence, a 6x spread in uncertainty, and relevance
# gates spanning near-irrelevant (0.08) to dominant (0.95). Nothing here is
# tuned; it is just varied enough that a rule which *isn't* scale-stable will
# visibly drift.
_SOURCE_MU = torch.tensor([0.9, -0.4, 0.2, -0.8, 0.5, -0.1], dtype=_DTYPE)
_SOURCE_E = torch.tensor([3.0, 0.5, 1.5, 2.0, 1.0, 0.8], dtype=_DTYPE)
_SOURCE_U = torch.tensor([0.3, 1.8, 0.7, 1.2, 0.5, 1.0], dtype=_DTYPE)
_SOURCE_G = torch.tensor([0.95, 0.08, 0.60, 0.30, 0.75, 0.15], dtype=_DTYPE)
_SOURCE_W = torch.tensor([1.0, 0.7, -0.5, 1.2, -0.9, 0.4], dtype=_DTYPE)
_BIAS0 = 0.0

_N_SOURCES = _SOURCE_MU.numel()


def _logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p / (1.0 - p))


def _layer_for(multiplicity: int, aggregation: str) -> BeliefLayer:
    """A `BeliefLayer(in_cells = N_SOURCES * multiplicity, out_cells = 1)`
    whose per-connection content weight and relevance logit are the source
    values tiled `multiplicity` times -- so duplicate connections are exact
    copies."""
    n = _N_SOURCES * multiplicity
    layer = BeliefLayer(in_cells=n, out_cells=1, aggregation=aggregation).to(_DTYPE)
    with torch.no_grad():
        layer.content_weight.copy_(_SOURCE_W.repeat(multiplicity).unsqueeze(0))
        layer.relevance_logit.copy_(_logit(_SOURCE_G).repeat(multiplicity).unsqueeze(0))
        layer.bias.fill_(_BIAS0)
    return layer


def _belief_for(multiplicity: int) -> BeliefCell:
    return BeliefCell(
        mu=_SOURCE_MU.repeat(multiplicity).unsqueeze(0),
        evidence=_SOURCE_E.repeat(multiplicity).unsqueeze(0),
        uncertainty=_SOURCE_U.repeat(multiplicity).unsqueeze(0),
    )


def run_for_aggregation(aggregation: str) -> list[dict]:
    rows: list[dict] = []
    base: dict[str, float] | None = None
    for m in MULTIPLICITIES:
        layer = _layer_for(m, aggregation)
        with torch.no_grad():
            out = layer(_belief_for(m))
        mu, e, u = out.mu.item(), out.evidence.item(), out.uncertainty.item()
        if base is None:
            base = {"mu": mu, "e": e, "u": u}
        rows.append(
            {
                "aggregation": aggregation,
                "multiplicity": m,
                "out_mu": mu,
                "out_e": e,
                "out_u": u,
                "abs_dmu_vs_x1": abs(mu - base["mu"]),
                "abs_de_vs_x1": abs(e - base["e"]),
                "abs_du_vs_x1": abs(u - base["u"]),
                "e_ratio_vs_x1": e / base["e"],
                "u_ratio_vs_x1": u / base["u"],
            }
        )
    return rows


def run_experiment() -> dict:
    results = {agg: run_for_aggregation(agg) for agg in AGGREGATIONS}

    # Machine-checkable invariance verdicts. `|delta| <= atol + rtol*|base|`
    # (torch.allclose convention) -- float64 accumulation over 16x more terms
    # leaves O(1e-9) relative noise, which is not a scale-dependence.
    atol, rtol = 1e-8, 1e-6
    verdicts: dict[str, dict] = {}
    for agg, rows in results.items():
        base = rows[0]
        max_dmu = max(r["abs_dmu_vs_x1"] for r in rows)
        max_de = max(r["abs_de_vs_x1"] for r in rows)
        max_du = max(r["abs_du_vs_x1"] for r in rows)
        verdicts[agg] = {
            "max_abs_dmu": max_dmu,
            "max_abs_de": max_de,
            "max_abs_du": max_du,
            "max_rel_de": max_de / abs(base["out_e"]),
            "mu_invariant": max_dmu <= atol + rtol * abs(base["out_mu"]),
            "e_invariant": max_de <= atol + rtol * abs(base["out_e"]),
            "u_invariant": max_du <= atol + rtol * abs(base["out_u"]),
            "e_ratio_at_x16": rows[-1]["e_ratio_vs_x1"],
            "u_ratio_at_x16": rows[-1]["u_ratio_vs_x1"],
        }

    return {
        "source_states": {
            "mu": _SOURCE_MU.tolist(),
            "evidence": _SOURCE_E.tolist(),
            "uncertainty": _SOURCE_U.tolist(),
            "relevance_g": _SOURCE_G.tolist(),
            "content_weight": _SOURCE_W.tolist(),
        },
        "multiplicities": list(MULTIPLICITIES),
        "results": results,
        "verdicts": verdicts,
    }


def _print_report(payload: dict) -> None:
    for agg, rows in payload["results"].items():
        print(f"\n=== {agg} ===")
        print(
            f"{'m':>3s}  {'out_mu':>12s}  {'out_e':>12s}  {'out_u':>12s}  "
            f"{'e/e(x1)':>10s}  {'u/u(x1)':>10s}"
        )
        for r in rows:
            print(
                f"{r['multiplicity']:>3d}  {r['out_mu']:>12.8f}  {r['out_e']:>12.8f}  "
                f"{r['out_u']:>12.8f}  {r['e_ratio_vs_x1']:>10.4f}  {r['u_ratio_vs_x1']:>10.4f}"
            )
        v = payload["verdicts"][agg]
        print(
            f"    invariance: mu={v['mu_invariant']} e={v['e_invariant']} "
            f"u={v['u_invariant']}  (max |Δmu|={v['max_abs_dmu']:.2e}, "
            f"|Δe|={v['max_abs_de']:.2e}, |Δu|={v['max_abs_du']:.2e})"
        )
    print(
        "\nExpected: scale_stable_precision and normalized_precision invariant in "
        "all three; precision inflates e ~m and shrinks u ~1/sqrt(m)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(_HERE / "results" / "processed" / "duplication_experiment.json"),
    )
    args = parser.parse_args()

    payload = run_experiment()
    _print_report(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
