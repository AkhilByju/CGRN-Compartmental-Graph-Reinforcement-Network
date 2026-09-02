#!/usr/bin/env python3
"""Experiment 004E -- the duplication test (docs/research_log.md
"Experiment 004E"). No training: a pure forward-pass diagnostic of
`BeliefLayer`'s aggregation formulas in isolation.

Feeds a `BeliefLayer` N identical copies of the exact same belief content
(same `mu`, `evidence`, `uncertainty`) through N "equivalent connections"
(every incoming connection given the identical content weight and relevance
gate, so N is the only thing that varies), for N in {1, 2, 4, 8, 16}, and
reports the resulting `mu`/`evidence`/`uncertainty`.

Since the N copies carry no new information relative to N=1, a principled
aggregation rule should leave `evidence`/`uncertainty` approximately
unchanged as N grows. `"precision"` is expected to fail this badly
(`evidence` is an unnormalized sum -- Experiment 004D already found this
during training; this formally demonstrates it as a property of the
formula, independent of any specific trained weights).
`"normalized_precision"` (Experiment 004F) was built to fix it -- rerun
this script after 004F lands to verify.

This exact property is also codified as an automated regression test in
`tests/test_duplication_invariance.py` -- this script is the
human-readable report version of the same check.

Usage:
    python experiments/004_cellv0_scaling/duplication_test.py
    python experiments/004_cellv0_scaling/duplication_test.py \\
        --aggregations precision normalized_precision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.integration import AGGREGATION_METHODS, BeliefLayer  # noqa: E402

N_COPIES: tuple[int, ...] = (1, 2, 4, 8, 16)

# One representative belief -- content 0.5, evidence/uncertainty both 1.0,
# matching BeliefCell.from_observed_features's convention for a raw input
# feature. All N copies share these exact values.
MU0, EVIDENCE0, UNCERTAINTY0 = 0.5, 1.0, 1.0
# "Equivalent connections": every incoming connection gets this identical
# weight, so varying N is the only thing that changes between conditions.
CONTENT_WEIGHT0, RELEVANCE_LOGIT0, BIAS0 = 1.0, 0.0, 0.0


def _duplicated_belief(n: int) -> BeliefCell:
    return BeliefCell(
        mu=torch.full((1, n), MU0),
        evidence=torch.full((1, n), EVIDENCE0),
        uncertainty=torch.full((1, n), UNCERTAINTY0),
    )


def _equivalent_layer(n: int, aggregation: str) -> BeliefLayer:
    layer = BeliefLayer(in_cells=n, out_cells=1, aggregation=aggregation)
    with torch.no_grad():
        layer.content_weight.fill_(CONTENT_WEIGHT0)
        layer.relevance_logit.fill_(RELEVANCE_LOGIT0)
        layer.bias.fill_(BIAS0)
    return layer


def run_duplication_test(aggregation: str) -> list[dict]:
    rows = []
    baseline_evidence: float | None = None
    baseline_uncertainty: float | None = None
    for n in N_COPIES:
        layer = _equivalent_layer(n, aggregation)
        with torch.no_grad():
            out = layer(_duplicated_belief(n))
        evidence, uncertainty, mu = out.evidence.item(), out.uncertainty.item(), out.mu.item()
        if n == 1:
            baseline_evidence, baseline_uncertainty = evidence, uncertainty
        rows.append(
            {
                "aggregation": aggregation,
                "n_copies": n,
                "mu": mu,
                "evidence": evidence,
                "uncertainty": uncertainty,
                "evidence_ratio_vs_n1": evidence / baseline_evidence,
                "uncertainty_ratio_vs_n1": uncertainty / baseline_uncertainty,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["precision", "normalized_precision"],
        choices=list(AGGREGATION_METHODS),
    )
    args = parser.parse_args()

    for aggregation in args.aggregations:
        print(f"=== {aggregation} ===")
        print(
            f"{'n_copies':>8s}  {'mu':>8s}  {'evidence':>10s}  {'uncertainty':>12s}  "
            f"{'evidence/N=1':>13s}  {'uncertainty/N=1':>16s}"
        )
        for row in run_duplication_test(aggregation):
            print(
                f"{row['n_copies']:>8d}  {row['mu']:>8.4f}  {row['evidence']:>10.4f}  "
                f"{row['uncertainty']:>12.4f}  {row['evidence_ratio_vs_n1']:>13.3f}  "
                f"{row['uncertainty_ratio_vs_n1']:>16.3f}"
            )
        print()

    print(
        "A duplicate-invariant aggregation rule keeps the last two columns near "
        "1.000 at every N -- 'precision' is expected to grow evidence "
        "~linearly with N and shrink uncertainty ~1/sqrt(N); "
        "'normalized_precision' is expected to stay near 1.000 throughout."
    )


if __name__ == "__main__":
    main()
