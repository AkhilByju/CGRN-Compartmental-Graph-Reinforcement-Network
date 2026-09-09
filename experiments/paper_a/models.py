"""The three Paper-A model families, plus the fixed-confidence ablation.

A. **CellV0.1** -- the frozen `BeliefNetwork` with `scale_stable_precision`
   aggregation (`src/models/architecture_v0/`). Not modified here in any way;
   this module only *sizes and constructs* it.
B. **Parameter-matched MLP** -- `input -> Linear -> SiLU -> Linear` whose
   hidden width is chosen so its trainable-parameter count lands within 2% of
   CellV0.1's actual count (Paper-A task Sec 4B).
C. **State-count MLP control** -- the same 1-hidden-layer SiLU MLP with hidden
   width `~= 3 * CellV0.1_hidden_cells` (CellV0.1 carries three runtime scalar
   states per cell). Deliberately *not* parameter-matched; its larger
   parameter count is reported, not hidden (Paper-A task Sec 4C).

Ablation. **CellV0.1-fixed-confidence** -- CellV0.1's exact content pathway
(same `BeliefLayer` parameters, same `content_weight`/`relevance_logit`/`bias`,
same count) but the evidence/uncertainty of every belief *entering* a fusion
is overwritten with ones, so the dynamically propagated `e`/`u` state cannot
influence anything downstream (Paper-A task Sec 11). Reuses the frozen
`BeliefLayer` unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.integration import BeliefLayer  # noqa: E402
from src.models.architecture_v0.precision_gain import (  # noqa: E402
    BeliefNetworkV02,
    effective_precision,
    relative_gain,
)
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim  # noqa: E402

CELLV01_AGGREGATION = "scale_stable_precision"
STATE_COUNT_MULTIPLIER = 3  # (mu, e, u) -- three runtime scalar states per cell

# The default Phase-1 screening grid -- FROZEN (Paper-A task Sec 12). CellV0.2
# is a later, separately-commissioned architecture line evaluated on the same
# frozen protocol; it is deliberately NOT a member of this tuple, so
# `run_phase1.py` never sweeps it into the original screen.
MODEL_FAMILIES: tuple[str, ...] = ("cellv0.1", "mlp_matched", "mlp_state_count")
CELLV02_FAMILY = "cellv0.2"

_MLP_SEARCH_RANGE = range(1, 4000)


# ---------------------------------------------------------------------------
# CellV0.1 sizing (closed form -- identical for every aggregation method,
# per docs/architecture_v0.md Sec 7 item 6)
# ---------------------------------------------------------------------------


def belief_param_count(hidden_cells: int, in_features: int, out_features: int) -> int:
    """Trainable parameters of a 2-`BeliefLayer` `BeliefNetwork` + linear
    readout: `layer1 + layer2 + readout`."""
    layer1 = hidden_cells * (2 * in_features + 1)
    layer2 = hidden_cells * (2 * hidden_cells + 1)
    readout = hidden_cells * out_features + out_features
    return layer1 + layer2 + readout


def belief_hidden_cells_for_budget(
    in_features: int, out_features: int, param_budget: int
) -> int:
    """Largest integer `hidden_cells` whose `BeliefNetwork` parameter count
    stays within `param_budget` (Paper-A task Sec 5)."""
    hc = 1
    while belief_param_count(hc + 1, in_features, out_features) <= param_budget:
        hc += 1
    return hc


# ---------------------------------------------------------------------------
# CellV0.2 sizing -- the Conservative Precision-Gain Cell
# (`src/models/architecture_v0/precision_gain.py`, docs/architecture_v0.md
# Sec 10). One signed connection matrix `V` [out, in] plus a per-output
# `gain_raw` and `bias` -- no relevance-gate matrix -- so ~half CellV0.1's
# per-connection parameter cost. Its hidden width is fitted to the SAME
# Phase-1 parameter budget; the resulting (larger) cell count is reported,
# not forced to match CellV0.1's.
# ---------------------------------------------------------------------------


def belief_v02_param_count(hidden_cells: int, in_features: int, out_features: int) -> int:
    """Trainable parameters of a 2-`PrecisionGainLayer` `BeliefNetworkV02` +
    linear readout.

        layer1  = hidden_cells * (in_features + 2)      # V + gain_raw + bias
        layer2  = hidden_cells * (hidden_cells + 2)
        readout = hidden_cells * out_features + out_features
    """
    layer1 = hidden_cells * (in_features + 2)
    layer2 = hidden_cells * (hidden_cells + 2)
    readout = hidden_cells * out_features + out_features
    return layer1 + layer2 + readout


def belief_v02_hidden_cells_for_budget(
    in_features: int, out_features: int, param_budget: int
) -> int:
    """Largest integer `hidden_cells` whose `BeliefNetworkV02` parameter
    count stays within `param_budget` (same budget as CellV0.1 / Paper-A
    task Sec 5)."""
    hc = 1
    while belief_v02_param_count(hc + 1, in_features, out_features) <= param_budget:
        hc += 1
    return hc


# ---------------------------------------------------------------------------
# Fixed-confidence ablation model
# ---------------------------------------------------------------------------


def _ones_confidence(belief: BeliefCell) -> BeliefCell:
    ones = torch.ones_like(belief.mu)
    return BeliefCell(mu=belief.mu, evidence=ones, uncertainty=ones)


class FixedConfidenceBeliefNetwork(nn.Module):
    """CellV0.1's architecture and parameters exactly, but the belief state
    entering every `BeliefLayer` has `e = u = 1` forced (the raw input
    already does; this additionally resets layer 1's *output* confidence
    before it reaches layer 2). Isolates whether the propagated evidence/
    uncertainty state contributes beyond the content pathway.

    Same module structure, same parameter count, same `content_weight` /
    `relevance_logit` / `bias` shapes as `BeliefNetwork` -- only the belief
    flowing between layers is intercepted.
    """

    def __init__(self, in_features: int, hidden_cells: int, out_features: int) -> None:
        super().__init__()
        self.layer1 = BeliefLayer(in_features, hidden_cells, aggregation=CELLV01_AGGREGATION)
        self.layer2 = BeliefLayer(hidden_cells, hidden_cells, aggregation=CELLV01_AGGREGATION)
        self.readout = nn.Linear(hidden_cells, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        belief = _ones_confidence(BeliefCell.from_observed_features(x))
        belief = _ones_confidence(self.layer1(belief))
        belief = self.layer2(belief)
        return self.readout(belief.mu)

    def forward_with_beliefs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCell]:
        belief = _ones_confidence(BeliefCell.from_observed_features(x))
        b1 = _ones_confidence(self.layer1(belief))
        b2 = self.layer2(b1)
        return self.readout(b2.mu), b2


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


@dataclass
class BuiltModel:
    model: nn.Module
    family: str
    parameter_count: int
    hidden_size: int  # hidden_cells for belief nets, hidden_dim for MLPs
    sizing: dict[str, float | int | str]


def _cellv01_hidden_cells(in_features: int, out_features: int, param_budget: int) -> int:
    return belief_hidden_cells_for_budget(in_features, out_features, param_budget)


def build_cellv01(
    in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    hidden_cells = _cellv01_hidden_cells(in_features, out_features, param_budget)
    model = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation=CELLV01_AGGREGATION
    )
    n_params = count_parameters(model)
    return BuiltModel(
        model=model,
        family="cellv0.1",
        parameter_count=n_params,
        hidden_size=hidden_cells,
        sizing={
            "hidden_cells": hidden_cells,
            "param_budget": param_budget,
            "params": n_params,
            "params_within_budget": bool(n_params <= param_budget),
            "aggregation": CELLV01_AGGREGATION,
        },
    )


def build_cellv02(
    in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    """CellV0.2 -- `BeliefNetworkV02` (two `PrecisionGainLayer`s + a
    confidence-scaled linear readout), largest hidden-cell count within the
    same Phase-1 parameter budget. Its lower per-connection cost buys more
    hidden cells than CellV0.1 at the same budget -- reported, not
    equalized (that is part of the architecture)."""
    hidden_cells = belief_v02_hidden_cells_for_budget(in_features, out_features, param_budget)
    model = BeliefNetworkV02(in_features, hidden_cells, out_features)
    n_params = count_parameters(model)
    return BuiltModel(
        model=model,
        family=CELLV02_FAMILY,
        parameter_count=n_params,
        hidden_size=hidden_cells,
        sizing={
            "hidden_cells": hidden_cells,
            "param_budget": param_budget,
            "params": n_params,
            "params_within_budget": bool(n_params <= param_budget),
            "params_formula": belief_v02_param_count(hidden_cells, in_features, out_features),
            "cellv01_hidden_cells": belief_hidden_cells_for_budget(
                in_features, out_features, param_budget
            ),
            "cellv01_params": belief_param_count(
                belief_hidden_cells_for_budget(in_features, out_features, param_budget),
                in_features,
                out_features,
            ),
        },
    )


def build_matched_mlp(
    in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    """Model B: 1-hidden-layer SiLU MLP matched to CellV0.1's *actual*
    parameter count (Paper-A task Sec 5: "match model B to the actual
    resulting CellV0.1 parameter count")."""
    hidden_cells = _cellv01_hidden_cells(in_features, out_features, param_budget)
    target = belief_param_count(hidden_cells, in_features, out_features)

    hidden_dim = match_hidden_dim(
        target, in_features, out_features, num_hidden_layers=1, search_range=_MLP_SEARCH_RANGE
    )
    model = MLPBaseline(
        in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
    )
    n_params = count_parameters(model)
    rel_diff = abs(n_params - target) / target
    return BuiltModel(
        model=model,
        family="mlp_matched",
        parameter_count=n_params,
        hidden_size=hidden_dim,
        sizing={
            "hidden_dim": hidden_dim,
            "target_params": target,
            "params": n_params,
            "param_diff": n_params - target,
            "param_rel_diff": rel_diff,
            "param_match_within_2pct": bool(rel_diff <= 0.02),
        },
    )


def build_state_count_mlp(
    in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    """Model C: 1-hidden-layer SiLU MLP with hidden width
    `3 * CellV0.1_hidden_cells`. Not parameter-matched; the (larger)
    parameter count is reported explicitly."""
    hidden_cells = _cellv01_hidden_cells(in_features, out_features, param_budget)
    cell_params = belief_param_count(hidden_cells, in_features, out_features)
    hidden_dim = STATE_COUNT_MULTIPLIER * hidden_cells
    model = MLPBaseline(
        in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
    )
    n_params = count_parameters(model)
    return BuiltModel(
        model=model,
        family="mlp_state_count",
        parameter_count=n_params,
        hidden_size=hidden_dim,
        sizing={
            "hidden_dim": hidden_dim,
            "cellv01_hidden_cells": hidden_cells,
            "state_count_multiplier": STATE_COUNT_MULTIPLIER,
            "params": n_params,
            "cellv01_params": cell_params,
            "params_vs_cellv01_ratio": n_params / cell_params,
            "params_more_than_cellv01": n_params - cell_params,
        },
    )


def build_fixed_confidence(
    in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    hidden_cells = _cellv01_hidden_cells(in_features, out_features, param_budget)
    model = FixedConfidenceBeliefNetwork(in_features, hidden_cells, out_features)
    n_params = count_parameters(model)
    return BuiltModel(
        model=model,
        family="cellv0.1_fixed_confidence",
        parameter_count=n_params,
        hidden_size=hidden_cells,
        sizing={
            "hidden_cells": hidden_cells,
            "params": n_params,
            "cellv01_params": belief_param_count(hidden_cells, in_features, out_features),
        },
    )


_BUILDERS = {
    "cellv0.1": build_cellv01,
    "cellv0.2": build_cellv02,
    "mlp_matched": build_matched_mlp,
    "mlp_state_count": build_state_count_mlp,
    "cellv0.1_fixed_confidence": build_fixed_confidence,
}


def build_model(family: str, in_features: int, out_features: int, param_budget: int) -> BuiltModel:
    if family not in _BUILDERS:
        raise ValueError(f"Unknown model family '{family}'. Expected one of {tuple(_BUILDERS)}.")
    return _BUILDERS[family](in_features, out_features, param_budget)


# `BeliefLayer` broadcasts to `(batch, out_cells, in_cells)`, so a full-set
# forward on a 10k-row MNIST/Fashion-MNIST split allocates multi-GB
# intermediates. Every evaluation forward (val during training, test, belief
# diagnostics) goes through this chunker instead.
EVAL_CHUNK = 1024


@torch.no_grad()
def batched_forward(model: nn.Module, x: torch.Tensor, chunk: int = EVAL_CHUNK) -> torch.Tensor:
    model.eval()
    if x.shape[0] <= chunk:
        return model(x)
    return torch.cat([model(x[i : i + chunk]) for i in range(0, x.shape[0], chunk)])


# ---------------------------------------------------------------------------
# CellV0.1 internal diagnostics (Paper-A task Sec 9 -- lightweight only, not
# an evaluation target)
# ---------------------------------------------------------------------------

BELIEF_FAMILIES: frozenset[str] = frozenset({"cellv0.1", "cellv0.1_fixed_confidence"})


def _forward_layers(model: nn.Module, x: torch.Tensor) -> tuple[BeliefCell, BeliefCell]:
    """Returns `(belief_after_layer1, belief_after_layer2)` for either a
    `BeliefNetwork` or a `FixedConfidenceBeliefNetwork`, mirroring each
    one's own `forward`."""
    if isinstance(model, FixedConfidenceBeliefNetwork):
        b0 = _ones_confidence(BeliefCell.from_observed_features(x))
        b1 = model.layer1(b0)
        b2 = model.layer2(_ones_confidence(b1))
        return b1, b2
    if isinstance(model, BeliefNetwork):
        b0 = BeliefCell.from_observed_features(x)
        b1 = model.layer1(b0)
        b2 = model.layer2(b1)
        return b1, b2
    raise TypeError(f"belief diagnostics not defined for {type(model).__name__}")


@torch.no_grad()
def belief_diagnostics(
    model: nn.Module, x: torch.Tensor, eps: float = 1e-8, chunk: int = EVAL_CHUNK
) -> dict[str, float]:
    """Mean `e`, mean `u`, mean `e / (u^2 + eps)`, and min/max of `e` and `u`
    for each `BeliefLayer` -- numerical-stability diagnostics, per the
    Paper-A task's explicit instruction not to treat these as headline
    metrics. Chunked (the `BeliefLayer` broadcast is memory-heavy at MNIST
    batch sizes)."""
    model.eval()
    acc: dict[str, list] = {}
    n_total = x.shape[0]
    for i in range(0, n_total, chunk):
        b1, b2 = _forward_layers(model, x[i : i + chunk])
        w = b1.evidence.shape[0]
        for tag, b in (("layer1", b1), ("layer2", b2)):
            e, u = b.evidence, b.uncertainty
            row = {
                "sum_e": float(e.sum()),
                "sum_u": float(u.sum()),
                "sum_e_over_u2": float((e / (u**2 + eps)).sum()),
                "n": e.numel(),
                "min_e": float(e.min()),
                "max_e": float(e.max()),
                "min_u": float(u.min()),
                "max_u": float(u.max()),
                "rows": w,
            }
            acc.setdefault(tag, []).append(row)

    out: dict[str, float] = {}
    for tag, rows in acc.items():
        n = sum(r["n"] for r in rows)
        out[f"diag_{tag}_mean_e"] = sum(r["sum_e"] for r in rows) / n
        out[f"diag_{tag}_mean_u"] = sum(r["sum_u"] for r in rows) / n
        out[f"diag_{tag}_mean_e_over_u2"] = sum(r["sum_e_over_u2"] for r in rows) / n
        out[f"diag_{tag}_min_e"] = min(r["min_e"] for r in rows)
        out[f"diag_{tag}_max_e"] = max(r["max_e"] for r in rows)
        out[f"diag_{tag}_min_u"] = min(r["min_u"] for r in rows)
        out[f"diag_{tag}_max_u"] = max(r["max_u"] for r in rows)
    return out


# ---------------------------------------------------------------------------
# CellV0.2 internal diagnostics (Paper-A task Sec 9 analogue -- lightweight,
# observational only, NOT an evaluation target and nothing is tuned on them).
# Reports the effective precision `pi = e / (1 + e u)` and the relative gain
# `2 pi / (pi + mean pi)` of each hidden layer's output belief state.
# ---------------------------------------------------------------------------


@torch.no_grad()
def precision_gain_diagnostics(
    model: nn.Module, x: torch.Tensor, chunk: int = EVAL_CHUNK
) -> dict[str, float]:
    """mean / std / min / max of effective precision, and mean / std of
    relative gain, for `layer1` and `layer2` of a `BeliefNetworkV02`.
    Chunked (running-moment accumulation) so it is memory-flat on the
    MNIST-sized test splits."""
    if not isinstance(model, BeliefNetworkV02):
        raise TypeError(
            f"precision-gain diagnostics not defined for {type(model).__name__}"
        )
    model.eval()
    acc: dict[str, dict[str, float]] = {}
    for i in range(0, x.shape[0], chunk):
        b1, b2 = model.layer_states(x[i : i + chunk])
        for tag, belief in (("layer1", b1), ("layer2", b2)):
            pi = effective_precision(belief.evidence, belief.uncertainty)
            rg = relative_gain(pi, model.eps)
            slot = acc.setdefault(
                tag,
                {"n": 0.0, "pi_sum": 0.0, "pi_sq": 0.0, "pi_min": float("inf"),
                 "pi_max": float("-inf"), "rg_sum": 0.0, "rg_sq": 0.0},
            )
            slot["n"] += pi.numel()
            slot["pi_sum"] += float(pi.sum())
            slot["pi_sq"] += float((pi * pi).sum())
            slot["pi_min"] = min(slot["pi_min"], float(pi.min()))
            slot["pi_max"] = max(slot["pi_max"], float(pi.max()))
            slot["rg_sum"] += float(rg.sum())
            slot["rg_sq"] += float((rg * rg).sum())

    def _std(total: float, sq: float, n: float) -> float:
        var = max(sq / n - (total / n) ** 2, 0.0)
        return var**0.5

    out: dict[str, float] = {}
    for tag, s in acc.items():
        n = s["n"]
        out[f"diag_{tag}_precision_mean"] = s["pi_sum"] / n
        out[f"diag_{tag}_precision_std"] = _std(s["pi_sum"], s["pi_sq"], n)
        out[f"diag_{tag}_precision_min"] = s["pi_min"]
        out[f"diag_{tag}_precision_max"] = s["pi_max"]
        out[f"diag_{tag}_relative_gain_mean"] = s["rg_sum"] / n
        out[f"diag_{tag}_relative_gain_std"] = _std(s["rg_sum"], s["rg_sq"], n)
    return out
