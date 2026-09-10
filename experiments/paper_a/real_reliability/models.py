"""The five Paper-A Phase-3 Part-B model families, plus the optional
`HistGradientBoosting` reference.

Reuses the Phase-2 wrappers unchanged (`PlainMLP`, `ConfidenceMLP`,
`ReliabilityCellV03` -- all `forward(x, c)`), sized here against the frozen
Part-B parameter budgets (APS ~100k, Air Quality ~50k):

======================  ==============================================  ================
Family                   input                                          reliability
======================  ==============================================  ================
``plain_mlp`` (A)        ``x_imputed``                                   none
``confidence_mlp`` (B)   ``concat(x_imputed, c)`` -- param-matched       exact ``c``
``confidence_mlp_same_width`` (C)  ``concat(x_imputed, c)`` -- CellV0.3 width  exact ``c``
``cellv0.3`` (D)         belief ``(mu=x_imputed, e=c, u=0)``             exact ``c``
``neumiss`` (E)          ``x_nan`` (NaN kept)                            handled natively
======================  ==============================================  ================

``neumiss`` wraps the **official** `NeuMissBlock` from
`marineLM/NeuMiss_sota` (installed as the external `neumiss` package, pinned
commit recorded in `NEUMISS_COMMIT`) followed by a one-hidden-layer prediction
head, matching `NeuMissMLP(mlp_depth=1)`'s structure. Its parameter count is
reported, not matched (the ``n_features^2`` block scales differently).

Every model emits **one** output value: a logit for APS (shared
`BCEWithLogitsLoss(pos_weight)`), a scalar for Air Quality.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from experiments.paper_a.models import (  # noqa: E402
    belief_v03_hidden_cells_for_budget,
    belief_v03_param_count,
)
from experiments.paper_a.reliability.models import (  # noqa: E402
    ConfidenceMLP,
    PlainMLP,
    ReliabilityCellV03,
    belief_diagnostics,
)
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.models.baselines.mlp import (  # noqa: E402
    match_hidden_dim,
    param_count_for_mlp,
)

PLAIN_MLP = "plain_mlp"
CONFIDENCE_MLP = "confidence_mlp"
CONFIDENCE_MLP_SAME_WIDTH = "confidence_mlp_same_width"
CELLV03 = "cellv0.3"
NEUMISS = "neumiss"
MODEL_FAMILIES: tuple[str, ...] = (
    PLAIN_MLP, CONFIDENCE_MLP, CONFIDENCE_MLP_SAME_WIDTH, CELLV03, NEUMISS,
)
RELIABILITY_AWARE: frozenset[str] = frozenset(
    {CONFIDENCE_MLP, CONFIDENCE_MLP_SAME_WIDTH, CELLV03, NEUMISS}
)

# Predeclared NeuMiss Neumann-iteration depths; the winner is chosen on
# validation only (Phase-3 task).
NEUMISS_DEPTHS: tuple[int, ...] = (1, 3, 5)

# Pinned commit of marineLM/NeuMiss_sota that the vendored/installed `neumiss`
# package must match (recorded, per the Phase-3 task).
NEUMISS_COMMIT = "7902b8dbe7114e8dc3010b5e8b132c35806a9d74"
NEUMISS_REPO = "https://github.com/marineLM/NeuMiss_sota"

_MLP_SEARCH_RANGE = range(1, 12000)
_MATCH_TOLERANCE = 0.02


class NeuMissIntegrationError(RuntimeError):
    """Raised when the official NeuMiss implementation cannot be imported /
    used. The Phase-3 task says to STOP and report the exact problem, never
    substitute an unofficial implementation."""


def _import_neumiss():
    try:
        import neumiss  # noqa: PLC0415
        from neumiss import NeuMissBlock  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise NeuMissIntegrationError(
            f"could not import the official `neumiss` package "
            f"(install: `pip install --no-deps git+{NEUMISS_REPO}@{NEUMISS_COMMIT}`): {exc!r}"
        ) from exc
    return neumiss, NeuMissBlock


class NeuMissWrapper(nn.Module):
    """Official `NeuMissBlock` (Neumann iterations, NaN handled natively) ->
    ``Linear -> ReLU -> Linear`` head, i.e. the `NeuMissMLP(mlp_depth=1)`
    structure with a configurable output width. Expects ``x`` with missing
    entries as ``NaN``; ``c`` is accepted for interface parity and ignored
    (NeuMiss learns the missingness response itself)."""

    def __init__(self, n_features: int, neumiss_depth: int, mlp_width: int, out_features: int = 1):
        super().__init__()
        _neumiss, NeuMissBlock = _import_neumiss()
        self.neumiss_depth = neumiss_depth
        self.mlp_width = mlp_width
        self.block = NeuMissBlock(n_features, neumiss_depth)
        self.head = nn.Sequential(
            nn.Linear(n_features, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, out_features),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        return self.head(self.block(x))


@dataclass
class BuiltModel:
    model: nn.Module
    family: str
    parameter_count: int
    hidden_size: int
    sizing: dict = field(default_factory=dict)
    neumiss_depth: int | None = None


def cellv03_target(in_features: int, param_budget: int) -> tuple[int, int]:
    """``(hidden_cells, actual_param_count)`` for CellV0.3 at this budget -- the
    target the MLP families are matched to (out_features is always 1)."""
    hc = belief_v03_hidden_cells_for_budget(in_features, 1, param_budget)
    return hc, belief_v03_param_count(hc, in_features, 1)


def _matched_mlp(cls, family, *, raw_in, net_in, target):
    hidden = match_hidden_dim(
        target, net_in, 1, num_hidden_layers=1, search_range=_MLP_SEARCH_RANGE
    )
    model = cls(raw_in, hidden, 1)
    n = count_parameters(model)
    rel = abs(n - target) / target
    return BuiltModel(
        model=model, family=family, parameter_count=n, hidden_size=hidden,
        sizing={
            "hidden_dim": hidden, "net_in_features": net_in, "target_params": target,
            "params": n, "param_rel_diff": rel,
            "param_match_within_2pct": bool(rel <= _MATCH_TOLERANCE),
            "params_formula": param_count_for_mlp(net_in, hidden, 1, num_hidden_layers=1),
        },
    )


def build_model(
    family: str, in_features: int, param_budget: int, *, neumiss_depth: int | None = None
) -> BuiltModel:
    hc, target = cellv03_target(in_features, param_budget)

    if family == CELLV03:
        model = ReliabilityCellV03(in_features, hc, 1)
        n = count_parameters(model)
        return BuiltModel(
            model=model, family=CELLV03, parameter_count=n, hidden_size=hc,
            sizing={
                "hidden_cells": hc, "param_budget": param_budget, "params": n,
                "target_params": target, "params_within_budget": bool(n <= param_budget),
            },
        )
    if family == PLAIN_MLP:
        return _matched_mlp(
            PlainMLP, PLAIN_MLP, raw_in=in_features, net_in=in_features, target=target
        )
    if family == CONFIDENCE_MLP:
        return _matched_mlp(
            ConfidenceMLP, CONFIDENCE_MLP, raw_in=in_features, net_in=2 * in_features, target=target
        )
    if family == CONFIDENCE_MLP_SAME_WIDTH:
        model = ConfidenceMLP(in_features, hc, 1)
        n = count_parameters(model)
        return BuiltModel(
            model=model, family=CONFIDENCE_MLP_SAME_WIDTH, parameter_count=n, hidden_size=hc,
            sizing={
                "hidden_dim": hc, "cellv03_hidden_cells": hc, "net_in_features": 2 * in_features,
                "target_params": target, "params": n,
                "param_ratio_vs_cellv03": n / target, "same_width_as_cellv03": True,
            },
        )
    if family == NEUMISS:
        if neumiss_depth is None:
            raise ValueError("neumiss requires an explicit neumiss_depth from NEUMISS_DEPTHS")
        # head width fitted so the total lands near the budget (the n_features^2
        # NeuMiss block is fixed); reported, not matched.
        _neumiss, NeuMissBlock = _import_neumiss()
        block_params = in_features + in_features * in_features  # mu + linear (no bias)
        remaining = max(param_budget - block_params, in_features + 4)
        # head params = in*w + w + w*1 + 1  ->  w ~= (remaining - 1) / (in + 2)
        mlp_width = max(1, round((remaining - 1) / (in_features + 2)))
        model = NeuMissWrapper(in_features, neumiss_depth, mlp_width, out_features=1)
        n = count_parameters(model)
        return BuiltModel(
            model=model, family=NEUMISS, parameter_count=n, hidden_size=mlp_width,
            neumiss_depth=neumiss_depth,
            sizing={
                "neumiss_depth": neumiss_depth, "mlp_width": mlp_width,
                "block_params": block_params, "params": n, "param_budget": param_budget,
                "params_within_budget": bool(n <= param_budget),
                "neumiss_commit": NEUMISS_COMMIT,
            },
        )
    raise ValueError(f"unknown family {family!r}; expected one of {MODEL_FAMILIES}")


__all__ = [
    "CELLV03",
    "CONFIDENCE_MLP",
    "CONFIDENCE_MLP_SAME_WIDTH",
    "MODEL_FAMILIES",
    "NEUMISS",
    "NEUMISS_COMMIT",
    "NEUMISS_DEPTHS",
    "NEUMISS_REPO",
    "PLAIN_MLP",
    "RELIABILITY_AWARE",
    "BuiltModel",
    "NeuMissIntegrationError",
    "NeuMissWrapper",
    "belief_diagnostics",
    "build_model",
    "cellv03_target",
]
