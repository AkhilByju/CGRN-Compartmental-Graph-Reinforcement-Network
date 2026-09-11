"""The six Architecture V2 frozen-benchmark model families (docs/architecture_v2.md
Sec N). All six expose the same interface -- `forward(x_corrupted, c)` -- so
the training/evaluation loop is model-agnostic.

=====================================  =============================  ===========
family                                 underlying net                 sees c how
=====================================  =============================  ===========
``plain_mlp``                          ``x_corrupted``                no
``confidence_mlp``                     ``concat(x_corrupted, c)``     exact
``cellv0.3``                           belief ``(mu=x,e=c,u=0)``      exact
``scalar_dendrite``                    ``x_corrupted``, sparse        no
``scalar_dendrite_reliability_gated``  ``c*x_corrupted``, sparse      once, at input
``belief_dendrite``                    belief, sparse, propagated     exact
=====================================  =============================  ===========

Every model targets ~150k trainable parameters (Sec K), independently sized
-- not matched to any one family's actual count. The three dendritic
families (`scalar_dendrite`, `scalar_dendrite_reliability_gated`,
`belief_dendrite`) share one `DendriticConnectivity` pair per seed, so the
"does hierarchical belief propagation help" comparisons are over identical
sparse topology, not just identical parameter count.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.conflict_normalized import BeliefNetworkV03  # noqa: E402
from src.models.architecture_v2.belief_dendrite import (  # noqa: E402
    BeliefDendriteNetwork,
    DendriticConnectivity,
    ScalarDendriteNetwork,
    ScalarDendriteReliabilityGated,
    make_connectivity_pair,
)
from src.models.architecture_v2.param_count import solve_hidden_width_for_budget  # noqa: E402
from src.models.baselines.mlp import (  # noqa: E402
    MLPBaseline,
    match_hidden_dim,
)

PLAIN_MLP = "plain_mlp"
CONFIDENCE_MLP = "confidence_mlp"
CELLV03 = "cellv0.3"
SCALAR_DENDRITE = "scalar_dendrite"
SCALAR_DENDRITE_GATED = "scalar_dendrite_reliability_gated"
BELIEF_DENDRITE = "belief_dendrite"
MODEL_FAMILIES: tuple[str, ...] = (
    PLAIN_MLP,
    CONFIDENCE_MLP,
    CELLV03,
    SCALAR_DENDRITE,
    SCALAR_DENDRITE_GATED,
    BELIEF_DENDRITE,
)
DENDRITE_FAMILIES: frozenset[str] = frozenset(
    {SCALAR_DENDRITE, SCALAR_DENDRITE_GATED, BELIEF_DENDRITE}
)
RELIABILITY_AWARE: frozenset[str] = frozenset(
    {CONFIDENCE_MLP, CELLV03, SCALAR_DENDRITE_GATED, BELIEF_DENDRITE}
)

IN_FEATURES = 784
OUT_FEATURES = 10
IMAGE_SHAPE = (1, 28, 28)
PARAM_BUDGET = 150_000

DENDRITE_BRANCHES = 4
DENDRITE_PATCH = 7
DENDRITE_K1 = IMAGE_SHAPE[0] * DENDRITE_PATCH * DENDRITE_PATCH  # 49
DENDRITE_K2 = 32

_MLP_SEARCH_RANGE = range(1, 6000)
_CELLV03_SEARCH_RANGE = range(1, 3000)


# ---------------------------------------------------------------------------
# Model wrappers -- uniform (x_corrupted, c) -> logits.
# ---------------------------------------------------------------------------


class PlainMLP(nn.Module):
    """Receives only `x_corrupted`; `c` is accepted so the training loop
    stays model-agnostic but is otherwise ignored."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.net = MLPBaseline(
            in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


class ConfidenceMLP(nn.Module):
    """Receives `concat(x_corrupted, c)` -- the exact reliability information
    CellV0.3/BeliefDendrite get, handed to an ordinary MLP."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.net = MLPBaseline(
            2 * in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=-1))


class DenseCellV03(nn.Module):
    """The dense (non-dendritic) `BeliefNetworkV03`, input belief `(mu=x,
    e=c, u=0)` -- the flat-fusion comparison point for `belief_dendrite`
    (Sec P Q3)."""

    def __init__(self, in_features: int, hidden_cells: int, out_features: int) -> None:
        super().__init__()
        self.net = BeliefNetworkV03(in_features, hidden_cells, out_features)

    def _input_belief(self, x: torch.Tensor, c: torch.Tensor) -> BeliefCell:
        return BeliefCell(mu=x, evidence=c, uncertainty=torch.zeros_like(x))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b1 = self.net.layer1(self._input_belief(x, c))
        b2 = self.net.layer2(b1)
        return self.net.readout(b2.mu)

    def belief_states_verbose(self, x: torch.Tensor, c: torch.Tensor):
        b1, cons1 = self.net.layer1.forward_verbose(self._input_belief(x, c))
        b2, cons2 = self.net.layer2.forward_verbose(b1)
        return (b1, cons1), (b2, cons2)


class ScalarDendriteModel(nn.Module):
    """`ScalarDendriteNetwork`, `c` ignored -- isolates dendritic *structure*
    from belief propagation (Sec P Q1)."""

    def __init__(
        self,
        connectivity1: DendriticConnectivity,
        connectivity2: DendriticConnectivity,
        out_features: int,
    ) -> None:
        super().__init__()
        self.net = ScalarDendriteNetwork(connectivity1, connectivity2, out_features)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


# `ScalarDendriteReliabilityGated.forward(x, reliability)` and
# `BeliefDendriteNetwork.forward(x, reliability)` already match the uniform
# `(x, c)` interface -- used directly, no wrapper needed.


# ---------------------------------------------------------------------------
# Sizing -- every family independently targets ~150k parameters (Sec K).
# ---------------------------------------------------------------------------


@dataclass
class BuiltModel:
    model: nn.Module
    family: str
    parameter_count: int
    hidden_size: int
    sizing: dict[str, Any] = field(default_factory=dict)


def _cellv03_param_count(hidden: int) -> int:
    # ConflictNormalizedLayer(in, out) has out*(in+2) parameters (V + gain_raw + bias).
    layer1 = hidden * (IN_FEATURES + 2)
    layer2 = hidden * (hidden + 2)
    readout = hidden * OUT_FEATURES + OUT_FEATURES
    return layer1 + layer2 + readout


def _match_cellv03_hidden(target: int, search_range: range) -> int:
    best_dim, best_diff = search_range[0], None
    for dim in search_range:
        diff = abs(_cellv03_param_count(dim) - target)
        if best_diff is None or diff < best_diff:
            best_dim, best_diff = dim, diff
    return best_dim


@functools.cache
def dendrite_target_hidden() -> int:
    """The shared uniform hidden width for all three dendritic families,
    solved once from the fixed structural hyperparameters (Sec K
    exploratory defaults: `B=4`, first-layer `7x7` patch, hidden-layer
    `K=min(32, H)`, which is exactly `DENDRITE_K2=32` once `H > 32`, checked
    below rather than assumed)."""
    sol = solve_hidden_width_for_budget(
        PARAM_BUDGET, DENDRITE_BRANCHES, DENDRITE_K1, DENDRITE_K2, OUT_FEATURES
    )
    if sol.hidden <= DENDRITE_K2:
        raise RuntimeError(
            f"solved hidden width {sol.hidden} <= DENDRITE_K2={DENDRITE_K2}; the "
            f"K2=min(32, H) default-coupling assumption baked into this budget solve "
            f"no longer holds."
        )
    return sol.hidden


@functools.cache
def _dendrite_connectivity(seed: int) -> tuple[DendriticConnectivity, DendriticConnectivity]:
    hidden = dendrite_target_hidden()
    return make_connectivity_pair(
        in_features=IN_FEATURES,
        hidden1=hidden,
        hidden2=hidden,
        branches_per_soma=DENDRITE_BRANCHES,
        seed=seed,
        image_shape=IMAGE_SHAPE,
        patch=DENDRITE_PATCH,
        sources_per_branch_1=DENDRITE_K1,
        sources_per_branch_2=DENDRITE_K2,
    )


def build_model(family: str, seed: int) -> BuiltModel:
    """Construct one Sec N family, sized to ~`PARAM_BUDGET` parameters.
    `seed` only affects the dendritic connectivity draw (same seed ->
    identical topology across all three dendritic families, so their
    comparison is over topology, not just parameter count) -- MLP/CellV0.3
    sizing is seed-independent."""
    if family == PLAIN_MLP:
        hidden = match_hidden_dim(
            PARAM_BUDGET,
            IN_FEATURES,
            OUT_FEATURES,
            num_hidden_layers=1,
            search_range=_MLP_SEARCH_RANGE,
        )
        model = PlainMLP(IN_FEATURES, hidden, OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(model, family, n, hidden, {"hidden_dim": hidden})

    if family == CONFIDENCE_MLP:
        hidden = match_hidden_dim(
            PARAM_BUDGET,
            2 * IN_FEATURES,
            OUT_FEATURES,
            num_hidden_layers=1,
            search_range=_MLP_SEARCH_RANGE,
        )
        model = ConfidenceMLP(IN_FEATURES, hidden, OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(model, family, n, hidden, {"hidden_dim": hidden})

    if family == CELLV03:
        hidden = _match_cellv03_hidden(PARAM_BUDGET, _CELLV03_SEARCH_RANGE)
        model = DenseCellV03(IN_FEATURES, hidden, OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(
            model,
            family,
            n,
            hidden,
            {"hidden_cells": hidden, "predicted_params": _cellv03_param_count(hidden)},
        )

    if family in DENDRITE_FAMILIES:
        conn1, conn2 = _dendrite_connectivity(seed)
        hidden = dendrite_target_hidden()
        if family == SCALAR_DENDRITE:
            model: nn.Module = ScalarDendriteModel(conn1, conn2, OUT_FEATURES)
        elif family == SCALAR_DENDRITE_GATED:
            model = ScalarDendriteReliabilityGated(conn1, conn2, OUT_FEATURES)
        else:
            model = BeliefDendriteNetwork(conn1, conn2, OUT_FEATURES)
        n = count_parameters(model)
        sizing = {
            "hidden": hidden,
            "branches_per_soma": DENDRITE_BRANCHES,
            "sources_per_branch_1": DENDRITE_K1,
            "sources_per_branch_2": DENDRITE_K2,
            "connectivity_seed": seed,
        }
        return BuiltModel(model, family, n, hidden, sizing)

    raise ValueError(f"unknown model family {family!r}; expected one of {MODEL_FAMILIES}")
