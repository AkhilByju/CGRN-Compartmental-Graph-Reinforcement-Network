"""The four Paper-A Phase-2 model families.

All four expose the **same** interface -- ``forward(x_corrupted, c)`` -- so the
training / evaluation loop is model-agnostic. What each does with ``c`` is the
whole experiment:

======================  ==============================================  ================
Family                   input to the underlying net                     reliability seen
======================  ==============================================  ================
``A`` plain_mlp          ``x_corrupted``                                  none
``B`` confidence_mlp     ``concat(x_corrupted, c)``  (dim ``2 * D``)      exact ``c``
``C`` reliability_gated  ``c * x_corrupted``         (dim ``D``)          exploited directly
``D`` cellv0.3           belief ``(mu=x_corrupted, e=c, u=0)``            exact ``c``
======================  ==============================================  ================

``A``/``B``/``C`` are the frozen Phase-1 ``Linear -> SiLU -> Linear`` MLP
(`src/models/baselines/mlp.py`, one hidden layer). ``D`` wraps the **frozen**
`BeliefNetworkV03` (`src/models/architecture_v0/conflict_normalized.py`) and
only changes the *input belief* -- `BeliefNetworkV03` itself is not touched.

Parameter budgets: CellV0.3's actual trainable-parameter count (fitted to the
Phase-1 per-dataset ``PARAM_BUDGET``) is the target; A/B/C's hidden width is
chosen to land within 2% of that count. Because B's input is twice as wide, it
needs a different hidden width to hit the same count -- fairness is by
parameters, not width (Phase-2 task Sec 7).
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
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.conflict_normalized import BeliefNetworkV03  # noqa: E402
from src.models.architecture_v0.precision_gain import effective_precision  # noqa: E402
from src.models.baselines.mlp import (  # noqa: E402
    MLPBaseline,
    match_hidden_dim,
    param_count_for_mlp,
)

PLAIN_MLP = "plain_mlp"
CONFIDENCE_MLP = "confidence_mlp"
RELIABILITY_GATED_MLP = "reliability_gated_mlp"
CELLV03 = "cellv0.3"
MODEL_FAMILIES: tuple[str, ...] = (PLAIN_MLP, CONFIDENCE_MLP, RELIABILITY_GATED_MLP, CELLV03)

RELIABILITY_AWARE: frozenset[str] = frozenset({CONFIDENCE_MLP, CELLV03})

_MLP_SEARCH_RANGE = range(1, 6000)
_MATCH_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Model wrappers -- uniform (x_corrupted, c) -> logits
# ---------------------------------------------------------------------------


class PlainMLP(nn.Module):
    """Model A. Receives only ``x_corrupted``; ``c`` is ignored (it is accepted
    so the training loop can stay model-agnostic). Establishes how hard the
    corruption is on its own."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.net = MLPBaseline(
            in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


class ConfidenceMLP(nn.Module):
    """Model B. Receives ``concat(x_corrupted, c)`` -- the exact same
    reliability information CellV0.3 gets, handed to an ordinary MLP. The
    critical baseline: can a conventional net simply *learn* what CellV0.3
    hard-codes?"""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.raw_in_features = in_features
        self.net = MLPBaseline(
            2 * in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=-1))


class ReliabilityGatedMLP(nn.Module):
    """Model C. Receives exactly ``c * x_corrupted`` -- the trivial engineered
    control: attenuate unreliable observations once at the input, then a plain
    MLP. No extra machinery. Asks whether hidden-state reliability propagation
    beats a single input-side gate."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.net = MLPBaseline(
            in_features, hidden_dim, out_features, num_hidden_layers=1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(c * x)


class ReliabilityCellV03(nn.Module):
    """Model D. The **frozen** `BeliefNetworkV03`, with its input belief
    initialized from the corruption metadata: ``mu = x_corrupted``,
    ``e = c``, ``u = 0`` (Phase-2 task Sec 1). Everything downstream -- the two
    `ConflictNormalizedLayer`s and the linear readout -- is untouched.
    """

    def __init__(self, in_features: int, hidden_cells: int, out_features: int) -> None:
        super().__init__()
        self.net = BeliefNetworkV03(in_features, hidden_cells, out_features)

    def _input_belief(self, x: torch.Tensor, c: torch.Tensor) -> BeliefCell:
        return BeliefCell(mu=x, evidence=c, uncertainty=torch.zeros_like(x))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b1 = self.net.layer1(self._input_belief(x, c))
        b2 = self.net.layer2(b1)
        return self.net.readout(b2.mu)

    def belief_states_verbose(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[tuple[BeliefCell, torch.Tensor], tuple[BeliefCell, torch.Tensor]]:
        """``((belief1, consensus1), (belief2, consensus2))`` from the
        corrupted input belief -- for the Phase-2 mechanism diagnostics."""
        b1, cons1 = self.net.layer1.forward_verbose(self._input_belief(x, c))
        b2, cons2 = self.net.layer2.forward_verbose(b1)
        return (b1, cons1), (b2, cons2)


# ---------------------------------------------------------------------------
# Sizing / parameter matching
# ---------------------------------------------------------------------------


@dataclass
class BuiltModel:
    model: nn.Module
    family: str
    parameter_count: int
    hidden_size: int
    sizing: dict[str, float | int | bool] = field(default_factory=dict)


def cellv03_target(in_features: int, out_features: int, param_budget: int) -> tuple[int, int]:
    """``(hidden_cells, actual_param_count)`` for the CellV0.3 arm at this
    budget -- the target every other family is matched to."""
    hc = belief_v03_hidden_cells_for_budget(in_features, out_features, param_budget)
    return hc, belief_v03_param_count(hc, in_features, out_features)


def _build_matched_mlp(
    cls: type[nn.Module],
    family: str,
    *,
    raw_in_features: int,
    net_in_features: int,
    out_features: int,
    target_params: int,
) -> BuiltModel:
    hidden_dim = match_hidden_dim(
        target_params, net_in_features, out_features,
        num_hidden_layers=1, search_range=_MLP_SEARCH_RANGE,
    )
    model = cls(raw_in_features, hidden_dim, out_features)
    n_params = count_parameters(model)
    rel_diff = abs(n_params - target_params) / target_params
    return BuiltModel(
        model=model,
        family=family,
        parameter_count=n_params,
        hidden_size=hidden_dim,
        sizing={
            "hidden_dim": hidden_dim,
            "net_in_features": net_in_features,
            "target_params": target_params,
            "params": n_params,
            "param_diff": n_params - target_params,
            "param_rel_diff": rel_diff,
            "param_match_within_2pct": bool(rel_diff <= _MATCH_TOLERANCE),
            "params_formula": param_count_for_mlp(
                net_in_features, hidden_dim, out_features, num_hidden_layers=1
            ),
        },
    )


def build_model(
    family: str, in_features: int, out_features: int, param_budget: int
) -> BuiltModel:
    """Construct one Phase-2 family sized against the CellV0.3 target."""
    hidden_cells, target = cellv03_target(in_features, out_features, param_budget)

    if family == CELLV03:
        model = ReliabilityCellV03(in_features, hidden_cells, out_features)
        n_params = count_parameters(model)
        return BuiltModel(
            model=model,
            family=CELLV03,
            parameter_count=n_params,
            hidden_size=hidden_cells,
            sizing={
                "hidden_cells": hidden_cells,
                "param_budget": param_budget,
                "params": n_params,
                "target_params": target,
                "params_within_budget": bool(n_params <= param_budget),
            },
        )
    if family == PLAIN_MLP:
        return _build_matched_mlp(
            PlainMLP, PLAIN_MLP, raw_in_features=in_features,
            net_in_features=in_features, out_features=out_features, target_params=target,
        )
    if family == CONFIDENCE_MLP:
        return _build_matched_mlp(
            ConfidenceMLP, CONFIDENCE_MLP, raw_in_features=in_features,
            net_in_features=2 * in_features, out_features=out_features, target_params=target,
        )
    if family == RELIABILITY_GATED_MLP:
        return _build_matched_mlp(
            ReliabilityGatedMLP, RELIABILITY_GATED_MLP, raw_in_features=in_features,
            net_in_features=in_features, out_features=out_features, target_params=target,
        )
    raise ValueError(f"unknown family {family!r}; expected one of {MODEL_FAMILIES}")


# ---------------------------------------------------------------------------
# CellV0.3 mechanism diagnostics from the corrupted input belief (Sec 13).
# Observational only -- never an auxiliary objective.
# ---------------------------------------------------------------------------

_DIAG_QUANTITIES = ("e", "u", "precision", "sqrt_precision", "abs_mu")
_DIAG_STATS = ("mean", "std", "min", "max")
BELIEF_DIAG_KEYS: frozenset[str] = frozenset(
    f"diag_{layer}_{q}_{s}"
    for layer in ("layer1", "layer2")
    for q in _DIAG_QUANTITIES
    for s in _DIAG_STATS
) | frozenset(
    f"diag_{layer}_{k}"
    for layer in ("layer1", "layer2")
    for k in ("precision_cv", "abs_consensus_mean")
)


def _rs_init() -> dict[str, float]:
    return {"n": 0.0, "sum": 0.0, "sq": 0.0, "min": float("inf"), "max": float("-inf")}


def _rs_update(slot: dict[str, float], t: torch.Tensor) -> None:
    slot["n"] += t.numel()
    slot["sum"] += float(t.sum())
    slot["sq"] += float((t * t).sum())
    slot["min"] = min(slot["min"], float(t.min()))
    slot["max"] = max(slot["max"], float(t.max()))


def _rs_final(slot: dict[str, float]) -> tuple[float, float, float, float]:
    n = slot["n"]
    mean = slot["sum"] / n
    var = max(slot["sq"] / n - mean * mean, 0.0)
    return mean, var**0.5, slot["min"], slot["max"]


@torch.no_grad()
def belief_diagnostics(
    model: ReliabilityCellV03,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    chunk: int = 1024,
) -> dict[str, float]:
    """Per-hidden-layer belief-state statistics on a corrupted batch: mean /
    std / min / max of ``e``, ``u``, ``pi = e/(1+e u)``, ``sqrt(pi)`` and
    ``|mu|``, plus the coefficient of variation of ``pi`` and the mean
    ``|consensus|``. Chunked running-moment accumulation (memory-flat on the
    10k-row image test splits)."""
    if not isinstance(model, ReliabilityCellV03):
        raise TypeError(f"belief diagnostics need a ReliabilityCellV03, got {type(model).__name__}")
    model.eval()
    acc: dict[tuple[str, str], dict[str, float]] = {}
    cons: dict[str, list[float]] = {}
    for i in range(0, x.shape[0], chunk):
        states = model.belief_states_verbose(x[i : i + chunk], c[i : i + chunk])
        for tag, (belief, consensus) in zip(("layer1", "layer2"), states):
            pi = effective_precision(belief.evidence, belief.uncertainty)
            values = {
                "e": belief.evidence,
                "u": belief.uncertainty,
                "precision": pi,
                "sqrt_precision": pi.clamp_min(0.0).sqrt(),
                "abs_mu": belief.mu.abs(),
            }
            for q, t in values.items():
                _rs_update(acc.setdefault((tag, q), _rs_init()), t)
            slot = cons.setdefault(tag, [0.0, 0.0])
            slot[0] += float(consensus.abs().sum())
            slot[1] += consensus.numel()

    out: dict[str, float] = {}
    for (tag, q), slot in acc.items():
        mean, std, lo, hi = _rs_final(slot)
        out[f"diag_{tag}_{q}_mean"] = mean
        out[f"diag_{tag}_{q}_std"] = std
        out[f"diag_{tag}_{q}_min"] = lo
        out[f"diag_{tag}_{q}_max"] = hi
    for tag, (abs_sum, n) in cons.items():
        out[f"diag_{tag}_abs_consensus_mean"] = abs_sum / n
    for tag in ("layer1", "layer2"):
        m = out[f"diag_{tag}_precision_mean"]
        sd = out[f"diag_{tag}_precision_std"]
        out[f"diag_{tag}_precision_cv"] = sd / (m + 1e-8)
    return out
