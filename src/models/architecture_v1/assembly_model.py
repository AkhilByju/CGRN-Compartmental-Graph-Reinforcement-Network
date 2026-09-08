"""CellV1.6 -- `PrecisionRegulatedAssemblyNetwork` (docs/architecture_v1.md
Sec 17). The assembled model:

    input
      -> CellV0.1 layer (scale_stable_precision BeliefLayer)
      -> PrecisionRegulatedAssemblyGate
      -> CellV0.1 layer (scale_stable_precision BeliefLayer, its learned
         relevance gate scaled by the gate's per-source participation)
      -> existing linear readout

This is `architecture_v0.belief_network.BeliefNetwork` with exactly one
mechanism added between the two layers -- the encoder
(`BeliefCell.from_observed_features`), the `(mu, e, u)` state, the
scale-stable precision fusion, and the readout are all untouched. With
`use_assembly_gate=False` (or the gate forced to participation-ones) the
forward pass reproduces the CellV0.1 `BeliefNetwork` computation exactly.

`.layer1` / `.layer2` / `.readout` are named to match
`BeliefNetwork`, so a CellV0.1 network's `state_dict` for those
submodules loads straight in (used by the equivalence tests).

Status: proposed and implemented, NOT evaluated. No experiment has been
run -- see docs/architecture_v1.md Sec 17's status line.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.integration import AggregationMethod, BeliefLayer
from src.models.architecture_v1.assembly_gate import (
    DEFAULT_DELTA_FLOOR,
    PrecisionRegulatedAssemblyGate,
)

_AGGREGATION: AggregationMethod = "scale_stable_precision"


class PrecisionRegulatedAssemblyNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_cells: int,
        out_features: int,
        *,
        assembly_hidden_dim: int = 8,
        delta_floor: float = DEFAULT_DELTA_FLOOR,
        eps: float = 1e-8,
        use_assembly_gate: bool = True,
        kappa_raw_init: float = 0.0,
        width_bias_init: float = 3.0,
    ) -> None:
        super().__init__()
        self.use_assembly_gate = use_assembly_gate
        self.layer1 = BeliefLayer(in_features, hidden_cells, aggregation=_AGGREGATION, eps=eps)
        self.layer2 = BeliefLayer(hidden_cells, hidden_cells, aggregation=_AGGREGATION, eps=eps)
        self.readout = nn.Linear(hidden_cells, out_features)
        self.assembly_gate = PrecisionRegulatedAssemblyGate(
            hidden_dim=assembly_hidden_dim,
            delta_floor=delta_floor,
            eps=eps,
            kappa_raw_init=kappa_raw_init,
            width_bias_init=width_bias_init,
        )
        # Last forward pass's assembly diagnostics (detached 0-dim
        # tensors), for a training loop to log. Empty until the first
        # `use_assembly_gate` forward.
        self.last_assembly_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_beliefs(x)[0]

    def forward_with_beliefs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCell]:
        belief = BeliefCell.from_observed_features(x)
        belief = self.layer1(belief)

        if self.use_assembly_gate:
            participation, diagnostics = self.assembly_gate(belief)
            self.last_assembly_diagnostics = diagnostics
            belief = self.layer2(belief, source_participation=participation)
        else:
            self.last_assembly_diagnostics = {}
            belief = self.layer2(belief)

        prediction = self.readout(belief.mu)
        return prediction, belief
