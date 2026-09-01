"""BeliefNetwork -- a minimal feedforward stack of BeliefLayers, built ONLY
for Experiment 002/003 (docs/experiment_protocol.md).

`Input -> BeliefLayer -> BeliefLayer -> linear readout -> output`. No
graph, no clusters, no recurrence, no stochasticity -- this is
deliberately NOT `ArchitectureV0` (see `model.py`, still unspecified); it
exists solely to test whether CellV0's belief-aggregation methods provide
any advantage over an ordinary MLP unit in isolation
(docs/architecture_v0.md Sec 1 "Test the cell before the graph").

The final linear readout (`BeliefLayer`'s last `mu` -> task output) is a
minimal, task-specific decode step, not the general "decoder" question in
docs/architecture_v0.md Sec 5 -- every `BeliefLayer` applies `tanh` to
`mu` internally, so an unconstrained output (e.g. for regression) needs a
linear projection on top of it, exactly as docs/research_log.md's
"Recommended Hidden Activation" note anticipated.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.integration import AggregationMethod, BeliefLayer


class BeliefNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_cells: int,
        out_features: int,
        aggregation: AggregationMethod = "reliability",
    ) -> None:
        super().__init__()
        self.layer1 = BeliefLayer(in_features, hidden_cells, aggregation=aggregation)
        self.layer2 = BeliefLayer(hidden_cells, hidden_cells, aggregation=aggregation)
        self.readout = nn.Linear(hidden_cells, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard `nn.Module` interface (raw prediction only), so this
        drops directly into `src.training.trainer.train_loop`/
        `evaluate_loop`. Use `forward_with_beliefs` to also inspect the
        final layer's evidence/uncertainty."""
        return self.forward_with_beliefs(x)[0]

    def forward_with_beliefs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCell]:
        belief = BeliefCell.from_observed_features(x)
        belief = self.layer1(belief)
        belief = self.layer2(belief)
        prediction = self.readout(belief.mu)
        return prediction, belief
