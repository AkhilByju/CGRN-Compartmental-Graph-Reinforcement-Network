"""`DynamicBeliefGraph` -- the assembled CellV1 architecture: encoder ->
`T` shared refinement steps -> decoder (docs/architecture_v1.md "The
complete reasoning cycle").

Standard `nn.Module` interface (`forward(x) -> prediction`), matching
`src.models.baselines.mlp.MLPBaseline` and
`src.models.architecture_v0.belief_network.BeliefNetwork` so it drops
directly into `src.training.trainer.train_loop`/`evaluate_loop`.
`forward_with_cells` additionally returns the final cell population;
`forward_with_graphs` additionally returns the per-step `(a_local,
a_global)` association matrices, for the graph-evolution analysis in
`experiments/v1_001_dynamic_groups`.

`encoder` accepts any `nn.Module` mapping raw input to a `BeliefCellV1`
population -- defaults to `PopulationEncoder` (dense projection to every
cell) but `experiments/v1_001_dynamic_groups` passes
`object_encoder.ObjectSeededEncoder` instead, so individual input objects
seed individual cells rather than every cell seeing the whole input at
once (docs/architecture_v1.md §6 / the user's 2026-09-02 note on why the
dense adapter can partially bypass what CellV1 is meant to study).
`use_global_routing=False` selects the CellV1-Local ablation
(`dynamics.py`).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.decoder import PopulationDecoder
from src.models.architecture_v1.dynamics import DynamicBeliefGraphCore, Graphs
from src.models.architecture_v1.encoder import PopulationEncoder


class DynamicBeliefGraph(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cells: int = 128,
        association_dim: int = 8,
        key_dim: int | None = None,
        hidden_dim: int = 32,
        num_steps: int = 6,
        lambda_: float = 1.0,
        tau: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-8,
        use_global_routing: bool = True,
        gate_init_bias: float = -2.0,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PopulationEncoder(
            in_features, n_cells, association_dim, eps=eps
        )
        self.core = DynamicBeliefGraphCore(
            n_cells,
            association_dim,
            num_steps=num_steps,
            key_dim=key_dim,
            hidden_dim=hidden_dim,
            lambda_=lambda_,
            tau=tau,
            gamma=gamma,
            eps=eps,
            use_global_routing=use_global_routing,
            gate_init_bias=gate_init_bias,
        )
        self.decoder = PopulationDecoder(association_dim, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cells(x)[0]

    def forward_with_cells(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1]:
        cells = self.encoder(x)
        cells = self.core(cells)
        prediction = self.decoder(cells)
        return prediction, cells

    def forward_with_graphs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1, list[Graphs]]:
        cells = self.encoder(x)
        cells, graphs = self.core.forward_with_graphs(cells)
        prediction = self.decoder(cells)
        return prediction, cells, graphs
