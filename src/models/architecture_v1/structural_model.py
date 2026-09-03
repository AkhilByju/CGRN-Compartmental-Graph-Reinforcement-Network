"""`StructuralBeliefGraph` -- the assembled CellV1.5 model: encoder ->
`T` structural-fusion/gated-write steps -> decoder (docs/
architecture_v1.md §16). Reuses `PopulationEncoder`/`PopulationDecoder`
directly, unwrapped -- like `association_model.py::LearnedAssociationField`
and unlike `field_model.py::FieldEncoder`, there is no per-input
coordinate for an encoder wrapper to compute; the structural addresses
`s_i` are model-level parameters (`structural.py::StructuralAddress`),
not something derived from the current input.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.decoder import PopulationDecoder
from src.models.architecture_v1.encoder import PopulationEncoder
from src.models.architecture_v1.structural_dynamics import StructuralRefinementCore
from src.models.architecture_v1.structural_plasticity import StructuralPlasticityConfig


class StructuralBeliefGraph(nn.Module):
    """See `StructuralRefinementCore`'s docstring for the full training-
    loop integration contract (`bootstrap_structural_graph` once before
    training, `update_edge_utility` after every `backward()`,
    `maybe_run_structural_plasticity` after every optimizer step)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cells: int = 128,
        association_dim: int = 8,
        d_s: int = 16,
        assoc_dim: int = 32,
        hidden_dim: int = 32,
        num_steps: int = 1,
        eps: float = 1e-8,
        gate_init_bias: float = -2.0,
        encoder: nn.Module | None = None,
        plasticity_config: StructuralPlasticityConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PopulationEncoder(
            in_features, n_cells, association_dim, eps=eps
        )
        self.core = StructuralRefinementCore(
            n_cells,
            association_dim,
            d_s=d_s,
            num_steps=num_steps,
            plasticity_config=plasticity_config,
            assoc_dim=assoc_dim,
            hidden_dim=hidden_dim,
            eps=eps,
            gate_init_bias=gate_init_bias,
        )
        self.decoder = PopulationDecoder(association_dim, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(x)[0]

    def forward_with_state(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1]:
        cells = self.encoder(x)
        cells = self.core(cells)
        prediction = self.decoder(cells)
        return prediction, cells

    def bootstrap_structural_graph(self, x: torch.Tensor, generator: torch.Generator | None = None) -> None:
        """§16.8: call once before training, on any batch of raw input
        `x` -- the resulting topology is disposable (learned
        representations are meaningless at init), just needs to exist so
        structural plasticity has something to refine."""
        with torch.no_grad():
            cells = self.encoder(x)
        self.core.bootstrap_structural_graph(cells, generator=generator)

    def maybe_run_structural_plasticity(
        self, x: torch.Tensor, step: int, total_steps: int, generator: torch.Generator | None = None
    ) -> bool:
        """§16.9: call once per optimizer step (after `optimizer.step()`)
        with the current training batch's raw input `x`; a no-op unless
        this step is due for a plasticity event."""
        with torch.no_grad():
            cells = self.encoder(x)
        return self.core.maybe_run_structural_plasticity(cells, step, total_steps, generator=generator)

    def update_edge_utility(self, decay: float = 0.99) -> None:
        """§16.6: call after `loss.backward()`, before
        `optimizer.zero_grad()`."""
        self.core.update_edge_utility(decay=decay)
