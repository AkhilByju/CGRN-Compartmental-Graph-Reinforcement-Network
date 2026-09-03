"""`SparseDynamicBeliefGraph` -- CellV1.1: encoder -> `T` shared sparse
refinement steps -> decoder. Sparse counterpart to `model.py`'s
`DynamicBeliefGraph` ("CellV1 Dense Reference" -- kept, unmodified, for
correctness checks at small `n_cells`, per docs/architecture_v1.md).
Same standard `nn.Module` interface; same pluggable `encoder` (the dense
`PopulationEncoder` and `object_encoder.ObjectSeededEncoder` both work
unchanged here -- LSH-based routing only changes how cells find each
other *after* encoding, not how the population is initialized) and
`use_global_routing` ablation flag.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.decoder import PopulationDecoder
from src.models.architecture_v1.encoder import PopulationEncoder
from src.models.architecture_v1.sparse_dynamics import SparseDynamicBeliefGraphCore, SparseGraphs


class SparseDynamicBeliefGraph(nn.Module):
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
        num_hashes_local: int = 2,
        bits_local: int = 6,
        chunk_size_local: int = 10,
        window_local: int = 0,
        num_hashes_global: int = 2,
        bits_global: int = 6,
        chunk_size_global: int = 4,
        window_global: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PopulationEncoder(
            in_features, n_cells, association_dim, eps=eps
        )
        self.core = SparseDynamicBeliefGraphCore(
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
            num_hashes_local=num_hashes_local,
            bits_local=bits_local,
            chunk_size_local=chunk_size_local,
            window_local=window_local,
            num_hashes_global=num_hashes_global,
            bits_global=bits_global,
            chunk_size_global=chunk_size_global,
            window_global=window_global,
        )
        self.decoder = PopulationDecoder(association_dim, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cells(x)[0]

    def forward_with_cells(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1]:
        cells = self.encoder(x)
        cells = self.core(cells)
        prediction = self.decoder(cells)
        return prediction, cells

    def forward_with_graphs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCellV1, list[SparseGraphs]]:
        cells = self.encoder(x)
        cells, graphs = self.core.forward_with_graphs(cells)
        prediction = self.decoder(cells)
        return prediction, cells, graphs
