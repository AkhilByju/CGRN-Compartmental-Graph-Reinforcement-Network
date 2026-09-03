"""BeliefCellV1 -- the CellV1 state primitive (docs/architecture_v1.md §5).

Extends `src.models.architecture_v0.cell.BeliefCell`'s `(mu, evidence,
uncertainty)` with a fourth field, `z`: a small association/key vector
that answers "what is this belief about / who should this cell talk to,"
kept deliberately separate from `mu` (content) because CellV0's
`(mu, evidence, uncertainty)` alone can't support meaningful dynamic
grouping -- two cells can have near-identical `mu` while representing
unrelated things (docs/architecture_v1.md §4).

This module defines only the state container and its invariants, mirroring
`architecture_v0/cell.py`'s scope -- the dynamic graph construction that
`z` enables lives in `routing.py`, and how `(mu, evidence, uncertainty, z)`
update over a refinement step lives in `dynamics.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BeliefCellV1:
    """A CellV1 belief state `(mu, evidence, uncertainty, z)` for a
    population of cells.

    `mu`, `evidence`, and `uncertainty` share one shape, e.g. `(batch,
    n_cells)`. `z` has one extra trailing dimension for the association
    vector: `(batch, n_cells, association_dim)`. `z.shape[:-1]` must equal
    `mu.shape`.
    """

    mu: torch.Tensor
    evidence: torch.Tensor
    uncertainty: torch.Tensor
    z: torch.Tensor

    def __post_init__(self) -> None:
        scalar_shapes = (self.mu.shape, self.evidence.shape, self.uncertainty.shape)
        if len(set(scalar_shapes)) != 1:
            raise ValueError(
                f"mu, evidence, and uncertainty must share a shape, got {scalar_shapes}."
            )
        if self.z.shape[:-1] != self.mu.shape:
            raise ValueError(
                f"z's shape (excluding its last, association-dim axis) must match "
                f"mu's shape. Got z={tuple(self.z.shape)}, mu={tuple(self.mu.shape)}."
            )
        if self.z.dim() < 1:
            raise ValueError("z must have at least one dimension (the association axis).")

        dtypes = (self.mu.dtype, self.evidence.dtype, self.uncertainty.dtype, self.z.dtype)
        if len(set(dtypes)) != 1:
            raise ValueError(f"mu, evidence, uncertainty, and z must share a dtype, got {dtypes}.")

        devices = (self.mu.device, self.evidence.device, self.uncertainty.device, self.z.device)
        if len(set(devices)) != 1:
            raise ValueError(
                f"mu, evidence, uncertainty, and z must share a device, got {devices}."
            )

        if bool((self.evidence < 0).any()):
            raise ValueError("evidence (e) must be non-negative.")
        if bool((self.uncertainty < 0).any()):
            raise ValueError("uncertainty (u) must be non-negative.")

    @property
    def shape(self) -> torch.Size:
        return self.mu.shape

    @property
    def n_cells(self) -> int:
        return self.mu.shape[-1]

    @property
    def association_dim(self) -> int:
        return self.z.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mu.dtype

    def to(self, *args, **kwargs) -> BeliefCellV1:
        """Moves/casts all four fields together, mirroring `torch.Tensor.to`."""
        return BeliefCellV1(
            mu=self.mu.to(*args, **kwargs),
            evidence=self.evidence.to(*args, **kwargs),
            uncertainty=self.uncertainty.to(*args, **kwargs),
            z=self.z.to(*args, **kwargs),
        )

    def detach(self) -> BeliefCellV1:
        return BeliefCellV1(
            mu=self.mu.detach(),
            evidence=self.evidence.detach(),
            uncertainty=self.uncertainty.detach(),
            z=self.z.detach(),
        )
