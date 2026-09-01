"""BeliefCell -- the CellV0 state primitive.

Design decision (docs/architecture_v0.md Sec 1, docs/research_log.md):
CellV0 replaces the single scalar activation `a_i` of an ordinary neuron
with a structured belief state:

    mu         in R    -- content: what the cell currently believes
    evidence   in R+   -- how much supporting information produced it
    uncertainty in R+  -- how (un)confident the cell is in `mu`

`tau` (persistence) is part of the eventual `(mu, evidence, uncertainty,
tau)` state but stays dormant until V0.1 introduces recurrence -- a
feedforward V0.0 cell fires once, so there is nothing yet for persistence
to persist across.

This module defines the belief-state container: its fields,
shape/dtype/value invariants, basic tensor-like ergonomics (`.to`,
`.detach`), and how a raw observed feature becomes an initial belief
(`from_observed_features`). It does NOT define how a layer combines N
incoming BeliefCells into one outgoing BeliefCell -- the belief-aggregation
operator lives in `integration.py`'s `BeliefLayer`, which currently
implements three competing candidate formulas pending an empirical
comparison (docs/research_log.md, docs/architecture_v0.md Sec 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BeliefCell:
    """A belief state `(mu, evidence, uncertainty)`.

    The three fields are tensors that must share a shape, dtype, and
    device. Shape is deliberately unconstrained beyond that agreement: a
    single cell is shape `()`, a layer of cells is `(num_cells,)`, and a
    batched layer is `(batch, num_cells)` -- the same object represents one
    conceptual belief cell or a whole vectorized layer's worth of them
    without redesign (see docs/architecture_v0.md "don't literally
    instantiate one Python object per cell").
    """

    mu: torch.Tensor
    evidence: torch.Tensor
    uncertainty: torch.Tensor

    def __post_init__(self) -> None:
        shapes = (self.mu.shape, self.evidence.shape, self.uncertainty.shape)
        if len(set(shapes)) != 1:
            raise ValueError(f"mu, evidence, and uncertainty must share a shape, got {shapes}.")

        dtypes = (self.mu.dtype, self.evidence.dtype, self.uncertainty.dtype)
        if len(set(dtypes)) != 1:
            raise ValueError(f"mu, evidence, and uncertainty must share a dtype, got {dtypes}.")

        devices = (self.mu.device, self.evidence.device, self.uncertainty.device)
        if len(set(devices)) != 1:
            raise ValueError(f"mu, evidence, and uncertainty must share a device, got {devices}.")

        if bool((self.evidence < 0).any()):
            raise ValueError("evidence (e) must be non-negative.")
        if bool((self.uncertainty < 0).any()):
            raise ValueError("uncertainty (u) must be non-negative.")

    @property
    def shape(self) -> torch.Size:
        return self.mu.shape

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mu.dtype

    def to(self, *args, **kwargs) -> BeliefCell:
        """Moves/casts all three fields together, mirroring `torch.Tensor.to`."""
        return BeliefCell(
            mu=self.mu.to(*args, **kwargs),
            evidence=self.evidence.to(*args, **kwargs),
            uncertainty=self.uncertainty.to(*args, **kwargs),
        )

    def detach(self) -> BeliefCell:
        return BeliefCell(
            mu=self.mu.detach(),
            evidence=self.evidence.detach(),
            uncertainty=self.uncertainty.detach(),
        )

    @classmethod
    def from_observed_features(cls, x: torch.Tensor) -> BeliefCell:
        """Initial belief for directly observed raw features: `mu = x`,
        `evidence = 1`, `uncertainty = 1` for every feature (docs/research_log.md
        "Input belief initialization"). Evidence and uncertainty start
        equal for every feature so that any useful evidence/uncertainty
        structure has to be learned by the network, not hand-given.

        This is deliberately narrow -- appropriate for the first
        experiments over directly observed feature vectors (Experiment
        002/006), not a general encoder for arbitrary observations or
        language (see `encoder.py`, still unspecified).
        """
        return cls(mu=x, evidence=torch.ones_like(x), uncertainty=torch.ones_like(x))
