"""CellV1.6 -- `PrecisionRegulatedAssemblyGate` (docs/architecture_v1.md
Sec 17). ONE input-dependent population-competition mechanism inserted
between two frozen CellV0.1 layers -- it does not touch the belief math,
the fusion rule, or the state `(mu, e, u)`. Given a `BeliefCell`
population it produces a per-source-cell *participation gain* in `[0, 1]`
that `integration.BeliefLayer` folds into its learned relevance gate
(`g_effective[j, i] = g[j, i] * participation[i]`) before running the
unchanged scale-stable precision equations.

Everything below is the user's specification (2026-09-06). The only
implementation-choice defaults -- flagged the same way Sec 6 / Sec 16.10
flag CellV1's own glue-code gaps -- are:

- `F_part`'s exact shape: a minimal `2 -> 8 -> 1` MLP with SiLU (the user
  did specify these numbers, so this is barely a choice), no activation on
  the output. The output `Linear` has `bias=False`: the drive is
  immediately per-example standardized (`_standardize`), which is exactly
  invariant to a uniform additive constant, so an output bias would be a
  structurally dead parameter (permanent zero gradient) -- the same
  BatchNorm-makes-the-prior-bias-redundant situation. `F_part` is
  therefore `2*8 + 8 + 8*1 = 32` params.
- `kappa_raw_init = 0.0` (so `kappa = softplus(0) = ln 2 ~= 0.69`) and
  `width_bias_init = 3.0` (so the initial `delta` is wide -- most cells
  participate at init and the network *learns* to sparsify rather than
  starting sparse, mirroring `WriteGateFunction`'s near-off bias init
  convention, CellV1.2). Neither is a tuned assembly-size target.
- `delta_floor = 0.05`: a numerical floor the user gave explicitly, kept
  as a config constant, not a size target.
- Diagnostics are returned as detached 0-dim tensors (no host sync in the
  forward path -- the caller `.item()`s them when it actually logs); they
  never enter the autograd graph.

There is deliberately no auxiliary sparsity loss, no target support
fraction, no top-k, no renormalization of participation to sum to one
(these are neural gains, not a probability distribution). If the network
learns to activate almost every cell, that is a legitimate empirical
result.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.architecture_v0.cell import BeliefCell

DEFAULT_DELTA_FLOOR = 0.05


class PrecisionRegulatedAssemblyGate(nn.Module):
    """Input: a CellV0.1 `BeliefCell` population, `mu`/`evidence`/
    `uncertainty` each `(batch, n_cells)`. Output: `(participation,
    diagnostics)` where `participation` is `(batch, n_cells)` in `[0, 1]`
    and `diagnostics` is a dict of detached floats.

    The module has NO `n_cells` parameter -- `F_part` and the two scalars
    `kappa_raw` / `width_bias` are its entire parameter set (34 params:
    `2*8+8` + `8*1` + 1 + 1), shared across every cell and every
    population size. That is the structural guarantee that there is no
    per-cell embedding or per-cell MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 8,
        delta_floor: float = DEFAULT_DELTA_FLOOR,
        eps: float = 1e-8,
        kappa_raw_init: float = 0.0,
        width_bias_init: float = 3.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.delta_floor = delta_floor

        # F_part: shared 2 -> hidden -> 1 MLP, applied independently to
        # every cell's [mu_i, log_precision_i]. SiLU on the hidden layer,
        # linear output (drive_i is unbounded).
        self.participation_drive = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.kappa_raw = nn.Parameter(torch.tensor(float(kappa_raw_init)))
        self.width_bias = nn.Parameter(torch.tensor(float(width_bias_init)))

    @staticmethod
    def _standardize(drive: torch.Tensor, eps: float) -> torch.Tensor:
        """Per-example (over the cell dim) zero-mean/unit-variance of the
        participation drive. Invariant to positive affine changes of
        `drive` up to the `eps` in the denominator -- `z(a*drive + b) ~=
        z(drive)` for `a > 0`."""
        mean = drive.mean(dim=-1, keepdim=True)
        var = ((drive - mean) ** 2).mean(dim=-1, keepdim=True)
        return (drive - mean) / torch.sqrt(var + eps)

    def delta_from_confidence(self, population_confidence: torch.Tensor) -> torch.Tensor:
        """`delta = delta_floor + softplus(width_bias - kappa * C)`, with
        `kappa = softplus(kappa_raw) >= 0`. Because `kappa >= 0` and
        `softplus` is monotone, `delta` is non-increasing in the population
        confidence `C` -- a more-confident population competes over a
        *narrower* window. Exposed as a method so the monotonicity can be
        unit-tested directly."""
        kappa = F.softplus(self.kappa_raw)
        return self.delta_floor + F.softplus(self.width_bias - kappa * population_confidence)

    def participation(
        self, belief: BeliefCell
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu = belief.mu
        e = belief.evidence
        u = belief.uncertainty

        # Same eps convention as integration.BeliefLayer.
        log_precision = torch.log(e + self.eps) - torch.log(u**2 + self.eps)

        # drive_i = F_part([mu_i, log_precision_i]); shared params, applied
        # over the last dim -> (batch, n_cells).
        features = torch.stack([mu, log_precision], dim=-1)
        drive = self.participation_drive(features).squeeze(-1)

        z = self._standardize(drive, self.eps)

        # Population confidence, per example.
        confidence = log_precision.mean(dim=-1)  # (batch,)
        delta = self.delta_from_confidence(confidence)  # (batch,)

        z_max = z.max(dim=-1).values  # (batch,)
        gap = z_max.unsqueeze(-1) - z  # (batch, n_cells), >= 0 by construction
        participation = torch.relu(1.0 - gap / (delta.unsqueeze(-1) + self.eps))
        # participation in [0, 1]: gap >= 0 gives the <= 1 bound; relu gives
        # the >= 0 bound. The max-drive cell has gap == 0 -> participation
        # exactly 1. Cells more than delta below the max have a negative
        # relu argument -> exact 0. Never renormalized.

        with torch.no_grad():
            active = participation > 0
            diagnostics = {
                "assembly_active_fraction": active.to(participation.dtype).mean().detach(),
                "assembly_mean_participation": participation.mean().detach(),
                "assembly_delta": delta.mean().detach(),
                "assembly_population_log_precision": confidence.mean().detach(),
            }

        return participation, diagnostics

    def forward(
        self, belief: BeliefCell
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.participation(belief)
