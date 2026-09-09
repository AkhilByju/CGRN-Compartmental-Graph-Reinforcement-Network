"""CellV0.3 -- the Conflict-Normalized Belief Cell.

A THIRD, separate CellV0-line aggregation operator, specified in full by the
user (docs/architecture_v0.md Sec 10). Like CellV0.2 it drops CellV0.1's
relevance-gate matrix and carries a single signed connection matrix ``V``
plus a per-output ``gain_raw`` / ``bias`` -- but it removes CellV0.2's
population-relative precision gain ``2 pi / (pi + mean pi)`` entirely.

CellV0.2 revealed a specific failure: that relative gain removes *absolute*
confidence. If every source becomes uniformly less reliable, the relative
gain stays ``1`` and nothing downstream can tell. CellV0.3 instead:

* forms a precision-weighted signed consensus,
* measures the disagreement (conflict) among the contributing beliefs,
* propagates support conservatively (``e_out <= max_j pi_j``),
* derives its own usable precision ``pi_out = e_out / (1 + e_out u_out)``,
* folds ``sqrt(pi_out)`` into *its own* activation in the same forward step.

So evidence/conflict is causally load-bearing -- it attenuates or amplifies
the very neuron whose belief it describes -- rather than passive metadata.

State primitive: the same ``BeliefCell`` ``(mu, e, u)`` as CellV0.1/CellV0.2,
with the *input* belief initialized ``e = 1, u = 0`` (reusing CellV0.2's
``initial_belief``). Here ``e`` is read as inherited support and ``u`` as an
internal conflict/disagreement state -- NOT calibrated predictive
uncertainty.

Per-layer computation (the user's spec, not tuned or reformulated); source
population ``(mu_j, e_j, u_j)``, output cell ``i``, ``eps`` a denominator
floor only:

    usable precision   pi_j    = e_j / (1 + e_j u_j)

    abs_V   = |V|                             V has shape [out_cells, in_cells]
    row_l1  = sum_j |V_ij|  (clamped to eps)
    A_ij    = |V_ij| / row_l1_i               unsigned weights, sum_j A_ij ~ 1
    S_ij    = V_ij  / row_l1_i                signed weights

    inherited support  e_i     = sum_j A_ij pi_j                 # GEMM  (convex -> <= max_j pi_j)
    signed consensus   c_i     = (sum_j S_ij pi_j mu_j) / e_i    # GEMM / e_i
    second moment      s_i     = (sum_j A_ij pi_j mu_j^2) / e_i  # GEMM / e_i
    conflict           u_i     = max(0, s_i - c_i^2)             # A-weighted var of sign(V_ij) mu_j
    output precision   pi_i    = e_i / (1 + e_i u_i)             # <= e_i
    confidence scale   k_i     = sqrt(pi_i)
    content            mu_i    = tanh(gamma_i c_i k_i + bias_i),  gamma_i = softplus(gain_raw_i)

``gain_raw`` is inverse-softplus-initialized so ``gamma_i = ||V_i||_1`` at
init (the same convention as CellV0.2). Unlike CellV0.2, a neutral-confidence
input does *not* generically reduce the layer to ``tanh(F.linear(mu, V, b))``
-- only the zero-conflict case does (``sqrt(pi_i) = 1`` needs ``pi_i = 1``,
i.e. ``e_i = 1`` and ``u_i = 0``).

Every reduction is a matmul or a dim-reduction -- three GEMMs per layer
(``(pi*mu) @ S.T``, ``pi @ A.T``, ``(pi*mu^2) @ A.T``). No
``(batch, out_cells, in_cells)`` edge tensor is ever materialized.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.precision_gain import (
    _inverse_softplus,
    effective_precision,
    initial_belief,
)

# `effective_precision(e, u) = e / (1 + e u)` is exactly CellV0.3's "usable
# precision"; `initial_belief(x)` is exactly CellV0.3's input belief
# (`mu = x, e = 1, u = 0`); `_inverse_softplus` is the shared gain-init helper.
# Reused from `precision_gain` rather than copied -- they are shared CellV0-line
# primitives and CellV0.2's equations are not modified by importing them.
_DEFAULT_EPS = 1e-8


class ConflictNormalizedLayer(nn.Module):
    """One CellV0.3 layer: a ``BeliefCell`` over ``in_cells`` -> a
    ``BeliefCell`` over ``out_cells``.

    Exactly three trainable parameter objects and **no** relevance/gate
    matrix:

    * ``V``        ``[out_cells, in_cells]`` -- signed connection directions
    * ``gain_raw`` ``[out_cells]``           -- pre-softplus output amplitude
    * ``bias``     ``[out_cells]``
    """

    def __init__(self, in_cells: int, out_cells: int, eps: float = _DEFAULT_EPS) -> None:
        super().__init__()
        self.in_cells = in_cells
        self.out_cells = out_cells
        self.eps = eps

        self.V = nn.Parameter(torch.empty(out_cells, in_cells))
        self.gain_raw = nn.Parameter(torch.empty(out_cells))
        self.bias = nn.Parameter(torch.zeros(out_cells))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Identical to CellV0.2's PrecisionGainLayer: Kaiming-uniform V
        # (a = sqrt(5)), then gain_raw set by inverse-softplus so
        # softplus(gain_raw) == ||V_i||_1 at init (amplitude convention shared
        # with CellV0.2 -- see docs/architecture_v0.md Sec 10).
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        with torch.no_grad():
            row_l1 = self.V.abs().sum(dim=1).clamp_min(self.eps)
            work_dtype = torch.float64 if row_l1.device.type == "cpu" else row_l1.dtype
            self.gain_raw.copy_(
                _inverse_softplus(row_l1.to(work_dtype), self.eps).to(self.V.dtype)
            )
            self.bias.zero_()

    def forward(self, incoming: BeliefCell) -> BeliefCell:
        return self.forward_verbose(incoming)[0]

    def forward_verbose(self, incoming: BeliefCell) -> tuple[BeliefCell, torch.Tensor]:
        """``forward`` plus the pre-activation signed ``consensus`` tensor,
        exposed for the lightweight CellV0.3 diagnostics only. ``consensus``
        does not affect training -- it is one of the tensors ``forward``
        already computes internally."""
        if incoming.mu.shape[-1] != self.in_cells:
            raise ValueError(
                f"Expected {self.in_cells} incoming cells, got shape {tuple(incoming.mu.shape)}."
            )

        pi = effective_precision(incoming.evidence, incoming.uncertainty)  # (B, in)
        mu = incoming.mu                                                   # (B, in)

        abs_V = self.V.abs()                                          # (out, in)
        row_l1 = abs_V.sum(dim=1).clamp_min(self.eps)                 # (out,)
        A = abs_V / row_l1.unsqueeze(1)                               # (out, in), rows ~ sum to 1
        S = self.V / row_l1.unsqueeze(1)                              # (out, in)

        # Three GEMMs -- (B, in) @ (in, out) -> (B, out) -- and no
        # (B, out, in) intermediate anywhere.
        e_out = (pi @ A.t()).clamp_min(self.eps)                      # (B, out)  convex comb of pi
        consensus = ((pi * mu) @ S.t()) / e_out                      # (B, out)
        second_moment = ((pi * mu.square()) @ A.t()) / e_out         # (B, out)
        u_out = (second_moment - consensus.square()).clamp_min(0.0)   # (B, out)

        pi_out = effective_precision(e_out, u_out)                    # (B, out)  <= e_out
        gamma = F.softplus(self.gain_raw)                             # (out,)
        preactivation = gamma * consensus * pi_out.sqrt() + self.bias  # (B, out)
        mu_out = torch.tanh(preactivation)                            # (B, out)

        return BeliefCell(mu=mu_out, evidence=e_out, uncertainty=u_out), consensus

    def extra_repr(self) -> str:
        return f"in_cells={self.in_cells}, out_cells={self.out_cells}"


class BeliefNetworkV03(nn.Module):
    """``input -> ConflictNormalizedLayer -> ConflictNormalizedLayer ->
    linear readout``.

    Mirrors CellV0.2's ``BeliefNetworkV02`` (two hidden belief layers + a
    linear readout) except the readout consumes ``final_mu`` **directly** --
    it is *not* re-scaled by precision, because every CellV0.3 cell has
    already folded its own output precision into its activation
    (docs/architecture_v0.md Sec 10). The readout does not emit a belief
    state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_cells: int,
        out_features: int,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.layer1 = ConflictNormalizedLayer(in_features, hidden_cells, eps=eps)
        self.layer2 = ConflictNormalizedLayer(hidden_cells, hidden_cells, eps=eps)
        self.readout = nn.Linear(hidden_cells, out_features)

    def layer_states(self, x: torch.Tensor) -> tuple[BeliefCell, BeliefCell]:
        """The two hidden belief states ``(after layer1, after layer2)``."""
        b1 = self.layer1(initial_belief(x))
        return b1, self.layer2(b1)

    def layer_states_verbose(
        self, x: torch.Tensor
    ) -> tuple[tuple[BeliefCell, torch.Tensor], tuple[BeliefCell, torch.Tensor]]:
        """``((belief1, consensus1), (belief2, consensus2))`` -- for the
        CellV0.3 diagnostics only."""
        b1, c1 = self.layer1.forward_verbose(initial_belief(x))
        b2, c2 = self.layer2.forward_verbose(b1)
        return (b1, c1), (b2, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, b2 = self.layer_states(x)
        return self.readout(b2.mu)
