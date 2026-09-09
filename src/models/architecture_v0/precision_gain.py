"""CellV0.2 -- the Conservative Precision-Gain Cell.

A SECOND, separate CellV0-line aggregation operator, specified in full by the
user (docs/architecture_v0.md Sec 10) and distinct from CellV0.1
(`integration.py`'s ``scale_stable_precision`` ``BeliefLayer``). It is **not**
a sixth ``BeliefLayer`` aggregation method: it drops CellV0.1's separate
relevance-gate matrix (``a_ij`` / ``g_ij``) entirely, carrying one signed
connection matrix ``V`` plus a per-output gain and bias, so it needs its own
layer/module. CellV0.1's equations, parameters, and recorded benchmark
results are untouched by this file.

State primitive: the same ``BeliefCell`` ``(mu, e, u)`` as CellV0.1, but the
*input* belief is initialized ``e = 1, u = 0`` (not ``e = u = 1``) -- see
``initial_belief``.

Per-layer computation (the user's spec, not tuned or reformulated here); for
a source population ``(mu_j, e_j, u_j)`` and output cell ``i``:

    effective precision   pi_j   = e_j / (1 + e_j u_j)
    relative gain         r_j    = 2 pi_j / (pi_j + mean_k pi_k)      in (0, 2)
    gain-modulated msg    x_j    = r_j mu_j
    signed consensus      c_i    = (sum_j x_j V_ij) / sum_j |V_ij|
    content               mu_i   = tanh(gamma_i c_i + b_i),  gamma_i = softplus(gain_raw_i)
    inherited support     e_i    = (sum_j pi_j |V_ij|) / sum_j |V_ij|
    disagreement          u_i    = max(0, (sum_j x_j^2 |V_ij|) / sum_j |V_ij| - c_i^2)

``e_i`` is a convex combination of the source precisions (weights
``|V_ij| / sum_j|V_ij| >= 0``), so it can never exceed ``max_j pi_j`` --
"conservative". ``u_i`` is the ``|V_ij|``-weighted variance of the signed,
gain-modulated messages ``sign(V_ij) x_j``, so conflicting sources raise it
and aligned sources leave it at 0. The effective precision the next layer
sees is again ``pi_i = e_i / (1 + e_i u_i) <= e_i``.

``gain_raw`` is initialized by inverse-softplus so that ``gamma_i = ||V_i||_1``
at initialization: the layer then reduces *exactly* to
``tanh(F.linear(mu, V, b))`` on a neutral-confidence input (``e = 1, u = 0``),
and the ``(V, gamma)`` pair can represent any conventional linear weight row
(set ``gamma_i = ||V_i||_1``). See ``tests/test_precision_gain.py``.

Every reduction is a matmul or a dim-reduction -- three GEMMs per layer
(content, inherited precision, second moment). No
``(batch, out_cells, in_cells)`` edge tensor is ever materialized (unlike
``BeliefLayer``'s broadcast).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell

_DEFAULT_EPS = 1e-8


def effective_precision(evidence: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """``pi = e / (1 + e u)``. With the ``BeliefCell`` invariant ``e, u >= 0``
    the denominator is ``>= 1``, so no epsilon is needed to keep it finite."""
    return evidence / (1.0 + evidence * uncertainty)


def relative_gain(precision: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """``r_j = 2 pi_j / (pi_j + mean_k pi_k)`` -- the population mean is taken
    over the last (cell) axis and is **not** detached, so gradients reach
    every source's ``e`` / ``u`` through it.

    In ``(0, 2)`` for positive precision; exactly ``1`` when every cell has
    equal precision. The denominator is clamped to ``eps`` only so it stays
    finite if precision underflows to all-zero -- the clamp never engages for
    normal values (both terms are strictly positive)."""
    precision_mean = precision.mean(dim=-1, keepdim=True)
    denom = (precision + precision_mean).clamp_min(eps)
    return 2.0 * precision / denom


def _inverse_softplus(y: torch.Tensor, eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """``softplus^{-1}(y) = log(exp(y) - 1) = y + log(-expm1(-y))``. The
    second form is used because it is stable for large ``y`` (no ``exp``
    overflow). ``y`` is clamped to ``eps`` to avoid ``log(0)``."""
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))


def initial_belief(x: torch.Tensor) -> BeliefCell:
    """CellV0.2 input belief: ``mu = x``, ``e = 1``, ``u = 0``
    (docs/architecture_v0.md Sec 10). Contrast CellV0.1's
    ``BeliefCell.from_observed_features`` (``e = u = 1``)."""
    return BeliefCell(
        mu=x,
        evidence=torch.ones_like(x),
        uncertainty=torch.zeros_like(x),
    )


class PrecisionGainLayer(nn.Module):
    """One CellV0.2 layer: a ``BeliefCell`` over ``in_cells`` -> a
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
        # Same linear-layer init the rest of the project uses (BeliefLayer,
        # nn.Linear): Kaiming-uniform with a = sqrt(5).
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        with torch.no_grad():
            row_l1 = self.V.abs().sum(dim=1).clamp_min(self.eps)
            # Round-trip in higher precision where possible so that
            # softplus(gain_raw) == row_l1 to ~1e-6 -- this is what makes the
            # neutral-confidence reduction to tanh(F.linear(mu, V, b)) exact.
            work_dtype = torch.float64 if row_l1.device.type == "cpu" else row_l1.dtype
            self.gain_raw.copy_(
                _inverse_softplus(row_l1.to(work_dtype), self.eps).to(self.V.dtype)
            )
            self.bias.zero_()

    def forward(self, incoming: BeliefCell) -> BeliefCell:
        if incoming.mu.shape[-1] != self.in_cells:
            raise ValueError(
                f"Expected {self.in_cells} incoming cells, got shape {tuple(incoming.mu.shape)}."
            )

        precision = effective_precision(incoming.evidence, incoming.uncertainty)  # (B, in)
        gain = relative_gain(precision, self.eps)                                 # (B, in)
        x = gain * incoming.mu                                                    # (B, in)

        abs_V = self.V.abs()                                                      # (out, in)
        row_l1 = abs_V.sum(dim=1).clamp_min(self.eps)                             # (out,)
        gamma = F.softplus(self.gain_raw)                                         # (out,)

        # Three GEMMs -- (B, in) @ (in, out) -> (B, out) -- and no
        # (B, out, in) intermediate anywhere.
        consensus = (x @ self.V.t()) / row_l1                                     # (B, out)
        mu_out = torch.tanh(gamma * consensus + self.bias)                        # (B, out)

        e_out = (precision @ abs_V.t()) / row_l1                                  # (B, out)
        second_moment = (x.square() @ abs_V.t()) / row_l1                         # (B, out)
        u_out = (second_moment - consensus.square()).clamp_min(0.0)               # (B, out)

        return BeliefCell(mu=mu_out, evidence=e_out, uncertainty=u_out)

    def extra_repr(self) -> str:
        return f"in_cells={self.in_cells}, out_cells={self.out_cells}"


class BeliefNetworkV02(nn.Module):
    """``input -> PrecisionGainLayer -> PrecisionGainLayer -> confidence-scaled
    linear readout``.

    Mirrors CellV0.1's ``BeliefNetwork`` (two hidden belief layers + a linear
    readout) except the readout **consumes confidence**: it is applied to
    ``relative_gain(final_precision) * final_mu``, not ``final_mu`` alone
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
        self.layer1 = PrecisionGainLayer(in_features, hidden_cells, eps=eps)
        self.layer2 = PrecisionGainLayer(hidden_cells, hidden_cells, eps=eps)
        self.readout = nn.Linear(hidden_cells, out_features)

    def layer_states(self, x: torch.Tensor) -> tuple[BeliefCell, BeliefCell]:
        """The two hidden belief states ``(after layer1, after layer2)`` --
        exposed for the lightweight CellV0.2 diagnostics only."""
        b1 = self.layer1(initial_belief(x))
        b2 = self.layer2(b1)
        return b1, b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_beliefs(x)[0]

    def forward_with_beliefs(self, x: torch.Tensor) -> tuple[torch.Tensor, BeliefCell]:
        _, b2 = self.layer_states(x)
        final_precision = effective_precision(b2.evidence, b2.uncertainty)
        readout_input = relative_gain(final_precision, self.eps) * b2.mu
        return self.readout(readout_input), b2
