"""CellV0.3 as a Transformer FFN hidden-neuron primitive (task Sec 6, 8, 12).

Frozen correctness reference: `ConflictNormalizedLayer`
(`src/models/architecture_v0/conflict_normalized.py`). `BeliefFFN` below
computes **the same equations** using the algebraically equivalent
raw-moment form the task spec asks for (Sec 6, 8) -- see the module
docstring on `_raw_moment_finish` for the identity this rests on. This is
exactly the same reduction `src/models/architecture_v2/belief_dendrite_fast.
py`'s `BeliefDendriteLayerDenseFast` already uses and this project has
already parity-tested once (`tests/test_architecture_v2_belief_dendrite_
fast.py`); `tests/test_nano_transformer_belief_ffn.py` re-verifies it here
against `ConflictNormalizedLayer` directly, not assumed transitively.

No CellV0.3 equation is changed anywhere in this file (CLAUDE.md Sec 2,
task Sec 2/12: "Do not modify CellV0.3 equations").
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture_v0.precision_gain import _inverse_softplus

_DEFAULT_EPS = 1e-8


def _raw_moment_finish(
    L: torch.Tensor, E: torch.Tensor, C: torch.Tensor, Q: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(L, E, C, Q) -> (e, consensus, u, pi)`, the frozen CellV0.3 equations
    (Sec 6's raw-moment form):

        e         = E / L
        consensus = C / E
        u         = clamp(Q / E - consensus^2, min=0)
        pi        = E / (L + E*u)

    Algebraically identical to `ConflictNormalizedLayer`'s `A = |V|/L`,
    `S = V/L` normalize-then-fuse sequence: dividing `E`/`C`/`Q` by `L`
    before taking the `consensus`/`second_moment` ratios would cancel `L`
    out of both numerator and denominator, and `pi = e/(1+e*u)` with
    `e = E/L` is `E/(L + E*u)` after multiplying through by `L`. Computing
    the un-normalized raw sums and finishing here (rather than normalizing
    `V` into `A`/`S` up front) is what lets Sec 8's kernel share one
    `abs(V)` GEMM between `E` and `Q` instead of normalizing twice.
    """
    E_safe = E.clamp_min(eps)
    e = E / L.clamp_min(eps)
    consensus = C / E_safe
    second_moment = Q / E_safe
    u = (second_moment - consensus.square()).clamp_min(0.0)
    pi = E / (L + E * u).clamp_min(eps)
    return e, consensus, u, pi


def _init_v_gain_bias(
    V: nn.Parameter, gain_raw: nn.Parameter, bias: nn.Parameter, eps: float
) -> None:
    """Identical init convention to `ConflictNormalizedLayer.reset_parameters`:
    Kaiming-uniform `V`, then `gain_raw` inverse-softplus-set so
    `softplus(gain_raw) == ||V_row||_1` at init."""
    nn.init.kaiming_uniform_(V, a=math.sqrt(5))
    with torch.no_grad():
        row_l1 = V.abs().sum(dim=1).clamp_min(eps)
        work_dtype = torch.float64 if row_l1.device.type == "cpu" else row_l1.dtype
        gain_raw.copy_(_inverse_softplus(row_l1.to(work_dtype), eps).to(V.dtype))
        bias.zero_()


class BeliefFFN(nn.Module):
    """CellV0.3 hidden layer + ordinary linear output projection (task Sec 6).

    Input belief is reset fresh on every invocation (`mu_in = x`, `e_in = 1`,
    `u_in = 0` -- Sec 1: "no externally supplied confidence signal", Sec 6),
    so `pi_in` is identically 1 and all subsequent belief structure
    (`e_hidden`, `u_hidden`, `pi_hidden`) is generated internally from `V`'s
    fan-in of the current token vector alone, never from any evidence
    carried across layers or tokens.

    Efficient kernel (Sec 8): flatten `[B, T, D] -> [B*T, D]`, one combined
    `cat([pi, Z]) @ |V|.T` GEMM for `(E, Q)` and one `X @ V.T` GEMM for `C`.
    No `[batch, seq, out, in]` edge tensor is ever materialized.
    """

    def __init__(self, d_model: int, d_hidden: int, eps: float = _DEFAULT_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.eps = eps

        self.V = nn.Parameter(torch.empty(d_hidden, d_model))
        self.gain_raw = nn.Parameter(torch.empty(d_hidden))
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        self.out_proj = nn.Linear(d_hidden, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_v_gain_bias(self.V, self.gain_raw, self.bias, self.eps)

    def _hidden_belief(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x` (already the neutral-precision `mu_in`) -> `(consensus, e, u,
        pi, gamma)`, each shaped like `x` but with the last dim `d_hidden`.
        `pi_in` is exactly 1 everywhere (Sec 6), so `mu = x`, `Z = x^2`."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.d_model)  # [N, d_model], N = batch * seq

        abs_V = self.V.abs()  # [d_hidden, d_model]
        L = abs_V.sum(dim=1)  # [d_hidden] -- row L1 norm; broadcasts against [N, d_hidden] below

        # pi_in == 1 everywhere (Sec 6), so P == ones_like(flat), X == flat, Z == flat^2.
        pi_flat = torch.ones_like(flat)
        Z = flat.square()
        n = flat.shape[0]
        EQ_input = torch.cat([pi_flat, Z], dim=0)  # [2N, d_model]
        EQ = EQ_input @ abs_V.T  # [2N, d_hidden] -- one GEMM shared by E and Q
        E, Q = EQ[:n], EQ[n:]  # each [N, d_hidden]
        C = flat @ self.V.T  # [N, d_hidden]

        e, consensus, u, pi = _raw_moment_finish(L, E, C, Q, self.eps)
        gamma = F.softplus(self.gain_raw)

        out_shape = (*orig_shape[:-1], self.d_hidden)
        return (
            consensus.reshape(out_shape),
            e.reshape(out_shape),
            u.reshape(out_shape),
            pi.reshape(out_shape),
            gamma,
        )

    def forward(
        self,
        x: torch.Tensor,
        neutralize_precision: bool = False,
        neutralize_conflict: bool = False,
        **_unused,
    ) -> torch.Tensor:
        mu_hidden, _, _ = self._activation(x, neutralize_precision, neutralize_conflict)
        return self.out_proj(mu_hidden)

    def _activation(
        self, x: torch.Tensor, neutralize_precision: bool, neutralize_conflict: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        consensus, e, u, pi, gamma = self._hidden_belief(x)

        if neutralize_conflict:
            # Sec 12: force u_hidden = 0 when computing hidden precision, i.e.
            # pi := effective_precision(e, 0) == e.
            precision_for_scale = e
        else:
            precision_for_scale = pi

        if neutralize_precision:
            # Sec 12: replace sqrt(pi_hidden) with 1 in the activation only.
            scale = torch.ones_like(precision_for_scale)
        else:
            scale = precision_for_scale.clamp_min(0.0).sqrt()

        mu_hidden = torch.tanh(gamma * consensus * scale + self.bias)
        return mu_hidden, pi, u

    def forward_with_diagnostics(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Normal (non-neutralized) forward, plus the Sec 11 belief-state
        diagnostics for this layer's hidden units on this batch: `pi` mean /
        std / coefficient-of-variation, `u` mean / std."""
        mu_hidden, pi, u = self._activation(x, False, False)
        pi_mean = pi.mean()
        pi_std = pi.std(unbiased=False)
        diagnostics = {
            "pi_mean": pi_mean.item(),
            "pi_std": pi_std.item(),
            "pi_cv": (pi_std / pi_mean.clamp_min(self.eps)).item(),
            "u_mean": u.mean().item(),
            "u_std": u.std(unbiased=False).item(),
        }
        return self.out_proj(mu_hidden), diagnostics

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_hidden={self.d_hidden}"


class FixedConfidenceFFN(nn.Module):
    """Mechanism-control model C (task Sec 7): identical `V` / `gain_raw` /
    `bias` / `out_proj` shapes as `BeliefFFN`, but no belief mechanism at
    all -- no `e`, `u`, `pi`, no second moment.

        L         = sum(abs(V))
        consensus = (mu @ V.T) / L
        mu_hidden = tanh(gamma * consensus + bias)

    Answers whether any CellV0.3 effect is caused by belief computation or
    merely by the normalized-weight / tanh parameterization it shares with
    `BeliefFFN`.
    """

    def __init__(self, d_model: int, d_hidden: int, eps: float = _DEFAULT_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.eps = eps

        self.V = nn.Parameter(torch.empty(d_hidden, d_model))
        self.gain_raw = nn.Parameter(torch.empty(d_hidden))
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        self.out_proj = nn.Linear(d_hidden, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_v_gain_bias(self.V, self.gain_raw, self.bias, self.eps)

    def forward(self, x: torch.Tensor, **_unused) -> torch.Tensor:
        L = self.V.abs().sum(dim=1).clamp_min(self.eps)  # [d_hidden]
        consensus = F.linear(x, self.V) / L  # [..., d_hidden]
        gamma = F.softplus(self.gain_raw)
        mu_hidden = torch.tanh(gamma * consensus + self.bias)
        return self.out_proj(mu_hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_hidden={self.d_hidden}"
