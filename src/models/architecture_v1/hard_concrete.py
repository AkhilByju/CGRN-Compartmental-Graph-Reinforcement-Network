"""Hard-concrete stochastic gate (Louizos, Welling & Kingma, 2017,
"Learning Sparse Neural Networks through L0 Regularization"), for the
global send/need gates in `docs/architecture_v1.md`'s "Self-Organizing
Refinement Field": a differentiable relaxation of a `{0, 1}` gate that can
land on an *exact* 0 or 1, not just values arbitrarily close to them the
way a plain `sigmoid` can. Used so "how many cells broadcast globally" is
learned (via the gate's distribution) rather than fixed at a chosen `K`.
"""

from __future__ import annotations

import torch


def hard_concrete_gate(
    log_alpha: torch.Tensor,
    training: bool,
    beta: float = 0.5,
    gamma: float = -0.1,
    zeta: float = 1.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """`log_alpha`: `(...)`, the gate's location parameter (output of a
    learned function of the cell's state, e.g. `SendGateFunction`).
    Returns a gate value in `[0, 1]`, with non-negligible probability mass
    at exactly `0` (and, less usefully here, exactly `1`).

    Training: sample `u ~ Uniform(0, 1)`, form a `Concrete` (binary
    Gumbel-softmax) relaxation at temperature `beta`, stretch it into
    `(gamma, zeta)` (`gamma < 0 < 1 < zeta`, so the stretched value can
    fall below 0 or above 1), then hard-clip to `[0, 1]` -- differentiable
    everywhere except exactly at the clip boundaries (standard
    straight-through-friendly behavior; gradients flow normally through
    the un-clipped region).

    Eval: skip sampling, use the deterministic value at `u = 0.5`
    (median of the relaxation), still stretched and clipped -- the usual
    hard-concrete convention for a reproducible test-time gate.
    """
    if training:
        u = torch.rand_like(log_alpha).clamp(eps, 1 - eps)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    else:
        s = torch.sigmoid(log_alpha)
    s_stretched = s * (zeta - gamma) + gamma
    return s_stretched.clamp(0.0, 1.0)
