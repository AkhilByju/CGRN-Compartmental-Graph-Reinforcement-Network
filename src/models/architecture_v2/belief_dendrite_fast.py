"""Optimized compute backends for Architecture V2's `BeliefDendriteLayer`
(frozen reference: `belief_dendrite.py`). This module changes **how** the
frozen equations are computed, never **what** they compute: it introduces no
new architecture, no new parameters, no changed connectivity/topology, and
no changed math. Every class here must reproduce the frozen
`BeliefDendriteLayer`'s forward outputs, `(mu, e, u, pi)` states, and
gradients within tight float32 tolerance -- verified in
`tests/test_architecture_v2_belief_dendrite_fast.py`, not assumed. The
frozen file is untouched and remains the correctness reference
(`docs/architecture_v2.md`; this module's own justification:
`experiments/belief_dendrite/backend_benchmark.py`'s measured results).

The algebraic identity this whole module rests on
---------------------------------------------------
The frozen layer normalizes its signed weights into `A = |v|/L`,
`S = v/L` (`L = sum_k |v_k|`) before fusing. Every one of its three
per-branch reductions (`e_branch`, `content_num`, `second_branch`) is linear
in `A` or `S`, so the shared factor `1/L` can be pulled out of all three
reductions and applied once at the end instead of inside each one. Define
four **raw** (un-normalized) sums per branch/soma row, over its `K`
(or `B`) sources:

    L = sum_k |v_k|                    (row_l1 -- identical to the frozen layer's)
    E = sum_k |v_k| * pi_k
    C = sum_k  v_k  * pi_k * mu_k
    Q = sum_k |v_k| * pi_k * mu_k^2

Then, reproducing the frozen layer's exact sequence of operations
(`e_branch = (A*pi).sum().clamp_min(eps)`, `consensus = content_num /
e_branch`, `second = (A*pi*mu^2).sum() / e_branch`, `u =
clamp(second - consensus^2, 0)`, `pi_out = effective_precision(e, u)`):

    e         = clamp_min(E / L, eps)
    consensus = (C / L) / e
    second    = (Q / L) / e
    u         = clamp_min(second - consensus^2, 0)
    pi_out    = effective_precision(e, u)          # e / (1 + e*u), same call
    mu_out    = tanh(gamma * consensus * sqrt(pi_out) + bias)

This is the same arithmetic the frozen layer performs, just factored so the
per-batch reductions run on the *raw* sums instead of on tensors that have
already been multiplied by a broadcast `A`/`S` -- and, in `SparseFast`, so
the two reductions that both need `|v| * pi` (`E` and `Q`) share that
product instead of recomputing it (the frozen layer computes `A*pi` twice:
once for `e_branch`, again inside `second_branch`'s expression, since eager
PyTorch does no common-subexpression elimination across separate Python
statements).

Two backends
------------
- `BeliefDendriteLayerSparseFast`: still gathers each branch's `K` sources
  via the *same* `DendriticConnectivity` (identical topology, identical
  parameters) as the frozen layer, but computes `L/E/C/Q` directly with the
  shared-subexpression reuse above -- fewer large-tensor passes, no `A`/`S`
  broadcast materialization.
- `BeliefDendriteLayerDenseFast`: scatters the *same* sparse `[M, K]`
  trainable weights into a temporary zero-filled `[M, input_dim]` buffer
  every forward call (no new trainable parameters -- the scatter is
  differentiable, so gradients flow back to the original sparse `V_branch`
  unchanged) and computes the branch-level `E`/`Q`/`C` with two GEMMs
  instead of a gather + elementwise reduction. This trades ~`1/density`
  more nominal FLOPs (density is ~6-12% for this benchmark's shapes) for
  two large, regular matrix multiplies instead of many irregular
  gather/reduce kernels -- whether that trade wins on MPS is an empirical
  question this module does not presume the answer to; see
  `experiments/belief_dendrite/backend_benchmark.py`.

Soma-level fusion never gathers in the frozen layer either (a soma's `B`
branches are already its own layer's branch outputs, in registered order),
so both backends share one soma-level implementation -- only the
branch-level computation differs between them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.precision_gain import effective_precision
from src.models.architecture_v2.belief_dendrite import (
    BranchSomaDiagnostics,
    DendriticConnectivity,
    _init_gain_bias,
)

_DEFAULT_EPS = 1e-8

__all__ = [
    "BeliefDendriteLayerDenseFast",
    "BeliefDendriteLayerSparseFast",
    "BeliefDendriteNetworkDenseFast",
    "BeliefDendriteNetworkSparseFast",
]


def _finish_from_raw_moments(
    L: torch.Tensor, E: torch.Tensor, C: torch.Tensor, Q: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(L, E, C, Q) -> (e, consensus, u, pi_out)`, replicating the frozen
    layer's exact operation sequence (see module docstring)."""
    e = (E / L).clamp_min(eps)
    consensus = (C / L) / e
    second = (Q / L) / e
    u = (second - consensus.square()).clamp_min(0.0)
    pi_out = effective_precision(e, u)
    return e, consensus, u, pi_out


def _soma_fusion(
    v_cable: torch.Tensor, pi_branch_hb: torch.Tensor, mu_branch_hb: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared soma-level fusion (never gathers in either backend): `v_cable`
    `[H, B]`, `pi_branch_hb`/`mu_branch_hb` `[batch, H, B]` -> `(e_soma,
    consensus, u_soma, pi_soma, A_cable)`."""
    abs_v = v_cable.abs()
    L = abs_v.sum(dim=-1).clamp_min(eps)  # [H]
    abs_v_pi = abs_v * pi_branch_hb  # [batch, H, B]
    v_pi = v_cable * pi_branch_hb  # [batch, H, B]
    E = abs_v_pi.sum(dim=-1)  # [batch, H]
    C = (v_pi * mu_branch_hb).sum(dim=-1)  # [batch, H]
    Q = (abs_v_pi * mu_branch_hb.square()).sum(dim=-1)  # [batch, H]
    e, consensus, u, pi_out = _finish_from_raw_moments(L, E, C, Q, eps)
    A_cable = abs_v / L.unsqueeze(-1)  # [H, B] -- needed by BranchSomaDiagnostics
    return e, consensus, u, pi_out, A_cable


class BeliefDendriteLayerSparseFast(nn.Module):
    """Drop-in replacement for `BeliefDendriteLayer` (identical parameters,
    connectivity, and init): raw-moment fusion computed from the same
    gathered `[batch, M, K]` tensors, with the `|v|*pi` product shared
    between the `E` and `Q` reductions instead of recomputed."""

    def __init__(self, connectivity: DendriticConnectivity, eps: float = _DEFAULT_EPS) -> None:
        super().__init__()
        self.connectivity = connectivity
        self.H, self.B, self.K, self.M = (
            connectivity.H,
            connectivity.B,
            connectivity.K,
            connectivity.M,
        )
        self.eps = eps

        self.V_branch = nn.Parameter(torch.empty(self.M, self.K))
        self.gain_raw_branch = nn.Parameter(torch.empty(self.M))
        self.bias_branch = nn.Parameter(torch.zeros(self.M))
        self.V_cable = nn.Parameter(torch.empty(self.H, self.B))
        self.gain_raw_soma = nn.Parameter(torch.empty(self.H))
        self.bias_soma = nn.Parameter(torch.zeros(self.H))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_gain_bias(self.V_branch, self.gain_raw_branch, self.bias_branch, self.eps)
        _init_gain_bias(self.V_cable, self.gain_raw_soma, self.bias_soma, self.eps)

    def forward(self, incoming: BeliefCell) -> BeliefCell:
        return self.forward_verbose(incoming)[0]

    def forward_verbose(self, incoming: BeliefCell) -> tuple[BeliefCell, BranchSomaDiagnostics]:
        pi = effective_precision(incoming.evidence, incoming.uncertainty)
        mu_g, pi_g = self.connectivity.gather(incoming.mu, pi)  # [batch, M, K]

        abs_v = self.V_branch.abs()  # [M, K]
        L = abs_v.sum(dim=-1).clamp_min(self.eps)  # [M]
        abs_v_pi = abs_v * pi_g  # [batch, M, K] -- shared by E and Q
        v_pi = self.V_branch * pi_g  # [batch, M, K] -- shared by C
        E = abs_v_pi.sum(dim=-1)  # [batch, M]
        C = (v_pi * mu_g).sum(dim=-1)  # [batch, M]
        Q = (abs_v_pi * mu_g.square()).sum(dim=-1)  # [batch, M]

        e_branch, consensus_branch, u_branch, pi_branch = _finish_from_raw_moments(
            L, E, C, Q, self.eps
        )
        gamma_branch = F.softplus(self.gain_raw_branch)
        mu_branch = torch.tanh(
            gamma_branch * consensus_branch * pi_branch.sqrt() + self.bias_branch
        )

        mu_branch_hb = mu_branch.reshape(-1, self.H, self.B)
        pi_branch_hb = pi_branch.reshape(-1, self.H, self.B)
        e_branch_hb = e_branch.reshape(-1, self.H, self.B)
        u_branch_hb = u_branch.reshape(-1, self.H, self.B)
        consensus_hb = consensus_branch.reshape(-1, self.H, self.B)

        e_soma, consensus_soma, u_soma, pi_soma, A_cable = _soma_fusion(
            self.V_cable, pi_branch_hb, mu_branch_hb, self.eps
        )
        gamma_soma = F.softplus(self.gain_raw_soma)
        mu_soma = torch.tanh(gamma_soma * consensus_soma * pi_soma.sqrt() + self.bias_soma)

        out = BeliefCell(mu=mu_soma, evidence=e_soma, uncertainty=u_soma)
        diag = BranchSomaDiagnostics(
            mu_branch=mu_branch_hb,
            e_branch=e_branch_hb,
            u_branch=u_branch_hb,
            pi_branch=pi_branch_hb,
            consensus_branch=consensus_hb,
            A_cable=A_cable,
            e_soma=e_soma,
        )
        return out, diag

    def extra_repr(self) -> str:
        return (
            f"H={self.H}, B={self.B}, K={self.K}, mode={self.connectivity.mode!r}, "
            "backend=sparse_fast"
        )


class BeliefDendriteLayerDenseFast(nn.Module):
    """Same parameters/connectivity/init as `BeliefDendriteLayer`, but the
    branch level scatters `V_branch` into a temporary dense `[M, input_dim]`
    buffer each forward call and computes `E`/`Q` (one GEMM, concatenated)
    and `C` (a second GEMM) instead of gathering. No new trainable
    parameters: the scatter is differentiable, so `V_branch.grad` is
    populated exactly as it is for the sparse backends."""

    def __init__(self, connectivity: DendriticConnectivity, eps: float = _DEFAULT_EPS) -> None:
        super().__init__()
        self.connectivity = connectivity
        self.H, self.B, self.K, self.M = (
            connectivity.H,
            connectivity.B,
            connectivity.K,
            connectivity.M,
        )
        self.input_dim = connectivity.input_dim
        self.eps = eps

        self.V_branch = nn.Parameter(torch.empty(self.M, self.K))
        self.gain_raw_branch = nn.Parameter(torch.empty(self.M))
        self.bias_branch = nn.Parameter(torch.zeros(self.M))
        self.V_cable = nn.Parameter(torch.empty(self.H, self.B))
        self.gain_raw_soma = nn.Parameter(torch.empty(self.H))
        self.bias_soma = nn.Parameter(torch.zeros(self.H))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_gain_bias(self.V_branch, self.gain_raw_branch, self.bias_branch, self.eps)
        _init_gain_bias(self.V_cable, self.gain_raw_soma, self.bias_soma, self.eps)

    def forward(self, incoming: BeliefCell) -> BeliefCell:
        return self.forward_verbose(incoming)[0]

    def _dense_branch_weights(self) -> torch.Tensor:
        zeros = torch.zeros(
            self.M, self.input_dim, dtype=self.V_branch.dtype, device=self.V_branch.device
        )
        return zeros.scatter(-1, self.connectivity.source_idx, self.V_branch)

    def forward_verbose(self, incoming: BeliefCell) -> tuple[BeliefCell, BranchSomaDiagnostics]:
        if incoming.mu.dim() != 2 or incoming.mu.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected incoming belief over {self.input_dim} cells, "
                f"got shape {tuple(incoming.mu.shape)}."
            )
        pi = effective_precision(incoming.evidence, incoming.uncertainty)  # [batch, input_dim]
        mu = incoming.mu  # [batch, input_dim]

        V_dense = self._dense_branch_weights()  # [M, input_dim]
        abs_V_dense = V_dense.abs()
        L = self.V_branch.abs().sum(dim=-1).clamp_min(self.eps)  # [M] -- cheap, from sparse weights

        batch = mu.shape[0]
        Z = pi * mu.square()  # [batch, input_dim]
        EQ_input = torch.cat([pi, Z], dim=0)  # [2*batch, input_dim]
        EQ = EQ_input @ abs_V_dense.T  # [2*batch, M]
        E, Q = EQ[:batch], EQ[batch:]  # each [batch, M]
        X = pi * mu  # [batch, input_dim]
        C = X @ V_dense.T  # [batch, M]

        e_branch, consensus_branch, u_branch, pi_branch = _finish_from_raw_moments(
            L, E, C, Q, self.eps
        )
        gamma_branch = F.softplus(self.gain_raw_branch)
        mu_branch = torch.tanh(
            gamma_branch * consensus_branch * pi_branch.sqrt() + self.bias_branch
        )

        mu_branch_hb = mu_branch.reshape(-1, self.H, self.B)
        pi_branch_hb = pi_branch.reshape(-1, self.H, self.B)
        e_branch_hb = e_branch.reshape(-1, self.H, self.B)
        u_branch_hb = u_branch.reshape(-1, self.H, self.B)
        consensus_hb = consensus_branch.reshape(-1, self.H, self.B)

        e_soma, consensus_soma, u_soma, pi_soma, A_cable = _soma_fusion(
            self.V_cable, pi_branch_hb, mu_branch_hb, self.eps
        )
        gamma_soma = F.softplus(self.gain_raw_soma)
        mu_soma = torch.tanh(gamma_soma * consensus_soma * pi_soma.sqrt() + self.bias_soma)

        out = BeliefCell(mu=mu_soma, evidence=e_soma, uncertainty=u_soma)
        diag = BranchSomaDiagnostics(
            mu_branch=mu_branch_hb,
            e_branch=e_branch_hb,
            u_branch=u_branch_hb,
            pi_branch=pi_branch_hb,
            consensus_branch=consensus_hb,
            A_cable=A_cable,
            e_soma=e_soma,
        )
        return out, diag

    def extra_repr(self) -> str:
        return (
            f"H={self.H}, B={self.B}, K={self.K}, mode={self.connectivity.mode!r}, "
            "backend=dense_fast"
        )


# ---------------------------------------------------------------------------
# Network-level wrappers -- same structure as the frozen `BeliefDendriteNetwork`
# (input belief -> layer1 -> layer2 -> linear readout), built from one of the
# two fast layer backends above instead of `BeliefDendriteLayer`.
# ---------------------------------------------------------------------------


class _BeliefDendriteNetworkFastBase(nn.Module):
    """Shared network-level plumbing; subclasses only pick the layer class."""

    _layer_cls: type[nn.Module]

    def __init__(
        self,
        connectivity1: DendriticConnectivity,
        connectivity2: DendriticConnectivity,
        out_features: int,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if connectivity2.input_dim != connectivity1.H:
            raise ValueError(
                f"connectivity2.input_dim ({connectivity2.input_dim}) must equal "
                f"connectivity1's H ({connectivity1.H})."
            )
        self.eps = eps
        self.layer1 = self._layer_cls(connectivity1, eps=eps)
        self.layer2 = self._layer_cls(connectivity2, eps=eps)
        self.readout = nn.Linear(connectivity2.H, out_features)

    def input_belief(self, x: torch.Tensor, reliability: torch.Tensor | None = None) -> BeliefCell:
        e = reliability if reliability is not None else torch.ones_like(x)
        return BeliefCell(mu=x, evidence=e, uncertainty=torch.zeros_like(x))

    def layer_states(self, incoming: BeliefCell) -> tuple[BeliefCell, BeliefCell]:
        b1 = self.layer1(incoming)
        return b1, self.layer2(b1)

    def layer_states_verbose(
        self, incoming: BeliefCell
    ) -> tuple[tuple[BeliefCell, BranchSomaDiagnostics], tuple[BeliefCell, BranchSomaDiagnostics]]:
        b1, d1 = self.layer1.forward_verbose(incoming)
        b2, d2 = self.layer2.forward_verbose(b1)
        return (b1, d1), (b2, d2)

    def forward(self, x: torch.Tensor, reliability: torch.Tensor | None = None) -> torch.Tensor:
        _, b2 = self.layer_states(self.input_belief(x, reliability))
        return self.readout(b2.mu)


class BeliefDendriteNetworkSparseFast(_BeliefDendriteNetworkFastBase):
    """`BeliefDendriteNetwork` built from `BeliefDendriteLayerSparseFast`."""

    _layer_cls = BeliefDendriteLayerSparseFast


class BeliefDendriteNetworkDenseFast(_BeliefDendriteNetworkFastBase):
    """`BeliefDendriteNetwork` built from `BeliefDendriteLayerDenseFast`."""

    _layer_cls = BeliefDendriteLayerDenseFast
