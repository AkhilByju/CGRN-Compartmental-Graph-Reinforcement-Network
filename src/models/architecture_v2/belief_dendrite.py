"""Architecture V2 -- the Belief Dendritic Network (docs/architecture_v2.md).

One artificial neuron is no longer one CellV0.3. It is several sparse
**belief dendrites** feeding one **belief soma**, and both levels run the
*same* CellV0.3-style conservative belief fusion -- the dendrite fuses its
upstream sources into a branch belief, then the soma fuses its dendrites'
branch beliefs into the neuron's output belief. Recursion, not a new
primitive.

State: the same `BeliefCell (mu, e, u)` as CellV0.1/0.2/0.3
(`src/models/architecture_v0/cell.py`), with usable precision
`pi = e / (1 + e u)` (`src/models/architecture_v0/precision_gain.py`'s
`effective_precision`, reused unmodified).

Per branch `b` of soma `i`, source set `S_ib` (exactly `K` indices, fixed and
non-trainable), signed synaptic weights `V_branch[b, :]`:

    A_bj = |V_bj| / (sum_k |V_bk}| + eps)      unsigned, rows sum to ~1
    S_bj =  V_bj  / (sum_k |V_bk}| + eps)      signed

    e_branch_b      = sum_j A_bj pi_j
    consensus_b     = (sum_j S_bj pi_j mu_j) / e_branch_b
    second_b        = (sum_j A_bj pi_j mu_j^2) / e_branch_b
    u_branch_b      = max(0, second_b - consensus_b^2)
    pi_branch_b     = e_branch_b / (1 + e_branch_b u_branch_b)
    mu_branch_b     = tanh(gamma_branch_b * consensus_b * sqrt(pi_branch_b) + bias_branch_b)

Then the soma runs the identical fusion over its `B` dendrites (cable weights
`V_cable[i, :]` in place of `V_branch`, branch beliefs in place of source
beliefs) to produce `(mu_i, e_i, u_i)`.

No router, no softmax, no top-k, no external context, no gate parameter
anywhere: a branch's influence at the soma is exactly its normalized cable
weight times its own `pi_branch` (`A_cable * pi_branch / e_soma`) -- reliable
branches are automatically louder, unreliable ones automatically quieter,
purely as a consequence of the belief-fusion arithmetic.

Efficiency: every reduction is `gather -> elementwise -> sum(dim=-1)` over
the `K` (or `B`) structural sources of a branch (or soma); the largest
tensor ever materialized is the gathered `[batch, H*B, K]` activation --
never a `[batch, H, input_dim]` or `[batch, H, B, input_dim]` dense edge
tensor. `DendriticConnectivity` stores only the `[H*B, K]` index buffer that
makes this possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.precision_gain import _inverse_softplus, effective_precision

_DEFAULT_EPS = 1e-8

__all__ = [
    "DendriticConnectivity",
    "BeliefDendriteLayer",
    "BeliefDendriteNetwork",
    "ScalarDendriteLayer",
    "ScalarDendriteNetwork",
    "ScalarDendriteReliabilityGated",
    "make_connectivity_pair",
]


# ---------------------------------------------------------------------------
# A. Dendritic connectivity -- fixed, non-trainable structural source indices.
# ---------------------------------------------------------------------------


def _balanced_random_indices(input_dim: int, num_branches: int, k: int, seed: int) -> torch.Tensor:
    """`[num_branches, k]` int64 source indices into `range(input_dim)`.

    Guarantees, always (not just typically):
    * exactly `k` **unique** sources within every branch (`k <= input_dim`
      required);
    * deterministic for a given `seed`.

    Approximately balanced global coverage: sources are drawn from a
    shuffled "deck" that is a permutation of `range(input_dim)`, consumed
    sequentially `k` at a time per branch. A branch that exhausts the deck
    mid-draw is topped up from a fresh permutation of *only the sources it
    hasn't already used this branch* (so per-branch uniqueness never
    breaks), and the next branch starts a clean full-range deck. Every deck
    cycle uses each source exactly once, so over `num_branches * k` draws
    each source is used within one draw of `floor(num_branches*k/input_dim)`
    of every other source -- balanced without a global reweighting pass.
    """
    if not (1 <= k <= input_dim):
        raise ValueError(f"k must be in [1, input_dim] (input_dim={input_dim}), got k={k}.")

    rng = torch.Generator().manual_seed(int(seed))

    def _perm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=rng)

    out = torch.empty(num_branches, k, dtype=torch.int64)
    deck = _perm(input_dim)
    pos = 0
    for b in range(num_branches):
        chosen: list[torch.Tensor] = []
        used: set[int] = set()
        filled = 0
        while filled < k:
            remaining = deck.shape[0] - pos
            take = min(k - filled, remaining)
            part = deck[pos : pos + take]
            chosen.append(part)
            used.update(part.tolist())
            pos += take
            filled += take
            if pos >= deck.shape[0] and filled < k:
                pool = torch.tensor(sorted(set(range(input_dim)) - used), dtype=torch.int64)
                deck = pool[_perm(pool.shape[0])]
                pos = 0
        out[b] = torch.cat(chosen)
        if pos >= deck.shape[0]:
            deck = _perm(input_dim)
            pos = 0
    return out


def _local_2d_positions(
    height: int, width: int, patch: int, num_branches: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(row0, col0)`, each `[num_branches]`, the deterministic top-left
    corner of each branch's `patch x patch` receptive field. Positions are
    laid out on a near-uniform grid over the valid corner range (so branch
    centers are spatially diverse and overlapping, not one big random
    cluster), then subsampled or seed-jittered to reach exactly
    `num_branches` -- deterministic, not a theoretical claim about optimal
    coverage."""
    max_r0 = height - patch
    max_c0 = width - patch
    if max_r0 < 0 or max_c0 < 0:
        raise ValueError(f"patch={patch} does not fit in a {height}x{width} image.")

    rng = torch.Generator().manual_seed(int(seed))
    if max_r0 == 0 and max_c0 == 0:
        z = torch.zeros(num_branches, dtype=torch.int64)
        return z, z

    aspect = (max_c0 + 1) / (max_r0 + 1) if max_r0 > 0 else float(max_c0 + 1)
    n_rows = max(1, min(max_r0 + 1, round(math.sqrt(num_branches / max(aspect, 1e-6)))))
    n_cols = max(1, math.ceil(num_branches / n_rows))
    r_vals = torch.unique(torch.linspace(0, max_r0, n_rows).round().to(torch.int64))
    c_vals = torch.unique(torch.linspace(0, max_c0, n_cols).round().to(torch.int64))
    grid = torch.stack(torch.meshgrid(r_vals, c_vals, indexing="ij"), dim=-1).reshape(-1, 2)

    if grid.shape[0] >= num_branches:
        pick = torch.randperm(grid.shape[0], generator=rng)[:num_branches]
        pick, _ = torch.sort(pick)
        positions = grid[pick]
    else:
        reps = math.ceil(num_branches / grid.shape[0])
        positions = grid.repeat(reps, 1)[:num_branches].clone()
        jitter_r = torch.randint(-1, 2, (num_branches,), generator=rng)
        jitter_c = torch.randint(-1, 2, (num_branches,), generator=rng)
        positions[:, 0] = (positions[:, 0] + jitter_r).clamp(0, max_r0)
        positions[:, 1] = (positions[:, 1] + jitter_c).clamp(0, max_c0)
    return positions[:, 0].contiguous(), positions[:, 1].contiguous()


class DendriticConnectivity(nn.Module):
    """Fixed, non-trainable source indices for `H` somas x `B` branches each
    (`M = H * B` total dendrites, `K` upstream sources per dendrite).

    Stores exactly one buffer, `source_idx: LongTensor[M, K]` -- flattened
    indices into the incoming layer's `input_dim` features. No dense
    `[H, input_dim]` (or `[M, input_dim]`) mask is ever created; `gather`
    below is the only way this connectivity touches a belief.
    """

    def __init__(
        self,
        source_idx: torch.Tensor,
        *,
        input_dim: int,
        num_somas: int,
        branches_per_soma: int,
        sources_per_branch: int,
        mode: str,
        seed: int | None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        expected_shape = (num_somas * branches_per_soma, sources_per_branch)
        if tuple(source_idx.shape) != expected_shape:
            raise ValueError(
                f"source_idx must have shape {expected_shape}, got {tuple(source_idx.shape)}."
            )
        if source_idx.dtype != torch.int64:
            source_idx = source_idx.to(torch.int64)
        if bool((source_idx < 0).any()) or bool((source_idx >= input_dim).any()):
            raise ValueError(f"source_idx must index into [0, {input_dim}).")

        self.register_buffer("source_idx", source_idx)
        self.input_dim = input_dim
        self.H = num_somas
        self.B = branches_per_soma
        self.K = sources_per_branch
        self.mode = mode
        self.seed = seed
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def M(self) -> int:
        return self.H * self.B

    @classmethod
    def balanced_random(
        cls,
        input_dim: int,
        num_somas: int,
        branches_per_soma: int,
        sources_per_branch: int,
        seed: int,
    ) -> DendriticConnectivity:
        idx = _balanced_random_indices(
            input_dim, num_somas * branches_per_soma, sources_per_branch, seed
        )
        return cls(
            idx,
            input_dim=input_dim,
            num_somas=num_somas,
            branches_per_soma=branches_per_soma,
            sources_per_branch=sources_per_branch,
            mode="balanced_random",
            seed=seed,
        )

    @classmethod
    def local_2d(
        cls,
        *,
        channels: int,
        height: int,
        width: int,
        num_somas: int,
        branches_per_soma: int,
        patch: int,
        seed: int,
    ) -> DendriticConnectivity:
        input_dim = channels * height * width
        num_branches = num_somas * branches_per_soma
        row0, col0 = _local_2d_positions(height, width, patch, num_branches, seed)

        # One contiguous patch x patch receptive field across every channel,
        # flattened row-major as `c * H * W + r * W + col` (matches how
        # experiments/paper_a's image loaders flatten (1, 28, 28) images).
        dr, dc = torch.meshgrid(torch.arange(patch), torch.arange(patch), indexing="ij")
        offsets = (dr * width + dc).reshape(-1)  # [patch*patch], within one channel
        channel_stride = height * width
        per_channel = (
            row0.unsqueeze(1) * width + col0.unsqueeze(1) + offsets.unsqueeze(0)
        )  # [M, patch*patch]
        idx = torch.cat(
            [per_channel + c * channel_stride for c in range(channels)], dim=1
        )  # [M, channels*patch*patch]

        return cls(
            idx,
            input_dim=input_dim,
            num_somas=num_somas,
            branches_per_soma=branches_per_soma,
            sources_per_branch=channels * patch * patch,
            mode="local_2d",
            seed=seed,
            meta={
                "channels": channels,
                "height": height,
                "width": width,
                "patch": patch,
                "row0": row0,
                "col0": col0,
            },
        )

    def gather(self, mu: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mu, pi): [batch, input_dim] -> (mu_g, pi_g): [batch, M, K]`.

        A single `index_select` per tensor (`[batch, M*K]`, a view-reshape to
        `[batch, M, K]`) -- never a `[batch, H, input_dim]` intermediate."""
        if mu.dim() != 2 or mu.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected mu of shape (batch, {self.input_dim}), got {tuple(mu.shape)}."
            )
        flat_idx = self.source_idx.reshape(-1)
        mu_g = mu.index_select(-1, flat_idx).reshape(mu.shape[0], self.M, self.K)
        pi_g = pi.index_select(-1, flat_idx).reshape(pi.shape[0], self.M, self.K)
        return mu_g, pi_g

    def gather_scalar(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.dim() != 2 or mu.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected mu of shape (batch, {self.input_dim}), got {tuple(mu.shape)}."
            )
        flat_idx = self.source_idx.reshape(-1)
        return mu.index_select(-1, flat_idx).reshape(mu.shape[0], self.M, self.K)

    def corruption_overlap(self, corrupted_mask: torch.Tensor) -> torch.Tensor:
        """Per-branch fraction of a branch's `K` sources that fall inside a
        boolean corruption mask over `input_dim` features -- the Sec O
        localization diagnostic's independent variable. `corrupted_mask`:
        `[input_dim]` or `[batch, input_dim]`; returns `[M]` or `[batch, M]`."""
        flat_idx = self.source_idx.reshape(-1)
        mask = corrupted_mask.to(torch.float32)
        gathered = mask.index_select(-1, flat_idx).reshape(*mask.shape[:-1], self.M, self.K)
        return gathered.mean(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, H={self.H}, B={self.B}, K={self.K}, "
            f"M={self.M}, mode={self.mode!r}"
        )


def make_connectivity_pair(
    *,
    in_features: int,
    hidden1: int,
    hidden2: int,
    branches_per_soma: int,
    seed: int,
    image_shape: tuple[int, int, int] | None = None,
    patch: int = 7,
    sources_per_branch_1: int | None = None,
    sources_per_branch_2: int | None = None,
) -> tuple[DendriticConnectivity, DendriticConnectivity]:
    """The default two-stage connectivity (Sec F): `local_2d` first layer for
    image-shaped input (else `balanced_random`), `balanced_random` second
    layer (hidden cells have no imposed 2D geometry). Exploratory
    compute-budget defaults, not theoretical constants: `K = min(32,
    input_dim)` when not given explicitly."""
    if image_shape is not None:
        channels, height, width = image_shape
        if channels * height * width != in_features:
            raise ValueError(
                f"image_shape {image_shape} has {channels * height * width} features, "
                f"expected in_features={in_features}."
            )
        conn1 = DendriticConnectivity.local_2d(
            channels=channels,
            height=height,
            width=width,
            num_somas=hidden1,
            branches_per_soma=branches_per_soma,
            patch=patch,
            seed=seed,
        )
    else:
        k1 = sources_per_branch_1 or min(32, in_features)
        conn1 = DendriticConnectivity.balanced_random(
            in_features, hidden1, branches_per_soma, k1, seed
        )

    k2 = sources_per_branch_2 or min(32, hidden1)
    conn2 = DendriticConnectivity.balanced_random(hidden1, hidden2, branches_per_soma, k2, seed + 1)
    return conn1, conn2


# ---------------------------------------------------------------------------
# B/C/D/E. BeliefDendriteLayer -- dendritic belief computation + somatic fusion.
# ---------------------------------------------------------------------------


def _row_normalize(v: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`v: [..., width]` -> `(A, S, row_l1)`, the unsigned/signed row-L1
    normalization shared by every belief-fusion level in this project."""
    abs_v = v.abs()
    row_l1 = abs_v.sum(dim=-1).clamp_min(eps)
    a = abs_v / row_l1.unsqueeze(-1)
    s = v / row_l1.unsqueeze(-1)
    return a, s, row_l1


def _init_gain_bias(
    v: torch.Tensor, gain_raw: nn.Parameter, bias: nn.Parameter, eps: float
) -> None:
    """Kaiming-uniform `v`, then inverse-softplus `gain_raw` so
    `softplus(gain_raw) == ||v_row||_1` at init (the CellV0.2/0.3 amplitude
    convention, reused unmodified) and `bias = 0`."""
    nn.init.kaiming_uniform_(v, a=math.sqrt(5))
    with torch.no_grad():
        row_l1 = v.abs().sum(dim=-1).clamp_min(eps)
        work_dtype = torch.float64 if row_l1.device.type == "cpu" else row_l1.dtype
        gain_raw.copy_(_inverse_softplus(row_l1.to(work_dtype), eps).to(v.dtype))
        bias.zero_()


@dataclass
class BranchSomaDiagnostics:
    """Non-differentiable-use diagnostics from one `BeliefDendriteLayer`
    forward pass, reshaped to `[batch, H, B]` where per-branch (Sec O)."""

    mu_branch: torch.Tensor  # [batch, H, B]
    e_branch: torch.Tensor  # [batch, H, B]
    u_branch: torch.Tensor  # [batch, H, B]
    pi_branch: torch.Tensor  # [batch, H, B]
    consensus_branch: torch.Tensor  # [batch, H, B]
    A_cable: torch.Tensor  # [H, B]
    e_soma: torch.Tensor  # [batch, H]

    def branch_contribution(self) -> torch.Tensor:
        """`r_b = A_cable_b * pi_branch_b / e_soma` (Sec P "routing by
        reliability"), `[batch, H, B]`; sums to ~1 over the branch axis."""
        return self.A_cable.unsqueeze(0) * self.pi_branch / self.e_soma.unsqueeze(-1)

    def effective_branches(self) -> torch.Tensor:
        """`1 / sum_b r_b^2`, `[batch, H]` -- diagnostics-only "how many
        branches is the soma effectively listening to" (Sec O)."""
        r = self.branch_contribution()
        return 1.0 / r.square().sum(dim=-1).clamp_min(1e-12)


class BeliefDendriteLayer(nn.Module):
    """One layer of Architecture V2: `H` somas x `B` sparse belief dendrites
    each, a `BeliefCell` over `input_dim` in -> a `BeliefCell` over `H` out.

    Trainable parameters, all flattened over `M = H*B` branches or `H` somas
    (never `[H, B, K]` or `[H, input_dim]`):

    * `V_branch`        `[M, K]`  -- per-branch signed synaptic weights
    * `gain_raw_branch` `[M]`
    * `bias_branch`     `[M]`
    * `V_cable`         `[H, B]`  -- per-soma signed cable weights
    * `gain_raw_soma`   `[H]`
    * `bias_soma`       `[H]`
    """

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
        if incoming.mu.dim() != 2 or incoming.mu.shape[-1] != self.connectivity.input_dim:
            raise ValueError(
                f"expected incoming belief over {self.connectivity.input_dim} cells, "
                f"got shape {tuple(incoming.mu.shape)}."
            )

        pi = effective_precision(incoming.evidence, incoming.uncertainty)  # [batch, input_dim]
        mu_g, pi_g = self.connectivity.gather(incoming.mu, pi)  # [batch, M, K]

        A, S, _ = _row_normalize(self.V_branch, self.eps)  # [M, K] each

        # --- dendritic (branch) belief fusion --------------------------------
        e_branch = (A * pi_g).sum(dim=-1).clamp_min(self.eps)  # [batch, M]
        content_num = (S * pi_g * mu_g).sum(dim=-1)  # [batch, M]
        consensus_branch = content_num / e_branch
        second_branch = (A * pi_g * mu_g.square()).sum(dim=-1) / e_branch
        u_branch = (second_branch - consensus_branch.square()).clamp_min(0.0)
        pi_branch = effective_precision(e_branch, u_branch)  # [batch, M]
        gamma_branch = F.softplus(self.gain_raw_branch)  # [M]
        mu_branch = torch.tanh(
            gamma_branch * consensus_branch * pi_branch.sqrt() + self.bias_branch
        )  # [batch, M]

        mu_branch_hb = mu_branch.reshape(-1, self.H, self.B)
        pi_branch_hb = pi_branch.reshape(-1, self.H, self.B)
        e_branch_hb = e_branch.reshape(-1, self.H, self.B)
        u_branch_hb = u_branch.reshape(-1, self.H, self.B)
        consensus_hb = consensus_branch.reshape(-1, self.H, self.B)

        # --- somatic belief fusion over dendrites -----------------------------
        A_cable, S_cable, _ = _row_normalize(self.V_cable, self.eps)  # [H, B] each

        e_soma = (A_cable * pi_branch_hb).sum(dim=-1).clamp_min(self.eps)  # [batch, H]
        content_num_soma = (S_cable * pi_branch_hb * mu_branch_hb).sum(dim=-1)  # [batch, H]
        consensus_soma = content_num_soma / e_soma
        second_soma = (A_cable * pi_branch_hb * mu_branch_hb.square()).sum(dim=-1) / e_soma
        u_soma = (second_soma - consensus_soma.square()).clamp_min(0.0)
        pi_soma = effective_precision(e_soma, u_soma)  # [batch, H]
        gamma_soma = F.softplus(self.gain_raw_soma)  # [H]
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
        return f"H={self.H}, B={self.B}, K={self.K}, mode={self.connectivity.mode!r}"


# ---------------------------------------------------------------------------
# F. BeliefDendriteNetwork -- default two-stage network.
# ---------------------------------------------------------------------------


class BeliefDendriteNetwork(nn.Module):
    """`input belief -> BeliefDendriteLayer -> BeliefDendriteLayer -> linear
    readout from mu` (Sec F). The readout consumes `mu` directly, not
    rescaled by precision -- every cell has already folded its own output
    precision into its activation (same convention as `BeliefNetworkV03`)."""

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
        self.layer1 = BeliefDendriteLayer(connectivity1, eps=eps)
        self.layer2 = BeliefDendriteLayer(connectivity2, eps=eps)
        self.readout = nn.Linear(connectivity2.H, out_features)

    @classmethod
    def build(
        cls,
        *,
        in_features: int,
        out_features: int,
        hidden1: int,
        hidden2: int,
        branches_per_soma: int = 4,
        seed: int = 0,
        image_shape: tuple[int, int, int] | None = None,
        patch: int = 7,
        sources_per_branch_1: int | None = None,
        sources_per_branch_2: int | None = None,
        eps: float = _DEFAULT_EPS,
    ) -> BeliefDendriteNetwork:
        conn1, conn2 = make_connectivity_pair(
            in_features=in_features,
            hidden1=hidden1,
            hidden2=hidden2,
            branches_per_soma=branches_per_soma,
            seed=seed,
            image_shape=image_shape,
            patch=patch,
            sources_per_branch_1=sources_per_branch_1,
            sources_per_branch_2=sources_per_branch_2,
        )
        return cls(conn1, conn2, out_features, eps=eps)

    def input_belief(self, x: torch.Tensor, reliability: torch.Tensor | None = None) -> BeliefCell:
        """Sec B: `mu = x`; `e = reliability` if given, else `e = 1`; `u = 0`."""
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


# ---------------------------------------------------------------------------
# G. Scalar dendritic controls -- identical topology, no (e, u) propagation.
# ---------------------------------------------------------------------------


class ScalarDendriteLayer(nn.Module):
    """Same branch/soma structure, synaptic parameter shapes, gains/biases
    and cable weights as `BeliefDendriteLayer` (and reuses the identical
    `_init_gain_bias` convention), but no belief state: a branch is
    `tanh(gamma_branch * sum_k S mu + bias_branch)`, a soma is
    `tanh(gamma_soma * sum_b S_cable branch + bias_soma)`. Isolates the
    effect of dendritic *structure* from belief propagation (Sec G)."""

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

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        mu_g = self.connectivity.gather_scalar(mu)  # [batch, M, K]
        _, S, _ = _row_normalize(self.V_branch, self.eps)
        gamma_branch = F.softplus(self.gain_raw_branch)
        branch = torch.tanh(gamma_branch * (S * mu_g).sum(dim=-1) + self.bias_branch)  # [batch, M]

        branch_hb = branch.reshape(-1, self.H, self.B)
        _, S_cable, _ = _row_normalize(self.V_cable, self.eps)
        gamma_soma = F.softplus(self.gain_raw_soma)
        soma = torch.tanh(gamma_soma * (S_cable * branch_hb).sum(dim=-1) + self.bias_soma)
        return soma  # [batch, H]

    def extra_repr(self) -> str:
        return f"H={self.H}, B={self.B}, K={self.K}, mode={self.connectivity.mode!r}"


class ScalarDendriteNetwork(nn.Module):
    """Same two-stage topology as `BeliefDendriteNetwork`, built from
    ordinary `ScalarDendriteLayer`s and reading only `x` (no reliability)."""

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
        self.layer1 = ScalarDendriteLayer(connectivity1, eps=eps)
        self.layer2 = ScalarDendriteLayer(connectivity2, eps=eps)
        self.readout = nn.Linear(connectivity2.H, out_features)

    @classmethod
    def build(
        cls,
        *,
        in_features: int,
        out_features: int,
        hidden1: int,
        hidden2: int,
        branches_per_soma: int = 4,
        seed: int = 0,
        image_shape: tuple[int, int, int] | None = None,
        patch: int = 7,
        sources_per_branch_1: int | None = None,
        sources_per_branch_2: int | None = None,
        eps: float = _DEFAULT_EPS,
    ) -> ScalarDendriteNetwork:
        conn1, conn2 = make_connectivity_pair(
            in_features=in_features,
            hidden1=hidden1,
            hidden2=hidden2,
            branches_per_soma=branches_per_soma,
            seed=seed,
            image_shape=image_shape,
            patch=patch,
            sources_per_branch_1=sources_per_branch_1,
            sources_per_branch_2=sources_per_branch_2,
        )
        return cls(conn1, conn2, out_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.readout(h2)


class ScalarDendriteReliabilityGated(nn.Module):
    """Sec G's critical control: reliability is applied **once**, at the
    input (`mu_input = reliability * x`), then fed through an ordinary
    `ScalarDendriteNetwork` -- the same dendritic topology as
    `BeliefDendriteNetwork`, but reliability never propagates past the first
    gate. Answers "is hierarchical belief propagation earning its keep, or
    would a single input-side gate inside the same sparse topology do just
    as well?" (Sec P Q2)."""

    def __init__(
        self,
        connectivity1: DendriticConnectivity,
        connectivity2: DendriticConnectivity,
        out_features: int,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        self.net = ScalarDendriteNetwork(connectivity1, connectivity2, out_features, eps=eps)

    @classmethod
    def build(
        cls,
        *,
        in_features: int,
        out_features: int,
        hidden1: int,
        hidden2: int,
        branches_per_soma: int = 4,
        seed: int = 0,
        image_shape: tuple[int, int, int] | None = None,
        patch: int = 7,
        sources_per_branch_1: int | None = None,
        sources_per_branch_2: int | None = None,
        eps: float = _DEFAULT_EPS,
    ) -> ScalarDendriteReliabilityGated:
        conn1, conn2 = make_connectivity_pair(
            in_features=in_features,
            hidden1=hidden1,
            hidden2=hidden2,
            branches_per_soma=branches_per_soma,
            seed=seed,
            image_shape=image_shape,
            patch=patch,
            sources_per_branch_1=sources_per_branch_1,
            sources_per_branch_2=sources_per_branch_2,
        )
        return cls(conn1, conn2, out_features, eps=eps)

    def forward(self, x: torch.Tensor, reliability: torch.Tensor | None = None) -> torch.Tensor:
        x_gated = x if reliability is None else reliability * x
        return self.net(x_gated)
