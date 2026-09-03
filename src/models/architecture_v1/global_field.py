"""Global communication for the Self-Organizing Refinement Field
(docs/architecture_v1.md §14, "CellV1.3.1"). Added *on top of* the frozen
local field (`field_fusion.py`, `field_dynamics.py`'s local-only path) --
nothing here changes local behavior when disabled.

The local field answers "who's in my current semantic neighborhood"
(similarity, via `field_fusion.py`'s kernel). The global field answers a
different question -- "which distant information would help my current
belief" (complementarity/relevance, via learned query/key matching, not
similarity). No explicit regions, no `K_global`, no `(n, n)` matrix: every
cell learns a continuous send gate (how useful is my information to
others) and need gate (how much outside information do I currently need),
and the actual retrieval is exact linear attention.

**Exact, not approximate.** `elu_feature_map` (Katharopoulos et al., 2020,
"Transformers are RNNs") is a deterministic positive feature map --
`K^G(i, j) := phi(q_i)^T phi(k_j)` *is* the compatibility kernel by
definition, not an estimate of some other kernel. Unlike
`random_features.py` (used for the local field), there is no random
projection, no variance, no "does this converge as R grows" question --
the reasoning that consumed most of the local field's debugging doesn't
apply here at all. `phi(x) = ELU(x) + 1 > 0` always (`ELU(x) > -1`
everywhere), which is what makes every reduction below provably
non-negative before any epsilon/clamp is even needed (see
`linear_global_belief_field`'s docstring).

Complexity: every tensor here is `(batch, n_cells, global_dim)` or
`(batch, global_dim)` -- `O(n_cells * global_dim)`, never `(n_cells,
n_cells)`. `global_dim` (`d_g`, suggested 8-16) is independent of
`n_cells`, so this doesn't add per-cell parameters as `n_cells` scales.
"""

from __future__ import annotations

import torch
from torch import nn

# See linear_global_belief_field's docstring: float32 catastrophic
# cancellation in the exact self-removal subtraction leaves a residual of
# ~1e-7 even for a mathematically-exact-zero true value, and `precision`
# (send*e/(u^2+eps)) can amplify that noise further when u is small; this
# floor must dominate the noise with a wide margin (not just be "small"),
# while staying far below any realistic aggregate mass once more than a
# couple of cells contribute real signal (~1000-10000x smaller at
# n_cells=128 with typical send/evidence magnitudes).
_DENOM_FLOOR = 1e-2


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """`phi(x) = ELU(x) + 1`. Strictly positive everywhere (`ELU(x) >
    -1`), unlike raw `x` -- this is what guarantees every dot product
    `phi(a)^T phi(b)` used below is strictly positive too, for any `a`,
    `b`."""
    return torch.nn.functional.elu(x) + 1.0


def _mlp(in_features: int, hidden_dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_features),
    )


class GlobalSendFunction(nn.Module):
    """`s_i = sigmoid(F_send(mu_i, e_i, u_i, mu_local_i, e_local_i,
    u_local_i))` -- how useful this cell's current information is to the
    rest of the population. Shared across all cells; trained end to end,
    same spirit as `field_functions.py::RoutingGateFunction`.

    `init_bias` (default `-2.0`, matching `shared_functions.py::
    WriteGateFunction`'s convention) starts every cell near-off
    (`sigmoid(-2.0) ~= 0.12`) rather than at the default-init `~0.5`: at
    `use_global=True`'s introduction, `send`/`need` both defaulted to
    `~0.5` and `field_dynamics.py`'s `source_gate_logit` weighted
    self/local/global equally from step 0 -- an unlearned, noisy global
    channel got a full vote in every cell's very first fused proposal.
    `docs/research_log.md`'s global-field 3-seed run diagnosed this: one
    seed improved over local-only Field-T2, the other two seeds regressed
    (one by ~0.15 R², well below every other model including CellV0.1),
    with ~6x the seed variance of any local-only model -- consistent with
    optimization noise from a loud, uninformative channel at the very
    start of training, not a stable, learned communication benefit."""

    def __init__(self, hidden_dim: int = 32, init_bias: float = -2.0) -> None:
        super().__init__()
        self.net = _mlp(in_features=6, hidden_dim=hidden_dim, out_features=1)
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)

    def forward(
        self,
        mu: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor,
        mu_local: torch.Tensor,
        e_local: torch.Tensor,
        u_local: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([mu, e, u, mu_local, e_local, u_local], dim=-1)
        return torch.sigmoid(self.net(features).squeeze(-1))


class GlobalNeedFunction(nn.Module):
    """`n_i = sigmoid(F_need(mu_i, e_i, u_i, mu_local_i, e_local_i,
    u_local_i))` -- how much non-local information this cell currently
    needs. Same input shape/`init_bias` convention as `GlobalSendFunction`
    (see its docstring), separate learned weights, different role."""

    def __init__(self, hidden_dim: int = 32, init_bias: float = -2.0) -> None:
        super().__init__()
        self.net = _mlp(in_features=6, hidden_dim=hidden_dim, out_features=1)
        with torch.no_grad():
            self.net[-1].bias.fill_(init_bias)

    def forward(
        self,
        mu: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor,
        mu_local: torch.Tensor,
        e_local: torch.Tensor,
        u_local: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([mu, e, u, mu_local, e_local, u_local], dim=-1)
        return torch.sigmoid(self.net(features).squeeze(-1))


class GlobalQueryKey(nn.Module):
    """`q_i = Wq z_i` (what information do I need), `k_i = Wk z_i` (what
    can I offer) -- separate learned linear projections (no bias,
    matching `routing.py::GlobalRouting`'s convention) of the semantic
    address into a `global_dim`-sized communication space. Two cells
    don't need similar `q`/`k` to communicate -- unlike the local field's
    similarity-based kernel, this is a complementarity match."""

    def __init__(self, association_dim: int, global_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(association_dim, global_dim, bias=False)
        self.key = nn.Linear(association_dim, global_dim, bias=False)

    def query_of(self, z: torch.Tensor) -> torch.Tensor:
        return self.query(z)

    def key_of(self, z: torch.Tensor) -> torch.Tensor:
        return self.key(z)


def linear_global_belief_field(
    mu: torch.Tensor,
    e: torch.Tensor,
    u: torch.Tensor,
    send: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The module's `LinearGlobalBeliefField` computation (a function
    here, not a class, matching this codebase's convention of plain
    functions for parameter-free math -- `fusion.py::precision_fusion`,
    `field_fusion.py::local_field_fusion`).

    `mu`/`e`/`u`/`send`: `(batch, n_cells)`. `q`/`k`: `(batch, n_cells,
    global_dim)`. Returns `(mu_G, evidence_G, uncertainty_G)`, each
    `(batch, n_cells)` -- the raw global proposal (no `tanh`/bias
    squashing here; the eventual self+local+global fuse re-applies
    `tanh` once, via the reused `fusion.py::precision_fusion`).

    Sender precision `pi_j = send_j * e_j / (u_j^2 + eps)` (reuses
    CellV0.1's exact semantics: more evidence -> more influence, more
    uncertainty -> less). One set of global reductions
    (`S_p/S_mu/S_mu2/S_s/S_e`, each `(batch, global_dim)`) is computed
    once for the whole population, then every receiver reads it through
    its own query -- `O(n_cells * global_dim)`, not `O(n_cells^2)`.

    **Exact self-removal, not approximate -- in exact arithmetic.** Every
    receiver's raw reduction includes its own (self) contribution;
    mathematically that subtracts out exactly (linear sums decompose
    exactly) so the global channel can't just retrieve the cell's own
    state back. `phi(x) = elu_feature_map(x) > 0` always, so `phi(q_i)^T
    phi(k_j) > 0` for *every* pair -- meaning every raw sum is provably
    `>=` its own self term, so the self-removed totals are non-negative
    *by construction*, not because of a defensive clamp.

    In float32, though, `raw` (a two-stage reduction: sum over cells,
    then a `d_g`-term dot product) and `self_compat * (...)` (a
    `d_g`-term dot product, then a scalar multiply) reach the same value
    via different rounding paths -- their difference doesn't cancel to
    *bit-exact* zero even when a receiver is (near-)isolated and the true
    value is zero. Measured residual ~1e-7 for `d_g=8`; the module-level
    `eps` (`1e-8`, this codebase's standard floor) is too small to
    dominate that, so `numerator / (near-zero-but-noisy denominator)` can
    land anywhere in `O(1)` instead of the intended "no information ->
    zero evidence" fallback (verified in
    `tests/test_global_field.py::test_self_contribution_excluded_for_isolated_cell`
    before this floor was added -- an n=1 cell's global evidence came
    back as ~its own evidence value, an artifact of both the numerator
    and denominator carrying the *same-shaped* rounding noise). See
    `_DENOM_FLOOR`'s comment for why `1e-2`, not something closer to
    `eps`, is what actually dominates that noise.

    Global evidence is a *ratio* (`(h . S_e) / (relevant sender mass)`),
    not a raw sum -- scale-stable by construction: more cells existing
    doesn't manufacture more evidence (the same lesson CellV0/CellV0.1
    already established for the belief-fusion formula itself).
    """
    phi_q = elu_feature_map(q)  # (batch, n, d_g) -- h_i
    phi_k = elu_feature_map(k)  # (batch, n, d_g) -- f_j

    precision = send * e / (u**2 + eps)  # pi_j, (batch, n)

    # --- one global reduction per quantity, O(n * d_g) ---
    s_p = torch.einsum("bnd,bn->bd", phi_k, precision)
    s_mu = torch.einsum("bnd,bn->bd", phi_k, precision * mu)
    s_mu2 = torch.einsum("bnd,bn->bd", phi_k, precision * mu**2)
    s_s = torch.einsum("bnd,bn->bd", phi_k, send)
    s_e = torch.einsum("bnd,bn->bd", phi_k, send * e)

    # --- per-receiver raw totals (still include self), O(n * d_g) ---
    p_raw = torch.einsum("bnd,bd->bn", phi_q, s_p)
    a_raw = torch.einsum("bnd,bd->bn", phi_q, s_s)
    mu_num_raw = torch.einsum("bnd,bd->bn", phi_q, s_mu)
    mu2_num_raw = torch.einsum("bnd,bd->bn", phi_q, s_mu2)
    e_num_raw = torch.einsum("bnd,bd->bn", phi_q, s_e)

    # --- exact self-removal ---
    self_compat = (phi_q * phi_k).sum(dim=-1)  # c_i = phi(q_i) . phi(k_i), (batch, n)
    p = (p_raw - self_compat * precision).clamp(min=0.0)
    a = (a_raw - self_compat * send).clamp(min=0.0)
    mu_num = mu_num_raw - self_compat * precision * mu
    mu2_num = mu2_num_raw - self_compat * precision * mu**2
    e_num = e_num_raw - self_compat * send * e

    denom_eps = max(eps, _DENOM_FLOOR)
    mu_g = mu_num / (p + denom_eps)
    evidence_g = e_num / (a + denom_eps)

    p_bar = p / (a + denom_eps)
    m2 = mu2_num / (p + denom_eps)
    disagreement = (m2 - mu_g**2).clamp(min=0.0)
    uncertainty_g = torch.sqrt(1.0 / (p_bar + eps) + disagreement)

    return mu_g, evidence_g, uncertainty_g
