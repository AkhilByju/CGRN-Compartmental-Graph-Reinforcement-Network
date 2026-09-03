"""Local field fusion for the Self-Organizing Refinement Field
(docs/architecture_v1.md §12), v2 -- corrected after the first version's
diagnostic (§12's "still not solid" finding: `z` reconstructed as a
kernel-average ratio blew up, and the `n_eff` second-moment term was
secretly `O(n * R^2)`, not `O(n * R)`). Three role corrections, not just
numerical patches:

- **`mu`/`e`/`u` (belief content)** stay a precision-weighted kernel
  average -- that *is* what `fusion.py`'s Method E/"CellV0.1" formula is
  for, and it's what converged cleanly in the diagnostic.
- **`r` (routing position)** now genuinely *self-organizes*: this module
  computes the density-weighted local mean `r_bar` (the numerator/
  denominator of a classic mean-shift step -- for a Gaussian kernel,
  moving toward `r_bar` is gradient ascent on the local log-density);
  `field_dynamics.py` applies the actual mean-shift update
  `r <- r + step_size * (r_bar - r)`. `r_bar` needs the *same* linear
  reduction machinery as `mu`'s content aggregate (no new complexity
  class), and is the *only* thing something 'summary of my neighborhood'
  is computed for now.
- **`z` (semantic identity) is no longer reconstructed here at all.** It
  was never actually the right shape for a kernel-weighted average --
  it's an identity a cell carries and updates, not a density to sit at
  the mean of. `field_dynamics.py` updates it with the same learned,
  gated mechanism CellV1 already uses elsewhere
  (`shared_functions.py::WriteGateFunction`/`SemanticUpdateFunction`),
  fed `r`/`r_bar` as its "what's around me" signal instead of a z-specific
  kernel average.
- **`n_eff` (effective source count) via a *second*, separately-fit
  random-feature map, not a `(R, R)` second-moment matrix.** `K(x,y)^2 =
  exp(-||x-y||^2 / sigma^2)` is *itself* a Gaussian kernel, just at
  bandwidth `sigma/sqrt(2)` instead of `sigma` -- so `sum_j(weight_j^2 *
  K_ij^2)` is a second *linear* reduction (`O(n * R)`), using a second
  `RandomFourierFeatures(sigma=field_sigma/sqrt(2))` instance applied to
  the same (already bandwidth-rescaled) routing coordinates, not a
  quadratic form. This is what keeps the whole field genuinely `O(n * R)`
  end to end -- the first version's `phi_ww` matrix was `O(n * R^2)`
  hiding inside an `O(n * R)` docstring claim.

Two implementations of the same math, as before: `local_field_fusion`
(scalable, random-feature-based) and `exact_local_field_fusion` (the
`O(n^2)` reference -- forms `K_ij` densely, reuses
`fusion.py::precision_fusion` directly, and computes `r_bar` via
`torch.matmul` on the same dense weights). `tests/test_field_exact_consistency.py`
checks the two against each other.
"""

from __future__ import annotations

import torch

from src.models.architecture_v1.fusion import precision_fusion


def local_field_fusion(
    mu: torch.Tensor,
    e: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    weight: torch.Tensor,
    psi: torch.Tensor,
    psi_sq: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mu`/`e`/`u`/`weight`: `(batch, n_cells)`. `r`: `(batch, n_cells,
    routing_dim)`. `psi`: `(batch, n_cells, R)` -- `RandomFourierFeatures`
    (Gaussian kernel, the field's own bandwidth) applied to the (already
    bandwidth-rescaled) routing coordinates. `psi_sq`: `(batch, n_cells,
    R)` -- a *separate* `RandomFourierFeatures` instance at
    `sigma/sqrt(2)`, applied to the *same* coordinates, approximating
    `K^2` for the `n_eff` denominator. `psi`/`psi_sq` can have different
    `R` than each other; both are `O(n * R)` regardless.

    Returns `(mu_out, evidence_out, uncertainty_out, r_bar)` -- `r_bar`
    is the mean-shift target (`field_dynamics.py` applies the actual
    step), not a final value itself.
    """
    precision = weight * e / (u**2 + eps)  # (batch, n)

    # --- linear reductions over senders (O(n * R) or O(n * R * routing_dim)) ---
    phi_w = torch.einsum("bnr,bn->br", psi, weight)
    phi_p = torch.einsum("bnr,bn->br", psi, precision)
    phi_pmu = torch.einsum("bnr,bn->br", psi, precision * mu)
    phi_pmu2 = torch.einsum("bnr,bn->br", psi, precision * mu**2)
    phi_mr = torch.einsum("bnr,bnd->brd", psi, weight.unsqueeze(-1) * r)
    phi_w2 = torch.einsum("bnr,bn->br", psi_sq, weight**2)  # K^2 sum -- linear, not an (R,R) matrix

    # --- self-anchored per-receiver evaluation ---
    # `psi_i . psi_i` / `psi_sq_i . psi_sq_i` is the RF's own (noisy) estimate
    # of K(i,i) -- which is *exactly* 1 for any Gaussian kernel, not something
    # that needs estimating. Every raw reduction below therefore gets its own
    # RF-estimated self-term subtracted out and replaced with the exact value,
    # before the (still-approximate, and -- only for the mass-like terms --
    # nonnegative-clamped) off-cell remainder is added back. This guarantees
    # a_sum/p_sum/a_sq_sum are each >= their exact self term (weight_i,
    # precision_i, weight_i^2 respectively, all strictly > 0), so none of
    # them can ever land at/near zero and blow up a division -- not an
    # adaptive-epsilon patch, a removal of the one term the approximation
    # never needed to guess at.
    self_kernel = (psi**2).sum(dim=-1)  # (batch, n) -- RF's own noisy K(i,i) estimate
    self_kernel_sq = (psi_sq**2).sum(dim=-1)

    def _self_anchor_mass(raw: torch.Tensor, self_value: torch.Tensor, exact_self: torch.Tensor) -> torch.Tensor:
        off_cell = torch.relu(raw - self_value * self_kernel)
        return exact_self + off_cell

    def _self_anchor_content(raw: torch.Tensor, self_value: torch.Tensor, exact_self: torch.Tensor) -> torch.Tensor:
        # Signed content (mu, r): self-anchor the same way, but no nonneg
        # clamp -- only "mass" (weight/precision-like) terms are guaranteed
        # nonnegative in the exact math.
        off_cell = raw - self_value * self_kernel
        return exact_self + off_cell

    raw_a_sum = torch.einsum("bnr,br->bn", psi, phi_w)
    a_sum = _self_anchor_mass(raw_a_sum, weight, exact_self=weight)

    raw_p_sum = torch.einsum("bnr,br->bn", psi, phi_p)
    p_sum = _self_anchor_mass(raw_p_sum, precision, exact_self=precision)

    raw_a_sq_sum = torch.einsum("bnr,br->bn", psi_sq, phi_w2)
    a_sq_sum = torch.relu(raw_a_sq_sum - (weight**2) * self_kernel_sq) + weight**2

    raw_s_mu = torch.einsum("bnr,br->bn", psi, phi_pmu)
    s_mu = _self_anchor_content(raw_s_mu, precision * mu, exact_self=precision * mu)

    raw_s_mu2 = torch.einsum("bnr,br->bn", psi, phi_pmu2)
    s_mu2 = _self_anchor_content(raw_s_mu2, precision * mu**2, exact_self=precision * mu**2)

    raw_s_mr = torch.einsum("bnr,brd->bnd", psi, phi_mr)
    self_mr = weight.unsqueeze(-1) * r
    s_mr = self_mr + (raw_s_mr - (weight * self_kernel).unsqueeze(-1) * r)

    c = s_mu / (p_sum + eps)
    mu_out = torch.tanh(c + bias)

    n_eff = (a_sum**2) / (a_sq_sum + eps)
    evidence_out = a_sum / (n_eff + eps)
    effective_precision = p_sum / (n_eff + eps)
    base_uncertainty_sq = 1.0 / (effective_precision + eps)
    disagreement = (s_mu2 / (p_sum + eps) - c**2).clamp(min=0.0)
    uncertainty_out = torch.sqrt(base_uncertainty_sq + disagreement)

    # Mean-shift displacement (r_bar - r), computed directly rather than
    # recovering an absolute r_bar and subtracting -- the exact self terms
    # in s_mr/a_sum cancel algebraically in this numerator (a cell
    # contributes nothing to its own displacement), so this also avoids ever
    # forming the two large, close-in-value numbers a naive
    # "r_bar = s_mr/a_sum; displacement = r_bar - r" would subtract.
    displacement_numerator = s_mr - r * a_sum.unsqueeze(-1)
    r_bar = r + displacement_numerator / (a_sum.unsqueeze(-1) + eps)

    return mu_out, evidence_out, uncertainty_out, r_bar


def exact_local_field_fusion(
    mu: torch.Tensor,
    e: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`O(n^2)` reference for `local_field_fusion`. `x`: `(batch, n_cells,
    dim)`, the bandwidth-rescaled routing coordinates (same tensor
    `local_field_fusion`'s `psi`/`psi_sq` are built from) -- forms
    `K_ij = exp(-||x_i - x_j||^2 / 2)` densely and reuses
    `fusion.py::precision_fusion` with `a_ij = weight_j * K_ij`, `m_ij =
    mu_j` (the field's "message" is the raw content, no separate F_msg --
    matching classic kernel-weighted averaging)."""
    from src.models.architecture_v1.random_features import exact_gaussian_kernel

    k = exact_gaussian_kernel(x, x)  # (batch, n, n), exact -- K^2 is just k**2, no separate approximation needed
    a = weight.unsqueeze(-2) * k  # (batch, n_receiver, n_sender)

    m = mu.unsqueeze(-2).expand_as(a)
    e_j = e.unsqueeze(-2)
    u_j = u.unsqueeze(-2)

    mu_out, evidence_out, uncertainty_out, alpha = precision_fusion(m, a, e_j, u_j, bias, eps)
    r_bar = torch.einsum("bij,bjd->bid", a, r) / (a.sum(dim=-1, keepdim=True) + eps)
    return mu_out, evidence_out, uncertainty_out, r_bar
