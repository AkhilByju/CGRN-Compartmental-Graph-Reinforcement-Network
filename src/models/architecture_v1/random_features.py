"""Random feature maps for approximating a kernel without ever forming an
`(n, n)` pairwise matrix (docs/architecture_v1.md's "Self-Organizing
Refinement Field" / CellV1.3).

**`RandomFourierFeatures` is what `field_fusion.py`'s local density/
attraction field actually uses.** The field is explicitly defined by a
Gaussian/RBF attraction kernel `K(x, y) = exp(-||x - y||^2 / (2 sigma^2))`,
and the textbook random-feature map for *that* kernel is Rahimi & Recht's
(2007) construction: real-valued, cosine-based, no positivity claim, no
numerical-stabilization trick needed (`cos` is bounded, unlike the `exp(...)`
terms `PositiveRandomFeatures` below needs to guard against overflowing).

`PositiveRandomFeatures` (Performer/FAVOR+-style) approximates a
*different* kernel -- `exp(x^T y)`, the exponential dot-product kernel
behind softmax attention -- and is kept here **unused for now**: an
earlier attempt applied it to the Gaussian-kernel use case by adjusting a
coefficient, which is *not* mathematically equivalent (`exp(x^Ty)` and
`exp(-||x-y||^2/2)` are different kernels; matching one pair's value at
carefully chosen inputs doesn't make the feature maps interchangeable) and
produced per-pair estimator variance that didn't concentrate even at
`R=2048` (`docs/architecture_v1.md` §12's diagnostic). Do not reach for it
again unless the underlying kernel actually becomes an exponential
dot-product kernel (e.g. for global attention, where that *is* the right
kernel) -- and even then, prefer the *orthogonal* random-feature variant
(FAVOR+ proper) over the plain i.i.d. version implemented below, for the
same variance reason.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _iid_directions(dim: int, num_features: int, generator: torch.Generator | None) -> torch.Tensor:
    """`(dim, num_features)`, each column an independent `N(0, I_dim)`
    draw -- standard (non-orthogonal) random features."""
    return torch.randn(dim, num_features, generator=generator)


def _orthogonal_directions(dim: int, num_features: int, generator: torch.Generator | None) -> torch.Tensor:
    """`(dim, num_features)`, marginally `N(0, I_dim)` per column (so
    still an unbiased kernel estimator by the same Bochner argument as
    `_iid_directions`), but columns within each `dim`-sized block are
    exactly orthogonal: QR-decompose an i.i.d. Gaussian `(dim, dim)`
    block to get an orthonormal (Haar-random, via the sign-correction on
    `R`'s diagonal) set of directions, then rescale each to an
    independently-drawn chi-distributed norm (sum of `dim` squared
    standard normals, square-rooted) so each column's *marginal* length
    distribution matches a plain Gaussian vector's -- only the *joint*
    distribution (the orthogonality) differs, which is what reduces
    variance without introducing bias."""
    num_blocks = (num_features + dim - 1) // dim
    blocks = []
    for _ in range(num_blocks):
        g = torch.randn(dim, dim, generator=generator)
        q, r = torch.linalg.qr(g)
        q = q * torch.diagonal(r).sign().unsqueeze(0)  # fix QR's sign ambiguity -> uniform (Haar) orthogonal matrix
        blocks.append(q)
    unit_directions = torch.cat(blocks, dim=1)[:, :num_features]  # (dim, num_features)
    chi_norms = torch.randn(dim, num_features, generator=generator).pow(2).sum(dim=0).sqrt()
    return unit_directions * chi_norms.unsqueeze(0)


class RandomFourierFeatures(nn.Module):
    """Rahimi & Recht (2007) random Fourier features for the Gaussian/RBF
    kernel `K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))`:

        phi(x)_r = sqrt(2/R) * cos(w_r^T x / sigma + b_r)

    `w_r ~ N(0, I_dim)` (fixed, not learned), `b_r ~ Uniform(0, 2*pi)`
    (fixed). `E[phi(x)^T phi(y)] = K(x, y)`, by Bochner's theorem (a
    stationary kernel is the Fourier transform of a probability measure --
    for the Gaussian kernel, that measure is itself Gaussian, which is
    exactly what `w_r` is drawn from) -- unbiased for any `R`, variance
    shrinking as `R` grows. Unlike `PositiveRandomFeatures`, entries can be
    negative (`cos` ranges over `[-1, 1]`); nothing here needs numerical
    stabilization, since `cos` never overflows.

    `orthogonal=True` (Yu et al., 2016, "Orthogonal Random Features")
    draws the `w_r` with the same marginal distribution (still `N(0,
    I_dim)` individually) but *not* independently: within each block of
    `dim` rows, they're constrained to be exactly orthogonal (via QR of an
    i.i.d. Gaussian block, each row then rescaled to a fresh chi-
    distributed norm so the marginal per-row distribution is unchanged --
    only the *joint* distribution differs). Still unbiased for the same
    reason (Bochner's theorem only needs the marginal to be right), but
    provably lower-variance -- needed here empirically:
    `docs/architecture_v1.md` §12's diagnostic found plain i.i.d. RFF's
    `mu` output converging correctly with `R`, but `z`/`uncertainty`
    (both recovered as a *ratio* of two noisy aggregated sums) blowing up
    by orders of magnitude whenever the denominator sum happened to land
    near zero from sampling noise -- exactly the failure mode orthogonal
    features are designed to suppress.
    """

    def __init__(
        self,
        dim: int,
        num_features: int,
        sigma: float = 1.0,
        orthogonal: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        directions = _orthogonal_directions if orthogonal else _iid_directions
        w = directions(dim, num_features, generator) / sigma
        self.register_buffer("w", w)
        self.register_buffer("b", torch.rand(num_features, generator=generator) * 2 * math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(..., n_cells, dim)`. Returns `phi(x)`: `(..., n_cells,
        num_features)`."""
        projection = x @ self.w + self.b  # (..., n_cells, R)
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


class PositiveRandomFeatures(nn.Module):
    """Performer/FAVOR+-style positive random features for the
    *exponential dot-product* kernel `exp(x^T y)` -- **not currently used**
    (see this module's docstring). Kept for a possible future global-
    attention use case, where `exp(q^Tk/sqrt(d))` genuinely is the right
    kernel to approximate.

        psi(x) = exp(x @ W - ||x||^2/2 - c) / sqrt(R)

    gives `E[psi(x)^T psi(y)] = exp(x^T y)`. `c`, a stabilizing constant
    shared across every cell and feature in a call (never computed
    per-row -- a per-row `c` would multiply each row by an uncancelled
    factor and silently corrupt every downstream sum), exists purely to
    keep `exp(...)` from overflowing.
    """

    def __init__(self, dim: int, num_features: int, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.num_features = num_features
        self.register_buffer("w", torch.randn(dim, num_features, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(..., n_cells, dim)`. Returns `psi(x)`: `(..., n_cells,
        num_features)`, all entries strictly positive."""
        projection = x @ self.w  # (..., n_cells, R)
        sq_norm = (x**2).sum(dim=-1, keepdim=True)  # (..., n_cells, 1)
        exponent = projection - 0.5 * sq_norm
        stabilizer = exponent.detach().amax(dim=(-2, -1), keepdim=True)
        return torch.exp(exponent - stabilizer) / (self.num_features**0.5)


def exact_gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Reference (not scalable): `K(x, y) = exp(-||x_i - y_j||^2 /
    (2*sigma^2))` for every pair, densely. `x`: `(..., n, dim)`, `y`:
    `(..., m, dim)`. Returns `(..., n, m)`. Used only by the exact
    `O(n^2)` field-fusion reference
    (`field_fusion.py::exact_kernel_weighted_fusion`) that validates
    `RandomFourierFeatures`' approximation at small `n`
    (docs/architecture_v1.md), not by the scalable model itself."""
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)  # (..., n, m, dim)
    return torch.exp(-0.5 * (diff**2).sum(dim=-1) / sigma**2)
