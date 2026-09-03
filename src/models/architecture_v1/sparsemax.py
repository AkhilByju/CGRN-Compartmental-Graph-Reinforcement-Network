"""Sparsemax (Martins & Astudillo, 2016) -- an implementation choice, not a
user-specified formula. `docs/architecture_v1.md` Part I proposes
"alpha-entmax" for turning association scores into an exactly-sparse
(some-entries-exactly-zero) distribution, citing "Adaptively Sparse
Transformers" (entmax-1.5). Sparsemax is the alpha=2 member of that same
family and has a simple closed form that needs no bisection/iteration and
no extra dependency (`entmax` is not in `pyproject.toml`) -- chosen here
purely as a dependency-free stand-in with the property that actually
matters for `docs/architecture_v1.md`: a differentiable projection that can
assign *exact* zero weight to low-scoring entries, unlike softmax. entmax
with alpha in (1, 2) would need iterative bisection; left as a possible
refinement, not required by anything specified so far.
"""

from __future__ import annotations

import torch

# Additive mask for excluded entries (e.g. self-loops). A large finite
# negative number rather than -inf: cumsum over an -inf entry would poison
# every later partial sum with -inf, and `scores - tau` could then hit
# `-inf - (-inf) = nan`. `-1e4` is far below any realistic association
# score here (scores are `-distance/tau` with distance >= 0, or a bounded
# dot-product-plus-log-need/offer term) so masked entries never enter the
# sparsemax support.
MASK_VALUE = -1e4


def sparsemax(scores: torch.Tensor) -> torch.Tensor:
    """Sparsemax over the last dimension. `scores`: `(..., k)`. Returns a
    tensor of the same shape, non-negative, summing to exactly 1 over the
    last dimension, with (generically) some entries exactly 0.

    Standard closed-form projection onto the probability simplex (Held,
    Wolfe & Crowder, 1974; popularized for attention by Martins &
    Astudillo, 2016): sort descending, find the largest support size `k*`
    for which the threshold-shifted sorted scores stay positive, then
    subtract that threshold and clamp at zero. Implemented with ordinary
    differentiable tensor ops (sort/cumsum/gather/clamp) rather than a
    hand-written backward pass -- autograd's gradient through
    `clamp(scores - tau, min=0)` already matches sparsemax's known
    Jacobian (identity minus a uniform-over-the-support averaging term) at
    every point where the support size is locally constant.
    """
    k = scores.size(-1)
    scores_sorted, _ = torch.sort(scores, dim=-1, descending=True)
    cumsum = scores_sorted.cumsum(dim=-1)
    ks = torch.arange(1, k + 1, device=scores.device, dtype=scores.dtype)
    support = 1.0 + ks * scores_sorted > cumsum
    k_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
    cumsum_support = cumsum.gather(-1, k_support - 1)
    tau = (cumsum_support - 1.0) / k_support.to(scores.dtype)
    return torch.clamp(scores - tau, min=0.0)


def sparse_association_with_null(scores: torch.Tensor, null_logit: torch.Tensor) -> torch.Tensor:
    """`scores`: `(..., n)`, already masked (e.g. diagonal set to
    `MASK_VALUE` to exclude self-association). Appends `null_logit` as an
    `(n+1)`-th competing option -- docs/architecture_v1.md's `s_i∅`,
    "I currently don't have a good match" -- runs sparsemax over all `n+1`
    options, then drops the null column. Because sparsemax's outputs sum to
    1 over *all* `n+1` options, a row can end up with total real-neighbor
    weight anywhere from 0 (everything routed to null) to 1 (null never
    entered the support) -- neighbor count is not fixed.

    `null_logit` is a learnable scalar (`nn.Parameter`, shape `()` or
    `(1,)`), shared across every cell and every pair -- one global
    "propensity to connect to nothing," not a per-cell or per-pair value.
    """
    null_column = null_logit.reshape(*([1] * (scores.dim() - 1)), 1).expand(*scores.shape[:-1], 1)
    augmented = torch.cat([scores, null_column], dim=-1)
    weights = sparsemax(augmented)
    return weights[..., :-1]
