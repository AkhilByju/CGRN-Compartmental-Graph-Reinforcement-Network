""""Dynamic Groups" -- a synthetic dataset designed specifically around
CellV1's hypothesis (docs/architecture_v1.md), for the first real CellV1
experiment (`experiments/v1_001_dynamic_groups/`). Unlike R2/C2/U2
(`regression.py`, `classification.py`, `uncertainty.py`), which weren't
designed with grouping in mind, this task is unsolvable by a model that
can't discover input-dependent structure: the correct grouping of objects
changes on every example and is never given as a label.

Each example contains `n_objects` (default 24) objects, each
`(k1, k2, v)`: `(k1, k2)` are 2D semantic coordinates, `v` is a scalar
value. For every example, a random number of hidden groups `K in
{k_min, ..., k_max}` (default `{2, 3, 4, 5}`) is sampled, `K` well-separated
centers are placed in semantic space, and every object's `(k1, k2)` is
drawn near one center -- but which group an object belongs to is never
observed by the model, only `(k1, k2, v)` per object.

The target requires both local and global computation, by construction:

    h_k = tanh(sum_{j in group k} v_j)                  -- per-group summary (LOCAL)
    y   = sum_k h_k + beta * h_left * h_right            -- cross-group term (GLOBAL)

where `h_left`/`h_right` are the summaries of the groups whose centers
have the smallest/largest `k1` (x-coordinate). Solving this requires first
discovering which objects form a group and summarizing within it (local),
then relating two specific, input-dependent groups to each other (global)
-- not reducible to per-object independence, and not solvable by summing
independent local compartments alone.

`x` is returned flattened, `(n_samples, n_objects * 3)`, in a fixed
per-object `[k1, k2, v]` order -- the same flatten order
`src.models.architecture_v1.object_encoder.ObjectSeededEncoder` expects, so
the identical `x` tensor drops into both a flat-input baseline (MLP,
`BeliefNetwork`) and CellV1's object-seeded encoder unchanged. `group_id`
(`(n_samples, n_objects)`, values in `[0, K)`, `K` varying per example) is
returned separately -- ground truth for evaluating whether CellV1's
*discovered* local graph agrees with the true hidden groups
(`docs/architecture_v1.md` §5's "agreement... local connectivity and the
true hidden groups"); it is never fed to any model.
"""

from __future__ import annotations

import torch

N_OBJECTS: int = 24
SEMANTIC_DIM: int = 2
FEATURES_PER_OBJECT: int = 3  # (k1, k2, v)


def _sample_separated_centers(
    k: int, min_dist: float, bound: float, generator: torch.Generator, max_tries: int = 20
) -> torch.Tensor:
    """`(k, 2)` centers in `[-bound, bound]^2`, resampled up to `max_tries`
    times if any pair is closer than `min_dist` (falls back to the last
    draw if separation still isn't met -- rare, and only makes that one
    example's groups noisier, not invalid)."""
    centers = torch.empty(k, SEMANTIC_DIM).uniform_(-bound, bound, generator=generator)
    if k == 1:
        return centers
    for _ in range(max_tries):
        dist = torch.cdist(centers, centers)
        dist.fill_diagonal_(float("inf"))
        if dist.min().item() >= min_dist:
            return centers
        centers = torch.empty(k, SEMANTIC_DIM).uniform_(-bound, bound, generator=generator)
    return centers


def dynamic_groups(
    n_samples: int,
    seed: int,
    n_objects: int = N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    noise_std: float = 0.15,
    value_scale: float = 1.0,
    beta: float = 1.0,
    center_bound: float = 1.0,
    min_center_dist: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns `(x, y, group_id)`: `x` is `(n_samples, n_objects * 3)`,
    `y` is `(n_samples, 1)`, `group_id` is `(n_samples, n_objects)` (long).

    Generation is a per-example Python loop (K, group sizes, and the
    per-group summary are inherently per-example combinatorics, not
    vectorizable the way a fixed-shape mean function is) -- fine at the
    sample counts this task needs (CellV1 training is the bottleneck, not
    data generation).
    """
    g = torch.Generator().manual_seed(seed)

    x = torch.empty(n_samples, n_objects, FEATURES_PER_OBJECT)
    y = torch.empty(n_samples, 1)
    group_id = torch.empty(n_samples, n_objects, dtype=torch.long)

    for i in range(n_samples):
        k = int(torch.randint(k_min, k_max + 1, (1,), generator=g).item())
        centers = _sample_separated_centers(k, min_center_dist, center_bound, g)

        groups = torch.empty(n_objects, dtype=torch.long)
        groups[:k] = torch.arange(k)  # guarantees every group has >= 1 member
        if n_objects > k:
            groups[k:] = torch.randint(0, k, (n_objects - k,), generator=g)
        groups = groups[torch.randperm(n_objects, generator=g)]

        semantic = centers[groups] + torch.randn(n_objects, SEMANTIC_DIM, generator=g) * noise_std
        values = torch.empty(n_objects, 1).uniform_(-value_scale, value_scale, generator=g)

        h = torch.zeros(k)
        for group in range(k):
            mask = groups == group
            h[group] = torch.tanh(values[mask].sum())

        left = int(torch.argmin(centers[:, 0]).item())
        right = int(torch.argmax(centers[:, 0]).item())
        target = h.sum() + beta * (h[left] * h[right])

        x[i] = torch.cat([semantic, values], dim=-1)
        y[i, 0] = target
        group_id[i] = groups

    return x.view(n_samples, n_objects * FEATURES_PER_OBJECT), y, group_id


def make_dynamic_groups_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Like `src.data.synthetic.uncertainty.make_uncertainty_splits`, but
    for `dynamic_groups`'s `(x, y, group_id)` triple."""
    n_total = n_train + n_val + n_test
    x, y, group_id = dynamic_groups(n_total, seed=seed, **kwargs)
    return {
        "train": (x[:n_train], y[:n_train], group_id[:n_train]),
        "val": (
            x[n_train : n_train + n_val],
            y[n_train : n_train + n_val],
            group_id[n_train : n_train + n_val],
        ),
        "test": (
            x[n_train + n_val :],
            y[n_train + n_val :],
            group_id[n_train + n_val :],
        ),
    }
