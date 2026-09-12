"""Parameter-budget solvers for the strong-baselines benchmark (spec Sec 6).

Every one of the seven required model families independently targets
``TARGET_PARAM_BUDGET`` (~250k) trainable parameters for CIFAR-10, matched by
adjusting only a width-like variable (CNN channel multiplier, ViT embedding
dimension, dendrite soma width) -- never CNN depth, Transformer depth/heads,
or dendrite B/K (spec Sec 6). Each solver's closed-form parameter-count
formula is checked against the real constructed ``nn.Module`` in
``tests/test_strong_baselines_param_budget.py`` -- a prediction to be
verified, not assumed, mirroring the convention already used by
``src/models/architecture_v2/param_count.py`` and
``src/models/baselines/mlp.py::match_hidden_dim``.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_PARAM_BUDGET = 250_000
MAX_DEVIATION = 0.05  # spec Sec 6: within +/-5% "where mathematically practical"


@dataclass(frozen=True)
class SizingSolution:
    width: int
    parameter_count: int
    budget: int

    @property
    def relative_deviation(self) -> float:
        return abs(self.parameter_count - self.budget) / self.budget


# ---------------------------------------------------------------------------
# A/B. CNN channel-width solver (SmallCNN, ConfidenceCNN).
# ---------------------------------------------------------------------------


def conv2d_param_count(in_channels: int, out_channels: int, kernel: int = 3) -> int:
    """`nn.Conv2d(in_channels, out_channels, kernel)`'s exact parameter count
    (weight + bias)."""
    return in_channels * out_channels * kernel * kernel + out_channels


def cnn_param_count(width: int, in_channels: int, out_features: int = 10) -> int:
    """Exact parameter count of the spec Sec 4A/4B topology at channel
    multiplier `width`:

        Conv3x3(in_channels -> width) -> SiLU
        Conv3x3(width -> width) -> SiLU
        MaxPool2d(2)                                    [downsample, no params]
        Conv3x3(width -> 2*width) -> SiLU
        Conv3x3(2*width -> 2*width) -> SiLU
        AdaptiveAvgPool2d(1)                            [no params]
        Linear(2*width -> out_features)

    `in_channels=3` for `SmallCNN`, `4` for `ConfidenceCNN` (image + one
    reliability channel) -- the two solve independently, per spec Sec 6
    ("Parameter-match by adjusting channel width, not by removing layers").
    """
    c1 = conv2d_param_count(in_channels, width)
    c2 = conv2d_param_count(width, width)
    c3 = conv2d_param_count(width, 2 * width)
    c4 = conv2d_param_count(2 * width, 2 * width)
    fc = 2 * width * out_features + out_features
    return c1 + c2 + c3 + c4 + fc


def solve_cnn_width(
    budget: int, in_channels: int, out_features: int = 10, search_range: range = range(1, 4001)
) -> SizingSolution:
    """The channel multiplier `width` (within `search_range`) whose
    `cnn_param_count` is closest to `budget`."""
    best_w, best_diff = search_range[0], None
    for w in search_range:
        diff = abs(cnn_param_count(w, in_channels, out_features) - budget)
        if best_diff is None or diff < best_diff:
            best_w, best_diff = w, diff
    return SizingSolution(best_w, cnn_param_count(best_w, in_channels, out_features), budget)


# ---------------------------------------------------------------------------
# C/D/E. ViT embedding-dimension solver (TinyViT and its two variants).
# ---------------------------------------------------------------------------


def vit_param_count(
    embed_dim: int,
    *,
    num_blocks: int = 3,
    num_patches: int = 64,
    patch: int = 4,
    in_channels: int = 3,
    out_features: int = 10,
    reliability_linear: bool = False,
) -> int:
    """Exact parameter count of the spec Sec 4C/4D/4E TinyViT topology at
    `embed_dim` (`d`):

        Conv2d patch embed (in_channels, patch x patch -> d)      in_channels*patch^2*d + d
        class token                                               d
        positional embeddings, num_patches + 1 tokens             (num_patches+1)*d
        num_blocks x [pre-norm block]:
            LayerNorm                                             2d
            qkv = Linear(d, 3d, bias)                              3d^2 + 3d
            out_proj = Linear(d, d, bias)                          d^2 + d
            LayerNorm                                             2d
            fc1 = Linear(d, 4d, bias)                              4d^2 + 4d
            fc2 = Linear(4d, d, bias)                              4d^2 + d
        final LayerNorm                                           2d
        class-token classifier = Linear(d, out_features, bias)    d*out_features + out_features
        [+ ConfidenceTinyViT only] reliability_proj = Linear(1, d, bias=False)   d

    `reliability_linear=True` adds the `ConfidenceTinyViT` (spec Sec 4D)
    per-patch reliability projector; `ReliabilityGatedTinyViT` (Sec 4E) adds
    no parameters over plain `TinyViT` (a multiplicative gate has none).
    """
    patch_embed = in_channels * patch * patch * embed_dim + embed_dim
    cls_token = embed_dim
    pos_embed = (num_patches + 1) * embed_dim
    per_block = 12 * embed_dim * embed_dim + 13 * embed_dim
    blocks = num_blocks * per_block
    final_norm = 2 * embed_dim
    head = embed_dim * out_features + out_features
    total = patch_embed + cls_token + pos_embed + blocks + final_norm + head
    if reliability_linear:
        total += embed_dim
    return total


def solve_vit_embed_dim(
    budget: int,
    *,
    num_blocks: int = 3,
    num_patches: int = 64,
    patch: int = 4,
    in_channels: int = 3,
    out_features: int = 10,
    reliability_linear: bool = False,
    num_heads: int = 4,
    max_embed_dim: int = 2000,
) -> SizingSolution:
    """The embedding dimension `d` (a positive multiple of `num_heads`,
    spec Sec 4C "`embedding_dim % 4 == 0`") whose `vit_param_count` is
    closest to `budget`."""
    candidates = range(num_heads, max_embed_dim + 1, num_heads)
    best_d, best_diff = candidates[0], None
    for d in candidates:
        diff = abs(
            vit_param_count(
                d,
                num_blocks=num_blocks,
                num_patches=num_patches,
                patch=patch,
                in_channels=in_channels,
                out_features=out_features,
                reliability_linear=reliability_linear,
            )
            - budget
        )
        if best_diff is None or diff < best_diff:
            best_d, best_diff = d, diff
    params = vit_param_count(
        best_d,
        num_blocks=num_blocks,
        num_patches=num_patches,
        patch=patch,
        in_channels=in_channels,
        out_features=out_features,
        reliability_linear=reliability_linear,
    )
    return SizingSolution(best_d, params, budget)


# ---------------------------------------------------------------------------
# F/G. Dendrite soma-width solver -- delegates to the frozen Architecture V2
# formula (`src/models/architecture_v2/param_count.py`); not reimplemented.
# ---------------------------------------------------------------------------


def solve_dendrite_hidden_width(
    budget: int,
    branches: int,
    sources_per_branch_1: int,
    sources_per_branch_2: int,
    out_features: int = 10,
):
    """Thin re-export of `solve_hidden_width_for_budget` (frozen formula) so
    every solver in this benchmark is reachable from one module."""
    from src.models.architecture_v2.param_count import solve_hidden_width_for_budget

    return solve_hidden_width_for_budget(
        budget, branches, sources_per_branch_1, sources_per_branch_2, out_features
    )
