"""The seven required strong-baselines model families (spec Sec 4), plus the
optional ConfidenceTinyViT capacity check (Sec 15). Every model exposes the
uniform interface `forward(x, c) -> logits`, `x`/`c`: `(batch, 3, 32, 32)`
(`c` is channel-constant -- the same spatial reliability value repeated over
R/G/B, matching `corruption.py`'s output) so the training/evaluation loop is
model-agnostic, mirroring `experiments/belief_dendrite/models.py`'s
convention for the frozen MNIST/Fashion-MNIST benchmark.

`ScalarDendriteModel` and `BeliefDendriteModel` wrap the frozen
`src/models/architecture_v2/belief_dendrite.py` classes completely
unmodified -- only a `(N, 3, 32, 32) -> (N, 3072)` reshape sits between this
module and that one, per spec Sec 4F/4G ("No changes" / "Do not otherwise
modify its equations").
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.belief_dendrite.strong_baselines.param_budget import (
    TARGET_PARAM_BUDGET,
    solve_cnn_width,
    solve_dendrite_hidden_width,
    solve_vit_embed_dim,
)
from src.evaluation.efficiency import count_parameters
from src.models.architecture_v2.belief_dendrite import (
    BeliefDendriteNetwork,
    DendriticConnectivity,
    ScalarDendriteNetwork,
    make_connectivity_pair,
)

IN_CHANNELS = 3
IMAGE_SIZE = 32
IMAGE_SHAPE: tuple[int, int, int] = (IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
OUT_FEATURES = 10

VIT_PATCH = 4
VIT_GRID = IMAGE_SIZE // VIT_PATCH  # 8
VIT_NUM_PATCHES = VIT_GRID * VIT_GRID  # 64, spec Sec 4C
VIT_NUM_BLOCKS = 3
VIT_NUM_HEADS = 4
VIT_MLP_RATIO = 4

DENDRITE_BRANCHES = 4
# Implementation-choice default (not user-specified, flagged per
# docs/architecture_v2.md's own convention for such defaults): an 8x8 local
# receptive field is 1/4 of CIFAR-10's 32-pixel side, proportionally the
# same as the frozen MNIST benchmark's 7x7 patch on a 28-pixel side.
DENDRITE_PATCH = 8
DENDRITE_K1 = IN_CHANNELS * DENDRITE_PATCH * DENDRITE_PATCH  # 192
DENDRITE_K2 = 32

SMALL_CNN = "small_cnn"
CONFIDENCE_CNN = "confidence_cnn"
TINY_VIT = "tiny_vit"
CONFIDENCE_TINY_VIT = "confidence_tiny_vit"
RELIABILITY_GATED_TINY_VIT = "reliability_gated_tiny_vit"
SCALAR_DENDRITE = "scalar_dendrite"
BELIEF_DENDRITE = "belief_dendrite"

MODEL_FAMILIES: tuple[str, ...] = (
    SMALL_CNN,
    CONFIDENCE_CNN,
    TINY_VIT,
    CONFIDENCE_TINY_VIT,
    RELIABILITY_GATED_TINY_VIT,
    SCALAR_DENDRITE,
    BELIEF_DENDRITE,
)
DENDRITE_FAMILIES: frozenset[str] = frozenset({SCALAR_DENDRITE, BELIEF_DENDRITE})
VIT_FAMILIES: frozenset[str] = frozenset(
    {TINY_VIT, CONFIDENCE_TINY_VIT, RELIABILITY_GATED_TINY_VIT}
)
RELIABILITY_AWARE: frozenset[str] = frozenset(
    {CONFIDENCE_CNN, CONFIDENCE_TINY_VIT, RELIABILITY_GATED_TINY_VIT, BELIEF_DENDRITE}
)


# ---------------------------------------------------------------------------
# A/B. SmallCNN / ConfidenceCNN.
# ---------------------------------------------------------------------------


class _ConvTrunk(nn.Module):
    """Spec Sec 4A's suggested topology, exactly matching
    `param_budget.cnn_param_count`'s formula. No normalization layer (spec:
    "Prefer GroupNorm or no normalization if batch-size dependence becomes
    an issue" -- omitting one avoids the issue outright, keeping "SmallCNN"
    simple as also asked)."""

    def __init__(self, in_channels: int, width: int, out_features: int = OUT_FEATURES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * width, out_features)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.pool(h)
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
        h = self.gap(h).flatten(1)
        return self.fc(h)


class SmallCNN(nn.Module):
    """Sees only the corrupted image; `c` is accepted (uniform interface)
    but ignored (spec Sec 4A)."""

    def __init__(self, width: int, out_features: int = OUT_FEATURES) -> None:
        super().__init__()
        self.net = _ConvTrunk(IN_CHANNELS, width, out_features)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x)


class ConfidenceCNN(nn.Module):
    """`concat(image, reliability_map)` -- 3 + 1 input channels (spec Sec
    4B). `c` is channel-constant (all three RGB channels carry the identical
    spatial reliability value), so only one channel of it is appended."""

    def __init__(self, width: int, out_features: int = OUT_FEATURES) -> None:
        super().__init__()
        self.net = _ConvTrunk(IN_CHANNELS + 1, width, out_features)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c[:, :1]], dim=1))


# ---------------------------------------------------------------------------
# C/D/E. TinyViT and its two reliability-aware variants.
# ---------------------------------------------------------------------------


def patch_reliability_map(c: torch.Tensor, patch: int = VIT_PATCH) -> torch.Tensor:
    """`c`: `(N, C, H, W)` (channel-constant) -> `(N, num_patches, 1)`, the
    mean reliability inside each non-overlapping `patch x patch` image
    region (spec Sec 4D: `patch_reliability = mean(pixel_reliability in
    patch)`). Shared by `ConfidenceTinyViT` and `ReliabilityGatedTinyViT`,
    and by the Sec 11 diagnostics."""
    pooled = F.avg_pool2d(c[:, :1], kernel_size=patch, stride=patch)  # (N, 1, grid, grid)
    return pooled.flatten(2).transpose(1, 2)  # (N, num_patches, 1)


class _Attention(nn.Module):
    """Standard multi-head self-attention, implemented from scratch (rather
    than `nn.MultiheadAttention`) so its parameter layout is exactly the one
    `param_budget.vit_param_count` predicts and stays stable across torch
    versions: one fused `qkv` projection + one output projection."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm Transformer block: `x + attn(norm(x))`, `x + mlp(norm(x))`,
    GELU, MLP expansion ratio 4 (spec Sec 4C)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = VIT_MLP_RATIO) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = mlp_ratio * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """A real, small, from-scratch Vision Transformer -- no pretrained
    weights, no convolutional stem beyond the patch-embedding projection
    itself, no stochastic depth (spec Sec 4C)."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_blocks: int = VIT_NUM_BLOCKS,
        num_heads: int = VIT_NUM_HEADS,
        patch: int = VIT_PATCH,
        image_size: int = IMAGE_SIZE,
        in_channels: int = IN_CHANNELS,
        out_features: int = OUT_FEATURES,
        mlp_ratio: int = VIT_MLP_RATIO,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.patch = patch
        self.grid = image_size // patch
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_features)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, C, H, W) -> (N, num_patches, embed_dim)`, before the class
        token/positional embedding are added -- overridable insertion point
        for the reliability-aware variants below."""
        h = self.patch_embed(x)  # (N, D, grid, grid)
        return h.flatten(2).transpose(1, 2)  # (N, num_patches, D)

    def encode(self, patch_tok: torch.Tensor) -> torch.Tensor:
        """`(N, num_patches, D) -> (N, num_patches+1, D)` post pos-embed,
        post-blocks, post final norm -- token 0 is the class token."""
        b = patch_tok.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tok = torch.cat([cls, patch_tok], dim=1) + self.pos_embed
        for block in self.blocks:
            tok = block(tok)
        return self.norm(tok)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        tok = self.encode(self.patch_tokens(x))
        return self.head(tok[:, 0])


class ConfidenceTinyViT(TinyViT):
    """`token = image_patch_embedding + reliability_embedding`,
    `reliability_embedding = Linear(1, embed_dim, bias=False)(patch_
    reliability)`; the class token gets no reliability term (spec Sec 4D)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_proj = nn.Linear(1, self.embed_dim, bias=False)

    def forward_verbose(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(logits, image_patch_embedding, reliability_embedding)`
        -- the latter two feed the Sec 11 norm-ratio diagnostic."""
        image_tok = self.patch_tokens(x)
        rel = patch_reliability_map(c, self.patch)
        rel_emb = self.reliability_proj(rel)
        tok = self.encode(image_tok + rel_emb)
        return self.head(tok[:, 0]), image_tok, rel_emb

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.forward_verbose(x, c)[0]


class ReliabilityGatedTinyViT(TinyViT):
    """`token = patch_reliability * image_patch_embedding`, then positional
    encoding as normal -- the Transformer analogue of input reliability
    gating; no propagated reliability state, no added parameters (spec Sec
    4E)."""

    def forward_verbose(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(logits, patch_reliability)` -- the latter feeds the Sec
        11 "mean patch reliability" diagnostic."""
        image_tok = self.patch_tokens(x)
        rel = patch_reliability_map(c, self.patch)
        tok = self.encode(rel * image_tok)
        return self.head(tok[:, 0]), rel

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.forward_verbose(x, c)[0]


# ---------------------------------------------------------------------------
# F/G. Dendritic families -- frozen Architecture V2 classes, reshaped input.
# ---------------------------------------------------------------------------


def build_dendrite_connectivity(
    hidden: int, seed: int
) -> tuple[DendriticConnectivity, DendriticConnectivity]:
    return make_connectivity_pair(
        in_features=IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2],
        hidden1=hidden,
        hidden2=hidden,
        branches_per_soma=DENDRITE_BRANCHES,
        seed=seed,
        image_shape=IMAGE_SHAPE,
        patch=DENDRITE_PATCH,
        sources_per_branch_1=DENDRITE_K1,
        sources_per_branch_2=DENDRITE_K2,
    )


class ScalarDendriteModel(nn.Module):
    """`ScalarDendriteNetwork` (frozen, unmodified), `c` ignored -- isolates
    dendritic *structure* from belief propagation (Sec 13 Q5)."""

    def __init__(
        self,
        connectivity1: DendriticConnectivity,
        connectivity2: DendriticConnectivity,
        out_features: int = OUT_FEATURES,
    ) -> None:
        super().__init__()
        self.net = ScalarDendriteNetwork(connectivity1, connectivity2, out_features)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.net(x.reshape(x.shape[0], -1))


class BeliefDendriteModel(nn.Module):
    """`BeliefDendriteNetwork` (frozen, unmodified): input belief `mu = x`,
    `e = c`, `u = 0` (spec Sec 4G). `c` is already broadcast identically
    across the three RGB channels by `corruption.py`, matching the spec's
    "broadcast the spatial reliability value across the corresponding RGB
    source cells."""

    def __init__(
        self,
        connectivity1: DendriticConnectivity,
        connectivity2: DendriticConnectivity,
        out_features: int = OUT_FEATURES,
    ) -> None:
        super().__init__()
        self.net = BeliefDendriteNetwork(connectivity1, connectivity2, out_features)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return self.net(x.reshape(n, -1), c.reshape(n, -1))


# ---------------------------------------------------------------------------
# Sizing -- every family independently targets ~TARGET_PARAM_BUDGET (Sec 6).
# ---------------------------------------------------------------------------


@dataclass
class BuiltModel:
    model: nn.Module
    family: str
    parameter_count: int
    width: int
    sizing: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_deviation(self) -> float:
        return abs(self.parameter_count - TARGET_PARAM_BUDGET) / TARGET_PARAM_BUDGET


@functools.cache
def dendrite_target_hidden() -> int:
    sol = solve_dendrite_hidden_width(
        TARGET_PARAM_BUDGET, DENDRITE_BRANCHES, DENDRITE_K1, DENDRITE_K2, OUT_FEATURES
    )
    if sol.hidden <= DENDRITE_K2:
        raise RuntimeError(
            f"solved hidden width {sol.hidden} <= DENDRITE_K2={DENDRITE_K2}; the "
            "K2=min(32, H) default-coupling assumption no longer holds at this budget."
        )
    return sol.hidden


@functools.cache
def _dendrite_connectivity(seed: int) -> tuple[DendriticConnectivity, DendriticConnectivity]:
    return build_dendrite_connectivity(dendrite_target_hidden(), seed)


def build_model(family: str, seed: int, param_budget: int = TARGET_PARAM_BUDGET) -> BuiltModel:
    """Constructs one of the seven required families (Sec 4), sized to
    `param_budget` independently (Sec 6: "not matched to any one family's
    exact count"). `seed` only affects the dendritic connectivity draw --
    `ScalarDendriteModel`/`BeliefDendriteModel` share identical topology at a
    fixed seed, so their comparison (Sec 13 Q5) is over belief propagation,
    not structure; every other family's sizing is seed-independent."""
    if family == SMALL_CNN:
        sol = solve_cnn_width(param_budget, IN_CHANNELS, OUT_FEATURES)
        model = SmallCNN(sol.width, OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(model, family, n, sol.width, {"channel_width": sol.width})

    if family == CONFIDENCE_CNN:
        sol = solve_cnn_width(param_budget, IN_CHANNELS + 1, OUT_FEATURES)
        model = ConfidenceCNN(sol.width, OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(model, family, n, sol.width, {"channel_width": sol.width})

    if family in VIT_FAMILIES:
        reliability_linear = family == CONFIDENCE_TINY_VIT
        sol = solve_vit_embed_dim(
            param_budget,
            num_blocks=VIT_NUM_BLOCKS,
            num_patches=VIT_NUM_PATCHES,
            patch=VIT_PATCH,
            in_channels=IN_CHANNELS,
            out_features=OUT_FEATURES,
            reliability_linear=reliability_linear,
            num_heads=VIT_NUM_HEADS,
        )
        cls = {
            TINY_VIT: TinyViT,
            CONFIDENCE_TINY_VIT: ConfidenceTinyViT,
            RELIABILITY_GATED_TINY_VIT: ReliabilityGatedTinyViT,
        }[family]
        model = cls(sol.width, out_features=OUT_FEATURES)
        n = count_parameters(model)
        return BuiltModel(model, family, n, sol.width, {"embed_dim": sol.width})

    if family in DENDRITE_FAMILIES:
        conn1, conn2 = _dendrite_connectivity(seed)
        hidden = dendrite_target_hidden()
        model = (
            ScalarDendriteModel(conn1, conn2, OUT_FEATURES)
            if family == SCALAR_DENDRITE
            else BeliefDendriteModel(conn1, conn2, OUT_FEATURES)
        )
        n = count_parameters(model)
        sizing = {
            "hidden": hidden,
            "branches_per_soma": DENDRITE_BRANCHES,
            "sources_per_branch_1": DENDRITE_K1,
            "sources_per_branch_2": DENDRITE_K2,
            "connectivity_seed": seed,
        }
        return BuiltModel(model, family, n, hidden, sizing)

    raise ValueError(f"unknown model family {family!r}; expected one of {MODEL_FAMILIES}")


def build_confidence_tiny_vit_500k(seed: int = 0) -> BuiltModel:
    """Sec 15's optional, exploratory Transformer-capacity check: a
    ConfidenceTinyViT re-sized to ~500k parameters. Not part of the primary
    seven-family benchmark; only ever run at CIFAR-10 seed 0, after the
    primary sweep completes, and never used to retrain/re-tune Architecture
    V2."""
    return build_model(CONFIDENCE_TINY_VIT, seed, param_budget=500_000)


__all__ = [
    "BELIEF_DENDRITE",
    "CONFIDENCE_CNN",
    "CONFIDENCE_TINY_VIT",
    "DENDRITE_FAMILIES",
    "MODEL_FAMILIES",
    "RELIABILITY_AWARE",
    "RELIABILITY_GATED_TINY_VIT",
    "SCALAR_DENDRITE",
    "SMALL_CNN",
    "TINY_VIT",
    "VIT_FAMILIES",
    "BeliefDendriteModel",
    "BuiltModel",
    "ConfidenceCNN",
    "ConfidenceTinyViT",
    "ReliabilityGatedTinyViT",
    "ScalarDendriteModel",
    "SmallCNN",
    "TinyViT",
    "build_confidence_tiny_vit_500k",
    "build_dendrite_connectivity",
    "build_model",
    "dendrite_target_hidden",
    "patch_reliability_map",
]
