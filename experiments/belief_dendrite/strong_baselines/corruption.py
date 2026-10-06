"""Local-missing-patch corruption for the CIFAR-10 strong-baselines
benchmark (spec Sec 3) and the training-time horizontal-flip augmentation
(spec Sec 2).

A single contiguous square patch per image (not independent per-pixel
corruption), same spatial mask applied to all three RGB channels. Missing
pixels are set to `0.0` in **standardized** pixel space (the same convention
`experiments/belief_dendrite/corruption.py`'s `missing_patch` family already
uses for MNIST) and marked `reliability = MISSING_CONFIDENCE`; everything
else keeps `reliability = 1`.

Determinism contract (identical to the frozen benchmark's corruption.py):
one realization is a pure function of `(experiment_seed, split, epoch,
replica)` plus the clean input -- model identity never enters the RNG
stream, so every model in a given `(seed,)` comparison sees byte-identical
corrupted inputs and patch geometry. Everything here is vectorized over the
batch dimension; there is no Python loop over individual examples.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

IMAGE_SHAPE: tuple[int, int, int] = (3, 32, 32)  # (channels, height, width)

TRAIN_PATCH_SIDES: tuple[int, ...] = (0, 4, 8, 12)
EVAL_PATCH_SIDES: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24)
IN_DIST_PATCH_SIDES: tuple[int, ...] = TRAIN_PATCH_SIDES

MISSING_VALUE = 0.0
MISSING_CONFIDENCE = 1e-3
OBSERVED_CONFIDENCE = 1.0

N_TEST_REPLICAS = 3

# OOD-drop reference points (spec Sec 9: acc(patch=12) - acc(patch=24)).
MAX_TRAIN_SEVERITY = 12.0
MAX_OOD_SEVERITY = 24.0


def _seed_key(*parts: object) -> int:
    key = "|".join(["strong_baselines", *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _corruption_seed(experiment_seed: int, split: str, epoch: int, replica: int) -> int:
    return _seed_key(experiment_seed, split, epoch, replica)


def _flip_seed(experiment_seed: int, split: str, epoch: int) -> int:
    return _seed_key("flip", experiment_seed, split, epoch)


class Corruption:
    """`x`: `(N, C, H, W)` corrupted pixels (standardized space). `c`:
    `(N, C, H, W)` reliability, identical across the channel dim (spec Sec
    4G: "broadcast the spatial reliability value across the corresponding
    RGB source cells"). `severity`: `(N,)` patch side actually applied per
    example. `mask`: `(N, H, W)` boolean -- the corrupted region, spatial
    only (shared across channels), used by the branch-overlap localization
    diagnostic."""

    __slots__ = ("x", "c", "severity", "mask")

    def __init__(self, x: torch.Tensor, c: torch.Tensor, severity: np.ndarray, mask: torch.Tensor):
        self.x = x
        self.c = c
        self.severity = severity
        self.mask = mask

    def flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x, c)` reshaped to `(N, C*H*W)`, channel-major -- matches
        `DendriticConnectivity.local_2d`'s flatten convention
        (`c * H*W + r*W + col`) exactly, since `x`/`c` are already laid out
        `(N, C, H, W)` and `.reshape` on a contiguous tensor preserves that
        order."""
        n = self.x.shape[0]
        return self.x.reshape(n, -1), self.c.reshape(n, -1)


def _patch_mask(sides: np.ndarray, r0: np.ndarray, c0: np.ndarray, h: int, w: int) -> np.ndarray:
    """Fully vectorized (N, H, W) boolean mask, one contiguous `side x side`
    square per example (empty when `side <= 0`)."""
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    r0_, c0_, side_ = r0[:, None, None], c0[:, None, None], sides[:, None, None]
    return (rows >= r0_) & (rows < r0_ + side_) & (cols >= c0_) & (cols < c0_ + side_)


def _draw_positions_for_sides(
    rng: np.random.Generator, sides: np.ndarray, h: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-example random top-left corner for a (possibly per-example
    varying) patch side, grouped by unique side value so each group draws
    with one vectorized `rng.integers` call."""
    n = sides.shape[0]
    r0 = np.zeros(n, dtype=np.int64)
    c0 = np.zeros(n, dtype=np.int64)
    for s in np.unique(sides):
        if s <= 0:
            continue
        sel = sides == s
        count = int(sel.sum())
        max_r0, max_c0 = h - int(s), w - int(s)
        r0[sel] = rng.integers(0, max_r0 + 1, size=count)
        c0[sel] = rng.integers(0, max_c0 + 1, size=count)
    return r0, c0


def horizontal_flip(
    x_clean: torch.Tensor, *, experiment_seed: int, split: str, epoch: int = 0
) -> torch.Tensor:
    """Random per-example horizontal flip (spec Sec 2: the only augmentation
    allowed), applied only where the caller chooses to call it (training
    split). `x_clean`: `(N, C, H, W)`. Deterministic given
    `(experiment_seed, split, epoch)`, independent of the corruption RNG
    stream. Fully vectorized: one boolean draw per example, then a single
    `torch.where` against the pre-flipped tensor (no Python loop)."""
    rng = np.random.default_rng(_flip_seed(experiment_seed, split, epoch))
    n = x_clean.shape[0]
    flip = torch.from_numpy(rng.random(n) < 0.5).to(x_clean.device)
    flipped = x_clean.flip(dims=(-1,))
    return torch.where(flip.view(-1, 1, 1, 1), flipped, x_clean)


def corrupt_missing_patch(
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    severity: float | None = None,
) -> Corruption:
    """`x_clean`: `(N, C, H, W)`, standardized. `severity=None` draws a
    per-example patch side `~ U(TRAIN_PATCH_SIDES)` (the training regime,
    spec Sec 3); an int/float pins every example to that side (evaluation)."""
    if x_clean.dim() != 4 or tuple(x_clean.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"expected (N, {IMAGE_SHAPE}) images, got shape {tuple(x_clean.shape)}")
    c, h, w = IMAGE_SHAPE
    rng = np.random.default_rng(_corruption_seed(experiment_seed, split, epoch, replica))
    n = x_clean.shape[0]

    if severity is None:
        sides = rng.choice(np.asarray(TRAIN_PATCH_SIDES, dtype=np.int64), size=n)
    else:
        if severity < 0:
            raise ValueError(f"patch side must be >= 0, got {severity}")
        sides = np.full(n, int(severity), dtype=np.int64)

    r0, c0 = _draw_positions_for_sides(rng, sides, h, w)
    mask_np = _patch_mask(sides, r0, c0, h, w)  # (N, H, W)
    mask = torch.from_numpy(mask_np).to(x_clean.device)
    mask_c = mask.unsqueeze(1).expand(-1, c, -1, -1)  # (N, C, H, W)

    xc = torch.where(mask_c, torch.full_like(x_clean, MISSING_VALUE), x_clean)
    conf_spatial = torch.where(
        mask,
        torch.full_like(mask, MISSING_CONFIDENCE, dtype=x_clean.dtype),
        torch.full_like(mask, OBSERVED_CONFIDENCE, dtype=x_clean.dtype),
    )
    conf = conf_spatial.unsqueeze(1).expand(-1, c, -1, -1).contiguous()

    return Corruption(x=xc, c=conf, severity=sides.astype(np.float64), mask=mask)


def eval_severities() -> tuple[float, ...]:
    return EVAL_PATCH_SIDES


def train_severities() -> tuple[float, ...]:
    return TRAIN_PATCH_SIDES


def in_dist_severities() -> tuple[float, ...]:
    return IN_DIST_PATCH_SIDES


def shuffle_reliability_spatially(c: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Sec 12's "shuffled spatial reliability" intervention: per example,
    permute the `(H, W)` reliability map's spatial positions while
    preserving its value histogram (the same set of scalar values, just
    relocated) and keeping the identical permutation across all channels
    (reliability is channel-constant by construction, so this preserves
    that). `c`: `(N, C, H, W)`."""
    n, ch, h, w = c.shape
    flat = c.reshape(n, ch, h * w)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.stack([torch.randperm(h * w, generator=generator) for _ in range(n)]).to(c.device)
    perm_c = perm.unsqueeze(1).expand(-1, ch, -1)
    shuffled = torch.gather(flat, dim=-1, index=perm_c)
    return shuffled.reshape(n, ch, h, w)
