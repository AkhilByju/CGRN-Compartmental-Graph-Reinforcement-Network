"""Structured image corruption for the Architecture V2 frozen benchmark
(docs/architecture_v2.md Sec K, spec Sec L/M).

Two families, both a single contiguous square patch per image (not
independent per-pixel corruption like Paper A's `missing`/`gaussian`):

* **`missing_patch`** (Sec L) -- one randomly located square region per
  image is zeroed and marked `reliability = 1e-3`; everything outside the
  patch is untouched, `reliability = 1`. Patch side is drawn per example
  from a fixed training regime (`{0, 4, 7, 10}`) or pinned for evaluation
  (`{0, 4, 7, 10, 14, 18}`).
* **`noisy_patch`** (Sec M) -- one randomly located square region of fixed
  side `10` gets additive `N(0, sigma^2)` noise (a single `sigma` per
  example, not per pixel); `reliability = 1 / (1 + sigma^2)` inside,
  `1` outside. `sigma` is drawn from `{0, 0.5, 1.0}` (train) or pinned to
  `{0, 0.5, 1.0, 1.5, 2.0}` (eval).

Determinism contract (matches `experiments/paper_a/reliability/corruption.py`):
a realization is a pure function of `(experiment_seed, split, epoch,
replica)` and the clean input -- model identity never enters the corruption
RNG stream, so every model in a `(dataset, corruption_family, seed)`
comparison sees byte-identical `x_corrupted` / `c` / patch geometry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

MISSING_PATCH = "missing_patch"
NOISY_PATCH = "noisy_patch"
CORRUPTION_FAMILIES: tuple[str, ...] = (MISSING_PATCH, NOISY_PATCH)

IMAGE_SHAPE: tuple[int, int, int] = (1, 28, 28)  # (channels, height, width)

TRAIN_PATCH_SIDES: tuple[int, ...] = (0, 4, 7, 10)
EVAL_PATCH_SIDES: tuple[int, ...] = (0, 4, 7, 10, 14, 18)
IN_DIST_PATCH_SIDES: tuple[int, ...] = TRAIN_PATCH_SIDES

NOISY_PATCH_SIDE = 10
TRAIN_SIGMAS: tuple[float, ...] = (0.0, 0.5, 1.0)
EVAL_SIGMAS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
IN_DIST_SIGMAS: tuple[float, ...] = TRAIN_SIGMAS

MISSING_VALUE = 0.0
MISSING_CONFIDENCE = 1e-3
OBSERVED_CONFIDENCE = 1.0

N_TEST_REPLICAS = 3

# --- OOD-drop reference points (max in-distribution severity -> most severe
#     evaluated severity), mirroring Paper A's MAX_TRAIN/MAX_OOD_SEVERITY.
MAX_TRAIN_SEVERITY: dict[str, float] = {MISSING_PATCH: 10.0, NOISY_PATCH: 1.0}
MAX_OOD_SEVERITY: dict[str, float] = {MISSING_PATCH: 18.0, NOISY_PATCH: 2.0}


def eval_severities(family: str) -> tuple[float, ...]:
    return EVAL_PATCH_SIDES if family == MISSING_PATCH else EVAL_SIGMAS


def train_severities(family: str) -> tuple[float, ...]:
    return TRAIN_PATCH_SIDES if family == MISSING_PATCH else TRAIN_SIGMAS


def in_dist_severities(family: str) -> tuple[float, ...]:
    return IN_DIST_PATCH_SIDES if family == MISSING_PATCH else IN_DIST_SIGMAS


def _corruption_seed(experiment_seed: int, split: str, epoch: int, replica: int) -> int:
    key = f"belief_dendrite|{experiment_seed}|{split}|{epoch}|{replica}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


@dataclass(frozen=True)
class Corruption:
    """`x`/`c`: `(N, 784)`, same dtype/device as the clean input.
    `severity`: `(N,)` per-example severity actually applied. `mask`:
    `(N, 784)` boolean -- exactly the corrupted patch, used by the Sec O
    localization diagnostic (`DendriticConnectivity.corruption_overlap`)."""

    x: torch.Tensor
    c: torch.Tensor
    severity: np.ndarray
    mask: torch.Tensor


def _as_2d_numpy(x_clean: torch.Tensor) -> np.ndarray:
    if x_clean.dim() != 2 or x_clean.shape[-1] != IMAGE_SHAPE[1] * IMAGE_SHAPE[2]:
        raise ValueError(
            f"expected a flattened (N, {IMAGE_SHAPE[1] * IMAGE_SHAPE[2]}) image tensor, "
            f"got shape {tuple(x_clean.shape)}"
        )
    return x_clean.detach().cpu().to(torch.float64).numpy()


def _finish(
    x_clean: torch.Tensor, xc: np.ndarray, conf: np.ndarray, mask: np.ndarray, sev: np.ndarray
) -> Corruption:
    x = torch.from_numpy(np.ascontiguousarray(xc, dtype=np.float32)).to(x_clean.device)
    c = torch.from_numpy(np.ascontiguousarray(conf, dtype=np.float32)).to(x_clean.device)
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(x_clean.device)
    return Corruption(x=x, c=c, severity=sev, mask=m)


def _patch_mask(sides: np.ndarray, r0: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """Fully vectorized (no Python loop over examples): `(N, H, W)` boolean,
    one contiguous `side x side` square per example (empty when `side<=0`)."""
    _, h, w = IMAGE_SHAPE
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    r0_, c0_, side_ = r0[:, None, None], c0[:, None, None], sides[:, None, None]
    return (rows >= r0_) & (rows < r0_ + side_) & (cols >= c0_) & (cols < c0_ + side_)


def _draw_positions_for_sides(
    rng: np.random.Generator, sides: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-example random top-left corner for a (possibly per-example
    varying) patch side, grouped by unique side value so each group can be
    drawn with a single vectorized `rng.integers` call."""
    _, h, w = IMAGE_SHAPE
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


def corrupt_missing_patch(
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    severity: float | None = None,
) -> Corruption:
    """`severity=None` -> per-example patch side `~ U(TRAIN_PATCH_SIDES)`
    (training regime); an int/float pins every example to that side."""
    rng = np.random.default_rng(_corruption_seed(experiment_seed, split, epoch, replica))
    xnp = _as_2d_numpy(x_clean)
    n = xnp.shape[0]

    if severity is None:
        sides = rng.choice(np.asarray(TRAIN_PATCH_SIDES, dtype=np.int64), size=n)
    else:
        if severity < 0:
            raise ValueError(f"patch side must be >= 0, got {severity}")
        sides = np.full(n, int(severity), dtype=np.int64)

    r0, c0 = _draw_positions_for_sides(rng, sides)
    mask = _patch_mask(sides, r0, c0).reshape(n, -1)

    xc = xnp.copy()
    xc[mask] = MISSING_VALUE
    conf = np.where(mask, MISSING_CONFIDENCE, OBSERVED_CONFIDENCE)
    return _finish(x_clean, xc, conf, mask, sides.astype(np.float64))


def corrupt_noisy_patch(
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    severity: float | None = None,
) -> Corruption:
    """Fixed `side = NOISY_PATCH_SIDE`; `severity=None` -> per-example
    `sigma ~ U(TRAIN_SIGMAS)` (training regime); a float pins `sigma`."""
    rng = np.random.default_rng(_corruption_seed(experiment_seed, split, epoch, replica))
    xnp = _as_2d_numpy(x_clean)
    n = xnp.shape[0]

    if severity is None:
        sigmas = rng.choice(np.asarray(TRAIN_SIGMAS, dtype=np.float64), size=n)
    else:
        if severity < 0:
            raise ValueError(f"sigma must be >= 0, got {severity}")
        sigmas = np.full(n, float(severity))

    sides = np.full(n, NOISY_PATCH_SIDE, dtype=np.int64)
    r0, c0 = _draw_positions_for_sides(rng, sides)
    mask = _patch_mask(sides, r0, c0).reshape(n, -1)

    noise = rng.standard_normal((n, xnp.shape[1])) * sigmas[:, None]
    xc = xnp + noise * mask.astype(xnp.dtype)
    conf_inside = 1.0 / (1.0 + sigmas**2)
    conf = np.where(mask, conf_inside[:, None], OBSERVED_CONFIDENCE)
    return _finish(x_clean, xc, conf, mask, sigmas)


def corrupt(
    family: str,
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    severity: float | None = None,
) -> Corruption:
    if family == MISSING_PATCH:
        return corrupt_missing_patch(
            x_clean,
            experiment_seed=experiment_seed,
            split=split,
            epoch=epoch,
            replica=replica,
            severity=severity,
        )
    if family == NOISY_PATCH:
        return corrupt_noisy_patch(
            x_clean,
            experiment_seed=experiment_seed,
            split=split,
            epoch=epoch,
            replica=replica,
            severity=severity,
        )
    raise ValueError(f"unknown corruption family {family!r}; expected one of {CORRUPTION_FAMILIES}")
