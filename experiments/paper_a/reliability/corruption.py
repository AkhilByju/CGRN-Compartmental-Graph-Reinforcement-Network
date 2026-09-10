"""Deterministic corruption generation for Paper A Phase 2.

Two corruption families, applied **after** Phase-1 preprocessing (so a Gaussian
`sigma` is expressed in standardized-feature units and a zero-imputed missing
value is the natural mean):

**A. Missing-feature** (`MISSING`) -- each feature is independently dropped to
`0` with a per-example probability `p`; the model is told
`c = 1.0` for an observed value and `c = 1e-3` for a missing one. The `1e-3`
is frozen, not tuned.

**B. Heterogeneous Gaussian** (`GAUSSIAN`) -- feature `j` of an example gets
additive `N(0, sigma_j^2)` noise with `sigma_j ~ U(0, s)` drawn *per feature*,
so features within a single example carry different noise levels; the model is
told the exact `c_j = 1 / (1 + sigma_j^2)`. This mapping is frozen.

Determinism contract (Paper-A Phase-2 task Sec 5)
------------------------------------------------
A corruption realization is a pure function of
`(experiment_seed, split, epoch, replica)` and the clean input tensor.
**Model identity never enters the corruption RNG stream**, so every model in a
`(dataset, corruption_family, seed)` comparison sees byte-identical
`x_corrupted` and `c`. Training corruption may vary between epochs (the `epoch`
key); validation and test corruption are fixed (`epoch = 0`, severity pinned).
Re-running the same experiment seed reproduces every corrupted value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

# --- families -------------------------------------------------------------
MISSING = "missing"
GAUSSIAN = "gaussian"
CORRUPTION_FAMILIES: tuple[str, ...] = (MISSING, GAUSSIAN)

# --- training severity regimes (per-example uniform draw over the set) ----
MISSING_P_TRAIN: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
GAUSSIAN_S_TRAIN: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)

# --- fixed evaluation severity grids -------------------------------------
MISSING_P_TEST: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7)
GAUSSIAN_S_TEST: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0, 1.5)

# --- eval severities inside the training regime (validation / checkpoint
#     selection, Sec 9) ----------------------------------------------------
MISSING_P_IN_DIST: tuple[float, ...] = (0.0, 0.1, 0.3)
GAUSSIAN_S_IN_DIST: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)

# --- OOD-drop reference points (Sec 11) ---------------------------------
MAX_TRAIN_SEVERITY: dict[str, float] = {MISSING: 0.3, GAUSSIAN: 0.75}
MAX_OOD_SEVERITY: dict[str, float] = {MISSING: 0.7, GAUSSIAN: 1.5}

# --- frozen missing-feature reliability values -------------------------
OBSERVED_CONFIDENCE = 1.0
MISSING_CONFIDENCE = 1e-3

# --- number of deterministic test-corruption replicas averaged per severity
N_TEST_REPLICAS = 3


def eval_severities(family: str) -> tuple[float, ...]:
    return MISSING_P_TEST if family == MISSING else GAUSSIAN_S_TEST


def train_severities(family: str) -> tuple[float, ...]:
    return MISSING_P_TRAIN if family == MISSING else GAUSSIAN_S_TRAIN


def in_dist_severities(family: str) -> tuple[float, ...]:
    return MISSING_P_IN_DIST if family == MISSING else GAUSSIAN_S_IN_DIST


def _corruption_seed(experiment_seed: int, split: str, epoch: int, replica: int) -> int:
    """A 64-bit seed derived only from the reproducible keys -- never from the
    model. SHA-256 so distinct keys are effectively independent streams."""
    key = f"paper_a_reliability|{experiment_seed}|{split}|{epoch}|{replica}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


@dataclass(frozen=True)
class Corruption:
    """One corruption realization for an `(N, D)` clean batch.

    * ``x`` -- corrupted observations, same shape/dtype/device as the input.
    * ``c`` -- reliability map in ``(0, 1]``, same shape.
    * ``severity`` -- ``(N,)`` per-example severity actually applied (the
      sampled ``p`` / ``s``); constant when a fixed severity was requested.
    """

    x: torch.Tensor
    c: torch.Tensor
    severity: np.ndarray


def _as_2d_numpy(x_clean: torch.Tensor) -> np.ndarray:
    if x_clean.dim() != 2:
        raise ValueError(f"expected a 2-D (N, D) tensor, got shape {tuple(x_clean.shape)}")
    return x_clean.detach().cpu().to(torch.float64).numpy()


def _finish(x_clean: torch.Tensor, xc: np.ndarray, conf: np.ndarray, sev: np.ndarray) -> Corruption:
    x = torch.from_numpy(np.ascontiguousarray(xc, dtype=np.float32)).to(x_clean.device)
    c = torch.from_numpy(np.ascontiguousarray(conf, dtype=np.float32)).to(x_clean.device)
    return Corruption(x=x, c=c, severity=sev)


def corrupt_missing(
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    p: float | None = None,
) -> Corruption:
    """Missing-feature corruption. `p=None` -> per-example `p ~ U(MISSING_P_TRAIN)`
    (the training regime); a float pins every example to that missing rate
    (evaluation)."""
    rng = np.random.default_rng(_corruption_seed(experiment_seed, split, epoch, replica))
    xnp = _as_2d_numpy(x_clean)
    n, d = xnp.shape
    if p is None:
        sev = rng.choice(np.asarray(MISSING_P_TRAIN, dtype=np.float64), size=n)
    else:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"missing probability p must be in [0, 1], got {p}")
        sev = np.full(n, float(p))
    missing = rng.random((n, d)) < sev[:, None]
    xc = xnp.copy()
    xc[missing] = 0.0
    conf = np.where(missing, MISSING_CONFIDENCE, OBSERVED_CONFIDENCE)
    return _finish(x_clean, xc, conf, sev)


def corrupt_gaussian(
    x_clean: torch.Tensor,
    *,
    experiment_seed: int,
    split: str,
    epoch: int = 0,
    replica: int = 0,
    s: float | None = None,
) -> Corruption:
    """Heterogeneous Gaussian corruption. `s=None` -> per-example max severity
    `s ~ U(GAUSSIAN_S_TRAIN)` (training regime); a float pins the max severity
    (evaluation). Per feature `sigma_j ~ U(0, s)`,
    `x_j <- x_j + N(0, sigma_j^2)`, `c_j = 1 / (1 + sigma_j^2)`."""
    rng = np.random.default_rng(_corruption_seed(experiment_seed, split, epoch, replica))
    xnp = _as_2d_numpy(x_clean)
    n, d = xnp.shape
    if s is None:
        sev = rng.choice(np.asarray(GAUSSIAN_S_TRAIN, dtype=np.float64), size=n)
    else:
        if s < 0.0:
            raise ValueError(f"gaussian max severity s must be >= 0, got {s}")
        sev = np.full(n, float(s))
    sigma = rng.uniform(0.0, 1.0, size=(n, d)) * sev[:, None]  # sigma_j ~ U(0, s)
    noise = rng.standard_normal((n, d)) * sigma
    xc = xnp + noise
    conf = 1.0 / (1.0 + sigma**2)
    return _finish(x_clean, xc, conf, sev)


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
    """Dispatch to the right family. `severity=None` -> training regime (random
    per-example severity); a float -> a fixed evaluation severity."""
    if family == MISSING:
        return corrupt_missing(
            x_clean, experiment_seed=experiment_seed, split=split,
            epoch=epoch, replica=replica, p=severity,
        )
    if family == GAUSSIAN:
        return corrupt_gaussian(
            x_clean, experiment_seed=experiment_seed, split=split,
            epoch=epoch, replica=replica, s=severity,
        )
    raise ValueError(f"unknown corruption family {family!r}; expected one of {CORRUPTION_FAMILIES}")


def shuffle_confidence(c: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Independently permute the reliability values across feature positions
    within each example (Sec 15 confidence-intervention test). The per-example
    multiset of `c` is preserved; its alignment to observations is destroyed.
    Deterministic in `seed`."""
    if c.dim() != 2:
        raise ValueError(f"expected a 2-D (N, D) tensor, got shape {tuple(c.shape)}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n, d = c.shape
    perm = torch.argsort(torch.rand(n, d, generator=g), dim=1).to(c.device)
    return torch.gather(c, 1, perm)
