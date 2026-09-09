"""Public-benchmark dataset loaders, deterministic splits, and train-only
preprocessing for Paper A Phase 1.

The dataset list is **frozen** (`BENCHMARK_DATASETS`) -- it was fixed before
looking at any CellV0.1 performance, per the Paper-A task's anti-cherry-picking
rule. Do not add or remove datasets based on results.

Split contract
--------------
* **sklearn tabular** (`breast_cancer`, `wine`, `digits`, `diabetes`,
  `california_housing`): a seeded 60/20/20 train-pool / val / test partition.
  Classification splits are stratified. The partition is a deterministic
  function of `seed` alone, so all three model families see the identical
  val/test sets for a given seed.
* **MNIST / Fashion-MNIST**: the ordinary official 60k/10k train/test split
  (openml serves all 70k rows in canonical order -- first 60k train, last 10k
  test). The validation subset is carved out of the official *training* set
  only, seeded and stratified; test is the untouched official 10k.

Low-data regime
---------------
The 25% condition subsamples the **training pool only** (stratified for
classification), seeded. Val and test are never subsampled.

Preprocessing
-------------
Every statistic (feature mean/std, regression-target mean/std) is fit on the
actually-used training subset and applied to val/test -- no leakage. Images
are flattened (already flat from openml) and standardized the same way as
tabular features (per-feature z-score, zero-variance pixels guarded by
`standardize`'s `clamp_min`). Regression targets are standardized for
optimization; `PreparedDataset.inverse_transform_targets` maps predictions
back to the original scale for RMSE/R^2.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.datasets import (  # noqa: E402
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from src.data.synthetic.utils import standardize  # noqa: E402

TaskType = Literal["classification", "regression"]

# ---------------------------------------------------------------------------
# Frozen benchmark list (fixed BEFORE inspecting CellV0.1 performance).
# ---------------------------------------------------------------------------
BENCHMARK_DATASETS: tuple[str, ...] = (
    "breast_cancer",
    "wine",
    "digits",
    "diabetes",
    "california_housing",
    "mnist",
    "fashion_mnist",
)

# Fixed parameter budget by dataset class (Paper-A task Sec 5). CellV0.1's
# hidden-cell count is the largest that stays within this; the matched MLP is
# then matched to CellV0.1's *actual* resulting parameter count.
PARAM_BUDGET: dict[str, int] = {
    "breast_cancer": 10_000,
    "wine": 10_000,
    "digits": 25_000,
    "diabetes": 10_000,
    "california_housing": 10_000,
    "mnist": 150_000,
    "fashion_mnist": 150_000,
}

_IMAGE_DATASETS: frozenset[str] = frozenset({"mnist", "fashion_mnist"})
_OPENML_IDS: dict[str, str] = {"mnist": "mnist_784", "fashion_mnist": "Fashion-MNIST"}

# Held-out validation size carved from the 60k official training set.
_IMAGE_VAL_SIZE: int = 10_000
_IMAGE_OFFICIAL_TRAIN: int = 60_000

# sklearn tabular 3-way partition fractions (train-pool / val / test).
_TEST_FRACTION: float = 0.2
_VAL_FRACTION_OF_DEV: float = 0.25  # 0.25 * 0.8 = 0.2 of the whole


class CaliforniaHousingUnavailable(RuntimeError):
    """Raised when California Housing cannot be downloaded in this
    environment. The Paper-A task says to report this explicitly and skip
    the dataset rather than silently substituting another."""


@dataclass
class PreparedDataset:
    name: str
    task_type: TaskType
    seed: int
    train_fraction: float

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    n_features: int
    n_classes: int | None
    is_binary: bool
    param_budget: int

    # Regression only: training-set target statistics for inverse-transform.
    target_mean: float | None = None
    target_std: float | None = None
    y_train_raw: torch.Tensor | None = None
    y_val_raw: torch.Tensor | None = None
    y_test_raw: torch.Tensor | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def out_features(self) -> int:
        """Readout width: number of classes for classification, 1 for
        regression."""
        return self.n_classes if self.task_type == "classification" else 1

    def inverse_transform_targets(self, standardized: torch.Tensor) -> torch.Tensor:
        """Map standardized regression predictions/targets back to the
        original scale. Identity for classification."""
        if self.task_type != "regression":
            return standardized
        return standardized * self.target_std + self.target_mean


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


@functools.cache
def _load_sklearn_xy(name: str) -> tuple[np.ndarray, np.ndarray, TaskType]:
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data.astype(np.float64), d.target.astype(np.int64), "classification"
    if name == "wine":
        d = load_wine()
        return d.data.astype(np.float64), d.target.astype(np.int64), "classification"
    if name == "digits":
        d = load_digits()
        return d.data.astype(np.float64), d.target.astype(np.int64), "classification"
    if name == "diabetes":
        d = load_diabetes()
        return d.data.astype(np.float64), d.target.astype(np.float64), "regression"
    if name == "california_housing":
        try:
            d = fetch_california_housing()
        except Exception as exc:  # noqa: BLE001 -- surface as a typed, explicit skip
            raise CaliforniaHousingUnavailable(
                f"Could not download California Housing: {exc!r}"
            ) from exc
        return d.data.astype(np.float64), d.target.astype(np.float64), "regression"
    raise ValueError(f"Unknown sklearn dataset '{name}'.")


@functools.cache
def _load_image_xy(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns all 70k rows in canonical order (first 60k = official train,
    last 10k = official test). Memoized -- the openml disk read is slow and
    the raw arrays are reused across every seed/fraction/model."""
    bunch = fetch_openml(_OPENML_IDS[name], version=1, as_frame=False)
    x = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target).astype(np.int64)
    expected_rows = _IMAGE_OFFICIAL_TRAIN + 10_000  # 70000
    if x.shape[0] != expected_rows:
        raise RuntimeError(
            f"{name}: expected {expected_rows} rows from openml, got {x.shape[0]}."
        )
    return x, y


def _standardize_global(
    train_x: torch.Tensor, *other_xs: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Single scalar mean/std over every element of `train_x`, applied to
    every tensor. Training-set statistics only."""
    mean = train_x.mean()
    std = train_x.std().clamp_min(1e-6)
    return tuple((x - mean) / std for x in (train_x, *other_xs))


def _relabel_contiguous(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Map class labels to a contiguous 0..K-1 integer range (openml targets
    arrive as strings/ints; sklearn ones are already contiguous, so this is
    a no-op there)."""
    classes, inverse = np.unique(y, return_inverse=True)
    return inverse.astype(np.int64), len(classes)


# ---------------------------------------------------------------------------
# Deterministic splitting
# ---------------------------------------------------------------------------


def _split_indices_sklearn(
    n: int, y: np.ndarray, task_type: TaskType, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded 60/20/20 train-pool / val / test index partition. Stratified
    for classification."""
    idx = np.arange(n)
    strat = y if task_type == "classification" else None
    dev_idx, test_idx = train_test_split(
        idx, test_size=_TEST_FRACTION, random_state=seed, stratify=strat
    )
    strat_dev = y[dev_idx] if task_type == "classification" else None
    train_pool_idx, val_idx = train_test_split(
        dev_idx, test_size=_VAL_FRACTION_OF_DEV, random_state=seed, stratify=strat_dev
    )
    return np.sort(train_pool_idx), np.sort(val_idx), np.sort(test_idx)


def _split_indices_image(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official 60k/10k train/test; val stratified-carved from the 60k."""
    official_train_idx = np.arange(_IMAGE_OFFICIAL_TRAIN)
    test_idx = np.arange(_IMAGE_OFFICIAL_TRAIN, y.shape[0])
    train_pool_idx, val_idx = train_test_split(
        official_train_idx,
        test_size=_IMAGE_VAL_SIZE,
        random_state=seed,
        stratify=y[official_train_idx],
    )
    return np.sort(train_pool_idx), np.sort(val_idx), np.sort(test_idx)


def _subsample_train(
    train_pool_idx: np.ndarray,
    y_pool: np.ndarray,
    task_type: TaskType,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    """Stratified (classification) subsample of the training pool. Full pool
    when `train_fraction >= 1`."""
    if train_fraction >= 1.0:
        return train_pool_idx
    strat = y_pool if task_type == "classification" else None
    kept, _ = train_test_split(
        train_pool_idx,
        train_size=train_fraction,
        random_state=seed,
        stratify=strat,
    )
    return np.sort(kept)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _raw_xy_and_task(name: str) -> tuple[np.ndarray, np.ndarray, TaskType]:
    if name in _IMAGE_DATASETS:
        x_all, y_all = _load_image_xy(name)
        return x_all, y_all, "classification"
    return _load_sklearn_xy(name)


def compute_split_indices(
    name: str, seed: int, train_fraction: float
) -> dict[str, np.ndarray]:
    """The deterministic `(train, val, test)` row-index partition for a
    `(name, seed, train_fraction)`, exposed separately from `prepare_dataset`
    so split determinism / disjointness can be checked without materializing
    the preprocessed tensors."""
    if name not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {BENCHMARK_DATASETS}.")
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}.")

    _, y_all, task_type = _raw_xy_and_task(name)
    if name in _IMAGE_DATASETS:
        train_pool_idx, val_idx, test_idx = _split_indices_image(y_all, seed)
    else:
        train_pool_idx, val_idx, test_idx = _split_indices_sklearn(
            y_all.shape[0], y_all, task_type, seed
        )
    train_idx = _subsample_train(
        train_pool_idx, y_all[train_pool_idx], task_type, train_fraction, seed
    )

    assert set(train_idx).isdisjoint(val_idx), f"{name}: train/val leakage"
    assert set(train_idx).isdisjoint(test_idx), f"{name}: train/test leakage"
    assert set(val_idx).isdisjoint(test_idx), f"{name}: val/test leakage"

    return {
        "train_pool": train_pool_idx,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def prepare_dataset(
    name: str, seed: int, train_fraction: float
) -> PreparedDataset:
    """Load, split, subsample, and preprocess `name` deterministically from
    `(seed, train_fraction)`. All three model families call this with the
    same arguments, so they train/evaluate on byte-identical tensors."""
    x_all, y_all, task_type = _raw_xy_and_task(name)
    idx = compute_split_indices(name, seed, train_fraction)
    train_pool_idx, train_idx, val_idx, test_idx = (
        idx["train_pool"], idx["train"], idx["val"], idx["test"]
    )

    x_train_np, x_val_np, x_test_np = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train_np, y_val_np, y_test_np = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    xt = torch.tensor(x_train_np, dtype=torch.float32)
    xv = torch.tensor(x_val_np, dtype=torch.float32)
    xs = torch.tensor(x_test_np, dtype=torch.float32)
    if name in _IMAGE_DATASETS:
        # "Standard normalization" for flattened images: a single global
        # mean/std (training subset only), matching torchvision's
        # `Normalize((mean,), (std,))` convention. Per-feature z-scoring
        # blows up on near-zero-variance border pixels.
        x_train, x_val, x_test = _standardize_global(xt, xv, xs)
    else:
        # Tabular: per-feature z-score, statistics from the training subset only.
        x_train, x_val, x_test = standardize(xt, xv, xs)

    n_classes: int | None = None
    is_binary = False
    target_mean = target_std = None
    y_train_raw = y_val_raw = y_test_raw = None

    if task_type == "classification":
        # Relabel against the full label set so class ids line up across
        # splits even if a rare class is absent from a 25% subsample.
        y_all_relabelled, n_classes = _relabel_contiguous(y_all)
        y_train = torch.tensor(y_all_relabelled[train_idx], dtype=torch.long)
        y_val = torch.tensor(y_all_relabelled[val_idx], dtype=torch.long)
        y_test = torch.tensor(y_all_relabelled[test_idx], dtype=torch.long)
        is_binary = n_classes == 2
    else:
        y_train_raw = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
        y_val_raw = torch.tensor(y_val_np, dtype=torch.float32).reshape(-1, 1)
        y_test_raw = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)
        target_mean = float(y_train_raw.mean())
        target_std = float(y_train_raw.std().clamp_min(1e-6))
        y_train = (y_train_raw - target_mean) / target_std
        y_val = (y_val_raw - target_mean) / target_std
        y_test = (y_test_raw - target_mean) / target_std

    meta = {
        "n_train_pool": int(train_pool_idx.shape[0]),
        "n_train_used": int(train_idx.shape[0]),
        "n_val": int(val_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "n_total": int(x_all.shape[0]),
        "official_split": name in _IMAGE_DATASETS,
    }

    return PreparedDataset(
        name=name,
        task_type=task_type,
        seed=seed,
        train_fraction=train_fraction,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        n_features=int(x_all.shape[1]),
        n_classes=n_classes,
        is_binary=is_binary,
        param_budget=PARAM_BUDGET[name],
        target_mean=target_mean,
        target_std=target_std,
        y_train_raw=y_train_raw,
        y_val_raw=y_val_raw,
        y_test_raw=y_test_raw,
        meta=meta,
    )


def dataset_available(name: str) -> bool:
    """Cheap check that a dataset can be loaded at all -- used to decide
    whether to skip California Housing without aborting the whole sweep."""
    try:
        if name in _IMAGE_DATASETS:
            _load_image_xy(name)
        else:
            _load_sklearn_xy(name)
        return True
    except CaliforniaHousingUnavailable:
        return False
    except Exception:  # noqa: BLE001
        return False
