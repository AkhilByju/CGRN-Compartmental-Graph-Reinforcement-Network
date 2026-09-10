"""Real missing-sensor dataset loaders, deterministic splits, and train-only
preprocessing for Paper A Phase 3 Part B.

The two datasets are **frozen** (`REAL_DATASETS`). Raw files are downloaded from
the UCI archive once and cached under ``~/.cache/cgrn_data/``.

Split contract
--------------
* **APS** (`aps`): the *official* 60,000/16,000 train/test split (two separate
  UCI CSVs) -- the 16k test set is never touched. A deterministic **stratified**
  80/20 train/validation partition is carved from the official training set,
  seeded (so all models see the identical val set for a given seed).
* **Air Quality** (`air_quality`): rows in chronological order (documented
  sensor drift is part of the real setting) -- **no shuffle before splitting**.
  After dropping rows whose *target* is missing: first 60% train, next 20%
  validation, final 20% test. The split does not depend on the seed.

Missingness
-----------
Missing values are the datasets' own -- APS blanks (``na``), Air Quality
``-200`` sentinels. They are **not** altered. Preprocessing fits every
statistic on **observed training values only**; observed values are
standardized; a missing standardized value becomes ``0`` for the MLP / CellV0.3
inputs (``x_imputed``) and stays ``NaN`` for NeuMiss (``x_nan``, which handles
missing entries directly). Reliability is ``c_j = 1.0`` for an observed feature
and ``c_j = 1e-3`` for a missing one -- derived from the missing mask alone,
never from the target.

Regression targets (Air Quality) are standardized on the training target
mean/std for optimization; `inverse_transform_targets` maps predictions back to
the original units for reported RMSE/MAE.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

TaskType = Literal["classification", "regression"]

REAL_DATASETS: tuple[str, ...] = ("aps", "air_quality")

# Frozen-before-running target neural parameter counts (Phase-3 task
# "Parameter budgets"). CellV0.3's hidden width is the largest that fits;
# Plain MLP and the parameter-matched Confidence MLP are matched to CellV0.3's
# actual count.
REAL_PARAM_BUDGET: dict[str, int] = {"aps": 100_000, "air_quality": 50_000}

OBSERVED_CONFIDENCE = 1.0
MISSING_CONFIDENCE = 1e-3

_CACHE = Path.home() / ".cache" / "cgrn_data"
_UCI = {
    "aps": "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip",
    "air_quality": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
}

_APS_FEATURE_COUNT = 170  # 171 CSV columns = class + 170 operational features
_AIR_QUALITY_INPUTS: tuple[str, ...] = (
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",   # Temperature
    "RH",  # Relative Humidity
    "AH",  # Absolute Humidity
)
_AIR_QUALITY_TARGET = "CO(GT)"
# Explicitly excluded: the other ground-truth pollutant columns and the target.
_AIR_QUALITY_FORBIDDEN: frozenset[str] = frozenset(
    {"CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"}
)
_AIR_QUALITY_MISSING_SENTINEL = -200


class DatasetUnavailable(RuntimeError):
    """Raised when a real dataset cannot be downloaded in this environment --
    the Phase-3 task says to report this explicitly, not silently substitute."""


@dataclass
class RealPreparedDataset:
    name: str
    task_type: TaskType
    seed: int

    # x_imputed (missing standardized value -> 0) for MLP / CellV0.3
    x_train: torch.Tensor
    x_val: torch.Tensor
    x_test: torch.Tensor
    # x with missing entries kept as NaN, for NeuMiss
    x_train_nan: torch.Tensor
    x_val_nan: torch.Tensor
    x_test_nan: torch.Tensor
    # reliability masks c (1.0 observed / 1e-3 missing)
    c_train: torch.Tensor
    c_val: torch.Tensor
    c_test: torch.Tensor
    # targets
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor

    n_features: int
    n_classes: int | None
    is_binary: bool
    param_budget: int

    # per-example missing fraction (missing_features / total_features)
    missing_frac_train: torch.Tensor
    missing_frac_val: torch.Tensor
    missing_frac_test: torch.Tensor

    # regression only
    target_mean: float | None = None
    target_std: float | None = None
    y_train_raw: torch.Tensor | None = None
    y_val_raw: torch.Tensor | None = None
    y_test_raw: torch.Tensor | None = None

    # classification only -- computed once from the training labels and applied
    # identically to every neural model. `class_weights` is the sklearn
    # `balanced` per-class vector; `pos_weight` = n_neg / n_pos is the scalar
    # for `BCEWithLogitsLoss` (every APS neural model emits one logit).
    class_weights: torch.Tensor | None = None
    pos_weight: float | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def out_features(self) -> int:
        """Readout width -- one logit for the (binary) real classification
        task, one value for regression."""
        return 1

    def inverse_transform_targets(self, standardized: torch.Tensor) -> torch.Tensor:
        if self.task_type != "regression":
            return standardized
        return standardized * self.target_std + self.target_mean


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------


def _download_zip(name: str) -> zipfile.ZipFile:
    try:
        req = urllib.request.Request(_UCI[name], headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=180).read()  # noqa: S310 -- pinned UCI URL
    except Exception as exc:  # noqa: BLE001
        raise DatasetUnavailable(f"could not download {name} from UCI: {exc!r}") from exc
    return zipfile.ZipFile(io.BytesIO(data))


def _cached_bytes(name: str, member: str) -> bytes:
    _CACHE.mkdir(parents=True, exist_ok=True)
    path = _CACHE / member
    if path.exists():
        return path.read_bytes()
    z = _download_zip(name)
    raw = z.read(member)
    path.write_bytes(raw)
    return raw


def dataset_available(name: str) -> bool:
    try:
        if name == "aps":
            _cached_bytes("aps", "aps_failure_training_set.csv")
        elif name == "air_quality":
            _cached_bytes("air_quality", "AirQualityUCI.csv")
        else:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Raw frames
# ---------------------------------------------------------------------------


def _aps_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The official APS train (60k) and test (16k) frames. `class` -> {0,1}
    (pos = 1), 170 operational feature columns, blanks -> NaN."""
    train = pd.read_csv(
        io.BytesIO(_cached_bytes("aps", "aps_failure_training_set.csv")),
        skiprows=20, na_values=["na"],
    )
    test = pd.read_csv(
        io.BytesIO(_cached_bytes("aps", "aps_failure_test_set.csv")),
        skiprows=20, na_values=["na"],
    )
    for f in (train, test):
        f["class"] = (f["class"] == "pos").astype(np.int64)
    feats = [c for c in train.columns if c != "class"]
    if len(feats) != _APS_FEATURE_COUNT:
        raise RuntimeError(f"expected {_APS_FEATURE_COUNT} APS features, got {len(feats)}")
    return train, test


def _air_quality_frame() -> pd.DataFrame:
    """Air Quality in chronological order; `-200` -> NaN; rows with a missing
    *target* dropped; input columns limited to the 8 real sensor/environment
    features (no other GT pollutant column, no CO(GT) as input)."""
    df = pd.read_csv(
        io.BytesIO(_cached_bytes("air_quality", "AirQualityUCI.csv")),
        sep=";", decimal=",",
    )
    df = df.dropna(how="all")  # drop the CSV's trailing all-empty rows
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    cols = list(_AIR_QUALITY_INPUTS) + [_AIR_QUALITY_TARGET]
    df = df[cols].replace(_AIR_QUALITY_MISSING_SENTINEL, np.nan)
    # drop rows whose target is missing; keep rows with missing *inputs*
    df = df[df[_AIR_QUALITY_TARGET].notna()].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _standardize_observed(
    x_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Per-feature z-score using **observed** training values only. Missing
    entries stay NaN. Zero-variance / all-missing columns are guarded."""
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return tuple((x - mean) / std for x in (x_train, *others))


def _mask_and_impute(x_std_nan: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(x_imputed, c, missing_mask)` -- missing standardized value -> 0,
    reliability 1.0 / 1e-3, boolean missing mask."""
    missing = np.isnan(x_std_nan)
    x_imputed = np.where(missing, 0.0, x_std_nan)
    c = np.where(missing, MISSING_CONFIDENCE, OBSERVED_CONFIDENCE)
    return x_imputed.astype(np.float32), c.astype(np.float32), missing


def _t(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(a), dtype=dtype)


def _missing_fraction(missing_mask: np.ndarray) -> torch.Tensor:
    return _t(missing_mask.mean(axis=1))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _prepare_aps(seed: int) -> RealPreparedDataset:
    train_df, test_df = _aps_frames()
    feats = [c for c in train_df.columns if c != "class"]

    y_dev = train_df["class"].to_numpy()
    dev_idx = np.arange(len(train_df))
    tr_idx, val_idx = train_test_split(
        dev_idx, test_size=0.2, random_state=seed, stratify=y_dev
    )
    tr_idx, val_idx = np.sort(tr_idx), np.sort(val_idx)

    x_all = train_df[feats].to_numpy(dtype=np.float64)
    x_tr_raw, x_val_raw = x_all[tr_idx], x_all[val_idx]
    x_te_raw = test_df[feats].to_numpy(dtype=np.float64)

    x_tr_std, x_val_std, x_te_std = _standardize_observed(x_tr_raw, x_val_raw, x_te_raw)
    xi_tr, c_tr, m_tr = _mask_and_impute(x_tr_std)
    xi_val, c_val, m_val = _mask_and_impute(x_val_std)
    xi_te, c_te, m_te = _mask_and_impute(x_te_std)

    y_tr = y_dev[tr_idx].astype(np.int64)
    y_val = y_dev[val_idx].astype(np.int64)
    y_te = test_df["class"].to_numpy().astype(np.int64)

    # class weights: sklearn's `class_weight="balanced"` recipe
    # (n_samples / (n_classes * bincount)), computed once from TRAIN labels and
    # applied identically to every neural model. pos_weight = n_neg / n_pos for
    # the shared BCEWithLogitsLoss.
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    w = counts.sum() / (2.0 * np.clip(counts, 1.0, None))
    pos_weight = float(counts[0] / max(counts[1], 1.0))

    return RealPreparedDataset(
        name="aps",
        task_type="classification",
        seed=seed,
        x_train=_t(xi_tr), x_val=_t(xi_val), x_test=_t(xi_te),
        x_train_nan=_t(x_tr_std), x_val_nan=_t(x_val_std), x_test_nan=_t(x_te_std),
        c_train=_t(c_tr), c_val=_t(c_val), c_test=_t(c_te),
        y_train=_t(y_tr, torch.long), y_val=_t(y_val, torch.long), y_test=_t(y_te, torch.long),
        n_features=len(feats),
        n_classes=2,
        is_binary=True,
        param_budget=REAL_PARAM_BUDGET["aps"],
        missing_frac_train=_missing_fraction(m_tr),
        missing_frac_val=_missing_fraction(m_val),
        missing_frac_test=_missing_fraction(m_te),
        class_weights=_t(w),
        pos_weight=pos_weight,
        meta={
            "n_train": int(tr_idx.size), "n_val": int(val_idx.size), "n_test": int(y_te.size),
            "official_test": True,
            "train_pos": int(y_tr.sum()), "val_pos": int(y_val.sum()), "test_pos": int(y_te.sum()),
            "overall_missing_frac": float(m_tr.mean()),
        },
    )


def _prepare_air_quality(seed: int) -> RealPreparedDataset:
    df = _air_quality_frame()
    n = len(df)
    n_tr, n_val = int(round(0.6 * n)), int(round(0.2 * n))
    sl_tr = slice(0, n_tr)
    sl_val = slice(n_tr, n_tr + n_val)
    sl_te = slice(n_tr + n_val, n)

    x_all = df[list(_AIR_QUALITY_INPUTS)].to_numpy(dtype=np.float64)
    y_all = df[_AIR_QUALITY_TARGET].to_numpy(dtype=np.float64)

    x_tr_std, x_val_std, x_te_std = _standardize_observed(
        x_all[sl_tr], x_all[sl_val], x_all[sl_te]
    )
    xi_tr, c_tr, m_tr = _mask_and_impute(x_tr_std)
    xi_val, c_val, m_val = _mask_and_impute(x_val_std)
    xi_te, c_te, m_te = _mask_and_impute(x_te_std)

    y_tr_raw, y_val_raw, y_te_raw = y_all[sl_tr], y_all[sl_val], y_all[sl_te]
    tgt_mean = float(y_tr_raw.mean())
    tgt_std = float(max(y_tr_raw.std(), 1e-6))
    y_tr = (y_tr_raw - tgt_mean) / tgt_std
    y_val = (y_val_raw - tgt_mean) / tgt_std
    y_te = (y_te_raw - tgt_mean) / tgt_std

    return RealPreparedDataset(
        name="air_quality",
        task_type="regression",
        seed=seed,
        x_train=_t(xi_tr), x_val=_t(xi_val), x_test=_t(xi_te),
        x_train_nan=_t(x_tr_std), x_val_nan=_t(x_val_std), x_test_nan=_t(x_te_std),
        c_train=_t(c_tr), c_val=_t(c_val), c_test=_t(c_te),
        y_train=_t(y_tr).reshape(-1, 1),
        y_val=_t(y_val).reshape(-1, 1),
        y_test=_t(y_te).reshape(-1, 1),
        n_features=len(_AIR_QUALITY_INPUTS),
        n_classes=None,
        is_binary=False,
        param_budget=REAL_PARAM_BUDGET["air_quality"],
        missing_frac_train=_missing_fraction(m_tr),
        missing_frac_val=_missing_fraction(m_val),
        missing_frac_test=_missing_fraction(m_te),
        target_mean=tgt_mean,
        target_std=tgt_std,
        y_train_raw=_t(y_tr_raw).reshape(-1, 1),
        y_val_raw=_t(y_val_raw).reshape(-1, 1),
        y_test_raw=_t(y_te_raw).reshape(-1, 1),
        meta={
            "n_train": n_tr, "n_val": n_val, "n_test": n - n_tr - n_val,
            "chronological": True,
            "inputs": list(_AIR_QUALITY_INPUTS), "target": _AIR_QUALITY_TARGET,
            "overall_missing_frac": float(m_tr.mean()),
        },
    )


def prepare_dataset(name: str, seed: int) -> RealPreparedDataset:
    if name == "aps":
        return _prepare_aps(seed)
    if name == "air_quality":
        return _prepare_air_quality(seed)
    raise ValueError(f"unknown real dataset {name!r}; expected one of {REAL_DATASETS}")


# ---------------------------------------------------------------------------
# Missingness strata (Phase-3 task "Real-missingness diagnostics")
# ---------------------------------------------------------------------------

# Fixed bins; an empty bin is skipped, never redefined.
MISSINGNESS_BINS: tuple[tuple[str, float, float], ...] = (
    ("0", -1e-9, 1e-9),
    ("(0, 0.10]", 1e-9, 0.10),
    ("(0.10, 0.25]", 0.10, 0.25),
    ("(0.25, 0.50]", 0.25, 0.50),
    (">0.50", 0.50, 1.0 + 1e-9),
)


def missingness_bin_index(frac: torch.Tensor) -> torch.Tensor:
    """Bucket each example's missing fraction into `MISSINGNESS_BINS` (returns
    the bin index; every example lands in exactly one bin)."""
    idx = torch.full_like(frac, -1, dtype=torch.long)
    for i, (_label, lo, hi) in enumerate(MISSINGNESS_BINS):
        in_bin = (frac > lo) & (frac <= hi)
        idx = torch.where(in_bin, torch.full_like(idx, i), idx)
    # frac == 0 exactly -> first bin (its lower edge is -1e-9)
    idx = torch.where(frac <= 1e-9, torch.zeros_like(idx), idx)
    return idx
