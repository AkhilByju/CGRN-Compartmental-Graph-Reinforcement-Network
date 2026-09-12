"""CIFAR-10 loading, deterministic stratified splitting, and train-only
channel standardization for the strong-baselines benchmark (spec Sec 2).

CIFAR-10 is fetched from its canonical source
(`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` -- the same
distribution `torchvision.datasets.CIFAR10` downloads; this project has no
torchvision dependency, so the pickle-format tarball is read directly) and
cached under `~/.cache/cgrn_data/`, alongside this project's other
externally-fetched datasets (`experiments/paper_a/datasets.py`'s
`fetch_openml`/`fetch_california_housing` caches use the same scikit-learn
data-home directory). An OpenML mirror (`CIFAR_10`, data id 40927) was tried
first and rejected: two independent downloads both failed OpenML's own MD5
check with two different (wrong) checksums, indicating a server-side/CDN
issue with that mirror, not a transient network blip -- switching sources
rather than retrying a corrupt one.

Official split: the 50,000-row training batches vs. the 10,000-row test
batch, untouched. Sec 2's `45,000 train / 5,000 validation` carve-out is a
seeded stratified partition of the official training set (one seed per
`{0, 1, 2}` -- three distinct 45k/5k partitions, not three shuffles of the
same one). Standardization uses per-channel training-subset mean/std only
(Sec 2), applied identically to val/test -- no leakage.
"""

from __future__ import annotations

import io
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

IMAGE_SHAPE: tuple[int, int, int] = (3, 32, 32)
N_CLASSES = 10
N_OFFICIAL_TRAIN = 50_000
N_OFFICIAL_TEST = 10_000
N_VAL = 5_000
N_TRAIN = N_OFFICIAL_TRAIN - N_VAL  # 45,000, spec Sec 2

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CACHE_DIR = Path.home() / ".cache" / "cgrn_data"
_TARBALL_PATH = _CACHE_DIR / "cifar-10-python.tar.gz"

_TRAIN_BATCH_NAMES = tuple(f"data_batch_{i}" for i in range(1, 6))
_TEST_BATCH_NAME = "test_batch"


def ensure_cifar10_tarball(dest: Path = _TARBALL_PATH, url: str = _CIFAR10_URL) -> Path:
    """Downloads the canonical CIFAR-10 tarball if not already cached.
    Idempotent -- a second call with an already-populated cache is a no-op."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 -- fixed, well-known dataset URL
    return dest


def _unpickle_member(tar: tarfile.TarFile, member_suffix: str) -> dict[str, Any]:
    member = next(m for m in tar.getmembers() if m.name.endswith(member_suffix))
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise RuntimeError(f"could not extract {member_suffix} from the CIFAR-10 tarball")
    return pickle.load(io.BytesIO(fileobj.read()), encoding="bytes")


def load_cifar10_raw(tarball_path: Path = _TARBALL_PATH) -> dict[str, np.ndarray]:
    """Returns `{"train_images": uint8 (50000,3,32,32), "train_labels": int64
    (50000,), "test_images": uint8 (10000,3,32,32), "test_labels": int64
    (10000,)}`. Each batch's raw `data` row is `3072` bytes -- 1024 red, then
    1024 green, then 1024 blue, each row-major over the 32x32 grid -- so
    `.reshape(-1, 3, 32, 32)` is exactly the channel-major layout this
    benchmark's `IMAGE_SHAPE` and `DendriticConnectivity.local_2d` already
    assume; no transpose needed."""
    ensure_cifar10_tarball(tarball_path)
    with tarfile.open(tarball_path, mode="r:gz") as tar:
        train_images_list, train_labels_list = [], []
        for name in _TRAIN_BATCH_NAMES:
            batch = _unpickle_member(tar, name)
            train_images_list.append(np.asarray(batch[b"data"], dtype=np.uint8))
            train_labels_list.append(np.asarray(batch[b"labels"], dtype=np.int64))
        test_batch = _unpickle_member(tar, _TEST_BATCH_NAME)
        test_images = np.asarray(test_batch[b"data"], dtype=np.uint8)
        test_labels = np.asarray(test_batch[b"labels"], dtype=np.int64)

    train_images = np.concatenate(train_images_list, axis=0).reshape(-1, 3, 32, 32)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_images = test_images.reshape(-1, 3, 32, 32)

    if train_images.shape[0] != N_OFFICIAL_TRAIN or test_images.shape[0] != N_OFFICIAL_TEST:
        raise RuntimeError(
            f"unexpected CIFAR-10 row counts: train={train_images.shape[0]}, "
            f"test={test_images.shape[0]} (expected {N_OFFICIAL_TRAIN}/{N_OFFICIAL_TEST})"
        )
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


@dataclass
class PreparedImageDataset:
    name: str
    seed: int

    x_train: torch.Tensor  # (N_TRAIN, C, H, W), standardized float32
    y_train: torch.Tensor  # (N_TRAIN,), int64
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    channel_mean: torch.Tensor  # (C,)
    channel_std: torch.Tensor  # (C,)
    image_shape: tuple[int, int, int] = IMAGE_SHAPE
    n_classes: int = N_CLASSES
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def in_features(self) -> int:
        c, h, w = self.image_shape
        return c * h * w


def stratified_train_val_split(
    labels: np.ndarray, seed: int, n_val: int = N_VAL
) -> dict[str, np.ndarray]:
    """Deterministic stratified `(train_idx, val_idx)` partition of the
    official 50k training set, one distinct partition per `seed` -- exposed
    separately from `prepare_cifar10` so split disjointness/determinism can
    be checked without materializing image tensors."""
    idx = np.arange(labels.shape[0])
    train_idx, val_idx = train_test_split(
        idx, test_size=n_val, random_state=seed, stratify=labels
    )
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    assert set(train_idx).isdisjoint(val_idx), "train/val leakage in the CIFAR-10 split"
    return {"train": train_idx, "val": val_idx}


def _standardize_per_channel(
    train_x: torch.Tensor, *other_xs: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Per-channel mean/std from `train_x` (`(N, C, H, W)`) only, applied to
    every given tensor -- spec Sec 2: "Standardize using training-set
    channel statistics only."""
    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    std = train_x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    out = tuple((x - mean) / std for x in (train_x, *other_xs))
    return out + (mean.reshape(-1), std.reshape(-1))


def prepare_cifar10(seed: int, tarball_path: Path = _TARBALL_PATH) -> PreparedImageDataset:
    """Loads CIFAR-10, applies the seeded 45k/5k stratified split, and
    standardizes with training-subset-only per-channel statistics. Pixel
    values are rescaled to `[0, 1]` before standardization (the conventional
    starting point; the mean/std themselves are still computed on this
    project's own training subset, not ImageNet constants)."""
    raw = load_cifar10_raw(tarball_path)
    split = stratified_train_val_split(raw["train_labels"], seed)
    train_idx, val_idx = split["train"], split["val"]

    to_float = lambda arr: torch.tensor(arr, dtype=torch.float32) / 255.0  # noqa: E731
    x_train_raw = to_float(raw["train_images"][train_idx])
    x_val_raw = to_float(raw["train_images"][val_idx])
    x_test_raw = to_float(raw["test_images"])

    x_train, x_val, x_test, mean, std = _standardize_per_channel(x_train_raw, x_val_raw, x_test_raw)

    y_train = torch.tensor(raw["train_labels"][train_idx], dtype=torch.long)
    y_val = torch.tensor(raw["train_labels"][val_idx], dtype=torch.long)
    y_test = torch.tensor(raw["test_labels"], dtype=torch.long)

    return PreparedImageDataset(
        name="cifar10",
        seed=seed,
        x_train=x_train.contiguous(),
        y_train=y_train,
        x_val=x_val.contiguous(),
        y_val=y_val,
        x_test=x_test.contiguous(),
        y_test=y_test,
        channel_mean=mean,
        channel_std=std,
        meta={
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "official_split": True,
        },
    )
