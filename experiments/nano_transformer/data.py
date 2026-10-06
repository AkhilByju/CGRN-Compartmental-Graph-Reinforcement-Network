"""TinyStories download, caching, and deterministic token-stream construction
(task Sec 2). One frozen tokenizer and one deterministic packed token stream
per split, shared by every model/seed/LR run -- "Do not allow examples to
differ between model families."

TinyStories ships official train/validation splits only (no test split).
Sec 10 needs a held-out validation set for LR selection *and* a separate
held-out test set for the single final evaluation. We derive both from the
official validation file by a fixed, deterministic partition (alternating
story index: even -> validation, odd -> test) -- both remain entirely held
out from training, and the split is recorded in the cached metadata so it is
never silently redone differently.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.nano_transformer.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    DOC_SEPARATOR,
    TrainedTokenizer,
    load_tokenizer,
    save_tokenizer,
    train_byte_level_bpe_tokenizer,
)

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "cache"
RAW_DIR = CACHE_DIR / "raw"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
TOKENS_DIR = CACHE_DIR / "tokens"

_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

_TOKEN_DTYPE = np.uint16  # vocab_size <= 512 fits comfortably


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 -- fixed, hardcoded HF dataset URL
    tmp.rename(dest)


def ensure_raw_downloaded() -> dict[str, Path]:
    """Downloads the official TinyStories train/valid text files if not
    already cached. Returns `{"train": path, "valid": path}`."""
    paths = {"train": RAW_DIR / "tinystories_train.txt", "valid": RAW_DIR / "tinystories_valid.txt"}
    if not paths["train"].exists():
        _download(_TRAIN_URL, paths["train"])
    if not paths["valid"].exists():
        _download(_VALID_URL, paths["valid"])
    return paths


def _split_stories(text: str) -> list[str]:
    stories = text.split(DOC_SEPARATOR)
    return [s for s in (story.strip() for story in stories) if s]


def split_validation_into_val_and_test(valid_text: str) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 split of the official validation file by story
    index parity -- even-indexed stories -> validation (used for LR
    selection during training), odd-indexed -> test (used once, at the end).
    Both are held out from training; neither depends on model, seed, or LR."""
    stories = _split_stories(valid_text)
    val = stories[0::2]
    test = stories[1::2]
    return val, test


def _encode_stories(tokenizer: TrainedTokenizer, stories: list[str]) -> np.ndarray:
    ids: list[int] = []
    for story in stories:
        ids.extend(tokenizer.encode(story))
        ids.append(tokenizer.doc_separator_id)
    return np.asarray(ids, dtype=_TOKEN_DTYPE)


@dataclass(frozen=True)
class DatasetBundle:
    tokenizer: TrainedTokenizer
    train_ids_path: Path
    val_ids_path: Path
    test_ids_path: Path
    meta: dict


def _meta_path() -> Path:
    return TOKENS_DIR / "meta.json"


def prepare_dataset(vocab_size: int = DEFAULT_VOCAB_SIZE, force: bool = False) -> DatasetBundle:
    """End-to-end, idempotent data pipeline: download (if needed) -> train
    tokenizer on train split only (if needed) -> encode train/val/test into
    cached `.npy` uint16 token-id arrays (if needed). Safe to call from every
    run; only does real work once per `(vocab_size,)` unless `force=True`."""
    raw = ensure_raw_downloaded()
    train_hash = _sha256_file(raw["train"])
    valid_hash = _sha256_file(raw["valid"])

    if not force and _meta_path().exists():
        cached_meta = json.loads(_meta_path().read_text())
        if (
            cached_meta.get("train_sha256") == train_hash
            and cached_meta.get("valid_sha256") == valid_hash
            and cached_meta.get("vocab_size") == vocab_size
            and (TOKENIZER_DIR / "tokenizer.json").exists()
            and Path(cached_meta["train_ids_path"]).exists()
        ):
            tokenizer = load_tokenizer(TOKENIZER_DIR)
            return DatasetBundle(
                tokenizer=tokenizer,
                train_ids_path=Path(cached_meta["train_ids_path"]),
                val_ids_path=Path(cached_meta["val_ids_path"]),
                test_ids_path=Path(cached_meta["test_ids_path"]),
                meta=cached_meta,
            )

    if not force and (TOKENIZER_DIR / "tokenizer.json").exists():
        tokenizer = load_tokenizer(TOKENIZER_DIR)
    else:
        tokenizer = train_byte_level_bpe_tokenizer(raw["train"], vocab_size=vocab_size)
        save_tokenizer(
            tokenizer,
            TOKENIZER_DIR,
            extra_meta={"train_sha256": train_hash, "train_file": str(raw["train"])},
        )

    valid_text = raw["valid"].read_text()
    val_stories, test_stories = split_validation_into_val_and_test(valid_text)

    train_text = raw["train"].read_text()
    train_stories = _split_stories(train_text)

    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    train_ids = _encode_stories(tokenizer, train_stories)
    val_ids = _encode_stories(tokenizer, val_stories)
    test_ids = _encode_stories(tokenizer, test_stories)

    train_ids_path = TOKENS_DIR / "train_ids.npy"
    val_ids_path = TOKENS_DIR / "val_ids.npy"
    test_ids_path = TOKENS_DIR / "test_ids.npy"
    np.save(train_ids_path, train_ids)
    np.save(val_ids_path, val_ids)
    np.save(test_ids_path, test_ids)

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "train_sha256": train_hash,
        "valid_sha256": valid_hash,
        "train_tokens": int(train_ids.shape[0]),
        "val_tokens": int(val_ids.shape[0]),
        "test_tokens": int(test_ids.shape[0]),
        "train_stories": len(train_stories),
        "val_stories": len(val_stories),
        "test_stories": len(test_stories),
        "train_ids_path": str(train_ids_path),
        "val_ids_path": str(val_ids_path),
        "test_ids_path": str(test_ids_path),
        "doc_separator_id": tokenizer.doc_separator_id,
    }
    _meta_path().write_text(json.dumps(meta, indent=2, sort_keys=True))

    return DatasetBundle(
        tokenizer=tokenizer,
        train_ids_path=train_ids_path,
        val_ids_path=val_ids_path,
        test_ids_path=test_ids_path,
        meta=meta,
    )


def load_token_ids(path: str | Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def iterate_batches(
    ids: np.ndarray,
    batch_size: int,
    context_length: int,
    start_block: int = 0,
):
    """Deterministic, non-shuffled, sequential batch iterator: walks
    non-overlapping `context_length + 1` windows through `ids` in a fixed
    order (never depends on model or seed -- task Sec 2/16), yields
    `(x, y)` numpy int64 arrays of shape `[batch_size, context_length]`
    (`y` is `x` shifted by one token), and wraps around to the start of
    `ids` when exhausted (never happens within this experiment's 30M-token
    budget for the sizes TinyStories provides, but is exact and deterministic
    if it ever did). `start_block` resumes deterministically from a given
    block offset instead of the beginning.
    """
    stride = context_length + 1
    if len(ids) < stride:
        raise ValueError("token stream too short for even one context-length block")
    n_blocks = (len(ids) - stride) // context_length + 1

    block = start_block % n_blocks
    while True:
        batch_x = np.empty((batch_size, context_length), dtype=np.int64)
        batch_y = np.empty((batch_size, context_length), dtype=np.int64)
        for row in range(batch_size):
            start = block * context_length
            window = ids[start : start + stride]
            batch_x[row] = window[:-1]
            batch_y[row] = window[1:]
            block = (block + 1) % n_blocks
        yield batch_x, batch_y
