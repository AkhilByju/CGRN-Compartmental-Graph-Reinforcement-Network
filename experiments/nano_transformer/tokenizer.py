"""Byte-level BPE tokenizer training (task Sec 2): one frozen tokenizer,
trained once on the training split only, vocab size 512, reused by every
model/run in this experiment. Generic infra (tokenizer training is not an
architecture decision) -- CLAUDE.md Sec 1's allowances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

DOC_SEPARATOR = "<|endoftext|>"
DEFAULT_VOCAB_SIZE = 512


@dataclass(frozen=True)
class TrainedTokenizer:
    tokenizer: Tokenizer
    vocab_size: int
    doc_separator_id: int

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


def train_byte_level_bpe_tokenizer(
    train_text_path: str | Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> TrainedTokenizer:
    """Train a byte-level BPE tokenizer on `train_text_path` only (task Sec 2:
    "Train a tokenizer on the TRAINING data only"). One special token,
    `<|endoftext|>`, matching TinyStories' own document-separator convention
    -- "standard special tokens only as required"."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[DOC_SEPARATOR],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(train_text_path)], trainer)

    doc_sep_id = tokenizer.token_to_id(DOC_SEPARATOR)
    if doc_sep_id is None:
        raise RuntimeError(f"{DOC_SEPARATOR!r} missing from trained vocabulary")

    return TrainedTokenizer(
        tokenizer=tokenizer, vocab_size=tokenizer.get_vocab_size(), doc_separator_id=doc_sep_id
    )


def save_tokenizer(trained: TrainedTokenizer, output_dir: str | Path, extra_meta: dict) -> None:
    """Saves the full tokenizer.json plus a human-inspectable vocab.json /
    merges.txt pair and a config.json recording vocab size, doc-separator id,
    and whatever provenance metadata the caller supplies (training-data hash,
    dataset version -- task Sec 2)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained.tokenizer.save(str(output_dir / "tokenizer.json"))
    trained.tokenizer.model.save(str(output_dir))  # vocab.json + merges.txt

    config = {
        "vocab_size": trained.vocab_size,
        "doc_separator": DOC_SEPARATOR,
        "doc_separator_id": trained.doc_separator_id,
        **extra_meta,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def load_tokenizer(output_dir: str | Path) -> TrainedTokenizer:
    output_dir = Path(output_dir)
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    config = json.loads((output_dir / "config.json").read_text())
    return TrainedTokenizer(
        tokenizer=tokenizer,
        vocab_size=config["vocab_size"],
        doc_separator_id=config["doc_separator_id"],
    )
