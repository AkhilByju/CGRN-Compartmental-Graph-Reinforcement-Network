from experiments.nano_transformer.tokenizer import (
    DOC_SEPARATOR,
    load_tokenizer,
    save_tokenizer,
    train_byte_level_bpe_tokenizer,
)

_SAMPLE_TEXT = (
    "Once upon a time there was a little dog named Spot. "
    "Spot liked to play in the park with his friend Kitty.\n"
    f"{DOC_SEPARATOR}\n"
    "Kitty and Spot found a shiny red ball. They played all day and became "
    "best friends forever.\n"
)


def test_train_byte_level_bpe_reaches_requested_vocab_size(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text(_SAMPLE_TEXT * 20)  # repeat so BPE has merges to find

    trained = train_byte_level_bpe_tokenizer(text_path, vocab_size=280)
    assert trained.vocab_size <= 280
    assert trained.doc_separator_id is not None


def test_encode_decode_roundtrip_and_persistence(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text(_SAMPLE_TEXT * 20)
    trained = train_byte_level_bpe_tokenizer(text_path, vocab_size=280)

    sample = "Spot and Kitty played with the ball."
    ids = trained.encode(sample)
    assert trained.decode(ids) == sample

    out_dir = tmp_path / "tok"
    save_tokenizer(trained, out_dir, extra_meta={"train_sha256": "deadbeef"})
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "merges.txt").exists()
    assert (out_dir / "config.json").exists()

    reloaded = load_tokenizer(out_dir)
    assert reloaded.vocab_size == trained.vocab_size
    assert reloaded.doc_separator_id == trained.doc_separator_id
    assert reloaded.encode(sample) == ids
    assert reloaded.decode(ids) == sample


def test_doc_separator_encodes_to_exactly_its_reserved_id(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text(_SAMPLE_TEXT * 20)
    trained = train_byte_level_bpe_tokenizer(text_path, vocab_size=280)
    ids = trained.encode(DOC_SEPARATOR)
    assert ids == [trained.doc_separator_id]
