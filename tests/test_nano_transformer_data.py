import numpy as np

from experiments.nano_transformer.data import (
    iterate_batches,
    split_validation_into_val_and_test,
)
from experiments.nano_transformer.tokenizer import DOC_SEPARATOR


def test_split_validation_into_val_and_test_is_disjoint_and_deterministic():
    text = DOC_SEPARATOR.join(f" story {i} " for i in range(10))
    val_a, test_a = split_validation_into_val_and_test(text)
    val_b, test_b = split_validation_into_val_and_test(text)

    assert val_a == val_b
    assert test_a == test_b
    assert set(val_a).isdisjoint(set(test_a))
    assert len(val_a) + len(test_a) == 10
    assert val_a[0] == "story 0"
    assert test_a[0] == "story 1"


def test_iterate_batches_shapes_and_shift_by_one():
    ids = np.arange(0, 2000, dtype=np.uint16)
    gen = iterate_batches(ids, batch_size=4, context_length=16)
    batch_x, batch_y = next(gen)
    assert batch_x.shape == (4, 16)
    assert batch_y.shape == (4, 16)
    # y is x shifted by exactly one position within each contiguous block.
    assert np.array_equal(batch_x[0][1:], batch_y[0][:-1])
    assert np.array_equal(batch_y[0], batch_x[0] + 1)


def test_iterate_batches_is_deterministic_across_independent_generators():
    ids = np.arange(0, 5000, dtype=np.uint16)
    gen_a = iterate_batches(ids, batch_size=8, context_length=32)
    gen_b = iterate_batches(ids, batch_size=8, context_length=32)

    for _ in range(5):
        xa, ya = next(gen_a)
        xb, yb = next(gen_b)
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, yb)


def test_iterate_batches_does_not_depend_on_any_model_or_seed_concept():
    """The task requires identical example ordering across model families --
    iterate_batches takes no model/seed argument at all, so two calls with
    the same ids/batch_size/context_length are byte-identical by
    construction (checked directly, not just asserted by API shape)."""
    ids = np.arange(0, 3000, dtype=np.uint16)
    seqs = []
    for _ in range(3):  # simulating "3 different models" reading the same stream
        gen = iterate_batches(ids, batch_size=4, context_length=8)
        seqs.append([next(gen) for _ in range(4)])

    for other in seqs[1:]:
        for (xa, ya), (xb, yb) in zip(seqs[0], other, strict=True):
            assert np.array_equal(xa, xb)
            assert np.array_equal(ya, yb)


def test_iterate_batches_wraps_around_deterministically():
    ids = np.arange(0, 100, dtype=np.uint16)  # small stream forces wraparound
    gen = iterate_batches(ids, batch_size=2, context_length=16)
    seen = [next(gen) for _ in range(10)]
    # After wraparound, the same block index must reproduce the same window.
    first_x, _ = seen[0]
    later_matches = [np.array_equal(x, first_x) for x, _ in seen[1:]]
    assert any(later_matches)  # wraparound must recur since the stream is short


def test_iterate_batches_rejects_stream_shorter_than_one_block():
    import pytest

    ids = np.arange(0, 5, dtype=np.uint16)
    with pytest.raises(ValueError):
        next(iterate_batches(ids, batch_size=2, context_length=16))
