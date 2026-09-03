import torch

from src.data.synthetic.dynamic_groups import (
    FEATURES_PER_OBJECT,
    N_OBJECTS,
    dynamic_groups,
    dynamic_groups_global,
    make_dynamic_groups_global_splits,
    make_dynamic_groups_splits,
)


def test_shapes() -> None:
    n_samples, n_objects = 16, N_OBJECTS
    x, y, group_id = dynamic_groups(n_samples, seed=0, n_objects=n_objects)
    assert x.shape == (n_samples, n_objects * FEATURES_PER_OBJECT)
    assert y.shape == (n_samples, 1)
    assert group_id.shape == (n_samples, n_objects)


def test_group_id_within_declared_range_and_every_group_nonempty() -> None:
    x, y, group_id = dynamic_groups(64, seed=1, k_min=2, k_max=5)
    for i in range(group_id.shape[0]):
        row = group_id[i]
        k = int(row.max().item()) + 1
        assert 2 <= k <= 5
        # every label from 0..k-1 must appear at least once (guaranteed by
        # construction: the first k objects, before shuffling, are one per group)
        assert set(row.tolist()) == set(range(k))


def test_deterministic_given_seed() -> None:
    x1, y1, g1 = dynamic_groups(20, seed=42)
    x2, y2, g2 = dynamic_groups(20, seed=42)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(g1, g2)


def test_different_seeds_differ() -> None:
    x1, _, _ = dynamic_groups(20, seed=1)
    x2, _, _ = dynamic_groups(20, seed=2)
    assert not torch.equal(x1, x2)


def test_target_equals_group_summary_formula_for_some_left_right_pair() -> None:
    # y = sum_k h_k + beta * h_left * h_right, where left/right are
    # whichever two (of the k <= 5) groups the generator picked by true
    # center x-coordinate -- not recoverable from x/group_id alone (the
    # true centers aren't returned), but the formula's *shape* is fully
    # checkable: y must equal h.sum() + beta*h[a]*h[b] for at least one
    # ordered pair (a, b) among the example's groups. Deterministic, no
    # noise-dependent flakiness.
    beta = 1.0
    x, y, group_id = dynamic_groups(30, seed=3, beta=beta)
    n_objects = group_id.shape[1]
    values = x.view(-1, n_objects, FEATURES_PER_OBJECT)[..., 2]

    for i in range(x.shape[0]):
        k = int(group_id[i].max().item()) + 1
        h = torch.stack(
            [torch.tanh(values[i][group_id[i] == group].sum()) for group in range(k)]
        )
        base = h.sum()
        candidates = [base + beta * h[a] * h[b] for a in range(k) for b in range(k) if a != b]
        assert any(torch.isclose(c, y[i, 0], atol=1e-5) for c in candidates)


def test_x_does_not_trivially_encode_group_id_via_object_order() -> None:
    # Objects are shuffled per example, so group membership isn't
    # recoverable from object *position* alone.
    _, _, group_id = dynamic_groups(50, seed=4)
    first_two_always_same_group = all(
        group_id[i, 0].item() == group_id[i, 1].item() for i in range(group_id.shape[0])
    )
    assert not first_two_always_same_group


def test_make_splits_partitions_correctly() -> None:
    splits = make_dynamic_groups_splits(n_train=10, n_val=5, n_test=5, seed=0)
    assert splits["train"][0].shape[0] == 10
    assert splits["val"][0].shape[0] == 5
    assert splits["test"][0].shape[0] == 5
    for x, y, group_id in splits.values():
        assert x.shape[1] == N_OBJECTS * FEATURES_PER_OBJECT
        assert y.shape[1] == 1
        assert group_id.shape[1] == N_OBJECTS


# --- dynamic_groups_global ---


def test_global_shapes() -> None:
    n_samples, n_objects = 16, N_OBJECTS
    x, y, group_id = dynamic_groups_global(n_samples, seed=0, n_objects=n_objects)
    assert x.shape == (n_samples, n_objects * FEATURES_PER_OBJECT)
    assert y.shape == (n_samples, 1)
    assert group_id.shape == (n_samples, n_objects)


def test_global_deterministic_given_seed() -> None:
    x1, y1, g1 = dynamic_groups_global(20, seed=42)
    x2, y2, g2 = dynamic_groups_global(20, seed=42)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(g1, g2)


def test_global_differs_from_spatial_pairing_dataset() -> None:
    # Same seed, same k range -- the *centers* and object shuffles are
    # sampled identically (same generator draw order for those), but the
    # target formula's pairing criterion differs, so y should differ.
    x1, y1, g1 = dynamic_groups(30, seed=7)
    x2, y2, g2 = dynamic_groups_global(30, seed=7)
    assert torch.equal(x1, x2)  # inputs/group assignment: identical draws
    assert torch.equal(g1, g2)
    assert not torch.equal(y1, y2)  # labels: different pairing criterion


def test_global_target_equals_argmax_argmin_h_formula() -> None:
    # Unlike dynamic_groups (whose left/right pairing depends on the true
    # centers, not returned), argmax/argmin of h is fully recoverable from
    # (x, group_id) alone -- an exact check, not an "any pair" check.
    beta = 1.0
    x, y, group_id = dynamic_groups_global(30, seed=3, beta=beta)
    n_objects = group_id.shape[1]
    values = x.view(-1, n_objects, FEATURES_PER_OBJECT)[..., 2]

    for i in range(x.shape[0]):
        k = int(group_id[i].max().item()) + 1
        h = torch.stack(
            [torch.tanh(values[i][group_id[i] == group].sum()) for group in range(k)]
        )
        expected = h.sum() + beta * h[torch.argmax(h)] * h[torch.argmin(h)]
        assert torch.isclose(expected, y[i, 0], atol=1e-5)


def test_global_make_splits_partitions_correctly() -> None:
    splits = make_dynamic_groups_global_splits(n_train=10, n_val=5, n_test=5, seed=0)
    assert splits["train"][0].shape[0] == 10
    assert splits["val"][0].shape[0] == 5
    assert splits["test"][0].shape[0] == 5
    for x, y, group_id in splits.values():
        assert x.shape[1] == N_OBJECTS * FEATURES_PER_OBJECT
        assert y.shape[1] == 1
        assert group_id.shape[1] == N_OBJECTS
