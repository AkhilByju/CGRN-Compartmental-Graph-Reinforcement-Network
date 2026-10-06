"""Architecture V2 -- `DendriticConnectivity` (docs/architecture_v2.md Sec C).

Both construction modes (`balanced_random`, `local_2d`) must produce fixed,
non-trainable, deterministic source indices with exactly `K` unique sources
per branch, and `gather`/`gather_scalar` must never materialize a dense
`[batch, H, input_dim]` edge tensor.
"""

from __future__ import annotations

import torch

from src.models.architecture_v2.belief_dendrite import DendriticConnectivity, make_connectivity_pair

# ---------------------------------------------------------------------------
# balanced_random
# ---------------------------------------------------------------------------


def test_balanced_random_is_deterministic_for_a_fixed_seed() -> None:
    a = DendriticConnectivity.balanced_random(
        input_dim=40, num_somas=6, branches_per_soma=4, sources_per_branch=7, seed=3
    )
    b = DendriticConnectivity.balanced_random(
        input_dim=40, num_somas=6, branches_per_soma=4, sources_per_branch=7, seed=3
    )
    assert torch.equal(a.source_idx, b.source_idx)


def test_balanced_random_different_seeds_differ() -> None:
    a = DendriticConnectivity.balanced_random(
        input_dim=40, num_somas=6, branches_per_soma=4, sources_per_branch=7, seed=1
    )
    b = DendriticConnectivity.balanced_random(
        input_dim=40, num_somas=6, branches_per_soma=4, sources_per_branch=7, seed=2
    )
    assert not torch.equal(a.source_idx, b.source_idx)


def test_balanced_random_no_repeated_source_within_a_branch() -> None:
    conn = DendriticConnectivity.balanced_random(
        input_dim=17, num_somas=9, branches_per_soma=5, sources_per_branch=11, seed=7
    )
    for row in conn.source_idx:
        assert len(set(row.tolist())) == row.numel()


def test_balanced_random_indices_in_range() -> None:
    conn = DendriticConnectivity.balanced_random(
        input_dim=23, num_somas=4, branches_per_soma=3, sources_per_branch=9, seed=0
    )
    assert int(conn.source_idx.min()) >= 0
    assert int(conn.source_idx.max()) < 23


def test_balanced_random_coverage_is_approximately_balanced() -> None:
    input_dim, h, b, k = 30, 40, 4, 12  # M*K = 1920, ~64 draws/source on average
    conn = DendriticConnectivity.balanced_random(
        input_dim=input_dim, num_somas=h, branches_per_soma=b, sources_per_branch=k, seed=5
    )
    counts = torch.bincount(conn.source_idx.reshape(-1), minlength=input_dim).float()
    mean = counts.mean()
    # Every source is drawn from repeated full-range decks; imbalance should
    # stay within a small band around the mean, not the wild variance a
    # uniform-iid draw would produce.
    assert (counts - mean).abs().max() <= max(3.0, 0.15 * mean)


def test_balanced_random_no_dense_mask_materialized_for_a_huge_input_dim() -> None:
    # A [batch, H, input_dim] dense mask at these sizes would be tens of GB;
    # a [batch, H*B, K] gather stays trivially small and fast. This is a
    # practical proxy for "no dense edge tensor is ever created" (the spec's
    # suggested TorchDispatch check) that doesn't depend on profiler internals.
    import time

    input_dim, h, b, k, batch = 2_000_000, 20, 4, 16, 16
    conn = DendriticConnectivity.balanced_random(
        input_dim=input_dim, num_somas=h, branches_per_soma=b, sources_per_branch=k, seed=0
    )
    mu = torch.randn(batch, input_dim)
    pi = torch.rand(batch, input_dim)
    t0 = time.perf_counter()
    mu_g, pi_g = conn.gather(mu, pi)
    elapsed = time.perf_counter() - t0
    assert mu_g.shape == (batch, h * b, k)
    assert pi_g.shape == (batch, h * b, k)
    assert elapsed < 2.0  # would be seconds-to-OOM if a dense mask were built


# ---------------------------------------------------------------------------
# local_2d
# ---------------------------------------------------------------------------


def test_local_2d_indices_never_leave_the_image() -> None:
    conn = DendriticConnectivity.local_2d(
        channels=3, height=16, width=12, num_somas=10, branches_per_soma=4, patch=5, seed=2
    )
    assert int(conn.source_idx.min()) >= 0
    assert int(conn.source_idx.max()) < 3 * 16 * 12
    assert conn.K == 3 * 5 * 5


def test_local_2d_branches_are_spatially_contiguous_per_channel() -> None:
    channels, height, width, patch = 2, 10, 14, 4
    conn = DendriticConnectivity.local_2d(
        channels=channels,
        height=height,
        width=width,
        num_somas=6,
        branches_per_soma=3,
        patch=patch,
        seed=4,
    )
    row0, col0 = conn.meta["row0"], conn.meta["col0"]
    for m in range(conn.M):
        idx = conn.source_idx[m].tolist()
        for c in range(channels):
            channel_idx = [
                i - c * height * width
                for i in idx
                if c * height * width <= i < (c + 1) * height * width
            ]
            rows = sorted({i // width for i in channel_idx})
            cols = sorted({i % width for i in channel_idx})
            assert rows == list(range(int(row0[m]), int(row0[m]) + patch))
            assert cols == list(range(int(col0[m]), int(col0[m]) + patch))


def test_local_2d_deterministic_for_a_fixed_seed() -> None:
    a = DendriticConnectivity.local_2d(
        channels=1, height=28, width=28, num_somas=8, branches_per_soma=4, patch=7, seed=9
    )
    b = DendriticConnectivity.local_2d(
        channels=1, height=28, width=28, num_somas=8, branches_per_soma=4, patch=7, seed=9
    )
    assert torch.equal(a.source_idx, b.source_idx)


def test_local_2d_coverage_is_spatially_diverse() -> None:
    conn = DendriticConnectivity.local_2d(
        channels=1, height=28, width=28, num_somas=32, branches_per_soma=4, patch=7, seed=1
    )
    row0, col0 = conn.meta["row0"], conn.meta["col0"]
    # not every branch centered at the same spot
    assert row0.unique().numel() > 1
    assert col0.unique().numel() > 1


def test_no_dense_structural_mask_created_during_forward() -> None:
    # Mirrors the balanced_random proxy for the local_2d path, at MNIST-scale
    # dims where a dense (batch, H, 784) mask would still be tiny, so this
    # checks output shape correctness instead of timing.
    conn = DendriticConnectivity.local_2d(
        channels=1, height=28, width=28, num_somas=16, branches_per_soma=4, patch=7, seed=0
    )
    mu = torch.randn(5, 784)
    pi = torch.rand(5, 784)
    mu_g, pi_g = conn.gather(mu, pi)
    assert mu_g.shape == (5, 64, 49)
    assert pi_g.shape == (5, 64, 49)


# ---------------------------------------------------------------------------
# make_connectivity_pair
# ---------------------------------------------------------------------------


def test_make_connectivity_pair_vector_mode_uses_balanced_random_twice() -> None:
    c1, c2 = make_connectivity_pair(
        in_features=50, hidden1=12, hidden2=8, branches_per_soma=4, seed=0
    )
    assert c1.mode == "balanced_random"
    assert c2.mode == "balanced_random"
    assert c2.input_dim == 12


def test_make_connectivity_pair_image_mode_uses_local_2d_then_balanced_random() -> None:
    c1, c2 = make_connectivity_pair(
        in_features=784,
        hidden1=16,
        hidden2=10,
        branches_per_soma=4,
        seed=0,
        image_shape=(1, 28, 28),
        patch=7,
    )
    assert c1.mode == "local_2d"
    assert c2.mode == "balanced_random"
    assert c2.input_dim == 16
