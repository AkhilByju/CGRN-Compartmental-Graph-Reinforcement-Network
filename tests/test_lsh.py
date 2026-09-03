import torch

from src.models.architecture_v1.lsh import (
    bucket_ids,
    gather_rows,
    gather_scalar,
    gather_vector,
    lookup_value,
    lsh_candidates,
    random_hyperplanes,
)


def test_bucket_ids_shape_and_range() -> None:
    g = torch.Generator().manual_seed(0)
    hp = random_hyperplanes(dim=8, num_hashes=3, bits=6, generator=g)
    z = torch.randn(2, 20, 8)
    buckets = bucket_ids(z, hp)
    assert buckets.shape == (2, 3, 20)
    assert (buckets >= 0).all()
    assert (buckets < 2**6).all()


def test_identical_vectors_hash_to_the_same_bucket() -> None:
    hp = random_hyperplanes(dim=8, num_hashes=2, bits=6)
    z = torch.randn(1, 5, 8)
    z[:, 1] = z[:, 0]  # cell 1 is an exact duplicate of cell 0
    buckets = bucket_ids(z, hp)
    assert torch.equal(buckets[:, :, 0], buckets[:, :, 1])


def test_lsh_candidates_shape() -> None:
    hp = random_hyperplanes(dim=8, num_hashes=2, bits=6)
    z = torch.randn(2, 30, 8)
    cand = lsh_candidates(z, z, hp, chunk_size=10, window=0)
    assert cand.shape == (2, 30, 2 * 1 * 10)
    assert (cand >= 0).all() and (cand < 30).all()


def test_lsh_candidates_with_window() -> None:
    hp = random_hyperplanes(dim=8, num_hashes=1, bits=4)
    z = torch.randn(1, 20, 8)
    cand = lsh_candidates(z, z, hp, chunk_size=5, window=1)
    assert cand.shape == (1, 20, 1 * 3 * 5)  # (2*window+1) = 3 chunks per hash


def test_gather_scalar_correctness() -> None:
    values = torch.arange(20).float().unsqueeze(0)  # (1, 20)
    idx = torch.tensor([[[0, 5, 19], [1, 2, 3]]])  # (1, 2, 3)
    gathered = gather_scalar(values, idx)
    assert torch.equal(gathered, torch.tensor([[[0.0, 5.0, 19.0], [1.0, 2.0, 3.0]]]))


def test_gather_vector_correctness() -> None:
    values = torch.arange(20 * 4).float().view(1, 20, 4)
    idx = torch.tensor([[[0, 5]]])  # (1, 1, 2)
    gathered = gather_vector(values, idx)
    assert torch.equal(gathered[0, 0, 0], values[0, 0])
    assert torch.equal(gathered[0, 0, 1], values[0, 5])


def test_gather_rows_finds_reciprocal_entries() -> None:
    # 3 cells, pool of 2 candidates each: cell 0 -> {1, 2}, cell 1 -> {0,
    # 2}, cell 2 -> {0, 1}.
    candidate_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])  # (1, 3, 2)
    weights = torch.tensor([[[0.3, 0.4], [0.7, 0.1], [0.2, 0.5]]])  # (1, 3, 2)

    rows_idx = gather_rows(candidate_idx, candidate_idx)  # for each (i,p)->j, j's own candidate row
    rows_weight = gather_rows(weights, candidate_idx)

    # (i=0, p=0) -> j=1; j=1's own candidate row is [0, 2] with weights [0.7, 0.1]
    assert torch.equal(rows_idx[0, 0, 0], torch.tensor([0, 2]))
    assert torch.allclose(rows_weight[0, 0, 0], torch.tensor([0.7, 0.1]))


def test_lookup_value_finds_and_defaults_to_zero() -> None:
    reference_idx = torch.tensor([[[2, 5, 9]]])  # (1, 1, 3)
    reference_value = torch.tensor([[[0.1, 0.2, 0.3]]])
    query_idx = torch.tensor([[[5, 7]]])  # 5 is in reference, 7 is not

    result = lookup_value(reference_idx, reference_value, query_idx)

    assert torch.allclose(result[0, 0, 0], torch.tensor(0.2))
    assert torch.allclose(result[0, 0, 1], torch.tensor(0.0))


def test_gradients_flow_through_gathers() -> None:
    hp = random_hyperplanes(dim=6, num_hashes=2, bits=5)
    z = torch.randn(2, 16, 6, requires_grad=True)
    mu = torch.randn(2, 16, requires_grad=True)

    cand = lsh_candidates(z.detach(), z.detach(), hp, chunk_size=6)
    mu_cand = gather_scalar(mu, cand)
    z_cand = gather_vector(z, cand)
    (mu_cand.sum() + z_cand.sum()).backward()

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(z.grad).all()
