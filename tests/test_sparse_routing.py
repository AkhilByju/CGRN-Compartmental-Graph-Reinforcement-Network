import torch

from src.models.architecture_v1.sparse_routing import SparseGlobalRouting, SparseLocalAssociation


def _unit_z(batch: int, n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    z = torch.randn(batch, n, d, generator=generator)
    return z / z.norm(dim=-1, keepdim=True)


def test_local_shapes_and_pool_size() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, d = 2, 30, 6
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=10, window=0)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local, candidate_idx = local(mu, u, z)

    assert a_local.shape == (batch, n, 20)  # 2 hashes * 1 chunk * chunk_size 10
    assert candidate_idx.shape == a_local.shape
    assert (a_local >= 0).all()
    assert torch.isfinite(a_local).all()
    assert (candidate_idx >= 0).all() and (candidate_idx < n).all()


def test_local_excludes_self() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, d = 2, 20, 4
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=8)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local, candidate_idx = local(mu, u, z)

    self_idx = torch.arange(n).view(1, n, 1)
    self_positions = candidate_idx == self_idx
    # wherever a candidate slot happens to be the cell itself, its weight must be 0
    assert torch.allclose(a_local[self_positions], torch.zeros_like(a_local[self_positions]))


def test_local_is_sparse() -> None:
    g = torch.Generator().manual_seed(2)
    batch, n, d = 2, 40, 6
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=10)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local, _ = local(mu, u, z)
    frac_zero = (a_local == 0).float().mean().item()
    assert frac_zero > 0.1  # loose bound; some mass routinely goes to null/non-reciprocal


def test_gradients_are_finite_through_local() -> None:
    g = torch.Generator().manual_seed(3)
    batch, n, d = 2, 20, 4
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=8)
    mu = torch.randn(batch, n, generator=g, requires_grad=True)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g).requires_grad_(True)

    a_local, _ = local(mu, u, z)
    a_local.sum().backward()

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(z.grad).all()
    for p in local.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_global_shapes_and_gradients() -> None:
    g = torch.Generator().manual_seed(4)
    batch, n, d = 2, 30, 6
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=10)
    glob = SparseGlobalRouting(association_dim=d, num_hashes=2, chunk_size=4)
    mu = torch.randn(batch, n, generator=g, requires_grad=True)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g).requires_grad_(True)
    need = torch.rand(batch, n, generator=g)
    offer = torch.rand(batch, n, generator=g)

    a_local, local_idx = local(mu, u, z)
    a_global, global_idx = glob(z, need, offer, local_idx, a_local)

    assert a_global.shape == (batch, n, 8)  # 2 hashes * 1 chunk * chunk_size 4
    assert global_idx.shape == a_global.shape
    assert torch.isfinite(a_global).all()

    a_global.sum().backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(z.grad).all()
    for p in glob.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_global_excludes_self() -> None:
    g = torch.Generator().manual_seed(5)
    batch, n, d = 2, 24, 4
    local = SparseLocalAssociation(association_dim=d, num_hashes=2, chunk_size=8)
    glob = SparseGlobalRouting(association_dim=d, num_hashes=2, chunk_size=4)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)
    need = torch.rand(batch, n, generator=g)
    offer = torch.rand(batch, n, generator=g)

    a_local, local_idx = local(mu, u, z)
    a_global, global_idx = glob(z, need, offer, local_idx, a_local)

    self_idx = torch.arange(n).view(1, n, 1)
    self_positions = global_idx == self_idx
    assert torch.allclose(a_global[self_positions], torch.zeros_like(a_global[self_positions]))
