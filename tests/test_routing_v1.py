import torch

from src.models.architecture_v1.routing import GlobalRouting, LocalAssociation


def _unit_z(batch: int, n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    z = torch.randn(batch, n, d, generator=generator)
    return z / z.norm(dim=-1, keepdim=True)


def test_local_association_is_symmetric() -> None:
    g = torch.Generator().manual_seed(0)
    batch, n, d = 2, 12, 4
    local = LocalAssociation(association_dim=d)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local = local(mu, u, z)

    assert torch.allclose(a_local, a_local.transpose(-1, -2), atol=1e-6)


def test_local_association_has_zero_diagonal() -> None:
    g = torch.Generator().manual_seed(1)
    batch, n, d = 2, 10, 4
    local = LocalAssociation(association_dim=d)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local = local(mu, u, z)

    diag = a_local.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))


def test_local_association_is_sparse() -> None:
    # With more than a handful of cells, sparsemax's exact-zero property
    # should mean most pairs end up with no association at all -- this is
    # the whole point of using it instead of softmax (docs/architecture_v1.md
    # Part I, "That's exactly what we DON'T want").
    g = torch.Generator().manual_seed(2)
    batch, n, d = 2, 30, 4
    local = LocalAssociation(association_dim=d)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)

    a_local = local(mu, u, z)

    frac_zero = (a_local == 0).float().mean().item()
    assert frac_zero > 0.3  # loose bound; exact fraction depends on random init


def test_identical_cells_maximize_local_association_relative_to_distant_ones() -> None:
    # Two cells with identical (mu, u, z) should end up more strongly
    # associated than either is with a cell whose z points a very different
    # direction and whose mu strongly disagrees.
    d = 4
    local = LocalAssociation(association_dim=d, lambda_=1.0, tau=1.0)
    z_a = torch.zeros(1, d)
    z_a[0, 0] = 1.0
    z_far = torch.zeros(1, d)
    z_far[0, 1] = 1.0

    z = torch.cat([z_a, z_a.clone(), z_far], dim=0).unsqueeze(0)  # (1, 3, d)
    mu = torch.tensor([[0.5, 0.5, -0.9]])
    u = torch.full((1, 3), 0.2)

    a_local = local(mu, u, z)
    assert a_local[0, 0, 1].item() >= a_local[0, 0, 2].item()


def test_global_routing_has_zero_diagonal_and_is_generally_directed() -> None:
    g = torch.Generator().manual_seed(3)
    batch, n, d = 2, 12, 4
    local = LocalAssociation(association_dim=d)
    routing = GlobalRouting(association_dim=d)
    mu = torch.randn(batch, n, generator=g)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g)
    need = torch.rand(batch, n, generator=g)
    offer = torch.rand(batch, n, generator=g)

    a_local = local(mu, u, z)
    a_global = routing(z, need, offer, a_local)

    diag = a_global.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))
    assert not torch.allclose(a_global, a_global.transpose(-1, -2), atol=1e-6)


def test_global_routing_penalizes_existing_local_edges() -> None:
    # Two cells that are already maximally locally associated should route
    # to each other globally less than two cells with the same content
    # compatibility but no local edge, all else equal (the `-gamma * A^L`
    # term in docs/architecture_v1.md Part III Step 3).
    d = 4
    routing = GlobalRouting(association_dim=d, gamma=5.0)
    z = torch.randn(1, 3, d)
    need = torch.tensor([[0.9, 0.5, 0.5]])
    offer = torch.tensor([[0.5, 0.9, 0.9]])

    a_local_high = torch.zeros(1, 3, 3)
    a_local_high[0, 0, 1] = 1.0
    a_local_high[0, 1, 0] = 1.0
    a_local_low = torch.zeros(1, 3, 3)

    a_global_high_local = routing(z, need, offer, a_local_high)
    a_global_low_local = routing(z, need, offer, a_local_low)

    assert a_global_high_local[0, 0, 1].item() <= a_global_low_local[0, 0, 1].item()


def test_gradients_are_finite_through_both_routing_modules() -> None:
    g = torch.Generator().manual_seed(4)
    batch, n, d = 2, 16, 4
    local = LocalAssociation(association_dim=d)
    routing = GlobalRouting(association_dim=d)
    mu = torch.randn(batch, n, generator=g, requires_grad=True)
    u = torch.rand(batch, n, generator=g) + 0.1
    z = _unit_z(batch, n, d, g).requires_grad_(True)
    need = torch.rand(batch, n, generator=g)
    offer = torch.rand(batch, n, generator=g)

    a_local = local(mu, u, z)
    a_global = routing(z, need, offer, a_local)
    (a_local.sum() + a_global.sum()).backward()

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(z.grad).all()
    for p in list(local.parameters()) + list(routing.parameters()):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
