import math

import torch

from src.models.architecture_v1.random_features import RandomFourierFeatures, exact_gaussian_kernel


def test_shape() -> None:
    rff = RandomFourierFeatures(dim=6, num_features=32)
    x = torch.randn(3, 10, 6)
    out = rff(x)
    assert out.shape == (3, 10, 32)


def test_no_overflow_for_large_inputs() -> None:
    # cos() never overflows, unlike PositiveRandomFeatures' exp() -- no
    # stabilization needed, confirmed directly with large-norm inputs.
    rff = RandomFourierFeatures(dim=4, num_features=16)
    x = torch.randn(2, 5, 4) * 1000
    out = rff(x)
    assert torch.isfinite(out).all()
    assert (out.abs() <= math.sqrt(2.0 / 16) + 1e-6).all()


def test_orthogonal_directions_are_orthogonal_within_a_block() -> None:
    from src.models.architecture_v1.random_features import _orthogonal_directions

    dim = 6
    w = _orthogonal_directions(dim, num_features=dim, generator=torch.Generator().manual_seed(0))
    # first `dim` columns form one QR block -- unit directions must be mutually orthogonal
    unit = w / w.norm(dim=0, keepdim=True)
    gram = unit.T @ unit
    off_diag = gram - torch.diag(torch.diagonal(gram))
    assert off_diag.abs().max().item() < 1e-5


def test_orthogonal_and_iid_both_unbiased_on_average() -> None:
    # Both should approximate the same kernel in expectation -- check the
    # mean approximation error over many trials is small for both.
    torch.manual_seed(0)
    dim, n = 4, 6
    x = torch.randn(1, n, dim) * 0.3  # modest norm, keeps variance manageable
    k_exact = exact_gaussian_kernel(x, x)

    for orthogonal in (False, True):
        errors = []
        for trial in range(20):
            g = torch.Generator().manual_seed(trial)
            rff = RandomFourierFeatures(dim=dim, num_features=64, orthogonal=orthogonal, generator=g)
            psi = rff(x)
            k_approx = torch.einsum("bir,bjr->bij", psi, psi)
            errors.append((k_approx - k_exact).mean().item())
        assert abs(sum(errors) / len(errors)) < 0.1, f"orthogonal={orthogonal} mean bias too large"


def test_gradients_flow_through_x_not_w() -> None:
    rff = RandomFourierFeatures(dim=5, num_features=16)
    x = torch.randn(2, 4, 5, requires_grad=True)
    out = rff(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert not rff.w.requires_grad  # fixed buffer, not learned
    assert not rff.b.requires_grad


def test_exact_gaussian_kernel_diagonal_is_one() -> None:
    x = torch.randn(2, 7, 4)
    k = exact_gaussian_kernel(x, x)
    diag = k.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
