import torch

from src.models.architecture_v1.sparsemax import sparse_association_with_null, sparsemax


def test_sums_to_one_over_last_dim() -> None:
    scores = torch.randn(3, 7)
    out = sparsemax(scores)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_nonnegative() -> None:
    scores = torch.randn(4, 10)
    out = sparsemax(scores)
    assert (out >= 0).all()


def test_produces_exact_zeros_for_a_clearly_dominated_entry() -> None:
    # One huge score, the rest far below it -> sparsemax should route
    # (close to) all mass to the winner and exactly zero everything else,
    # unlike softmax.
    scores = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])
    out = sparsemax(scores)
    assert torch.isclose(out[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[0, 1:], torch.zeros(4))


def test_uniform_scores_split_evenly() -> None:
    scores = torch.zeros(1, 4)
    out = sparsemax(scores)
    assert torch.allclose(out, torch.full((1, 4), 0.25))


def test_gradient_is_finite_even_with_exact_zero_outputs() -> None:
    scores = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    out = sparsemax(scores)
    assert (out == 0).any()  # confirms this case actually exercises exact-zero entries
    out.sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_null_option_can_absorb_all_mass() -> None:
    # A very negative null logit should make the null option lose to any
    # real (finite) score; a very positive one should let it dominate,
    # driving every real association toward zero.
    scores = torch.randn(2, 6)
    high_null = sparse_association_with_null(scores, torch.tensor([100.0]))
    assert torch.allclose(high_null, torch.zeros_like(high_null))


def test_null_option_shape_matches_input_without_the_extra_column() -> None:
    scores = torch.randn(2, 5, 6)
    out = sparse_association_with_null(scores, torch.zeros(1))
    assert out.shape == scores.shape
