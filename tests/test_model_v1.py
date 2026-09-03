import torch

from src.models.architecture_v1.model import DynamicBeliefGraph


def _small_model(**overrides) -> DynamicBeliefGraph:
    defaults = dict(
        in_features=6,
        out_features=1,
        n_cells=16,
        association_dim=4,
        num_steps=3,
        hidden_dim=8,
    )
    defaults.update(overrides)
    return DynamicBeliefGraph(**defaults)


def test_forward_shape_regression() -> None:
    model = _small_model(out_features=1)
    x = torch.randn(5, 6)
    out = model(x)
    assert out.shape == (5, 1)


def test_forward_shape_classification_style_output() -> None:
    model = _small_model(out_features=3)
    x = torch.randn(5, 6)
    out = model(x)
    assert out.shape == (5, 3)


def test_forward_with_cells_returns_valid_population() -> None:
    model = _small_model()
    x = torch.randn(4, 6)
    pred, cells = model.forward_with_cells(x)

    assert pred.shape == (4, 1)
    assert cells.mu.shape == (4, 16)
    assert cells.z.shape == (4, 16, 4)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(cells.mu).all()
    assert (cells.evidence >= 0).all()
    assert (cells.uncertainty >= 0).all()


def test_batch_size_one() -> None:
    model = _small_model()
    x = torch.randn(1, 6)
    out = model(x)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


def test_full_forward_backward_has_finite_gradients() -> None:
    # Regression test for the `sqrt(0)` gradient-blowup bug caught during
    # implementation (routing.py's `_safe_sqrt`): sparsemax's exact zeros
    # feeding into `sqrt(a_ij * a_ji)` produced nan/inf gradients on every
    # parameter in the model before the fix.
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(8, 6)
    y = torch.randn(8, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_optimizer_step_does_not_diverge_immediately() -> None:
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(8, 6)
    y = torch.randn(8, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
