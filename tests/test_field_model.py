import torch

from src.models.architecture_v1.field_model import SelfOrganizingRefinementField


def _small_model(**overrides) -> SelfOrganizingRefinementField:
    defaults = dict(
        in_features=6,
        out_features=1,
        n_cells=24,
        association_dim=6,
        routing_dim=5,
        num_features=64,
        hidden_dim=8,
        num_steps=1,
    )
    defaults.update(overrides)
    return SelfOrganizingRefinementField(**defaults)


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


def test_forward_with_state_agrees_with_forward() -> None:
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(4, 6)

    pred_a = model(x)
    pred_b, state = model.forward_with_state(x)

    assert torch.allclose(pred_a, pred_b)
    assert state.cells.mu.shape == (4, 24)
    assert state.r.shape == (4, 24, 5)


def test_batch_size_one() -> None:
    model = _small_model()
    x = torch.randn(1, 6)
    out = model(x)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


def test_full_forward_backward_has_finite_gradients() -> None:
    torch.manual_seed(0)
    model = _small_model()  # default num_steps=1
    x = torch.randn(8, 6)
    y = torch.randn(8, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    for name, p in model.named_parameters():
        if name.startswith("core.step.routing_gate"):
            # r's movement only matters for a *future* step -- at the
            # default num_steps=1 there is none, so this legitimately gets
            # no gradient (test_two_step_model_gives_routing_gate_a_gradient
            # confirms it does at num_steps=2).
            continue
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


def test_two_step_model_gives_routing_gate_a_gradient() -> None:
    torch.manual_seed(0)
    model = _small_model(num_steps=2)
    x = torch.randn(4, 6)
    y = torch.randn(4, 1)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    grad = model.core.step.routing_gate.net[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
