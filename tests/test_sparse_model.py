import torch

from src.models.architecture_v1.sparse_model import SparseDynamicBeliefGraph


def _small_model(**overrides) -> SparseDynamicBeliefGraph:
    defaults = dict(
        in_features=6,
        out_features=1,
        n_cells=24,
        association_dim=4,
        num_steps=3,
        hidden_dim=8,
        chunk_size_local=8,
        chunk_size_global=4,
    )
    defaults.update(overrides)
    return SparseDynamicBeliefGraph(**defaults)


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


def test_forward_with_cells_and_graphs_agree() -> None:
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(4, 6)

    pred_a, cells_a = model.forward_with_cells(x)
    pred_b, cells_b, graphs = model.forward_with_graphs(x)

    assert torch.allclose(pred_a, pred_b)
    assert torch.allclose(cells_a.mu, cells_b.mu)
    assert len(graphs) == 3


def test_batch_size_one() -> None:
    model = _small_model()
    x = torch.randn(1, 6)
    out = model(x)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


def test_full_forward_backward_has_finite_gradients() -> None:
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


def test_local_only_model_has_fewer_params_than_full() -> None:
    from src.evaluation.efficiency import count_parameters

    full = _small_model(use_global_routing=True)
    local = _small_model(use_global_routing=False)
    assert count_parameters(local) < count_parameters(full)
