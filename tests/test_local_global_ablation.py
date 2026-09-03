"""`use_global_routing=False` (docs/architecture_v1.md §9's CellV1-Local
ablation for `experiments/v1_001_dynamic_groups`) should be a genuinely
smaller model -- no global-routing/need/offer parameters at all -- not a
full model with its global output zeroed out."""

import torch

from src.evaluation.efficiency import count_parameters
from src.models.architecture_v1.cell import BeliefCellV1
from src.models.architecture_v1.dynamics import DynamicBeliefGraphStep
from src.models.architecture_v1.model import DynamicBeliefGraph


def _random_population(batch: int, n: int, d: int, generator: torch.Generator) -> BeliefCellV1:
    z = torch.randn(batch, n, d, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return BeliefCellV1(
        mu=torch.tanh(torch.randn(batch, n, generator=generator)),
        evidence=torch.ones(batch, n),
        uncertainty=torch.ones(batch, n),
        z=z,
    )


def test_local_only_step_has_no_global_modules() -> None:
    step = DynamicBeliefGraphStep(n_cells=10, association_dim=4, hidden_dim=8, use_global_routing=False)
    assert step.global_routing is None
    assert step.need is None
    assert step.offer is None
    assert step.global_bias is None
    assert step.source_gate_logit.numel() == 2  # self + local only


def test_full_step_has_global_modules() -> None:
    step = DynamicBeliefGraphStep(n_cells=10, association_dim=4, hidden_dim=8, use_global_routing=True)
    assert step.global_routing is not None
    assert step.need is not None
    assert step.offer is not None
    assert step.source_gate_logit.numel() == 3  # self + local + global


def test_local_only_step_returns_none_for_a_global() -> None:
    g = torch.Generator().manual_seed(0)
    step = DynamicBeliefGraphStep(n_cells=8, association_dim=4, hidden_dim=8, use_global_routing=False)
    cells = _random_population(2, 8, 4, g)

    out, (a_local, a_global) = step.forward_with_graphs(cells)

    assert a_global is None
    assert a_local.shape == (2, 8, 8)
    assert torch.isfinite(out.mu).all()


def test_local_model_has_strictly_fewer_parameters_than_full_model() -> None:
    kwargs = dict(in_features=6, out_features=1, n_cells=16, association_dim=4, num_steps=3, hidden_dim=8)
    full = DynamicBeliefGraph(use_global_routing=True, **kwargs)
    local = DynamicBeliefGraph(use_global_routing=False, **kwargs)

    assert count_parameters(local) < count_parameters(full)


def test_local_model_forward_and_backward() -> None:
    torch.manual_seed(0)
    model = DynamicBeliefGraph(
        in_features=6, out_features=1, n_cells=16, association_dim=4, num_steps=3, hidden_dim=8,
        use_global_routing=False,
    )
    x = torch.randn(5, 6)
    y = torch.randn(5, 1)

    pred = model(x)
    assert pred.shape == (5, 1)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_forward_with_graphs_matches_forward_with_cells_prediction() -> None:
    torch.manual_seed(0)
    model = DynamicBeliefGraph(
        in_features=6, out_features=1, n_cells=16, association_dim=4, num_steps=3, hidden_dim=8
    )
    x = torch.randn(5, 6)

    pred_a, cells_a = model.forward_with_cells(x)
    pred_b, cells_b, graphs = model.forward_with_graphs(x)

    assert torch.allclose(pred_a, pred_b)
    assert torch.allclose(cells_a.mu, cells_b.mu)
    assert len(graphs) == 3
    for a_local, a_global in graphs:
        assert a_local.shape == (5, 16, 16)
        assert a_global.shape == (5, 16, 16)
