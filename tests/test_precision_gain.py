"""CellV0.2 -- Conservative Precision-Gain Cell (docs/architecture_v0.md Sec 10).

Correctness properties the user's spec calls out explicitly: shape/dtype/
device (incl. MPS), positivity/finiteness of the belief state, the
``relative_gain`` bounds, the neutral-confidence reduction to
``tanh(F.linear(mu, V, b))``, linear-row expressivity, uniform replication
invariance, the conservative-precision inequality, conflict -> higher
uncertainty -> lower effective precision, full gradient reach, row-scaling
invariance of the normalized quantities, the no-edge-tensor guarantee, and a
short AdamW smoke train.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.precision_gain import (
    BeliefNetworkV02,
    PrecisionGainLayer,
    _inverse_softplus,
    effective_precision,
    initial_belief,
    relative_gain,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _random_belief(batch: int, cells: int, *, dtype=torch.float32, device="cpu") -> BeliefCell:
    g = torch.Generator(device="cpu").manual_seed(cells * 1000 + batch)
    return BeliefCell(
        mu=torch.randn(batch, cells, generator=g).to(dtype=dtype, device=device),
        evidence=(torch.rand(batch, cells, generator=g) * 4 + 0.5).to(dtype=dtype, device=device),
        uncertainty=(torch.rand(batch, cells, generator=g) * 3).to(dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# shape / dtype / device / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_shape_dtype_and_valid_state(dtype: torch.dtype) -> None:
    layer = PrecisionGainLayer(10, 4).to(dtype)
    out = layer(_random_belief(8, 10, dtype=dtype))

    assert out.shape == (8, 4)
    assert out.mu.dtype == dtype and out.evidence.dtype == dtype
    assert torch.isfinite(out.mu).all()
    assert torch.isfinite(out.evidence).all()
    assert torch.isfinite(out.uncertainty).all()
    assert (out.evidence > 0).all(), "e must stay strictly positive for a non-degenerate V"
    assert (out.uncertainty >= 0).all(), "u must stay non-negative"


def test_network_forward_shape_and_finite() -> None:
    net = BeliefNetworkV02(12, 16, 3)
    x = torch.randn(20, 12)
    out = net(x)
    assert out.shape == (20, 3)
    assert torch.isfinite(out).all()

    pred, belief = net.forward_with_beliefs(x)
    assert torch.equal(pred, out) or torch.allclose(pred, out)
    assert belief.shape == (20, 16)
    assert (belief.evidence > 0).all() and (belief.uncertainty >= 0).all()


def test_wrong_input_cell_count_raises() -> None:
    layer = PrecisionGainLayer(5, 3)
    with pytest.raises(ValueError):
        layer(initial_belief(torch.zeros(2, 7)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_forward_backward() -> None:
    device = torch.device("mps")
    net = BeliefNetworkV02(10, 12, 3).to(device)
    x = torch.randn(16, 10, device=device)

    out = net(x)
    assert out.shape == (16, 3)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()

    out.pow(2).mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# relative_gain bounds
# ---------------------------------------------------------------------------


def test_relative_gain_strictly_between_zero_and_two() -> None:
    torch.manual_seed(0)
    e = torch.rand(6, 32) * 5 + 0.2
    u = torch.rand(6, 32) * 4
    r = relative_gain(effective_precision(e, u))
    assert (r > 0.0).all()
    assert (r < 2.0).all()


def test_relative_gain_is_one_for_equal_precisions() -> None:
    for value in (0.05, 0.7, 3.0, 50.0):
        p = torch.full((4, 9), value)
        r = relative_gain(p)
        assert torch.allclose(r, torch.ones_like(r), atol=1e-6)


def test_relative_gain_population_mean_is_not_detached() -> None:
    p = (torch.rand(3, 7) + 0.1).requires_grad_(True)
    relative_gain(p).sum().backward()
    # If the mean were detached, d/dp would be 2/(p+mean_const); keeping it
    # attached changes every entry's gradient and couples the cells.
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert (p.grad != 0).any()


# ---------------------------------------------------------------------------
# neutral-confidence reduction + linear-row expressivity
# ---------------------------------------------------------------------------


def test_neutral_confidence_reduces_to_linear_tanh() -> None:
    torch.manual_seed(1)
    layer = PrecisionGainLayer(11, 7)
    mu = torch.randn(5, 11)

    out = layer(initial_belief(mu))  # e = 1, u = 0  ->  precision = 1, gain = 1

    reference = torch.tanh(F.linear(mu, layer.V, layer.bias))
    assert torch.allclose(out.mu, reference, atol=1e-5, rtol=1e-5)
    # every source precision is exactly 1 -> convex combination is exactly 1
    assert torch.allclose(out.evidence, torch.ones_like(out.evidence), atol=1e-5)


def test_normalized_V_gamma_reproduces_arbitrary_linear_weight_row() -> None:
    torch.manual_seed(2)
    out_cells, in_cells = 4, 9
    W = torch.randn(out_cells, in_cells)
    b = torch.randn(out_cells)

    layer = PrecisionGainLayer(in_cells, out_cells)
    with torch.no_grad():
        layer.V.copy_(W)
        row_l1 = W.abs().sum(dim=1)
        layer.gain_raw.copy_(_inverse_softplus(row_l1.double()).to(W.dtype))
        layer.bias.copy_(b)

    mu = torch.randn(6, in_cells)
    out = layer(initial_belief(mu))

    # pre-activation matches an ordinary linear layer exactly
    gamma = F.softplus(layer.gain_raw)
    row_l1f = layer.V.abs().sum(dim=1)
    consensus = (mu @ layer.V.t()) / row_l1f
    pre_activation = gamma * consensus + layer.bias
    assert torch.allclose(pre_activation, F.linear(mu, W, b), atol=1e-4)
    assert torch.allclose(out.mu, torch.tanh(F.linear(mu, W, b)), atol=1e-4)


def test_inverse_softplus_round_trips() -> None:
    y = torch.tensor([1e-4, 0.1, 1.0, 5.0, 14.0, 40.0], dtype=torch.float64)
    assert torch.allclose(F.softplus(_inverse_softplus(y)), y, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# uniform replication invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_uniform_replication_invariance(k: int) -> None:
    in_cells, out_cells, batch = 5, 3, 4
    layer = PrecisionGainLayer(in_cells, out_cells).double()

    base = _random_belief(batch, in_cells, dtype=torch.float64)
    out_base = layer(base)

    big = PrecisionGainLayer(in_cells * k, out_cells).double()
    with torch.no_grad():
        big.V.copy_(layer.V.repeat(1, k))          # duplicate the columns of V k*
        big.gain_raw.copy_(layer.gain_raw)
        big.bias.copy_(layer.bias)
    replicated = BeliefCell(
        mu=base.mu.repeat(1, k),
        evidence=base.evidence.repeat(1, k),
        uncertainty=base.uncertainty.repeat(1, k),
    )
    out_big = big(replicated)

    assert torch.allclose(out_big.mu, out_base.mu, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_big.evidence, out_base.evidence, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_big.uncertainty, out_base.uncertainty, atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# conservative precision
# ---------------------------------------------------------------------------


def test_e_out_is_a_convex_combination_of_source_precisions() -> None:
    torch.manual_seed(3)
    layer = PrecisionGainLayer(12, 6)
    belief = _random_belief(16, 12)
    src_precision = effective_precision(belief.evidence, belief.uncertainty)  # (B, in)

    out = layer(belief)

    src_max = src_precision.max(dim=1, keepdim=True).values
    src_min = src_precision.min(dim=1, keepdim=True).values
    assert (out.evidence <= src_max + 1e-6).all(), "e_out exceeded max source precision"
    assert (out.evidence >= src_min - 1e-6).all(), "e_out below min source precision"

    precision_out = effective_precision(out.evidence, out.uncertainty)
    assert (precision_out <= out.evidence + 1e-6).all(), "precision_out exceeded e_out"


# ---------------------------------------------------------------------------
# conflict raises uncertainty and lowers effective output precision
# ---------------------------------------------------------------------------


def test_conflicting_sources_raise_u_out_and_lower_effective_precision() -> None:
    layer = PrecisionGainLayer(2, 1)
    with torch.no_grad():
        layer.V.copy_(torch.tensor([[1.0, 1.0]]))  # equal, positive -> both sources agree in sign
        layer.gain_raw.copy_(_inverse_softplus(torch.tensor([2.0])))
        layer.bias.zero_()

    # identical inherited support in both examples (same e, same u per source)
    e = torch.tensor([[1.5, 1.5]])
    u = torch.tensor([[0.3, 0.3]])
    aligned = BeliefCell(mu=torch.tensor([[0.6, 0.6]]), evidence=e, uncertainty=u)
    conflicting = BeliefCell(mu=torch.tensor([[0.6, -0.6]]), evidence=e, uncertainty=u)

    out_aligned = layer(aligned)
    out_conflicting = layer(conflicting)

    # inherited support pathway is identical
    assert torch.allclose(out_aligned.evidence, out_conflicting.evidence, atol=1e-6)
    # ...but disagreement is not
    assert out_aligned.uncertainty.item() == pytest.approx(0.0, abs=1e-6)
    assert out_conflicting.uncertainty.item() > 0.1
    # ...so the effective precision handed to the next layer is lower
    p_aligned = effective_precision(out_aligned.evidence, out_aligned.uncertainty).item()
    p_conflicting = effective_precision(
        out_conflicting.evidence, out_conflicting.uncertainty
    ).item()
    assert p_conflicting < p_aligned


# ---------------------------------------------------------------------------
# gradient reach
# ---------------------------------------------------------------------------


def test_gradients_reach_source_mu_evidence_uncertainty() -> None:
    layer = PrecisionGainLayer(5, 3)
    mu = torch.randn(4, 5, requires_grad=True)
    evidence = (torch.rand(4, 5) * 3 + 0.5).requires_grad_(True)
    uncertainty = (torch.rand(4, 5) * 2 + 0.1).requires_grad_(True)

    out = layer(BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty))
    (out.mu.sum() + out.evidence.sum() + out.uncertainty.sum()).backward()

    for name, t in (("mu", mu), ("evidence", evidence), ("uncertainty", uncertainty)):
        assert t.grad is not None, name
        assert torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, name


def test_gradients_reach_every_parameter_through_task_loss() -> None:
    torch.manual_seed(4)
    net = BeliefNetworkV02(8, 10, 3)
    x = torch.randn(32, 8)
    y = torch.randint(0, 3, (32,))

    loss = F.cross_entropy(net(x), y)
    loss.backward()

    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for name in ("layer1.V", "layer2.V", "layer1.gain_raw", "layer2.gain_raw",
                 "layer1.bias", "layer2.bias", "readout.weight"):
        g = dict(net.named_parameters())[name].grad
        assert g.abs().sum() > 0, f"{name} received a zero gradient"


# ---------------------------------------------------------------------------
# row-scaling invariance of the normalized quantities
# ---------------------------------------------------------------------------


def test_scaling_a_V_row_by_a_positive_constant_changes_nothing_normalized() -> None:
    torch.manual_seed(5)
    layer = PrecisionGainLayer(6, 3)
    belief = _random_belief(4, 6)

    out0 = layer(belief)
    with torch.no_grad():
        layer.V[0].mul_(2.5)  # positive scalar; gain_raw / bias untouched

    out1 = layer(belief)

    # normalized consensus is embedded in mu_out (gamma_0, bias_0 unchanged)
    assert torch.allclose(out0.mu[:, 0], out1.mu[:, 0], atol=1e-5)
    assert torch.allclose(out0.evidence[:, 0], out1.evidence[:, 0], atol=1e-5)
    assert torch.allclose(out0.uncertainty[:, 0], out1.uncertainty[:, 0], atol=1e-5)
    # untouched rows are bit-for-bit unaffected
    assert torch.allclose(out0.mu[:, 1:], out1.mu[:, 1:], atol=1e-6)
    assert torch.allclose(out0.evidence[:, 1:], out1.evidence[:, 1:], atol=1e-6)


# ---------------------------------------------------------------------------
# no (batch, out_cells, in_cells) edge tensor
# ---------------------------------------------------------------------------


def test_forward_and_backward_materialize_no_rank3_tensor() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    batch, in_cells, out_cells = 7, 5, 3  # mutually distinct so a (7,3,5) tensor is unmistakable
    layer = PrecisionGainLayer(in_cells, out_cells)
    belief = _random_belief(batch, in_cells)

    class _Recorder(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.max_dim = 0
            self.op_names: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            self.op_names.append(str(func))
            for leaf in tree_leaves(result):
                if isinstance(leaf, torch.Tensor):
                    self.max_dim = max(self.max_dim, leaf.dim())
            return result

    with _Recorder() as rec:
        out = layer(belief)
        out.mu.pow(2).sum().backward()

    assert rec.max_dim <= 2, f"a rank-{rec.max_dim} tensor was materialized"
    assert not any("bmm" in n for n in rec.op_names), "a batched (rank-3) matmul was used"


# ---------------------------------------------------------------------------
# AdamW smoke train
# ---------------------------------------------------------------------------


def test_short_adamw_train_reduces_loss_without_nans() -> None:
    torch.manual_seed(6)
    n, d = 512, 8
    x = torch.randn(n, d)
    teacher = torch.randn(d, 1)
    y = x @ teacher + 0.1 * torch.randn(n, 1)

    net = BeliefNetworkV02(d, 16, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(300):
        opt.zero_grad()
        loss = F.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(math.isfinite(v) for v in losses)
    early = statistics.mean(losses[:20])
    late = statistics.mean(losses[-20:])
    assert late < 0.75 * early, f"loss did not broadly decrease ({early:.4f} -> {late:.4f})"
