"""CellV0.3 -- Conflict-Normalized Belief Cell (docs/architecture_v0.md Sec 10).

The correctness properties the user's spec calls out explicitly: shape/dtype/
device (incl. real MPS), positivity/finiteness of the belief state, the
conservative-precision inequality, perfect-agreement -> zero conflict,
conflict sensitivity, the **absolute-confidence sensitivity** that CellV0.2
lost, confidence being causally load-bearing, uniform replication invariance,
positive row-scaling invariance, the zero-conflict reduction to an ordinary
linear+tanh neuron, full gradient reach, the no-edge-tensor guarantee, and a
short AdamW smoke train.

CellV0.1 / CellV0.2 are untouched -- their test suites still pass unchanged.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.conflict_normalized import (
    BeliefNetworkV03,
    ConflictNormalizedLayer,
)
from src.models.architecture_v0.precision_gain import (
    _inverse_softplus,
    effective_precision,
    initial_belief,
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


def _confidence_scale(belief: BeliefCell) -> torch.Tensor:
    return effective_precision(belief.evidence, belief.uncertainty).sqrt()


# ---------------------------------------------------------------------------
# 1. shape / dtype / device / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_shape_dtype_and_valid_state(dtype: torch.dtype) -> None:
    layer = ConflictNormalizedLayer(10, 4).to(dtype)
    out = layer(_random_belief(8, 10, dtype=dtype))

    assert out.shape == (8, 4)
    assert out.mu.dtype == dtype and out.evidence.dtype == dtype and out.uncertainty.dtype == dtype
    assert torch.isfinite(out.mu).all()
    assert torch.isfinite(out.evidence).all()
    assert torch.isfinite(out.uncertainty).all()


def test_network_forward_shape_and_finite() -> None:
    net = BeliefNetworkV03(12, 16, 3)
    x = torch.randn(20, 12)
    out = net(x)
    assert out.shape == (20, 3)
    assert torch.isfinite(out).all()

    (b1, _), (b2, _) = net.layer_states_verbose(x)
    assert b1.shape == (20, 16) and b2.shape == (20, 16)


def test_wrong_input_cell_count_raises() -> None:
    layer = ConflictNormalizedLayer(5, 3)
    with pytest.raises(ValueError):
        layer(initial_belief(torch.zeros(2, 7)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_forward_backward() -> None:
    device = torch.device("mps")
    net = BeliefNetworkV03(10, 12, 3).to(device)
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
# 2. state validity
# ---------------------------------------------------------------------------


def test_state_is_strictly_positive_and_nonnegative_and_finite() -> None:
    torch.manual_seed(0)
    layer = ConflictNormalizedLayer(16, 9)
    out = layer(_random_belief(24, 16))
    pi_out = effective_precision(out.evidence, out.uncertainty)

    assert (out.evidence > 0).all(), "e_out must stay strictly positive"
    assert (out.uncertainty >= 0).all(), "u_out must stay non-negative"
    assert (pi_out > 0).all(), "pi_out must stay strictly positive"
    for t in (out.mu, out.evidence, out.uncertainty, pi_out):
        assert torch.isfinite(t).all()


# ---------------------------------------------------------------------------
# 3. conservative precision
# ---------------------------------------------------------------------------


def test_e_out_never_exceeds_max_source_precision_and_pi_out_le_e_out() -> None:
    torch.manual_seed(3)
    layer = ConflictNormalizedLayer(12, 6)
    belief = _random_belief(16, 12)
    src_pi = effective_precision(belief.evidence, belief.uncertainty)  # (B, in)

    out = layer(belief)
    src_max = src_pi.max(dim=1, keepdim=True).values
    src_min = src_pi.min(dim=1, keepdim=True).values

    assert (out.evidence <= src_max + 1e-6).all(), "e_out exceeded max source precision"
    assert (out.evidence >= src_min - 1e-6).all(), "e_out below min source precision"

    pi_out = effective_precision(out.evidence, out.uncertainty)
    assert (pi_out <= out.evidence + 1e-6).all(), "pi_out exceeded e_out"


# ---------------------------------------------------------------------------
# 4. perfect agreement -> zero conflict -> pi_out ~= e_out
# ---------------------------------------------------------------------------


def test_perfectly_agreeing_signed_messages_give_zero_conflict() -> None:
    # V all positive + mu constant across sources -> every signed message
    # sign(V_ij) mu_j is the same value -> the A-weighted variance is 0.
    torch.manual_seed(4)
    layer = ConflictNormalizedLayer(7, 5).double()
    with torch.no_grad():
        layer.V.abs_()

    mu = torch.full((6, 7), 0.4, dtype=torch.float64)
    belief = BeliefCell(
        mu=mu,
        evidence=(torch.rand(6, 7, dtype=torch.float64) * 3 + 0.5),
        uncertainty=(torch.rand(6, 7, dtype=torch.float64) * 2),
    )
    out = layer(belief)

    assert out.uncertainty.abs().max().item() == pytest.approx(0.0, abs=1e-9)
    pi_out = effective_precision(out.evidence, out.uncertainty)
    assert torch.allclose(pi_out, out.evidence, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. conflict sensitivity (consensus controlled)
# ---------------------------------------------------------------------------


def test_more_disagreement_raises_u_out_lowers_pi_out_and_shrinks_the_activation() -> None:
    # 2 sources, equal positive weight -> A = S = [0.5, 0.5]. Feed
    # mu = [c + d, c - d]: the weighted mean of the signed messages stays c
    # (consensus controlled) while their weighted variance is d^2.
    layer = ConflictNormalizedLayer(2, 1).double()
    with torch.no_grad():
        layer.V.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        layer.gain_raw.copy_(_inverse_softplus(torch.tensor([2.0], dtype=torch.float64)))
        layer.bias.zero_()

    c = 0.5
    e = torch.full((1, 2), 1.5, dtype=torch.float64)
    u = torch.full((1, 2), 0.3, dtype=torch.float64)

    prev_u = -1.0
    prev_pi = math.inf
    prev_mag = math.inf
    consensus_seen: list[float] = []
    for d in (0.0, 0.3, 0.6, 1.0):
        mu = torch.tensor([[c + d, c - d]], dtype=torch.float64)
        out, consensus = layer.forward_verbose(BeliefCell(mu=mu, evidence=e, uncertainty=u))
        pi_out = effective_precision(out.evidence, out.uncertainty).item()
        # confidence-scaled pre-activation magnitude, consensus held ~fixed
        mag = abs(out.mu.item())  # bias = 0, tanh monotone -> tracks |gamma c sqrt(pi)|

        consensus_seen.append(consensus.item())
        if d > 0.0:
            assert out.uncertainty.item() > prev_u, f"u_out did not rise at d={d}"
            assert pi_out < prev_pi, f"pi_out did not fall at d={d}"
            assert mag < prev_mag, f"activation magnitude did not shrink at d={d}"
        prev_u, prev_pi, prev_mag = out.uncertainty.item(), pi_out, mag

    assert max(consensus_seen) - min(consensus_seen) < 1e-9, "consensus was not controlled"


# ---------------------------------------------------------------------------
# 6. absolute-confidence sensitivity -- the property CellV0.2 lost (critical)
# ---------------------------------------------------------------------------


def test_uniformly_lower_source_precision_lowers_output_confidence_and_changes_activation() -> None:
    # Two source populations with identical mu, identical (uniform) relative
    # precision pattern and identical weight structure -- one just has all
    # source precision scaled down. CellV0.2's relative gain 2pi/(pi+mean pi)
    # cancels this exactly; CellV0.3 must NOT.
    torch.manual_seed(6)
    layer = ConflictNormalizedLayer(5, 4).double()
    with torch.no_grad():
        layer.V.copy_(layer.V.abs() + 0.1)  # positive -> non-trivial consensus

    mu = torch.rand(3, 5, dtype=torch.float64) + 0.2
    z = torch.zeros(3, 5, dtype=torch.float64)
    ones = torch.ones(3, 5, dtype=torch.float64)
    high = BeliefCell(mu=mu, evidence=ones, uncertainty=z)
    low = BeliefCell(mu=mu, evidence=ones * 0.1, uncertainty=z)

    out_high, cons_high = layer.forward_verbose(high)
    out_low, cons_low = layer.forward_verbose(low)

    # the precision-normalised consensus is (correctly) identical...
    assert torch.allclose(cons_high, cons_low, atol=1e-9)
    # ...but the confidence scale is strictly lower for the less-reliable pop...
    assert (_confidence_scale(out_low) < _confidence_scale(out_high) - 1e-6).all()
    # ...and therefore the activation differs (this is exactly the pi=[1,1,1]
    # vs pi=[0.1,0.1,0.1] case the spec calls out).
    assert not torch.allclose(out_high.mu, out_low.mu, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. confidence is causally load-bearing
# ---------------------------------------------------------------------------


def test_changing_only_source_evidence_uncertainty_moves_mu_out() -> None:
    torch.manual_seed(7)
    layer = ConflictNormalizedLayer(6, 5)
    mu = torch.randn(4, 6)

    a = BeliefCell(mu=mu, evidence=torch.ones(4, 6), uncertainty=torch.zeros(4, 6))
    b = BeliefCell(
        mu=mu,
        evidence=torch.full((4, 6), 2.0),
        uncertainty=torch.full((4, 6), 1.3),
    )
    out_a = layer(a)
    out_b = layer(b)
    assert not torch.allclose(out_a.mu, out_b.mu, atol=1e-4), "mu_out ignored the belief state"


# ---------------------------------------------------------------------------
# 8. uniform replication invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_uniform_replication_invariance(k: int) -> None:
    in_cells, out_cells, batch = 5, 3, 4
    layer = ConflictNormalizedLayer(in_cells, out_cells).double()

    base = _random_belief(batch, in_cells, dtype=torch.float64)
    out_base, cons_base = layer.forward_verbose(base)

    big = ConflictNormalizedLayer(in_cells * k, out_cells).double()
    with torch.no_grad():
        big.V.copy_(layer.V.repeat(1, k))  # duplicate the columns of V k*
        big.gain_raw.copy_(layer.gain_raw)
        big.bias.copy_(layer.bias)
    replicated = BeliefCell(
        mu=base.mu.repeat(1, k),
        evidence=base.evidence.repeat(1, k),
        uncertainty=base.uncertainty.repeat(1, k),
    )
    out_big, cons_big = big.forward_verbose(replicated)

    pi_base = effective_precision(out_base.evidence, out_base.uncertainty)
    pi_big = effective_precision(out_big.evidence, out_big.uncertainty)
    assert torch.allclose(cons_big, cons_base, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_big.evidence, out_base.evidence, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_big.uncertainty, out_base.uncertainty, atol=1e-9, rtol=1e-7)
    assert torch.allclose(pi_big, pi_base, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_big.mu, out_base.mu, atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# 9. positive row-scaling invariance of the normalized quantities
# ---------------------------------------------------------------------------


def test_scaling_a_V_row_by_a_positive_constant_changes_nothing_normalized() -> None:
    torch.manual_seed(9)
    layer = ConflictNormalizedLayer(6, 3).double()
    belief = _random_belief(4, 6, dtype=torch.float64)

    out0, cons0 = layer.forward_verbose(belief)
    with torch.no_grad():
        layer.V[0].mul_(3.5)  # positive scalar; gain_raw / bias untouched
    out1, cons1 = layer.forward_verbose(belief)

    pi0 = effective_precision(out0.evidence, out0.uncertainty)
    pi1 = effective_precision(out1.evidence, out1.uncertainty)
    for a, b in (
        (cons0[:, 0], cons1[:, 0]),
        (out0.evidence[:, 0], out1.evidence[:, 0]),
        (out0.uncertainty[:, 0], out1.uncertainty[:, 0]),
        (pi0[:, 0], pi1[:, 0]),
        (out0.mu[:, 0], out1.mu[:, 0]),  # gamma_0, bias_0 unchanged too
    ):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-7)
    # untouched rows are bit-for-bit unaffected
    assert torch.allclose(out0.mu[:, 1:], out1.mu[:, 1:], atol=1e-12)


# ---------------------------------------------------------------------------
# 10. standard-neuron limiting case (zero conflict only)
# ---------------------------------------------------------------------------


def test_zero_conflict_unit_precision_reduces_to_linear_tanh() -> None:
    # all source precision = 1 (e = 1, u = 0), zero conflict (V positive +
    # constant mu), gamma = ||V_i||_1  ->  mu_out == tanh(F.linear(mu, V, b)).
    torch.manual_seed(10)
    out_cells, in_cells = 4, 9
    layer = ConflictNormalizedLayer(in_cells, out_cells).double()
    with torch.no_grad():
        W = torch.rand(out_cells, in_cells, dtype=torch.float64) + 0.1
        b = torch.randn(out_cells, dtype=torch.float64)
        layer.V.copy_(W)
        layer.gain_raw.copy_(_inverse_softplus(W.abs().sum(dim=1)))
        layer.bias.copy_(b)

    mu = torch.full((6, in_cells), 0.7, dtype=torch.float64)
    out = layer(initial_belief(mu))

    assert torch.allclose(out.mu, torch.tanh(F.linear(mu, W, b)), atol=1e-10)
    assert torch.allclose(out.evidence, torch.ones_like(out.evidence), atol=1e-10)


# ---------------------------------------------------------------------------
# 11. gradient reach
# ---------------------------------------------------------------------------


def test_gradients_reach_source_mu_evidence_uncertainty() -> None:
    layer = ConflictNormalizedLayer(5, 3)
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
    torch.manual_seed(11)
    net = BeliefNetworkV03(8, 10, 3)
    x = torch.randn(32, 8)
    y = torch.randint(0, 3, (32,))

    F.cross_entropy(net(x), y).backward()

    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for name in (
        "layer1.V", "layer2.V", "layer1.gain_raw", "layer2.gain_raw",
        "layer1.bias", "layer2.bias", "readout.weight",
    ):
        g = dict(net.named_parameters())[name].grad
        assert g.abs().sum() > 0, f"{name} received a zero gradient"


def test_task_loss_gradient_flows_through_the_precision_path() -> None:
    # Freeze the content path: hold mu and every parameter fixed, and check a
    # loss on mu_out still produces a non-zero gradient w.r.t. source e and u
    # (i.e. it flows mu_out -> pi_out -> e_out/u_out -> input e/u).
    torch.manual_seed(12)
    layer = ConflictNormalizedLayer(6, 4)
    mu = torch.randn(5, 6)
    evidence = (torch.rand(5, 6) * 3 + 0.5).requires_grad_(True)
    uncertainty = (torch.rand(5, 6) * 2 + 0.1).requires_grad_(True)

    out = layer(BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty))
    out.mu.pow(2).sum().backward()  # a pure-content loss

    assert evidence.grad.abs().sum() > 0, "no gradient reached source e via the precision path"
    assert uncertainty.grad.abs().sum() > 0, "no gradient reached source u via the precision path"


# ---------------------------------------------------------------------------
# 12. no (batch, out_cells, in_cells) edge tensor
# ---------------------------------------------------------------------------


def test_forward_and_backward_materialize_no_rank3_tensor() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    batch, in_cells, out_cells = 7, 5, 3  # mutually distinct -> a (7,3,5) tensor is unmistakable
    layer = ConflictNormalizedLayer(in_cells, out_cells)
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
# 13. AdamW smoke train
# ---------------------------------------------------------------------------


def test_short_adamw_train_reduces_loss_without_nans() -> None:
    torch.manual_seed(13)
    n, d = 512, 8
    x = torch.randn(n, d)
    teacher = torch.randn(d, 1)
    y = x @ teacher + 0.1 * torch.randn(n, 1)

    net = BeliefNetworkV03(d, 16, 1)
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


# ---------------------------------------------------------------------------
# CellV0.1 / CellV0.2 are untouched
# ---------------------------------------------------------------------------


def test_cellv01_and_cellv02_modules_are_not_perturbed_by_import() -> None:
    # importing conflict_normalized must not mutate precision_gain's helpers.
    from src.models.architecture_v0 import precision_gain

    p = torch.full((4, 5), 0.7)
    assert torch.allclose(precision_gain.relative_gain(p), torch.ones_like(p), atol=1e-6)
    b = precision_gain.initial_belief(torch.zeros(2, 3))
    assert torch.equal(b.evidence, torch.ones(2, 3))
    assert torch.equal(b.uncertainty, torch.zeros(2, 3))
