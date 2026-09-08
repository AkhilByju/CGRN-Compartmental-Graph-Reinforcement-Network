"""CellV1.6 -- `PrecisionRegulatedAssemblyGate` (docs/architecture_v1.md
Sec 17). "Correctness" here means "matches the specified formulas and
produces valid, finite output," not "produces good results" -- that is an
experiment question, and no experiment has been run for CellV1.6.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch.utils._pytree import tree_leaves

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v1.assembly_gate import PrecisionRegulatedAssemblyGate


def _population(batch: int, n_cells: int, seed: int = 0) -> BeliefCell:
    g = torch.Generator().manual_seed(seed)
    return BeliefCell(
        mu=torch.randn(batch, n_cells, generator=g),
        evidence=torch.rand(batch, n_cells, generator=g) * 3.0 + 0.2,
        uncertainty=torch.rand(batch, n_cells, generator=g) + 0.2,
    )


def _log_precision(gate: PrecisionRegulatedAssemblyGate, belief: BeliefCell) -> torch.Tensor:
    eps = gate.eps
    return torch.log(belief.evidence + eps) - torch.log(belief.uncertainty**2 + eps)


def _drive(gate: PrecisionRegulatedAssemblyGate, belief: BeliefCell) -> torch.Tensor:
    features = torch.stack([belief.mu, _log_precision(gate, belief)], dim=-1)
    return gate.participation_drive(features).squeeze(-1)


# --- shape / dtype / device preservation ---------------------------------


def test_shape_dtype_device_preservation() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    belief = _population(4, 32)
    participation, diagnostics = gate(belief)

    assert participation.shape == belief.mu.shape
    assert participation.dtype == belief.mu.dtype
    assert participation.device == belief.mu.device
    for value in diagnostics.values():
        assert value.ndim == 0
        assert not value.requires_grad


def test_runs_on_various_population_sizes_with_one_shared_module() -> None:
    # The gate constructor takes no `n_cells` -- one module, every size.
    gate = PrecisionRegulatedAssemblyGate()
    n_params = sum(p.numel() for p in gate.parameters())
    for n_cells in (1, 2, 8, 64, 512):
        participation, _ = gate(_population(3, n_cells, seed=n_cells))
        assert participation.shape == (3, n_cells)
    assert sum(p.numel() for p in gate.parameters()) == n_params  # unchanged
    assert n_params == 34  # (2*8+8) + 8*1 + kappa_raw + width_bias


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_compatibility() -> None:
    device = torch.device("mps")
    gate = PrecisionRegulatedAssemblyGate().to(device)
    belief = _population(4, 48).to(device)
    participation, diagnostics = gate(belief)

    assert participation.device.type == "mps"
    assert participation.shape == (4, 48)
    assert torch.isfinite(participation).all()
    assert (participation >= 0).all() and (participation <= 1).all()
    for value in diagnostics.values():
        assert torch.isfinite(value).all()


# --- participation value invariants -------------------------------------


def test_participation_finite_and_in_unit_interval() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    for seed in range(8):
        participation, _ = gate(_population(6, 40, seed=seed))
        assert torch.isfinite(participation).all()
        assert (participation >= 0.0).all()
        assert (participation <= 1.0).all()


def test_extreme_beliefs_stay_finite_and_bounded() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    belief = BeliefCell(
        mu=torch.tensor([[-50.0, 0.0, 50.0, 1e-6, -1e-6, 3.0]]),
        evidence=torch.tensor([[0.0, 1e-9, 1e6, 2.0, 0.0, 1e-3]]),
        uncertainty=torch.tensor([[0.0, 1e6, 1e-9, 0.0, 3.0, 1e-4]]),
    )
    participation, diagnostics = gate(belief)
    assert torch.isfinite(participation).all()
    assert (participation >= 0.0).all() and (participation <= 1.0).all()
    for value in diagnostics.values():
        assert torch.isfinite(value).all()


def test_max_drive_cell_gets_participation_one() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    for seed in range(5):
        belief = _population(4, 32, seed=seed)
        participation, _ = gate(belief)
        argmax = _drive(gate, belief).argmax(dim=-1, keepdim=True)
        top = participation.gather(-1, argmax).squeeze(-1)
        assert torch.allclose(top, torch.ones_like(top))


def test_cells_outside_the_window_get_exact_zero() -> None:
    # A tight competition window (small width_bias -> small delta) plus a
    # wide spread of drives guarantees some cells fall more than delta
    # below the max -- those must be EXACTLY 0.0, not merely small.
    gate = PrecisionRegulatedAssemblyGate(width_bias_init=-2.0)
    belief = BeliefCell(
        mu=torch.randn(2, 128, generator=torch.Generator().manual_seed(1)) * 4.0,
        evidence=torch.rand(2, 128, generator=torch.Generator().manual_seed(2)) + 0.5,
        uncertainty=torch.rand(2, 128, generator=torch.Generator().manual_seed(3)) + 0.5,
    )
    participation, _ = gate(belief)
    assert (participation == 0.0).any()
    # and the ones that are zero are genuinely below (max - delta)
    z = PrecisionRegulatedAssemblyGate._standardize(_drive(gate, belief), gate.eps)
    delta = gate.delta_from_confidence(_log_precision(gate, belief).mean(dim=-1))
    gap = z.max(dim=-1, keepdim=True).values - z
    below_window = gap > (delta.unsqueeze(-1) + gate.eps)
    assert torch.equal(participation[below_window], torch.zeros_like(participation[below_window]))


def test_no_fixed_winner_count() -> None:
    # Two populations engineered to give different numbers of active cells:
    # one with drives tightly clustered near the max (many within delta),
    # one with drives spread out (few within delta). Nothing in the gate
    # fixes how many cells win.
    gate = PrecisionRegulatedAssemblyGate()

    tight = BeliefCell(
        mu=torch.full((1, 64), 0.0),
        evidence=torch.full((1, 64), 1.0) + 0.001 * torch.arange(64).float(),
        uncertainty=torch.full((1, 64), 1.0),
    )
    spread = BeliefCell(
        mu=torch.linspace(-6.0, 6.0, 64).unsqueeze(0),
        evidence=torch.linspace(0.1, 5.0, 64).unsqueeze(0),
        uncertainty=torch.linspace(2.0, 0.1, 64).unsqueeze(0),
    )
    n_active_tight = int((gate(tight)[0] > 0).sum())
    n_active_spread = int((gate(spread)[0] > 0).sum())
    assert n_active_tight != n_active_spread


def test_participation_not_renormalized_to_sum_to_one() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    participation, _ = gate(_population(4, 32, seed=7))
    row_sums = participation.sum(dim=-1)
    # These are neural gains, not probabilities: a sparsemax/softmax would
    # pin every row-sum to exactly 1. Here the sums are well above 1 and
    # vary from example to example -- no simplex projection anywhere.
    assert row_sums.mean() > 2.0
    assert (row_sums >= 1.0 - 1e-4).all()  # max-drive cell alone contributes 1
    assert row_sums.std() > 1e-3


# --- delta responds correctly to precision -----------------------------


def test_delta_is_non_increasing_in_population_confidence() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    confidence = torch.linspace(-6.0, 6.0, 200)
    delta = gate.delta_from_confidence(confidence)
    assert torch.all(delta[1:] <= delta[:-1] + 1e-6)
    assert torch.all(delta >= gate.delta_floor - 1e-9)


def test_uniformly_more_evidence_does_not_increase_delta() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    belief = _population(4, 48, seed=11)
    more_evidence = BeliefCell(belief.mu, belief.evidence * 4.0, belief.uncertainty)

    _, base = gate(belief)
    _, scaled = gate(more_evidence)
    key = "assembly_population_log_precision"
    assert scaled[key] > base[key]
    assert scaled["assembly_delta"] <= base["assembly_delta"] + 1e-6


def test_uniformly_less_uncertainty_does_not_increase_delta() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    belief = _population(4, 48, seed=12)
    less_uncertainty = BeliefCell(belief.mu, belief.evidence, belief.uncertainty * 0.25)

    _, base = gate(belief)
    _, sharpened = gate(less_uncertainty)
    key = "assembly_population_log_precision"
    assert sharpened[key] > base[key]
    assert sharpened["assembly_delta"] <= base["assembly_delta"] + 1e-6


# --- normalized drive is affine-invariant -----------------------------


def test_standardized_drive_invariant_to_positive_affine_logit_change() -> None:
    drive = torch.randn(4, 50, generator=torch.Generator().manual_seed(0))
    z = PrecisionRegulatedAssemblyGate._standardize(drive, 1e-8)
    for a, b in [(3.5, 7.0), (0.5, -2.0), (100.0, 1.0)]:
        z_affine = PrecisionRegulatedAssemblyGate._standardize(a * drive + b, 1e-8)
        assert torch.allclose(z, z_affine, atol=1e-4)


# --- gradients reach every learned component -------------------------


def test_gradients_reach_beliefs_and_all_gate_parameters() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    mu = torch.randn(4, 48, requires_grad=True)
    evidence = (torch.rand(4, 48) + 1.5).requires_grad_()
    uncertainty = (torch.rand(4, 48) * 0.3 + 0.3).requires_grad_()

    participation, _ = gate(BeliefCell(mu, evidence, uncertainty))
    participation.sum().backward()

    for name, tensor in [("mu", mu), ("evidence", evidence), ("uncertainty", uncertainty)]:
        assert tensor.grad is not None, f"no grad for {name}"
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0, f"zero grad for {name}"

    for name, param in gate.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0, f"zero grad for {name}"


def test_beliefs_are_not_detached() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    mu = torch.randn(2, 16, requires_grad=True)
    evidence = (torch.rand(2, 16) + 1.0).requires_grad_()
    uncertainty = (torch.rand(2, 16) + 0.5).requires_grad_()
    participation, _ = gate(BeliefCell(mu, evidence, uncertainty))
    assert participation.requires_grad


# --- diagnostics ------------------------------------------------------


def test_diagnostics_do_not_touch_autograd_and_match_definitions() -> None:
    gate = PrecisionRegulatedAssemblyGate()
    belief = _population(5, 40, seed=3)
    participation, diagnostics = gate(belief)

    assert set(diagnostics) == {
        "assembly_active_fraction",
        "assembly_mean_participation",
        "assembly_delta",
        "assembly_population_log_precision",
    }
    assert diagnostics["assembly_active_fraction"] == pytest.approx(
        (participation > 0).to(participation.dtype).mean().item()
    )
    assert diagnostics["assembly_mean_participation"] == pytest.approx(participation.mean().item())
    for value in diagnostics.values():
        assert not value.requires_grad


# --- memory / no [N, N] tensor --------------------------------------


def test_assembly_computation_builds_no_quadratic_tensor() -> None:
    """The gate must never allocate anything of size ~n_cells^2 -- every
    intermediate is (batch, n_cells) or (batch, n_cells, small_const)."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:  # pragma: no cover - dispatch mode always present on >=2.2
        pytest.skip("TorchDispatchMode unavailable")

    class _MaxNumel(TorchDispatchMode):
        def __init__(self) -> None:
            self.max_numel = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_leaves(out):
                if isinstance(leaf, torch.Tensor):
                    self.max_numel = max(self.max_numel, leaf.numel())
            return out

    gate = PrecisionRegulatedAssemblyGate()
    batch, n_cells = 2, 256
    belief = _population(batch, n_cells, seed=1)

    recorder = _MaxNumel()
    with recorder, torch.no_grad():
        gate.participation(belief)

    # A full (batch, n_cells, n_cells) tensor would be 131072 elements;
    # the largest thing the gate actually builds is F_part's hidden
    # activation, (batch, n_cells, hidden_dim) = 4096.
    assert recorder.max_numel < batch * n_cells * 16
    assert recorder.max_numel <= batch * n_cells * 8


def test_activation_memory_scales_linearly_in_cell_count() -> None:
    """Same empirical-scaling discipline as
    `test_learned_association.py::test_no_quadratic_scaling_in_cell_count`."""
    gate = PrecisionRegulatedAssemblyGate()

    def bench(n_cells: int, n_iters: int = 20) -> float:
        belief = _population(4, n_cells, seed=n_cells)
        with torch.no_grad():
            gate.participation(belief)  # warmup
            start = time.perf_counter()
            for _ in range(n_iters):
                gate.participation(belief)
        return (time.perf_counter() - start) / n_iters

    small = bench(512)
    large = bench(4096)  # 8x the cells
    ratio = large / small
    assert ratio < 25.0, f"scaling looks quadratic: {small=:.5f}s {large=:.5f}s {ratio=:.1f}"
