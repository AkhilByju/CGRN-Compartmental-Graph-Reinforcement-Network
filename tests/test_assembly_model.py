"""CellV1.6 -- `PrecisionRegulatedAssemblyNetwork` (docs/architecture_v1.md
Sec 17): end-to-end shape/gradient checks, exact equivalence to the
CellV0.1 `BeliefNetwork` when the gate is disabled or forced to
participation-ones, and a short AdamW smoke-training run. No
experiment-quality claim -- "this doesn't crash and can fit noise."
"""

from __future__ import annotations

import pytest
import torch

from src.evaluation.efficiency import count_parameters
from src.models.architecture_v0.belief_network import BeliefNetwork
from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v1.assembly_model import PrecisionRegulatedAssemblyNetwork


def _copy_shared_layers(dst: PrecisionRegulatedAssemblyNetwork, src: BeliefNetwork) -> None:
    dst.layer1.load_state_dict(src.layer1.state_dict())
    dst.layer2.load_state_dict(src.layer2.state_dict())
    dst.readout.load_state_dict(src.readout.state_dict())


def test_forward_shape_and_validity() -> None:
    torch.manual_seed(0)
    model = PrecisionRegulatedAssemblyNetwork(in_features=5, hidden_cells=16, out_features=2)
    x = torch.randn(8, 5)
    pred, belief = model.forward_with_beliefs(x)
    assert pred.shape == (8, 2)
    assert belief.shape == (8, 16)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(belief.evidence).all()
    assert torch.isfinite(belief.uncertainty).all()
    assert set(model.last_assembly_diagnostics) == {
        "assembly_active_fraction",
        "assembly_mean_participation",
        "assembly_delta",
        "assembly_population_log_precision",
    }


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_end_to_end() -> None:
    device = torch.device("mps")
    model = PrecisionRegulatedAssemblyNetwork(4, 12, 1).to(device)
    x = torch.randn(6, 4, device=device)
    out = model(x)
    assert out.device.type == "mps"
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_param_count_increase_over_cellv0_1_is_exactly_34() -> None:
    cellv0_1 = BeliefNetwork(6, 24, 3, aggregation="scale_stable_precision")
    cellv1_6 = PrecisionRegulatedAssemblyNetwork(6, 24, 3)
    # F_part: (2*8+8) + (8*1, no output bias) = 32; plus kappa_raw + width_bias.
    assert count_parameters(cellv1_6) - count_parameters(cellv0_1) == 34


def test_gate_disabled_reproduces_cellv0_1_exactly() -> None:
    torch.manual_seed(1)
    cellv0_1 = BeliefNetwork(7, 20, 2, aggregation="scale_stable_precision")
    model = PrecisionRegulatedAssemblyNetwork(7, 20, 2, use_assembly_gate=False)
    _copy_shared_layers(model, cellv0_1)

    x = torch.randn(16, 7)
    assert torch.equal(model(x), cellv0_1(x))


def test_participation_ones_reproduces_cellv0_1_to_tolerance() -> None:
    torch.manual_seed(2)
    cellv0_1 = BeliefNetwork(5, 16, 1, aggregation="scale_stable_precision")
    model = PrecisionRegulatedAssemblyNetwork(5, 16, 1)
    _copy_shared_layers(model, cellv0_1)

    x = torch.randn(12, 5)
    belief_in = model.layer1(BeliefCell.from_observed_features(x))
    ones = torch.ones_like(belief_in.mu)
    forced = model.readout(model.layer2(belief_in, source_participation=ones).mu)

    reference = cellv0_1(x)
    assert torch.allclose(forced, reference, atol=1e-6, rtol=0)


def test_gradients_reach_every_parameter() -> None:
    torch.manual_seed(3)
    model = PrecisionRegulatedAssemblyNetwork(4, 12, 1)
    x = torch.randn(8, 4)
    model(x).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_short_adamw_smoke_training_decreases_loss() -> None:
    torch.manual_seed(4)
    model = PrecisionRegulatedAssemblyNetwork(6, 16, 1)
    x = torch.randn(32, 6)
    y = torch.randn(32, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_assembly_gate_changes_the_output() -> None:
    # Sanity: the mechanism is actually wired in -- with the same shared
    # weights, gate-on and gate-off must differ.
    torch.manual_seed(5)
    model = PrecisionRegulatedAssemblyNetwork(5, 16, 1, width_bias_init=-1.0)
    x = torch.randn(8, 5)
    with_gate = model(x)
    model.use_assembly_gate = False
    without_gate = model(x)
    assert not torch.allclose(with_gate, without_gate)
