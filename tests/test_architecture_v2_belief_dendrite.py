"""Architecture V2 -- `BeliefDendriteLayer` / `BeliefDendriteNetwork`
(docs/architecture_v2.md Sec D/E/F).

Covers: dendritic-state validity, the hierarchical conservative-precision
inequality through both levels, replication invariance at the branch and the
soma level, the defining local-corruption-isolation test, within- vs
between-dendrite conflict, reliability-only routing (no separate routing
parameter), and full gradient reach.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v0.precision_gain import effective_precision
from src.models.architecture_v2.belief_dendrite import (
    BeliefDendriteLayer,
    BeliefDendriteNetwork,
    DendriticConnectivity,
)


def _random_belief(batch: int, cells: int, *, dtype=torch.float32, seed: int = 0) -> BeliefCell:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch, cells, generator=g, dtype=dtype)
    evidence = torch.rand(batch, cells, generator=g, dtype=dtype) * 3 + 0.5
    uncertainty = torch.rand(batch, cells, generator=g, dtype=dtype) * 2
    return BeliefCell(mu=mu, evidence=evidence, uncertainty=uncertainty)


def _small_layer(**overrides) -> BeliefDendriteLayer:
    cfg = dict(input_dim=20, num_somas=5, branches_per_soma=4, sources_per_branch=6, seed=1)
    cfg.update(overrides)
    conn = DendriticConnectivity.balanced_random(**cfg)
    return BeliefDendriteLayer(conn)


# ---------------------------------------------------------------------------
# 8-12. dendritic / somatic state validity
# ---------------------------------------------------------------------------


def test_branch_and_soma_state_are_positive_finite_and_shaped_right() -> None:
    layer = _small_layer().double()
    belief = _random_belief(7, 20, dtype=torch.float64)
    out, diag = layer.forward_verbose(belief)

    assert out.mu.shape == (7, 5)
    assert torch.isfinite(out.mu).all()
    assert bool((out.evidence > 0).all())
    assert bool((out.uncertainty >= 0).all())
    assert torch.isfinite(diag.pi_branch).all()
    assert bool((diag.e_branch > 0).all())
    assert bool((diag.u_branch >= 0).all())
    assert bool((diag.pi_branch > 0).all())
    assert diag.pi_branch.shape == (7, 5, 4)


def test_wrong_input_width_raises() -> None:
    layer = _small_layer()
    with pytest.raises(ValueError):
        layer(_random_belief(3, 19))


# ---------------------------------------------------------------------------
# hierarchical conservative precision: pi_branch <= e_branch <= max source pi,
# and pi_soma <= e_soma <= max branch pi, therefore pi_soma <= max source pi.
# ---------------------------------------------------------------------------


def test_hierarchical_conservative_precision_through_both_levels() -> None:
    layer = _small_layer(
        input_dim=30, num_somas=6, branches_per_soma=5, sources_per_branch=8, seed=2
    ).double()
    belief = _random_belief(9, 30, dtype=torch.float64, seed=3)
    out, diag = layer.forward_verbose(belief)

    pi_source = effective_precision(belief.evidence, belief.uncertainty)  # [batch, 30]
    max_source_pi = pi_source.max(dim=-1).values  # [batch]
    pi_soma = effective_precision(out.evidence, out.uncertainty)  # [batch, H]
    max_branch_pi_per_soma = diag.pi_branch.max(dim=-1).values  # [batch, H]

    tol = 1e-9
    assert bool((diag.pi_branch <= diag.e_branch + tol).all())
    assert bool((diag.e_soma <= max_branch_pi_per_soma + tol).all())
    assert bool((pi_soma <= diag.e_soma + tol).all())
    assert bool((pi_soma <= max_source_pi.unsqueeze(-1) + tol).all())


# ---------------------------------------------------------------------------
# 13. branch-level replication invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4, 8])
def test_branch_replication_invariance(k: int) -> None:
    in_cells = 6
    conn = DendriticConnectivity(
        torch.tensor([list(range(in_cells))]),
        input_dim=in_cells,
        num_somas=1,
        branches_per_soma=1,
        sources_per_branch=in_cells,
        mode="custom",
        seed=None,
    )
    base = BeliefDendriteLayer(conn).double()
    belief = _random_belief(4, in_cells, dtype=torch.float64, seed=k)
    out_base, diag_base = base.forward_verbose(belief)

    conn_rep = DendriticConnectivity(
        torch.tensor([list(range(in_cells)) * k]),
        input_dim=in_cells,
        num_somas=1,
        branches_per_soma=1,
        sources_per_branch=in_cells * k,
        mode="custom",
        seed=None,
    )
    replicated = BeliefDendriteLayer(conn_rep).double()
    with torch.no_grad():
        replicated.V_branch.copy_(base.V_branch.repeat(1, k))
        replicated.gain_raw_branch.copy_(base.gain_raw_branch)
        replicated.bias_branch.copy_(base.bias_branch)
        replicated.V_cable.copy_(base.V_cable)
        replicated.gain_raw_soma.copy_(base.gain_raw_soma)
        replicated.bias_soma.copy_(base.bias_soma)
    out_rep, diag_rep = replicated.forward_verbose(belief)

    assert torch.allclose(out_base.mu, out_rep.mu, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_base.evidence, out_rep.evidence, atol=1e-9, rtol=1e-7)
    assert torch.allclose(out_base.uncertainty, out_rep.uncertainty, atol=1e-9, rtol=1e-7)
    assert torch.allclose(diag_base.pi_branch, diag_rep.pi_branch, atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# 14. soma-level replication invariance -- duplicate equivalent dendrites.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 4])
def test_soma_replication_invariance(k: int) -> None:
    in_cells, branches = 8, 3
    conn = DendriticConnectivity.balanced_random(
        input_dim=in_cells, num_somas=1, branches_per_soma=branches, sources_per_branch=5, seed=11
    )
    base = BeliefDendriteLayer(conn).double()
    belief = _random_belief(4, in_cells, dtype=torch.float64, seed=100 + k)
    out_base, diag_base = base.forward_verbose(belief)

    # k identical copies of the same `branches` sources -> M = branches * k.
    idx_rep = conn.source_idx.repeat(k, 1)
    conn_rep = DendriticConnectivity(
        idx_rep,
        input_dim=in_cells,
        num_somas=1,
        branches_per_soma=branches * k,
        sources_per_branch=5,
        mode="custom",
        seed=None,
    )
    replicated = BeliefDendriteLayer(conn_rep).double()
    with torch.no_grad():
        replicated.V_branch.copy_(base.V_branch.repeat(k, 1))
        replicated.gain_raw_branch.copy_(base.gain_raw_branch.repeat(k))
        replicated.bias_branch.copy_(base.bias_branch.repeat(k))
        replicated.V_cable.copy_(base.V_cable.repeat(1, k))
        replicated.gain_raw_soma.copy_(base.gain_raw_soma)
        replicated.bias_soma.copy_(base.bias_soma)
    out_rep, _ = replicated.forward_verbose(belief)

    assert torch.allclose(out_base.mu, out_rep.mu, atol=1e-8, rtol=1e-6)
    assert torch.allclose(out_base.evidence, out_rep.evidence, atol=1e-8, rtol=1e-6)
    assert torch.allclose(out_base.uncertainty, out_rep.uncertainty, atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------
# 15-18. the defining local-corruption-isolation test.
# ---------------------------------------------------------------------------


def test_degrading_one_branchs_sources_only_lowers_that_branchs_precision_and_influence() -> None:
    idx = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])  # soma 0: branch A, branch B
    conn = DendriticConnectivity(
        idx,
        input_dim=8,
        num_somas=1,
        branches_per_soma=2,
        sources_per_branch=4,
        mode="custom",
        seed=None,
    )
    layer = BeliefDendriteLayer(conn).double()

    belief = _random_belief(6, 8, dtype=torch.float64, seed=42)
    belief_clean = BeliefCell(
        mu=belief.mu, evidence=torch.ones_like(belief.mu), uncertainty=torch.zeros_like(belief.mu)
    )
    out_clean, diag_clean = layer.forward_verbose(belief_clean)

    e_degraded = belief_clean.evidence.clone()
    e_degraded[:, 0:4] = 1e-4  # branch A's sources only
    belief_degraded = BeliefCell(
        mu=belief.mu, evidence=e_degraded, uncertainty=torch.zeros_like(belief.mu)
    )
    out_degraded, diag_degraded = layer.forward_verbose(belief_degraded)

    pi_a_clean, pi_a_deg = diag_clean.pi_branch[:, 0, 0], diag_degraded.pi_branch[:, 0, 0]
    pi_b_clean, pi_b_deg = diag_clean.pi_branch[:, 0, 1], diag_degraded.pi_branch[:, 0, 1]

    assert bool((pi_a_deg < pi_a_clean * 0.5).all())  # branch A precision falls substantially
    assert torch.allclose(pi_b_clean, pi_b_deg, atol=1e-9)  # branch B untouched

    r_clean = diag_clean.branch_contribution()[:, 0, 0]
    r_degraded = diag_degraded.branch_contribution()[:, 0, 0]
    assert bool((r_degraded < r_clean).all())  # branch A's somatic influence falls


# ---------------------------------------------------------------------------
# 19-21. within- vs between-dendrite conflict.
# ---------------------------------------------------------------------------


def test_within_branch_disagreement_raises_u_branch_and_lowers_pi_branch() -> None:
    idx = torch.tensor([[0, 1, 2, 3]])
    conn = DendriticConnectivity(
        idx,
        input_dim=4,
        num_somas=1,
        branches_per_soma=1,
        sources_per_branch=4,
        mode="custom",
        seed=None,
    )
    layer = BeliefDendriteLayer(conn).double()
    with torch.no_grad():
        layer.V_branch.copy_(torch.ones(1, 4, dtype=torch.float64))  # all-positive, equal weight

    agree = torch.full((3, 4), 0.6, dtype=torch.float64)
    e1 = torch.ones(3, 4, dtype=torch.float64)
    _, diag_agree = layer.forward_verbose(
        BeliefCell(mu=agree, evidence=e1, uncertainty=torch.zeros(3, 4, dtype=torch.float64))
    )

    disagree = torch.tensor([[0.9, -0.9, 0.9, -0.9]] * 3, dtype=torch.float64)
    _, diag_disagree = layer.forward_verbose(
        BeliefCell(mu=disagree, evidence=e1, uncertainty=torch.zeros(3, 4, dtype=torch.float64))
    )

    assert bool((diag_disagree.u_branch > diag_agree.u_branch).all())
    assert bool((diag_disagree.pi_branch < diag_agree.pi_branch).all())


def test_between_branch_disagreement_raises_u_soma_and_lowers_pi_soma() -> None:
    idx = torch.tensor([[0, 1], [2, 3]])
    conn = DendriticConnectivity(
        idx,
        input_dim=4,
        num_somas=1,
        branches_per_soma=2,
        sources_per_branch=2,
        mode="custom",
        seed=None,
    )
    layer = BeliefDendriteLayer(conn).double()
    with torch.no_grad():
        layer.V_branch.copy_(torch.ones(2, 2, dtype=torch.float64))
        layer.V_cable.copy_(torch.ones(1, 2, dtype=torch.float64))

    e1 = torch.ones(3, 4, dtype=torch.float64)
    u0 = torch.zeros(3, 4, dtype=torch.float64)

    agree = torch.full(
        (3, 4), 0.5, dtype=torch.float64
    )  # both branches will land on the same consensus
    out_agree, diag_agree = layer.forward_verbose(BeliefCell(mu=agree, evidence=e1, uncertainty=u0))

    disagree = torch.tensor(
        [[0.9, 0.9, -0.9, -0.9]] * 3, dtype=torch.float64
    )  # branch A vs branch B disagree
    out_disagree, diag_disagree = layer.forward_verbose(
        BeliefCell(mu=disagree, evidence=e1, uncertainty=u0)
    )

    pi_soma_agree = effective_precision(out_agree.evidence, out_agree.uncertainty)
    pi_soma_disagree = effective_precision(out_disagree.evidence, out_disagree.uncertainty)

    assert bool((out_disagree.uncertainty > out_agree.uncertainty).all())
    assert bool((pi_soma_disagree < pi_soma_agree).all())


# ---------------------------------------------------------------------------
# routing by reliability: normalized somatic contribution sums to ~1, no
# separate routing parameter exists anywhere in the module.
# ---------------------------------------------------------------------------


def test_branch_contributions_sum_to_one_and_have_no_dedicated_routing_parameter() -> None:
    layer = _small_layer(
        input_dim=25, num_somas=4, branches_per_soma=5, sources_per_branch=7, seed=6
    ).double()
    belief = _random_belief(5, 25, dtype=torch.float64, seed=8)
    _, diag = layer.forward_verbose(belief)

    totals = diag.branch_contribution().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)

    param_names = {name for name, _ in layer.named_parameters()}
    assert param_names == {
        "V_branch",
        "gain_raw_branch",
        "bias_branch",
        "V_cable",
        "gain_raw_soma",
        "bias_soma",
    }


def test_lowering_one_branch_precision_lowers_only_its_own_contribution() -> None:
    layer = _small_layer(
        input_dim=25, num_somas=4, branches_per_soma=5, sources_per_branch=7, seed=6
    ).double()
    belief = _random_belief(5, 25, dtype=torch.float64, seed=8)
    _, diag_before = layer.forward_verbose(belief)

    degraded_evidence = belief.evidence.clone()
    branch_sources = layer.connectivity.source_idx[0]  # soma 0, branch 0's sources
    degraded_evidence[:, branch_sources] *= 1e-4
    degraded = BeliefCell(mu=belief.mu, evidence=degraded_evidence, uncertainty=belief.uncertainty)
    _, diag_after = layer.forward_verbose(degraded)

    before = diag_before.branch_contribution()[:, 0, :]
    after = diag_after.branch_contribution()[:, 0, :]
    assert bool((after[:, 0] < before[:, 0]).all())
    assert bool((after[:, 1:].sum(dim=-1) > before[:, 1:].sum(dim=-1)).all())


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def test_gradients_reach_source_state_and_every_branch_and_soma_parameter() -> None:
    layer = _small_layer()
    belief = _random_belief(6, 20)
    mu = belief.mu.clone().requires_grad_(True)
    e = belief.evidence.clone().requires_grad_(True)
    u = belief.uncertainty.clone().requires_grad_(True)

    out = layer(BeliefCell(mu=mu, evidence=e, uncertainty=u))
    (out.mu.sum() + out.evidence.sum() + out.uncertainty.sum()).backward()

    for name, t in (("mu", mu), ("e", e), ("u", u)):
        assert t.grad is not None, name
        assert torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, name

    for name, p in layer.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_network_gradients_reach_both_layers_and_readout_through_task_loss() -> None:
    torch.manual_seed(0)
    net = BeliefDendriteNetwork.build(
        in_features=16, out_features=3, hidden1=8, hidden2=6, branches_per_soma=4, seed=0
    )
    x = torch.randn(10, 16)
    rel = torch.rand(10, 16) * 0.8 + 0.2
    y = torch.randint(0, 3, (10,))

    F.cross_entropy(net(x, rel), y).backward()

    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_no_reliability_defaults_to_e_equals_one() -> None:
    torch.manual_seed(1)
    net = BeliefDendriteNetwork.build(
        in_features=10, out_features=2, hidden1=6, hidden2=4, branches_per_soma=3, seed=1
    )
    x = torch.randn(4, 10)
    belief = net.input_belief(x)
    assert torch.allclose(belief.evidence, torch.ones_like(x))
    assert torch.allclose(belief.uncertainty, torch.zeros_like(x))
    out = net(x)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# network-level smoke: image + vector construction, shapes, finiteness.
# ---------------------------------------------------------------------------


def test_image_network_forward_shape_and_finite() -> None:
    torch.manual_seed(2)
    net = BeliefDendriteNetwork.build(
        in_features=784,
        out_features=10,
        hidden1=16,
        hidden2=12,
        branches_per_soma=4,
        seed=0,
        image_shape=(1, 28, 28),
        patch=7,
    )
    x = torch.randn(9, 784)
    rel = torch.rand(9, 784)
    out = net(x, rel)
    assert out.shape == (9, 10)
    assert torch.isfinite(out).all()
