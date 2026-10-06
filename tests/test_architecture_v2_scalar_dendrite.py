"""Architecture V2 -- scalar dendritic controls (docs/architecture_v2.md
Sec G): `ScalarDendriteLayer`/`Network` (same topology, no belief state) and
`ScalarDendriteReliabilityGated` (reliability applied once at the input,
never propagated internally).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.models.architecture_v2.belief_dendrite import (
    BeliefDendriteNetwork,
    DendriticConnectivity,
    ScalarDendriteLayer,
    ScalarDendriteNetwork,
    ScalarDendriteReliabilityGated,
)


def _conn(seed=0, input_dim=20, h=5, b=4, k=6) -> DendriticConnectivity:
    return DendriticConnectivity.balanced_random(
        input_dim=input_dim, num_somas=h, branches_per_soma=b, sources_per_branch=k, seed=seed
    )


def test_scalar_layer_forward_shape_and_finite() -> None:
    conn = _conn()
    layer = ScalarDendriteLayer(conn)
    x = torch.randn(7, 20)
    out = layer(x)
    assert out.shape == (7, 5)
    assert torch.isfinite(out).all()


def test_scalar_layer_same_parameter_shapes_as_belief_layer() -> None:
    from src.models.architecture_v2.belief_dendrite import BeliefDendriteLayer

    conn = _conn()
    scalar = ScalarDendriteLayer(conn)
    belief = BeliefDendriteLayer(conn)
    scalar_shapes = {n: tuple(p.shape) for n, p in scalar.named_parameters()}
    belief_shapes = {n: tuple(p.shape) for n, p in belief.named_parameters()}
    assert scalar_shapes == belief_shapes


def test_scalar_network_gradients_reach_every_parameter() -> None:
    torch.manual_seed(0)
    net = ScalarDendriteNetwork.build(
        in_features=16, out_features=3, hidden1=8, hidden2=6, branches_per_soma=4, seed=0
    )
    x = torch.randn(10, 16)
    y = torch.randint(0, 3, (10,))
    F.cross_entropy(net(x), y).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_reliability_gated_matches_scalar_network_on_pre_gated_input() -> None:
    """`ScalarDendriteReliabilityGated(x, c)` must be exactly
    `ScalarDendriteNetwork(c * x)` on the identical topology -- the gate is
    applied once, at the input, and nothing downstream sees `c` again."""
    torch.manual_seed(0)
    conn1, conn2 = (
        DendriticConnectivity.balanced_random(
            input_dim=12, num_somas=7, branches_per_soma=4, sources_per_branch=5, seed=0
        ),
        DendriticConnectivity.balanced_random(
            input_dim=7, num_somas=4, branches_per_soma=4, sources_per_branch=5, seed=1
        ),
    )
    gated = ScalarDendriteReliabilityGated(conn1, conn2, out_features=3)
    plain = ScalarDendriteNetwork(conn1, conn2, out_features=3)
    with torch.no_grad():
        plain.layer1.load_state_dict(gated.net.layer1.state_dict())
        plain.layer2.load_state_dict(gated.net.layer2.state_dict())
        plain.readout.load_state_dict(gated.net.readout.state_dict())

    x = torch.randn(6, 12)
    c = torch.rand(6, 12) * 0.8 + 0.1
    out_gated = gated(x, c)
    out_plain = plain(c * x)
    assert torch.allclose(out_gated, out_plain, atol=1e-6)


def test_reliability_gated_ignores_reliability_when_none() -> None:
    torch.manual_seed(1)
    net = ScalarDendriteReliabilityGated.build(
        in_features=10, out_features=2, hidden1=6, hidden2=4, branches_per_soma=3, seed=1
    )
    x = torch.randn(5, 10)
    assert torch.allclose(net(x), net(x, None))


def test_belief_and_scalar_networks_can_share_identical_topology() -> None:
    torch.manual_seed(0)
    belief_net = BeliefDendriteNetwork.build(
        in_features=784,
        out_features=10,
        hidden1=16,
        hidden2=12,
        branches_per_soma=4,
        seed=0,
        image_shape=(1, 28, 28),
        patch=7,
    )
    scalar_net = ScalarDendriteNetwork(
        belief_net.layer1.connectivity, belief_net.layer2.connectivity, out_features=10
    )
    gated_net = ScalarDendriteReliabilityGated(
        belief_net.layer1.connectivity, belief_net.layer2.connectivity, out_features=10
    )

    assert scalar_net.layer1.connectivity is belief_net.layer1.connectivity
    assert torch.equal(
        scalar_net.layer1.connectivity.source_idx, belief_net.layer1.connectivity.source_idx
    )
    assert torch.equal(
        gated_net.net.layer2.connectivity.source_idx, belief_net.layer2.connectivity.source_idx
    )
