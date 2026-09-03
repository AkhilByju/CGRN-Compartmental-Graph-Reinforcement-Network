import torch

from src.models.architecture_v1.field_functions import (
    BandwidthFunction,
    FieldSemanticUpdateFunction,
    MassFunction,
    RoutingFunction,
    RoutingGateFunction,
)


def _state(batch=3, n=8, d=6):
    mu = torch.tanh(torch.randn(batch, n))
    e = torch.rand(batch, n) + 0.1
    u = torch.rand(batch, n) + 0.1
    z = torch.randn(batch, n, d)
    z = z / z.norm(dim=-1, keepdim=True)
    return mu, e, u, z


def test_routing_function_shape() -> None:
    mu, e, u, z = _state(d=6)
    f = RoutingFunction(association_dim=6, routing_dim=5, hidden_dim=8)
    r = f(mu, e, u, z)
    assert r.shape == (3, 8, 5)


def test_bandwidth_function_respects_h_min() -> None:
    mu, e, u, z = _state(d=6)
    f = BandwidthFunction(association_dim=6, hidden_dim=8, h_min=0.3)
    h = f(mu, e, u, z)
    assert h.shape == (3, 8)
    assert (h >= 0.3).all()


def test_mass_function_is_nonnegative() -> None:
    mu, e, u, z = _state(d=6)
    f = MassFunction(association_dim=6, hidden_dim=8)
    m = f(mu, e, u, z)
    assert (m >= 0).all()


def test_routing_gate_is_bounded() -> None:
    mu, e, u, _ = _state(d=6)
    f = RoutingGateFunction(hidden_dim=8)
    beta = f(mu, e, u, mu, e, u)
    assert beta.shape == (3, 8)
    assert (beta >= 0).all() and (beta <= 1).all()


def test_field_semantic_update_shape() -> None:
    mu, e, u, z = _state(d=6)
    routing_dim = 5
    r = torch.randn(3, 8, routing_dim)
    r_bar = torch.randn(3, 8, routing_dim)
    f = FieldSemanticUpdateFunction(association_dim=6, routing_dim=routing_dim, hidden_dim=8)
    delta = f(z, r, r_bar, mu, e, u)
    assert delta.shape == z.shape


def test_gradients_flow_through_all_field_functions() -> None:
    mu, e, u, z = _state(d=6)
    mu.requires_grad_(True)
    z.requires_grad_(True)
    routing_fn = RoutingFunction(6, 5, hidden_dim=8)
    bandwidth_fn = BandwidthFunction(6, hidden_dim=8)
    mass_fn = MassFunction(6, hidden_dim=8)
    gate_fn = RoutingGateFunction(hidden_dim=8)
    update_fn = FieldSemanticUpdateFunction(6, 5, hidden_dim=8)

    r = routing_fn(mu, e, u, z)
    h = bandwidth_fn(mu, e, u, z)
    m = mass_fn(mu, e, u, z)
    beta = gate_fn(mu, e, u, mu, e, u)
    delta = update_fn(z, r, r, mu, e, u)

    (r.sum() + h.sum() + m.sum() + beta.sum() + delta.sum()).backward()

    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(z.grad).all()
    for module in (routing_fn, bandwidth_fn, mass_fn, gate_fn, update_fn):
        for p in module.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
