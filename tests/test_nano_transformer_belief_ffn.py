import torch

from experiments.nano_transformer.belief_ffn import BeliefFFN, FixedConfidenceFFN
from src.models.architecture_v0.conflict_normalized import ConflictNormalizedLayer
from src.models.architecture_v0.precision_gain import effective_precision, initial_belief


def _copy_weights(src_layer: ConflictNormalizedLayer, dst_ffn: BeliefFFN) -> None:
    with torch.no_grad():
        dst_ffn.V.copy_(src_layer.V)
        dst_ffn.gain_raw.copy_(src_layer.gain_raw)
        dst_ffn.bias.copy_(src_layer.bias)


def test_belief_ffn_forward_parity_with_frozen_conflict_normalized_layer():
    torch.manual_seed(0)
    d_model, d_hidden, batch = 16, 24, 5
    frozen = ConflictNormalizedLayer(d_model, d_hidden)
    ffn = BeliefFFN(d_model, d_hidden)
    _copy_weights(frozen, ffn)

    x = torch.randn(batch, d_model, dtype=torch.float64)
    frozen = frozen.double()
    ffn.V.data = ffn.V.data.double()
    ffn.gain_raw.data = ffn.gain_raw.data.double()
    ffn.bias.data = ffn.bias.data.double()

    frozen_belief, frozen_consensus = frozen.forward_verbose(initial_belief(x))
    frozen_pi = effective_precision(frozen_belief.evidence, frozen_belief.uncertainty)

    consensus, e, u, pi, gamma = ffn._hidden_belief(x)

    assert torch.allclose(consensus, frozen_consensus, atol=1e-9)
    assert torch.allclose(e, frozen_belief.evidence, atol=1e-9)
    assert torch.allclose(u, frozen_belief.uncertainty, atol=1e-9)
    assert torch.allclose(pi, frozen_pi, atol=1e-9)

    mu_hidden, _, _ = ffn._activation(x, neutralize_precision=False, neutralize_conflict=False)
    assert torch.allclose(mu_hidden, frozen_belief.mu, atol=1e-9)


def test_belief_ffn_gradient_parity_with_frozen_layer():
    torch.manual_seed(1)
    d_model, d_hidden, batch = 12, 20, 4

    frozen = ConflictNormalizedLayer(d_model, d_hidden).double()
    ffn = BeliefFFN(d_model, d_hidden)
    _copy_weights(frozen, ffn)
    ffn = ffn.double()

    x0 = torch.randn(batch, d_model, dtype=torch.float64)
    x_frozen = x0.clone().requires_grad_(True)
    x_ffn = x0.clone().requires_grad_(True)

    frozen_belief = frozen(initial_belief(x_frozen))
    frozen_belief.mu.pow(2).sum().backward()

    mu_hidden, _, _ = ffn._activation(x_ffn, False, False)
    mu_hidden.pow(2).sum().backward()

    assert torch.allclose(ffn.V.grad, frozen.V.grad, atol=1e-8)
    assert torch.allclose(ffn.gain_raw.grad, frozen.gain_raw.grad, atol=1e-8)
    assert torch.allclose(ffn.bias.grad, frozen.bias.grad, atol=1e-8)
    assert torch.allclose(x_ffn.grad, x_frozen.grad, atol=1e-8)


def test_belief_ffn_input_belief_is_neutral_precision():
    """Sec 6: mu_in=x, e_in=1, u_in=0 -> pi_in == 1 identically, so e_hidden
    reduces to the plain row-L1-normalized sum of |V| (no per-token
    weighting) while u_hidden still varies with x."""
    torch.manual_seed(2)
    d_model, d_hidden = 8, 10
    ffn = BeliefFFN(d_model, d_hidden)
    x1 = torch.randn(3, d_model)
    x2 = torch.randn(3, d_model) * 5  # different input

    _, e1, _, _, _ = ffn._hidden_belief(x1)
    _, e2, _, _, _ = ffn._hidden_belief(x2)
    # e_hidden must be input-independent (always == 1) given the neutral input belief.
    assert torch.allclose(e1, torch.ones_like(e1), atol=1e-6)
    assert torch.allclose(e2, torch.ones_like(e2), atol=1e-6)


def test_belief_ffn_end_to_end_output_shape_and_finite():
    ffn = BeliefFFN(d_model=16, d_hidden=32)
    x = torch.randn(2, 6, 16)
    out = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_precision_neutralization_changes_output_and_matches_manual_formula():
    torch.manual_seed(3)
    ffn = BeliefFFN(d_model=8, d_hidden=6)
    x = torch.randn(4, 8)

    normal = ffn(x)
    neutralized = ffn(x, neutralize_precision=True)
    assert not torch.allclose(normal, neutralized)

    consensus, e, u, pi, gamma = ffn._hidden_belief(x)
    expected = ffn.out_proj(torch.tanh(gamma * consensus * 1.0 + ffn.bias))
    assert torch.allclose(neutralized, expected, atol=1e-6)


def test_conflict_neutralization_changes_output_and_matches_manual_formula():
    torch.manual_seed(4)
    ffn = BeliefFFN(d_model=8, d_hidden=6)
    x = torch.randn(4, 8)

    normal = ffn(x)
    neutralized = ffn(x, neutralize_conflict=True)
    consensus, e, u, pi, gamma = ffn._hidden_belief(x)
    expected = ffn.out_proj(torch.tanh(gamma * consensus * e.clamp_min(0).sqrt() + ffn.bias))
    assert torch.allclose(neutralized, expected, atol=1e-6)
    # u should be nontrivial for a random layer, so this should differ from normal.
    if not torch.allclose(u, torch.zeros_like(u), atol=1e-6):
        assert not torch.allclose(normal, neutralized)


def test_forward_with_diagnostics_reports_expected_keys():
    ffn = BeliefFFN(d_model=8, d_hidden=6)
    x = torch.randn(4, 8)
    out, diag = ffn.forward_with_diagnostics(x)
    assert out.shape == x.shape
    for key in ("pi_mean", "pi_std", "pi_cv", "u_mean", "u_std"):
        assert key in diag
        assert diag[key] == diag[key]  # not NaN


def test_no_edge_tensor_allocated_during_belief_ffn_forward():
    """Sec 8: no [batch, seq, out, in] tensor may ever be materialized."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _MaxNumelTracker(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.max_numel = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            outputs = out if isinstance(out, (tuple, list)) else (out,)
            for t in outputs:
                if isinstance(t, torch.Tensor):
                    self.max_numel = max(self.max_numel, t.numel())
            return out

    batch, seq, d_model, d_hidden = 4, 8, 16, 200
    n = batch * seq
    forbidden_edge_numel = n * d_hidden * d_model  # the [B,T,O,I]-style tensor size

    ffn = BeliefFFN(d_model, d_hidden)
    x = torch.randn(batch, seq, d_model)

    tracker = _MaxNumelTracker()
    with tracker:
        ffn(x)

    assert tracker.max_numel < forbidden_edge_numel
    # Largest legitimate tensor is the combined EQ GEMM output, [2N, d_hidden].
    assert tracker.max_numel <= 2 * n * d_hidden * 2  # generous slack for internal copies


def test_fixed_confidence_ffn_has_no_belief_state():
    ffn = FixedConfidenceFFN(d_model=8, d_hidden=6)
    x = torch.randn(3, 8)
    out = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    param_names = {name for name, _ in ffn.named_parameters()}
    assert param_names == {"V", "gain_raw", "bias", "out_proj.weight"}


def test_belief_ffn_and_fixed_confidence_ffn_share_parameter_shapes():
    d_model, d_hidden = 10, 14
    belief = BeliefFFN(d_model, d_hidden)
    fixed = FixedConfidenceFFN(d_model, d_hidden)
    belief_shapes = {n: tuple(p.shape) for n, p in belief.named_parameters()}
    fixed_shapes = {n: tuple(p.shape) for n, p in fixed.named_parameters()}
    assert belief_shapes == fixed_shapes
