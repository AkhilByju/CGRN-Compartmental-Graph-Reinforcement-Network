import torch

from experiments.nano_transformer.backbone import CausalSelfAttention
from experiments.nano_transformer.belief_ffn import BeliefFFN, FixedConfidenceFFN
from experiments.nano_transformer.models import (
    NanoTransformerSpec,
    SwiGLUFFN,
    build_cellv03_transformer_1m,
    build_fixed_confidence_transformer_1m,
    build_modern_transformer_1m,
)

_SPEC = NanoTransformerSpec(vocab_size=512, d_model=128, n_heads=4, n_layers=5, context_length=256)
_BUILDERS = {
    "swiglu": (build_modern_transformer_1m, SwiGLUFFN),
    "cellv0.3": (build_cellv03_transformer_1m, BeliefFFN),
    "fixed_confidence": (build_fixed_confidence_transformer_1m, FixedConfidenceFFN),
}


def test_all_three_models_within_one_percent_of_target():
    target = 1_000_000
    for name, (builder, _) in _BUILDERS.items():
        built = builder(_SPEC, target=target)
        actual = sum(p.numel() for p in built.model.parameters())
        assert abs(actual - target) / target <= 0.01, f"{name}: {actual} vs {target}"


def test_param_solver_prediction_matches_real_module_count_exactly():
    """Sec 16: 'parameter solver accuracy' -- the closed-form prediction used
    to pick d_hidden must equal the real nn.Module's parameter count, not
    merely land close to the target."""
    for name, (builder, _) in _BUILDERS.items():
        built = builder(_SPEC, target=1_000_000)
        actual = sum(p.numel() for p in built.model.parameters())
        assert built.width_solution.parameter_count == actual, name


def test_correct_ffn_kind_used_per_model():
    for name, (builder, ffn_cls) in _BUILDERS.items():
        built = builder(_SPEC, target=1_000_000)
        for block in built.model.blocks:
            assert isinstance(block.ffn, ffn_cls), name


def test_identical_attention_implementation_across_primary_models():
    for name, (builder, _) in _BUILDERS.items():
        built = builder(_SPEC, target=1_000_000)
        for block in built.model.blocks:
            assert type(block.attn) is CausalSelfAttention, name
            assert block.attn.n_heads == _SPEC.n_heads
            assert block.attn.head_dim == _SPEC.d_model // _SPEC.n_heads


def test_attention_weight_copy_gives_identical_attention_output_across_models():
    """Same attention class + same weights + same input -> same output,
    regardless of which FFN kind the rest of the block uses."""
    torch.manual_seed(0)
    built_a = build_modern_transformer_1m(_SPEC, target=1_000_000)
    built_b = build_cellv03_transformer_1m(_SPEC, target=1_000_000)

    attn_a = built_a.model.blocks[0].attn
    attn_b = built_b.model.blocks[0].attn
    attn_b.load_state_dict(attn_a.state_dict())

    x = torch.randn(2, 16, _SPEC.d_model)
    rope = built_a.model.rope_freqs[:16]
    out_a = attn_a(x, rope)
    out_b = attn_b(x, rope)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_tied_embeddings_for_all_models():
    for name, (builder, _) in _BUILDERS.items():
        built = builder(_SPEC, target=1_000_000)
        names = {n for n, _ in built.model.named_parameters()}
        assert not any("lm_head" in n for n in names), name


def test_forward_backward_finite_for_all_models():
    torch.manual_seed(0)
    idx = torch.randint(0, _SPEC.vocab_size, (2, 32))
    for name, (builder, _) in _BUILDERS.items():
        built = builder(_SPEC, target=1_000_000)
        logits = built.model(idx)
        assert torch.isfinite(logits).all(), name
        loss = logits.float().pow(2).mean()
        loss.backward()
        for pname, p in built.model.named_parameters():
            assert p.grad is not None, f"{name}.{pname}"
            assert torch.isfinite(p.grad).all(), f"{name}.{pname}"


def test_param_report_breakdown_keys():
    built = build_cellv03_transformer_1m(_SPEC, target=1_000_000)
    report = built.param_report()
    for key in ("ffn_kind", "ffn_hidden_width", "total", "embedding", "attention", "ffn", "norms"):
        assert key in report
