import torch
from torch import nn

from experiments.nano_transformer.backbone import (
    CausalSelfAttention,
    RMSNorm,
    TransformerLM,
    apply_rope,
    precompute_rope_freqs,
)


def _identity_ffn(_d_model_ignored=None):
    class _Identity(nn.Module):
        def forward(self, x, **_unused):
            return torch.zeros_like(x)

    return _Identity()


def _make_model(vocab_size=64, d_model=16, n_heads=4, n_layers=2, context_length=8):
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_length=context_length,
        make_ffn=lambda _i: _identity_ffn(),
    )


def test_rmsnorm_unit_scale_has_rms_one():
    norm = RMSNorm(dim=8)
    x = torch.randn(4, 8) * 5
    out = norm(x)
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_rope_freqs_shape_and_unit_magnitude():
    head_dim, max_seq_len = 32, 256
    freqs = precompute_rope_freqs(head_dim, max_seq_len)
    assert freqs.shape == (max_seq_len, head_dim // 2)
    assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-5)


def test_rope_rejects_odd_head_dim():
    import pytest

    with pytest.raises(ValueError):
        precompute_rope_freqs(head_dim=31, max_seq_len=16)


def test_apply_rope_preserves_vector_norm():
    b, h, t, d = 2, 4, 8, 32
    x = torch.randn(b, h, t, d)
    freqs = precompute_rope_freqs(d, t)
    rotated = apply_rope(x, freqs)
    orig_norm = x.reshape(b, h, t, d // 2, 2).pow(2).sum(-1).sqrt()
    rot_norm = rotated.reshape(b, h, t, d // 2, 2).pow(2).sum(-1).sqrt()
    assert torch.allclose(orig_norm, rot_norm, atol=1e-3)


def test_attention_output_shape():
    attn = CausalSelfAttention(d_model=32, n_heads=4)
    freqs = precompute_rope_freqs(8, 16)
    x = torch.randn(2, 16, 32)
    out = attn(x, freqs)
    assert out.shape == x.shape


def test_attention_rejects_bad_head_split():
    import pytest

    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=30, n_heads=4)


def test_causal_masking_no_future_leakage():
    """Changing a future token must not change logits at earlier positions."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits_a = model(idx)

    idx_modified = idx.clone()
    idx_modified[0, -1] = (idx_modified[0, -1] + 1) % 64
    with torch.no_grad():
        logits_b = model(idx_modified)

    # positions 0..T-2 must be identical; only position T-1's *input* changed,
    # so only outputs that could depend on position T-1 (i.e. logits at
    # position T-1 itself) are allowed to differ.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)


def test_causal_masking_output_depends_on_all_prior_tokens():
    """Sanity check the leakage test isn't vacuous: an early-token change
    must change later logits."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits_a = model(idx)

    idx_modified = idx.clone()
    idx_modified[0, 0] = (idx_modified[0, 0] + 1) % 64
    with torch.no_grad():
        logits_b = model(idx_modified)

    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-6)


def test_rejects_sequence_longer_than_context():
    import pytest

    model = _make_model(context_length=8)
    idx = torch.randint(0, 64, (1, 9))
    with pytest.raises(ValueError):
        model(idx)


def test_tied_embedding_and_lm_head():
    model = _make_model()
    # No separate lm_head parameter should exist; logits are computed via
    # F.linear(x, tok_emb.weight) in TransformerLM.forward (see source).
    param_names = {name for name, _ in model.named_parameters()}
    assert not any("lm_head" in name for name in param_names)
    assert "tok_emb.weight" in param_names


def test_param_breakdown_sums_to_total():
    model = _make_model()
    breakdown = model.param_breakdown()
    total_from_module = sum(p.numel() for p in model.parameters())
    assert breakdown["total"] == total_from_module
    assert breakdown["total"] == (
        breakdown["embedding"] + breakdown["attention"] + breakdown["ffn"] + breakdown["norms"]
    )
