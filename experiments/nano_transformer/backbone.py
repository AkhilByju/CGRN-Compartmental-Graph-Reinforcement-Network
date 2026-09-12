"""Shared decoder-only Transformer backbone for the CellV0.3 nano-transformer
comparison (see the task spec's Sec 4: "Common Transformer backbone").

Every primary model (`ModernTransformer1M`, `CellV03Transformer1M`,
`FixedConfidenceCellTransformer`, `models.py`) is built from exactly this
backbone -- pre-norm blocks, RMSNorm, RoPE, causal self-attention via
`F.scaled_dot_product_attention(is_causal=True)`, tied token embedding / LM
head, no dropout, no attention bias -- and differs *only* in which FFN module
each block wraps. This isolates the FFN neuron primitive as the sole
architectural variable (Sec 4, Sec 5-7).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """`x / rms(x) * weight`. No bias, no mean-subtraction (that's LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """`[max_seq_len, head_dim // 2]` complex tensor of unit-magnitude RoPE
    rotations, cached once (Sec 3: "Cache ... RoPE frequencies ... on the
    appropriate device")."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim // 2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """Apply cached RoPE rotations to `x` of shape `[B, n_heads, T, head_dim]`.
    `rope_freqs` is `[T, head_dim // 2]` complex (a slice of the cached full
    table, already on the same device)."""
    b, h, t, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, h, t, d // 2, 2))
    rotated = x_complex * rope_freqs.view(1, 1, t, d // 2)
    return torch.view_as_real(rotated).reshape(b, h, t, d).type_as(x)


class CausalSelfAttention(nn.Module):
    """Ordinary full multi-head causal self-attention (Sec 4: "not GQA/MQA").
    No biases anywhere; RoPE on q/k; `F.scaled_dot_product_attention` with
    `is_causal=True` -- no `[T, T]` mask is ever materialized."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """`x = x + Attn(RMSNorm(x))`; `x = x + FFN(RMSNorm(x))`.

    `ffn` is any module with `forward(x, **kwargs) -> Tensor` -- this is the
    single point where the three primary models differ (SwiGLU / BeliefFFN /
    FixedConfidenceFFN, `belief_ffn.py` and `models.py`). Unknown `**kwargs`
    (e.g. the neutralization flags in Sec 12) are accepted and ignored by FFNs
    that don't use them, so the block/model interface stays uniform.
    """

    def __init__(self, d_model: int, n_heads: int, ffn: nn.Module) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = ffn

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor, **ffn_kwargs) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x), **ffn_kwargs)
        return x


class TransformerLM(nn.Module):
    """Decoder-only LM: tied embedding -> `n_layers` `TransformerBlock`s ->
    final RMSNorm -> tied LM head. `make_ffn` is called once per layer,
    `make_ffn(layer_index) -> nn.Module`, so a model can (but need not) vary
    the FFN by depth -- every model in this experiment uses the same FFN
    kind and hidden width at every layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        context_length: int,
        make_ffn: Callable[[int], nn.Module],
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.context_length = context_length
        self.head_dim = d_model // n_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, make_ffn(i)) for i in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

        rope_freqs = precompute_rope_freqs(self.head_dim, context_length, base=rope_base)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Tied embedding / LM head (Sec 4, Sec 9): no separate head parameters.
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, **ffn_kwargs) -> torch.Tensor:
        b, t = idx.shape
        if t > self.context_length:
            raise ValueError(f"sequence length {t} exceeds context_length {self.context_length}")
        x = self.tok_emb(idx)
        rope_freqs = self.rope_freqs[:t].to(x.device)
        for block in self.blocks:
            x = block(x, rope_freqs, **ffn_kwargs)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)  # tied head, no separate weight matrix
        return logits

    def param_breakdown(self) -> dict[str, int]:
        """Total / embedding / attention / FFN parameter counts (Sec 9)."""
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        attention = sum(
            p.numel() for block in self.blocks for p in block.attn.parameters()
        )
        ffn = sum(p.numel() for block in self.blocks for p in block.ffn.parameters())
        norms = sum(p.numel() for p in self.parameters()) - embedding - attention - ffn
        total = embedding + attention + ffn + norms
        return {
            "total": total,
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "norms": norms,
        }
