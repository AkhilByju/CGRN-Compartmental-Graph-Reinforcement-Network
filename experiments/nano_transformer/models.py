"""The three primary nano-transformer models (task Sec 5, 6, 7): identical
backbone (`backbone.TransformerLM`), differing only in FFN neuron primitive.
Each is parameter-matched to ~1,000,000 total parameters (+/-1%) by solving
for the FFN hidden width `d_hidden` -- `param_solver.py` -- since the three
FFN kinds have different parameter-per-hidden-unit costs (Sec 6: "Because
CellV0.3 and SwiGLU have different parameter scaling, their hidden widths
are allowed to differ").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.nano_transformer.backbone import TransformerLM
from experiments.nano_transformer.belief_ffn import BeliefFFN, FixedConfidenceFFN
from experiments.nano_transformer.param_solver import WidthSolution, solve_hidden_width_for_target


class SwiGLUFFN(nn.Module):
    """Model A's FFN (task Sec 5): `SiLU(W_gate x) * (W_value x) -> W_out`.
    No biases, matching the rest of the backbone's no-bias convention."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_value = nn.Linear(d_model, d_hidden, bias=False)
        self.w_out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor, **_unused) -> torch.Tensor:
        hidden = F.silu(self.w_gate(x)) * self.w_value(x)
        return self.w_out(hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_hidden={self.d_hidden}"


def swiglu_ffn_param_count(d_model: int, d_hidden: int) -> int:
    """`W_gate + W_value + W_out`, no bias: `3 * d_hidden * d_model`."""
    return 3 * d_hidden * d_model


def belief_or_fixed_confidence_ffn_param_count(d_model: int, d_hidden: int) -> int:
    """Shared by `BeliefFFN` and `FixedConfidenceFFN` (identical parameter
    shapes -- task Sec 7): `V` (`d_hidden*d_model`) + `gain_raw` + `bias`
    (`2*d_hidden`) + `out_proj` (`d_hidden*d_model`, no bias) =
    `d_hidden*(2*d_model + 2)`."""
    return d_hidden * (2 * d_model + 2)


def backbone_fixed_param_count(vocab_size: int, d_model: int, n_layers: int) -> int:
    """Every parameter in `TransformerLM` that does not depend on the FFN:
    tied embedding (`vocab_size*d_model`), `n_layers` attention blocks
    (`4*d_model^2` each -- q/k/v/o, no bias), `2*d_model` per-layer RMSNorm
    weights (attn-norm + ffn-norm) plus one final RMSNorm (`d_model`)."""
    embedding = vocab_size * d_model
    attention = n_layers * 4 * d_model * d_model
    norms = n_layers * 2 * d_model + d_model
    return embedding + attention + norms


@dataclass(frozen=True)
class NanoTransformerSpec:
    """The Sec 4 "common architecture" fields shared by all three models."""

    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 5
    context_length: int = 256
    rope_base: float = 10000.0


@dataclass(frozen=True)
class BuiltModel:
    model: TransformerLM
    width_solution: WidthSolution
    ffn_kind: str

    def param_report(self) -> dict[str, int | str]:
        breakdown = self.model.param_breakdown()
        return {
            "ffn_kind": self.ffn_kind,
            "ffn_hidden_width": self.width_solution.hidden,
            **breakdown,
        }


def build_modern_transformer_1m(spec: NanoTransformerSpec, target: int = 1_000_000) -> BuiltModel:
    """Model A (Sec 5): SwiGLU FFN, width-solved to `target` total params."""
    fixed = backbone_fixed_param_count(spec.vocab_size, spec.d_model, spec.n_layers)
    coef = spec.n_layers * 3 * spec.d_model
    solution = solve_hidden_width_for_target(target, fixed, coef)
    model = TransformerLM(
        vocab_size=spec.vocab_size,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        context_length=spec.context_length,
        make_ffn=lambda _i: SwiGLUFFN(spec.d_model, solution.hidden),
        rope_base=spec.rope_base,
    )
    return BuiltModel(model=model, width_solution=solution, ffn_kind="swiglu")


def build_cellv03_transformer_1m(spec: NanoTransformerSpec, target: int = 1_000_000) -> BuiltModel:
    """Model B (Sec 6): CellV0.3 (`BeliefFFN`) hidden layer, width-solved to
    `target` total params."""
    fixed = backbone_fixed_param_count(spec.vocab_size, spec.d_model, spec.n_layers)
    coef = spec.n_layers * (2 * spec.d_model + 2)
    solution = solve_hidden_width_for_target(target, fixed, coef)
    model = TransformerLM(
        vocab_size=spec.vocab_size,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        context_length=spec.context_length,
        make_ffn=lambda _i: BeliefFFN(spec.d_model, solution.hidden),
        rope_base=spec.rope_base,
    )
    return BuiltModel(model=model, width_solution=solution, ffn_kind="cellv0.3")


def build_fixed_confidence_transformer_1m(
    spec: NanoTransformerSpec, target: int = 1_000_000
) -> BuiltModel:
    """Model C (Sec 7): mechanism control, identical parameter shapes to
    `BeliefFFN` so the same width solve applies."""
    fixed = backbone_fixed_param_count(spec.vocab_size, spec.d_model, spec.n_layers)
    coef = spec.n_layers * (2 * spec.d_model + 2)
    solution = solve_hidden_width_for_target(target, fixed, coef)
    model = TransformerLM(
        vocab_size=spec.vocab_size,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        context_length=spec.context_length,
        make_ffn=lambda _i: FixedConfidenceFFN(spec.d_model, solution.hidden),
        rope_base=spec.rope_base,
    )
    return BuiltModel(model=model, width_solution=solution, ffn_kind="fixed_confidence")


BUILDERS = {
    "swiglu": build_modern_transformer_1m,
    "cellv0.3": build_cellv03_transformer_1m,
    "fixed_confidence": build_fixed_confidence_transformer_1m,
}
