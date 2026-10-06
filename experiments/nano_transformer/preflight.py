"""MPS speed preflight (task Sec 15): 50 warmup + 200 measured training
steps for each of the three models, reporting ms/step and tokens/sec. Also
benchmarks the `BeliefFFN` kernel alone (Sec 8: "Benchmark this kernel
independently on MPS"), mirroring the existing
`experiments/paper_a/bench_cellv03_layer.py` isolated-layer pattern.

If CellV0.3 is >3x slower than the SwiGLU model, Sec 15 permits
implementation-only optimizations (GEMM fusion, contiguous layouts, cached
`abs(V)`, fewer allocations, `torch.compile`) but forbids touching the
equations, approximating precision, dropping second moments, or changing
depth -- this module only measures; it does not itself apply any of those
optimizations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from experiments.nano_transformer.belief_ffn import BeliefFFN
from experiments.nano_transformer.models import (
    BUILDERS,
    NanoTransformerSpec,
    SwiGLUFFN,
    build_cellv03_transformer_1m,
)
from src.utilities.device import get_device

_DEFAULT_OUT = (
    Path(__file__).resolve().parent / "results" / "processed" / "mps_speed_preflight.json"
)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mps_memory_bytes(device: torch.device) -> int | None:
    if device.type != "mps" or not hasattr(torch.mps, "current_allocated_memory"):
        return None
    return torch.mps.current_allocated_memory()


def _time_steps(step_fn, device: torch.device, warmup: int, measured: int) -> float:
    for _ in range(warmup):
        step_fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(measured):
        step_fn()
    _sync(device)
    return time.perf_counter() - start


def preflight_one_model(
    model_kind: str,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 64,
    context_length: int = 256,
    warmup: int = 50,
    measured: int = 200,
    lr: float = 1e-3,
) -> dict[str, Any]:
    spec = NanoTransformerSpec(vocab_size=vocab_size, context_length=context_length)
    built = BUILDERS[model_kind](spec, target=1_000_000)
    model = built.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def step() -> None:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if device.type == "mps" and hasattr(torch.mps, "reset_peak_memory_stats"):
        torch.mps.reset_peak_memory_stats()

    elapsed = _time_steps(step, device, warmup, measured)
    ms_per_step = elapsed / measured * 1000
    tokens_per_step = batch_size * context_length
    tokens_per_sec = tokens_per_step * measured / elapsed

    return {
        "model_kind": model_kind,
        "params": built.param_report()["total"],
        "ffn_hidden_width": built.width_solution.hidden,
        "ms_per_step": ms_per_step,
        "tokens_per_sec": tokens_per_sec,
        "peak_mps_memory_bytes": _peak_mps_memory_bytes(device),
        "device": device.type,
    }


def benchmark_belief_ffn_kernel_alone(
    d_model: int,
    d_hidden: int,
    batch: int,
    context_length: int,
    device: torch.device,
    warmup: int = 20,
    measured: int = 100,
) -> dict[str, Any]:
    """Sec 8's "benchmark this kernel independently" -- BeliefFFN vs a
    SwiGLU FFN at the *same* hidden width (not parameter-matched here; this
    isolates the per-FFN-call cost, not a fair total-model comparison, which
    `run_preflight` already provides)."""
    belief = BeliefFFN(d_model, d_hidden).to(device)
    swiglu = SwiGLUFFN(d_model, d_hidden).to(device)
    x = torch.randn(batch, context_length, d_model, device=device)

    def _fwd_bwd(module: torch.nn.Module):
        def run() -> None:
            module.zero_grad(set_to_none=True)
            module(x).pow(2).sum().backward()

        return run

    belief_ms = _time_steps(_fwd_bwd(belief), device, warmup, measured) / measured * 1000
    swiglu_ms = _time_steps(_fwd_bwd(swiglu), device, warmup, measured) / measured * 1000

    return {
        "d_model": d_model,
        "d_hidden": d_hidden,
        "batch": batch,
        "context_length": context_length,
        "belief_ffn_ms_per_fwd_bwd": belief_ms,
        "swiglu_ffn_ms_per_fwd_bwd": swiglu_ms,
        "belief_over_swiglu_ratio": belief_ms / swiglu_ms,
    }


def run_preflight(
    vocab_size: int,
    device: torch.device | None = None,
    batch_size: int = 64,
    context_length: int = 256,
    warmup: int = 50,
    measured: int = 200,
) -> dict[str, Any]:
    device = device or get_device()
    per_model = {
        kind: preflight_one_model(
            kind, vocab_size, device, batch_size, context_length, warmup, measured
        )
        for kind in BUILDERS
    }
    swiglu_ms = per_model["swiglu"]["ms_per_step"]
    for r in per_model.values():
        r["slowdown_vs_swiglu"] = r["ms_per_step"] / swiglu_ms

    cellv03_hidden = build_cellv03_transformer_1m(
        NanoTransformerSpec(vocab_size=vocab_size, context_length=context_length)
    ).width_solution.hidden
    kernel_bench = benchmark_belief_ffn_kernel_alone(
        d_model=128, d_hidden=cellv03_hidden, batch=batch_size, context_length=context_length,
        device=device,
    )

    flagged = per_model["cellv0.3"]["slowdown_vs_swiglu"] > 3.0
    return {
        "device": device.type,
        "batch_size": batch_size,
        "context_length": context_length,
        "warmup": warmup,
        "measured": measured,
        "per_model": per_model,
        "belief_ffn_kernel_alone": kernel_bench,
        "cellv03_over_3x_slower_than_swiglu": flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    device = get_device(args.device)
    results = run_preflight(
        args.vocab_size, device, args.batch_size, args.context_length, args.warmup, args.measured
    )

    print(f"device={device.type}  batch={args.batch_size}  context={args.context_length}\n")
    header = (
        f"{'model':>18s} {'params':>9s} {'d_hidden':>9s} {'ms/step':>9s} "
        f"{'tok/s':>10s} {'vs swiglu':>10s}"
    )
    print(header)
    print("-" * len(header))
    for kind, r in results["per_model"].items():
        print(
            f"{kind:>18s} {r['params']:>9d} {r['ffn_hidden_width']:>9d} "
            f"{r['ms_per_step']:>9.2f} {r['tokens_per_sec']:>10.0f} "
            f"{r['slowdown_vs_swiglu']:>9.2f}x"
        )
    kernel_json = json.dumps(results["belief_ffn_kernel_alone"], indent=2)
    print(f"\nBeliefFFN-alone kernel: {kernel_json}")
    if results["cellv03_over_3x_slower_than_swiglu"]:
        print("\nWARNING: CellV0.3 is >3x slower than ModernTransformer (Sec 15).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
