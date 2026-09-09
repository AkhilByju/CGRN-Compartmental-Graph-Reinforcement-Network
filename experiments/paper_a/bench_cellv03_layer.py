#!/usr/bin/env python3
"""Isolated-layer forward / backward timing: CellV0.3's
`ConflictNormalizedLayer` vs CellV0.2's `PrecisionGainLayer` vs CellV0.1's
`BeliefLayer(scale_stable_precision)` vs a plain `Linear -> Tanh`.

Purely a cost measurement -- no training, no accuracy. CellV0.1 and CellV0.2
are timed exactly as they stand; nothing in this script optimizes or
modifies them (Paper-A task Sec 15: "Do not optimize V0.1 as part of this
task").

Also prints the closed-form parameter-count formulas.

Usage:
    python experiments/paper_a/bench_cellv03_layer.py
    python experiments/paper_a/bench_cellv03_layer.py --device cpu --iters 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.architecture_v0.conflict_normalized import (  # noqa: E402
    ConflictNormalizedLayer,
)
from src.models.architecture_v0.integration import BeliefLayer  # noqa: E402
from src.models.architecture_v0.precision_gain import (  # noqa: E402
    PrecisionGainLayer,
    initial_belief,
)

# (in_cells, out_cells, batch) -- spanning the Phase-1 regimes at the CellV0.3
# (== CellV0.2) hidden widths.
CONFIGS: tuple[tuple[int, int, int], ...] = (
    (30, 83, 128),     # breast_cancer-ish
    (64, 123, 128),    # digits
    (784, 157, 128),   # mnist layer 1
    (157, 157, 128),   # mnist layer 2
    (256, 256, 256),   # square control
)

_KINDS = ("cellv0.3", "cellv0.2", "cellv0.1", "linear_tanh")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time(fn, device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e3  # ms/iter


def _make_runners(in_c: int, out_c: int, batch: int, device: torch.device):
    cn = ConflictNormalizedLayer(in_c, out_c).to(device)
    pg = PrecisionGainLayer(in_c, out_c).to(device)
    bl = BeliefLayer(in_c, out_c, aggregation="scale_stable_precision").to(device)
    lin = nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()).to(device)

    x = torch.randn(batch, in_c, device=device)
    belief_neutral = initial_belief(x)  # e=1, u=0 -- CellV0.2 / CellV0.3 input
    belief_v01 = BeliefCell(
        mu=x,
        evidence=torch.rand(batch, in_c, device=device) * 3 + 0.5,
        uncertainty=torch.rand(batch, in_c, device=device) + 0.1,
    )

    fwd = {
        "cellv0.3": lambda: cn(belief_neutral),
        "cellv0.2": lambda: pg(belief_neutral),
        "cellv0.1": lambda: bl(belief_v01),
        "linear_tanh": lambda: lin(x),
    }

    def _fb(layer, belief):
        def run():
            layer.zero_grad(set_to_none=True)
            layer(belief).mu.pow(2).sum().backward()
        return run

    def _fb_lin():
        lin.zero_grad(set_to_none=True)
        lin(x).pow(2).sum().backward()

    fwd_bwd = {
        "cellv0.3": _fb(cn, belief_neutral),
        "cellv0.2": _fb(pg, belief_neutral),
        "cellv0.1": _fb(bl, belief_v01),
        "linear_tanh": _fb_lin,
    }
    n_params = {
        "cellv0.3": sum(p.numel() for p in cn.parameters()),
        "cellv0.2": sum(p.numel() for p in pg.parameters()),
        "cellv0.1": sum(p.numel() for p in bl.parameters()),
        "linear_tanh": sum(p.numel() for p in lin.parameters()),
    }
    return n_params, fwd, fwd_bwd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--out", default=str(_HERE / "results" / "processed" / "cellv03_layer_bench.json")
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(0)

    print(
        "Parameter-count formulas (per layer):\n"
        "  CellV0.3 ConflictNormalizedLayer : out*(in + 2)    (V + gain_raw + bias)\n"
        "  CellV0.2 PrecisionGainLayer      : out*(in + 2)    (V + gain_raw + bias)\n"
        "  CellV0.1 BeliefLayer             : out*(2*in + 1)  "
        "(content_weight + relevance_logit + bias)\n"
    )
    print(f"device={device.type}  iters={args.iters}  warmup={args.warmup}\n")

    rows = []
    header = (
        f"{'in':>5s} {'out':>5s} {'batch':>6s} | "
        f"{'V0.3 f+b':>9s} {'V0.2 f+b':>9s} {'V0.1 f+b':>9s} {'lin f+b':>9s} | "
        f"{'V0.3/V0.2':>9s} {'V0.3/V0.1':>9s} {'V0.3/lin':>9s}"
    )
    print(header)
    print("-" * len(header))
    for in_c, out_c, batch in CONFIGS:
        n_params, fwd, fwd_bwd = _make_runners(in_c, out_c, batch, device)
        fwd_ms = {k: _time(fwd[k], device, args.iters, args.warmup) for k in _KINDS}
        fb_ms = {k: _time(fwd_bwd[k], device, args.iters, args.warmup) for k in _KINDS}
        rows.append(
            {
                "in_cells": in_c, "out_cells": out_c, "batch": batch,
                "params": n_params,
                "forward_ms": fwd_ms, "forward_backward_ms": fb_ms,
                "v03_over_v02_fwdbwd": fb_ms["cellv0.3"] / fb_ms["cellv0.2"],
                "v03_over_v01_fwdbwd": fb_ms["cellv0.3"] / fb_ms["cellv0.1"],
                "v03_over_linear_fwdbwd": fb_ms["cellv0.3"] / fb_ms["linear_tanh"],
            }
        )
        print(
            f"{in_c:>5d} {out_c:>5d} {batch:>6d} | "
            f"{fb_ms['cellv0.3']:>9.3f} {fb_ms['cellv0.2']:>9.3f} "
            f"{fb_ms['cellv0.1']:>9.3f} {fb_ms['linear_tanh']:>9.3f} | "
            f"{fb_ms['cellv0.3'] / fb_ms['cellv0.2']:>8.2f}x "
            f"{fb_ms['cellv0.3'] / fb_ms['cellv0.1']:>8.2f}x "
            f"{fb_ms['cellv0.3'] / fb_ms['linear_tanh']:>8.2f}x"
        )

    payload = {
        "device": device.type,
        "iters": args.iters,
        "warmup": args.warmup,
        "torch_threads": torch.get_num_threads(),
        "param_formulas": {
            "cellv0.3_per_layer": "out*(in + 2)",
            "cellv0.2_per_layer": "out*(in + 2)",
            "cellv0.1_per_layer": "out*(2*in + 1)",
        },
        "configs": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
