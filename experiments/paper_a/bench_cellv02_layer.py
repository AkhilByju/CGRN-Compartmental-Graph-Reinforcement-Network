#!/usr/bin/env python3
"""Isolated-layer forward / backward timing: CellV0.2's `PrecisionGainLayer`
vs CellV0.1's `BeliefLayer(scale_stable_precision)` vs a plain
`Linear -> Tanh`.

Purely a cost measurement -- no training, no accuracy. CellV0.1 is timed
exactly as it stands (`src/models/architecture_v0/integration.py`); nothing
in this script optimizes or modifies it.

Also prints the closed-form parameter-count formulas for the two cells.

Usage:
    python experiments/paper_a/bench_cellv02_layer.py
    python experiments/paper_a/bench_cellv02_layer.py --device cpu --iters 100
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
from src.models.architecture_v0.integration import BeliefLayer  # noqa: E402
from src.models.architecture_v0.precision_gain import (  # noqa: E402
    PrecisionGainLayer,
    initial_belief,
)

# (in_cells, out_cells, batch) -- spanning the Phase-1 regimes: small tabular,
# the Digits mid-size, and the MNIST-scale hidden widths of each cell.
CONFIGS: tuple[tuple[int, int, int], ...] = (
    (30, 83, 128),     # breast_cancer-ish, CellV0.2 hidden width
    (64, 123, 128),    # digits, CellV0.2 hidden width
    (784, 157, 128),   # mnist layer 1, CellV0.2 hidden width
    (157, 157, 128),   # mnist layer 2, CellV0.2 hidden width
    (256, 256, 256),   # square control
)


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
    pg = PrecisionGainLayer(in_c, out_c).to(device)
    bl = BeliefLayer(in_c, out_c, aggregation="scale_stable_precision").to(device)
    lin = nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()).to(device)

    x = torch.randn(batch, in_c, device=device)
    belief_v02 = initial_belief(x)  # e=1, u=0
    belief_v01 = BeliefCell(
        mu=x,
        evidence=torch.rand(batch, in_c, device=device) * 3 + 0.5,
        uncertainty=torch.rand(batch, in_c, device=device) + 0.1,
    )

    def fwd_pg():
        pg(belief_v02)

    def fwd_bl():
        bl(belief_v01)

    def fwd_lin():
        lin(x)

    def fb_pg():
        pg.zero_grad(set_to_none=True)
        pg(belief_v02).mu.pow(2).sum().backward()

    def fb_bl():
        bl.zero_grad(set_to_none=True)
        bl(belief_v01).mu.pow(2).sum().backward()

    def fb_lin():
        lin.zero_grad(set_to_none=True)
        lin(x).pow(2).sum().backward()

    n_params = {
        "cellv0.2": sum(p.numel() for p in pg.parameters()),
        "cellv0.1": sum(p.numel() for p in bl.parameters()),
        "linear_tanh": sum(p.numel() for p in lin.parameters()),
    }
    forward = {"cellv0.2": fwd_pg, "cellv0.1": fwd_bl, "linear_tanh": fwd_lin}
    fwd_bwd = {"cellv0.2": fb_pg, "cellv0.1": fb_bl, "linear_tanh": fb_lin}
    return n_params, forward, fwd_bwd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--out", default=str(_HERE / "results" / "processed" / "cellv02_layer_bench.json")
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(0)

    print(
        "Parameter-count formulas (per layer):\n"
        "  CellV0.2 PrecisionGainLayer : out*(in + 2)    (V + gain_raw + bias)\n"
        "  CellV0.1 BeliefLayer        : out*(2*in + 1)  "
        "(content_weight + relevance_logit + bias)\n"
    )
    print(f"device={device.type}  iters={args.iters}  warmup={args.warmup}\n")

    rows = []
    header = (
        f"{'in':>5s} {'out':>5s} {'batch':>6s} | "
        f"{'V0.2 fwd':>9s} {'V0.1 fwd':>9s} {'lin fwd':>9s} | "
        f"{'V0.2 f+b':>9s} {'V0.1 f+b':>9s} {'lin f+b':>9s} | "
        f"{'V0.2/V0.1':>9s} {'V0.2/lin':>9s}"
    )
    print(header)
    print("-" * len(header))
    for in_c, out_c, batch in CONFIGS:
        n_params, forward, fwd_bwd = _make_runners(in_c, out_c, batch, device)
        fwd = {k: _time(forward[k], device, args.iters, args.warmup) for k in forward}
        fb = {k: _time(fwd_bwd[k], device, args.iters, args.warmup) for k in fwd_bwd}
        rows.append(
            {
                "in_cells": in_c, "out_cells": out_c, "batch": batch,
                "params": n_params,
                "forward_ms": fwd, "forward_backward_ms": fb,
                "v02_over_v01_fwdbwd": fb["cellv0.2"] / fb["cellv0.1"],
                "v02_over_linear_fwdbwd": fb["cellv0.2"] / fb["linear_tanh"],
            }
        )
        print(
            f"{in_c:>5d} {out_c:>5d} {batch:>6d} | "
            f"{fwd['cellv0.2']:>9.3f} {fwd['cellv0.1']:>9.3f} {fwd['linear_tanh']:>9.3f} | "
            f"{fb['cellv0.2']:>9.3f} {fb['cellv0.1']:>9.3f} {fb['linear_tanh']:>9.3f} | "
            f"{fb['cellv0.2'] / fb['cellv0.1']:>8.2f}x {fb['cellv0.2'] / fb['linear_tanh']:>8.2f}x"
        )

    payload = {
        "device": device.type,
        "iters": args.iters,
        "warmup": args.warmup,
        "torch_threads": torch.get_num_threads(),
        "param_formulas": {
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
