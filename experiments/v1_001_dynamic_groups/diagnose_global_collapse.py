#!/usr/bin/env python3
"""Targeted diagnostic: reproduce the `very_hard` (`n_objects=192`),
`seed=0` `field_local_global_t2` collapse found by
`run_complexity_scaling.py` (test R^2=0.224 there, vs local-only 0.781
and CellV0.1 0.816 -- the "enormous failure... clearly amplified by the
global pathway," not present in the local field), and test which of three
explanations holds, per the user's exact protocol. Exactly three
variants, `n_objects=192`/`k_min=12`/`k_max=20`/`seed=0` fixed throughout:

    A) current            -- 1500 steps, R=256 (reproduces the collapse)
    B) longer training     -- 4500 steps, R=256 (tests: undertraining?)
    C) more accurate field -- 1500 steps, R=512 (tests: RFF resolution?)

**No architecture-math changes.** Instrumentation only: forward hooks on
the existing (frozen) `GlobalSendFunction`/`GlobalNeedFunction`
instances capture their per-step outputs as they're already computed;
`source_gate_logit` is read directly (an existing parameter, not a new
one). Nothing about the model's forward pass changes.

Logs, every `--log-every` steps: train R^2 (evaluated on a *fixed* 500-
example subset of the training set, so the trajectory isn't confounded
by which minibatch happened to be sampled), mean send, mean need, the
global-source fuse weight (`sigmoid(source_gate_logit[2])`), and the
fraction of send/need values saturated near 0 or 1 (`<0.05` or `>0.95`).

Interpretation (the user's own framing):
  - B fixes it, C doesn't  -> undertraining; the architecture needs more
    optimization as the population grows, not a redesign.
  - C fixes it, B doesn't  -> RFF resolution (R relative to n_cells)
    becomes relevant at scale.
  - neither fixes it       -> the global learning dynamics themselves are
    not scale-stable; needs a redesign, not tuning.

Usage:
    python experiments/v1_001_dynamic_groups/diagnose_global_collapse.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from harness import _build_global_comparison_models
from src.data.synthetic.dynamic_groups import make_dynamic_groups_global_splits
from src.evaluation.regression import r_squared
from src.utilities.device import get_device
from src.utilities.seeding import set_seed

N_OBJECTS = 192
K_MIN, K_MAX = 12, 20
SEED = 0

VARIANTS: list[dict] = [
    {"name": "A_current_1500steps_R256", "steps": 1500, "num_features": 256},
    {"name": "B_longer_4500steps_R256", "steps": 4500, "num_features": 256},
    {"name": "C_moreaccurate_1500steps_R512", "steps": 1500, "num_features": 512},
]


def _install_send_need_hooks(step: torch.nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pure instrumentation -- captures GlobalSendFunction/
    GlobalNeedFunction's outputs as the existing forward pass already
    computes them, no change to what's computed."""
    send_outputs: list[torch.Tensor] = []
    need_outputs: list[torch.Tensor] = []
    step.send_fn.register_forward_hook(lambda module, inputs, output: send_outputs.append(output.detach()))
    step.need_fn.register_forward_hook(lambda module, inputs, output: need_outputs.append(output.detach()))
    return send_outputs, need_outputs


def _saturated_frac(x: torch.Tensor) -> float:
    return ((x < 0.05) | (x > 0.95)).float().mean().item()


def run_variant(
    name: str,
    steps: int,
    num_features: int,
    log_every: int,
    lr: float = 1e-2,
    batch_size: int = 64,
    n_train: int = 3_000,
    n_val: int = 500,
    n_test: int = 500,
) -> dict:
    set_seed(SEED)
    device = get_device()

    splits = make_dynamic_groups_global_splits(
        n_train, n_val, n_test, seed=SEED, n_objects=N_OBJECTS, k_min=K_MIN, k_max=K_MAX
    )
    x_train, y_train, _ = splits["train"]
    x_test, y_test, _ = splits["test"]

    in_features = x_train.shape[1]
    models, _ = _build_global_comparison_models(
        "dynamic_groups_global",
        in_features,
        1,
        SEED,
        n_cells=N_OBJECTS,
        n_objects=N_OBJECTS,
        num_features=num_features,
        include_field_t1=False,
    )
    model = models["field_local_global_t2"].to(device)
    send_outputs, need_outputs = _install_send_need_hooks(model.core.step)

    # Fixed subset so the train-R^2 trajectory reflects training progress,
    # not which minibatch happened to land on a given logging step.
    eval_x = x_train[:500].to(device)
    eval_y = y_train[:500].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loss_fn = torch.nn.MSELoss()

    log: list[dict] = []
    step_count = 0
    model.train()
    while step_count < steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            send_outputs.clear()
            need_outputs.clear()
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            step_count += 1

            if step_count == 1 or step_count % log_every == 0:
                all_send = torch.cat([s.flatten() for s in send_outputs])
                all_need = torch.cat([n.flatten() for n in need_outputs])
                model.eval()
                with torch.no_grad():
                    train_r2 = r_squared(model(eval_x), eval_y)
                model.train()
                global_gate = torch.sigmoid(model.core.step.source_gate_logit)[2].item()
                entry = {
                    "step": step_count,
                    "train_r2": train_r2,
                    "mean_send": all_send.mean().item(),
                    "mean_need": all_need.mean().item(),
                    "global_gate_weight": global_gate,
                    "send_saturated_frac": _saturated_frac(all_send),
                    "need_saturated_frac": _saturated_frac(all_need),
                }
                log.append(entry)
                print(
                    f"[{name}] step={step_count:5d} train_r2={entry['train_r2']:+.4f} "
                    f"mean_send={entry['mean_send']:.4f} mean_need={entry['mean_need']:.4f} "
                    f"global_gate={global_gate:.4f} "
                    f"send_sat={entry['send_saturated_frac']:.2f} need_sat={entry['need_saturated_frac']:.2f}"
                )

            if step_count >= steps:
                break

    model.eval()
    with torch.no_grad():
        final_test_r2 = r_squared(model(x_test.to(device)), y_test.to(device))
    print(f"[{name}] FINAL test_r2={final_test_r2:.4f}\n")

    return {
        "name": name,
        "steps": steps,
        "num_features": num_features,
        "final_test_r2": final_test_r2,
        "log": log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--summary-out", default="results/processed/v1_001_global_collapse_diagnostic.json")
    args = parser.parse_args()

    results = []
    for v in VARIANTS:
        print(f"=== {v['name']} ===")
        results.append(run_variant(v["name"], v["steps"], v["num_features"], log_every=args.log_every))

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote diagnostic log to {summary_path}")

    print("\n=== Final test R^2, all variants ===")
    for r in results:
        print(f"    {r['name']:32s} steps={r['steps']:5d} R={r['num_features']:4d}  test_r2={r['final_test_r2']:.4f}")


if __name__ == "__main__":
    main()
