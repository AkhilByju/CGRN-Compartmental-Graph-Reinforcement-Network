#!/usr/bin/env python3
"""One-off diagnostic: why is CellV1(.1) flatlining near R^2=0 on
dynamic_groups at V1-S0 while mlp/cellv0.1 learn fine (2026-09-02 user
report: mlp ~0.61-0.68, cellv0.1 ~0.74-0.77, cellv1_local/full ~0.00-0.03,
two seeds)? Isolates *where* the failure is, per the user's explicit
fork:

  - dense and full-pool-sparse learn, normal-sparse doesn't -> LSH
    candidate routing is destroying the learning signal.
  - all three stay near 0 -> the architecture/training setup itself is
    the issue (init, recurrent update, readout, z dynamics).
  - train R^2 high, test R^2 ~0 -> generalization/overfitting.
  - train R^2 also ~0 -> optimization/signal-propagation issue.

Three CellV1 variants, same n_cells/association_dim/num_steps/hidden_dim,
same data/seed/steps/optimizer, differing ONLY in routing:

  1. dense       -- model.py::DynamicBeliefGraph (CellV1 Dense Reference)
  2. full_pool   -- sparse_model.py::SparseDynamicBeliefGraph with the
                    candidate pool covering every cell (1 hash round,
                    chunk_size == n_cells) -- same config
                    tests/test_sparse_dense_consistency.py uses to prove
                    sparse reduces to dense exactly.
  3. normal      -- SparseDynamicBeliefGraph at the harness's normal LSH
                    settings (2 hashes, chunk_size 10/4).

mlp/cellv0.1 included only as a floor/ceiling reference, matched to
`dense`'s parameter count. Not wired into the RunRecord/results-dir
machinery -- this is a quick, standalone diagnostic, not a tracked
experiment.

Usage:
    python experiments/v1_001_dynamic_groups/diagnose_v1_learning.py
    python experiments/v1_001_dynamic_groups/diagnose_v1_learning.py --steps 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from harness import (  # noqa: E402
    ASSOCIATION_DIM,
    CELLV1_HIDDEN_DIM,
    N_CELLS,
    NUM_STEPS,
    _build_splits,
    _compute_metrics,
    _match_belief_hidden_cells,
    _train,
)
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v1.model import DynamicBeliefGraph  # noqa: E402
from src.models.architecture_v1.object_encoder import ObjectSeededEncoder  # noqa: E402
from src.models.architecture_v1.sparse_model import SparseDynamicBeliefGraph  # noqa: E402
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

N_OBJECTS = 24


@torch.no_grad()
def _batched_predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    """Unbatched eval on thousands of examples can blow up memory for the
    sparse modules: `gather_rows`'s `(batch, n_cells, pool, pool)`
    intermediate is `O(batch * n_cells * pool^2)`, and with
    `full_pool_sparse` (`pool == n_cells`) that's effectively
    `O(batch * n_cells^3)` -- fine at training's `batch_size=64`, not at
    3000 examples in one shot (hit a 46.88 GiB allocation doing exactly
    this). Chunk instead."""
    outputs = []
    for start in range(0, x.shape[0], batch_size):
        outputs.append(model(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=["mlp", "cellv0.1", "dense", "full_pool_sparse", "normal_sparse"],
        help="Run only these models (e.g. to resume after a partial run without redoing finished ones).",
    )
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    train_split, _val_split, test_split = _build_splits(
        "dynamic_groups", args.seed, args.n_train, args.n_val, args.n_test
    )
    x_train, y_train, _ = train_split
    x_test, y_test, _ = test_split
    in_features = x_train.shape[1]
    loss_fn = torch.nn.MSELoss()

    def make_encoder() -> ObjectSeededEncoder:
        return ObjectSeededEncoder(n_objects=N_OBJECTS, n_cells=N_CELLS, association_dim=ASSOCIATION_DIM)

    set_seed(args.seed)
    dense = DynamicBeliefGraph(
        in_features=in_features, out_features=1, n_cells=N_CELLS, association_dim=ASSOCIATION_DIM,
        hidden_dim=CELLV1_HIDDEN_DIM, num_steps=NUM_STEPS, use_global_routing=True, encoder=make_encoder(),
    )

    set_seed(args.seed)
    full_pool = SparseDynamicBeliefGraph(
        in_features=in_features, out_features=1, n_cells=N_CELLS, association_dim=ASSOCIATION_DIM,
        hidden_dim=CELLV1_HIDDEN_DIM, num_steps=NUM_STEPS, use_global_routing=True, encoder=make_encoder(),
        num_hashes_local=1, bits_local=6, chunk_size_local=N_CELLS, window_local=0,
        num_hashes_global=1, bits_global=6, chunk_size_global=N_CELLS, window_global=0,
    )

    set_seed(args.seed)
    normal_sparse = SparseDynamicBeliefGraph(
        in_features=in_features, out_features=1, n_cells=N_CELLS, association_dim=ASSOCIATION_DIM,
        hidden_dim=CELLV1_HIDDEN_DIM, num_steps=NUM_STEPS, use_global_routing=True, encoder=make_encoder(),
    )

    target_params = count_parameters(dense)
    hidden_cells = _match_belief_hidden_cells(target_params, in_features, 1)
    set_seed(args.seed)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, 1, aggregation="scale_stable_precision")
    mlp_hidden = match_hidden_dim(target_params, in_features, 1, num_hidden_layers=2)
    set_seed(args.seed)
    mlp = MLPBaseline(in_features, mlp_hidden, 1, num_hidden_layers=2)

    models = {
        "mlp": mlp,
        "cellv0.1": cellv0_1,
        "dense": dense,
        "full_pool_sparse": full_pool,
        "normal_sparse": normal_sparse,
    }
    if args.only:
        models = {k: v for k, v in models.items() if k in args.only}

    print(f"seed={args.seed} steps={args.steps} n_train={args.n_train} n_cells={N_CELLS} "
          f"(dense_params={target_params}, cellv0.1_hidden_cells={hidden_cells}, mlp_hidden_dim={mlp_hidden})\n")
    print(f"{'model':18s} {'params':>8s} {'train_r2':>10s} {'test_r2':>10s} {'wall_s':>8s}")
    for name, model in models.items():
        model.to(device)
        t0 = time.perf_counter()
        _train(model, loss_fn, x_train, y_train, device, args.seed, args.steps, args.batch_size, args.lr)
        wall = time.perf_counter() - t0

        model.eval()
        train_pred = _batched_predict(model, x_train, device)
        test_pred = _batched_predict(model, x_test, device)
        train_r2 = _compute_metrics(True, train_pred, y_train)["r2"]
        test_r2 = _compute_metrics(True, test_pred, y_test)["r2"]

        print(f"{name:18s} {count_parameters(model):8d} {train_r2:10.4f} {test_r2:10.4f} {wall:8.1f}")


if __name__ == "__main__":
    main()
