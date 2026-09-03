"""Shared logic for the first real CellV1 experiment: "Does self-organizing
computation actually help?" (user's Experiment 005, filed here as
`v1_001_dynamic_groups` -- see the folder README for why the number
differs from the user's "005"). One experiment, four models, four tasks --
not five separate diagnostic projects.

**Wired to CellV1.1 (sparse), not dense.** Dense `DynamicBeliefGraph`
(`model.py`) doesn't scale (`O(n_cells^2)`) and is kept only as
`docs/architecture_v1.md` §11's correctness reference
(`tests/test_sparse_dense_consistency.py`); this harness always builds
`SparseDynamicBeliefGraph` (`sparse_model.py`).

**Models** (`_build_models`): `mlp` (conventional baseline), `cellv0.1`
(`BeliefNetwork` with `scale_stable_precision` aggregation -- fixed 2-layer
stack), `cellv1_local` (`SparseDynamicBeliefGraph`,
`use_global_routing=False`), `cellv1_full` (`use_global_routing=True`).
`cellv1_full`'s actual parameter count -- at whichever `n_cells` is passed
in (128/256/512, the `V1_SCALES` the user asked to sweep) -- is the
parameter-matching target for `mlp` and `cellv0.1`, recomputed at every
scale; `cellv1_local`'s count is reported, not matched to (a real
structural ablation, fewer modules entirely, naturally smaller).

**Tasks** (`_build_splits`): `r2_interaction`/`c2_interaction`/
`u2_heteroscedastic_interaction` (existing CellV0 datasets, reused as-is)
and `dynamic_groups` (`src.data.synthetic.dynamic_groups`, designed
specifically around the self-organization hypothesis -- the task this
experiment is actually about; run this one first). `dynamic_groups`'
`n_objects`/`k_min`/`k_max` are threaded all the way through here (not
hardcoded) so a harder-complexity run (more objects, more groups) can
follow later without touching this file again.

**Input adapter:** every task except `dynamic_groups` uses CellV1's
default dense `PopulationEncoder`. `dynamic_groups` uses
`object_encoder.ObjectSeededEncoder` instead -- each cell sees one object,
not the whole input.

**Graph analysis** (`_graph_metrics`, `_save_graph_evolution`): CellV1.1's
routing returns gathered `(weights, candidate_idx)` pairs, not dense
`(n_cells, n_cells)` matrices -- a receiver's `p`-th pool slot can refer to
a *different* cell at every step (the LSH hash depends on `z`, which
changes every step), so "graph change over time" and "does this edge
agree with the true group" are computed via candidate-identity lookups
(`lsh.lookup_value`/`gather_scalar`), not positional indexing into a fixed
matrix the way the dense version could. Metrics are otherwise exactly what
the user asked for and nothing more: local sparsity, local-graph change
over `T`, AUROC of local-association strength vs. true group membership,
fraction of global edges crossing true groups.

Not a `src/` module -- experiment-specific composition of already-existing
generic pieces, following `experiments/004_cellv0_scaling/scaling_harness.py`'s
pattern.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.synthetic.classification import make_classification_splits  # noqa: E402
from src.data.synthetic.dynamic_groups import N_OBJECTS as DEFAULT_N_OBJECTS  # noqa: E402
from src.data.synthetic.dynamic_groups import make_dynamic_groups_global_splits  # noqa: E402
from src.data.synthetic.dynamic_groups import make_dynamic_groups_splits  # noqa: E402
from src.data.synthetic.regression import make_regression_splits  # noqa: E402
from src.data.synthetic.uncertainty import make_uncertainty_splits  # noqa: E402
from src.data.synthetic.utils import standardize  # noqa: E402
from src.evaluation.classification import accuracy as accuracy_metric  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.regression import mae as mae_metric  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.evaluation.regression import rmse as rmse_metric  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v1.association_model import LearnedAssociationField  # noqa: E402
from src.models.architecture_v1.field_model import SelfOrganizingRefinementField  # noqa: E402
from src.models.architecture_v1.lsh import gather_scalar, lookup_value  # noqa: E402
from src.models.architecture_v1.object_encoder import ObjectSeededEncoder  # noqa: E402
from src.models.architecture_v1.sparse_model import SparseDynamicBeliefGraph  # noqa: E402
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

TASKS: tuple[str, ...] = (
    "r2_interaction",
    "c2_interaction",
    "u2_heteroscedastic_interaction",
    "dynamic_groups",
    "dynamic_groups_global",
)

# Tasks using the per-object `ObjectSeededEncoder` (object_encoder.py) and
# returning `(x, y, group_id)` splits, not `PopulationEncoder`'s dense
# `(x, y)`. `dynamic_groups_global` (src.data.synthetic.dynamic_groups)
# shares `dynamic_groups`' exact object layout/flatten order -- only the
# label's cross-group pairing criterion differs.
OBJECT_SEEDED_TASKS: tuple[str, ...] = ("dynamic_groups", "dynamic_groups_global")
ARCHITECTURES: tuple[str, ...] = ("mlp", "cellv0.1", "cellv1_local", "cellv1_full")

# CellV1.3 (Self-Organizing Refinement Field) comparison -- a separate,
# additive path (`_build_field_models`/`run_field_task` below), not a
# replacement for the cellv1_local/cellv1_full comparison above. Kept
# apart because the field model has no discrete candidate-graph output to
# feed the cellv1_local/cellv1_full graph-analysis metrics
# (`_graph_metrics`) -- comparing field_t1/field_t2 is purely on
# prediction performance, params, and wall-clock, per the user's spec.
FIELD_ARCHITECTURES: tuple[str, ...] = ("mlp", "cellv0.1", "field_t1", "field_t2")
FIELD_NUM_FEATURES = 256  # R, the user's stated experimental default

# CellV1.3's global communication field (global_field.py), added on top of
# the local field -- a separate, additive comparison
# (_build_global_comparison_models/run_global_field_comparison below),
# not a replacement for FIELD_ARCHITECTURES/run_field_task above (which
# stays untouched -- its 3-seed T1-vs-T2 result is already recorded).
# "Does learned non-local communication improve beyond the already-
# successful local self-organizing refinement?" -- no mlp in this
# comparison, matching the user's exact point-18 listing.
GLOBAL_FIELD_ARCHITECTURES: tuple[str, ...] = ("cellv0.1", "field_t1", "field_t2", "field_local_global_t2")
GLOBAL_DIM = 16  # d_g, the user's suggested "8 or 16"

# Learned Association Field (association_model.py) -- the user's narrower
# redesign replacing field_dynamics.py's ORFF-approximated Gaussian local
# field with an exact, learned low-rank association kernel
# (learned_association.py). Now "the leading architecture" per the
# user's own framing; field_t2/field_local_global_t2 above are retired
# from active development, kept only as the research baseline showing
# why this redesign was necessary -- not deleted, not modified.
ASSOCIATION_ARCHITECTURES: tuple[str, ...] = ("cellv0.1", "association_local_global_t2")
ASSOC_DIM = 32  # D, the user's suggested "16-64," matching learned_association.py's own default

# The three scales the user asked to sweep -- T fixed at 6 for all of them.
V1_SCALES: dict[str, int] = {"V1-S0": 128, "V1-S1": 256, "V1-S2": 512}

# CellV1 is otherwise fixed across every arm/task/scale, per the user's
# spec -- no hyperparameter hunting yet. N_CELLS is the single-scale
# default (overridden per V1_SCALES entry when sweeping).
N_CELLS = 128
ASSOCIATION_DIM = 8
NUM_STEPS = 6
CELLV1_HIDDEN_DIM = 32  # width of CellV1's small shared MLPs (F_msg/F_need/F_offer/F_z)

# LSH routing hyperparameters (docs/architecture_v1.md §11) -- fixed pool
# sizes regardless of n_cells, which is the entire point of the sparse
# design. Not tuned; reasonable defaults matching the user's suggested
# orders of magnitude.
NUM_HASHES_LOCAL, BITS_LOCAL, CHUNK_SIZE_LOCAL, WINDOW_LOCAL = 2, 6, 10, 0
NUM_HASHES_GLOBAL, BITS_GLOBAL, CHUNK_SIZE_GLOBAL, WINDOW_GLOBAL = 2, 6, 4, 0


def _is_regression(task: str) -> bool:
    return task != "c2_interaction"


def _belief_total_params(hidden_cells: int, in_features: int, out_features: int) -> int:
    """Closed-form 2-`BeliefLayer` `BeliefNetwork` parameter count (same
    formula as `experiments/004_cellv0_scaling/scaling_harness.py`'s
    private helper of the same name -- duplicated rather than imported
    across experiment folders, this repo's existing convention for
    experiment-local composition)."""
    layer1 = hidden_cells * (2 * in_features + 1)
    layer2 = hidden_cells * (2 * hidden_cells + 1)
    readout = hidden_cells * out_features + out_features
    return layer1 + layer2 + readout


def _match_belief_hidden_cells(
    target_params: int, in_features: int, out_features: int, search_range: range = range(1, 2000)
) -> int:
    best_hc, best_diff = search_range[0], None
    for hc in search_range:
        diff = abs(_belief_total_params(hc, in_features, out_features) - target_params)
        if best_diff is None or diff < best_diff:
            best_diff, best_hc = diff, hc
    return best_hc


def _build_splits(
    task: str,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
):
    """Returns `(train, val, test)`. Each split is `(x, y)` except
    `dynamic_groups`, which is `(x, y, group_id)` -- `group_id` is never
    fed to a model, only used by `_graph_metrics`/`_save_graph_evolution`.
    `n_objects`/`k_min`/`k_max` only affect `dynamic_groups`."""
    if task in OBJECT_SEEDED_TASKS:
        splits_fn = make_dynamic_groups_splits if task == "dynamic_groups" else make_dynamic_groups_global_splits
        splits = splits_fn(n_train, n_val, n_test, seed=seed, n_objects=n_objects, k_min=k_min, k_max=k_max)
        return splits["train"], splits["val"], splits["test"]
    if task == "u2_heteroscedastic_interaction":
        splits = make_uncertainty_splits(task, n_train, n_val, n_test, seed=seed)
        return (
            splits["train"][:2],
            splits["val"][:2],
            splits["test"][:2],
        )  # drop true_std -- not used by this experiment
    if _is_regression(task):
        splits = make_regression_splits(task, n_train, n_val, n_test, seed=seed)
    else:
        splits = make_classification_splits(task, n_train, n_val, n_test, seed=seed)
    return splits["train"], splits["val"], splits["test"]


def _compute_metrics(regression: bool, pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    if regression:
        return {"mae": mae_metric(pred, y), "rmse": rmse_metric(pred, y), "r2": r_squared(pred, y)}
    return {"accuracy": accuracy_metric(pred, y)}


def _build_models(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    association_dim: int = ASSOCIATION_DIM,
    num_steps: int = NUM_STEPS,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    num_hashes_local: int = NUM_HASHES_LOCAL,
    bits_local: int = BITS_LOCAL,
    chunk_size_local: int = CHUNK_SIZE_LOCAL,
    window_local: int = WINDOW_LOCAL,
    num_hashes_global: int = NUM_HASHES_GLOBAL,
    bits_global: int = BITS_GLOBAL,
    chunk_size_global: int = CHUNK_SIZE_GLOBAL,
    window_global: int = WINDOW_GLOBAL,
) -> tuple[dict[str, torch.nn.Module], dict[str, int | str]]:
    """Builds all four architectures, `cellv1_full`-parameter-matched
    `mlp`/`cellv0.1` (recomputed at whatever `n_cells` is passed in);
    `cellv1_local` is a genuine ablation (fewer modules), left unmatched.
    Returns `(models, sizing_info)`.

    `n_cells` is the main lever for both the real experiment (`V1_SCALES`:
    128/256/512) and a fast correctness check (e.g. 30) -- CellV1.1's own
    cost, not `mlp`/`cellv0.1` (always cheap regardless of scale), is what
    makes a run slow."""
    set_seed(seed)

    encoder_kind = task in OBJECT_SEEDED_TASKS

    def make_encoder() -> torch.nn.Module | None:
        if not encoder_kind:
            return None
        return ObjectSeededEncoder(n_objects=n_objects, n_cells=n_cells, association_dim=association_dim)

    sparse_kwargs = dict(
        num_hashes_local=num_hashes_local,
        bits_local=bits_local,
        chunk_size_local=chunk_size_local,
        window_local=window_local,
        num_hashes_global=num_hashes_global,
        bits_global=bits_global,
        chunk_size_global=chunk_size_global,
        window_global=window_global,
    )

    cellv1_full = SparseDynamicBeliefGraph(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        use_global_routing=True,
        encoder=make_encoder(),
        **sparse_kwargs,
    )
    target_params = count_parameters(cellv1_full)

    cellv1_local = SparseDynamicBeliefGraph(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        use_global_routing=False,
        encoder=make_encoder(),
        **sparse_kwargs,
    )

    hidden_cells = _match_belief_hidden_cells(target_params, in_features, out_features)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, out_features, aggregation="scale_stable_precision")

    mlp_hidden_dim = match_hidden_dim(target_params, in_features, out_features, num_hidden_layers=2)
    mlp = MLPBaseline(in_features, mlp_hidden_dim, out_features, num_hidden_layers=2)

    models = {"mlp": mlp, "cellv0.1": cellv0_1, "cellv1_local": cellv1_local, "cellv1_full": cellv1_full}
    sizing = {
        "cellv1_full__params": target_params,
        "cellv1_local__params": count_parameters(cellv1_local),
        "cellv0.1__params": count_parameters(cellv0_1),
        "cellv0.1__hidden_cells": hidden_cells,
        "mlp__params": count_parameters(mlp),
        "mlp__hidden_dim": mlp_hidden_dim,
    }
    return models, sizing


def _train(
    model: torch.nn.Module,
    loss_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
) -> float:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    start = time.perf_counter()
    step = 0
    while step < steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break
    return time.perf_counter() - start


def _train_until_convergence(
    model: torch.nn.Module,
    loss_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
    regression: bool,
    batch_size: int,
    lr: float,
    val_every: int = 100,
    patience_steps: int = 500,
    max_steps: int = 5_000,
) -> dict[str, float | int]:
    """Early-stopping training loop, replacing `_train`'s fixed step
    count for the complexity-scaling comparison
    (`run_complexity_scaling_convergence.py`) -- a 24-object and a
    192-object problem don't converge on the same optimization timescale
    (`docs/research_log.md`'s global-collapse diagnostic: the same
    architecture that looked "collapsed" at a fixed 1500 steps reached
    0.80-0.82 R^2 given enough steps or a better-resolved field), so
    comparing architectures at one arbitrary step count confounds
    capability with convergence speed. Evaluates validation performance
    every `val_every` steps; stops once `patience_steps` worth of checks
    pass with no improvement, or `max_steps` is reached. Restores the
    model to its best-validation state before returning (so the caller's
    subsequent test-set evaluation reflects the best checkpoint, not
    wherever training happened to stop) and reports both *when* the best
    checkpoint was found (steps/wall-clock "to convergence") and the
    total steps/wall-clock actually spent (best-checkpoint time plus the
    patience tail) -- the efficiency question the user's protocol is
    specifically after, not just whether the architecture eventually
    gets there."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_dev, y_val_dev = x_val.to(device), y_val.to(device)

    best_val = -float("inf")
    best_step = 0
    best_wall_clock = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    steps_since_improvement = 0

    start = time.perf_counter()
    step = 0
    data_iter = iter(loader)
    while step < max_steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        step += 1

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_dev)
                val_metric = r_squared(val_pred, y_val_dev) if regression else accuracy_metric(val_pred, y_val_dev)
            model.train()

            if val_metric > best_val:
                best_val = val_metric
                best_step = step
                best_wall_clock = time.perf_counter() - start
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every

            if steps_since_improvement >= patience_steps:
                break

    total_wall_clock = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "steps_to_convergence": best_step,
        "wall_clock_to_convergence_seconds": best_wall_clock,
        "total_steps_run": step,
        "total_wall_clock_seconds": total_wall_clock,
        "best_val_metric": best_val,
    }


# ---------------------------------------------------------------------------
# Graph analysis for CellV1.1's gathered (weights, candidate_idx) routing --
# see this module's docstring for why these can't be positional the way the
# dense version's (n_cells, n_cells) analysis was.
# ---------------------------------------------------------------------------


def _off_self_mask(candidate_idx: torch.Tensor) -> torch.Tensor:
    n = candidate_idx.shape[1]
    self_idx = torch.arange(n, device=candidate_idx.device).view(1, n, 1)
    return candidate_idx != self_idx


def _sparse_sparsity(weights: torch.Tensor, candidate_idx: torch.Tensor) -> float:
    vals = weights[_off_self_mask(candidate_idx)]
    return float((vals == 0).float().mean().item()) if vals.numel() else float("nan")


def _sparse_change(
    idx_prev: torch.Tensor, w_prev: torch.Tensor, idx_curr: torch.Tensor, w_curr: torch.Tensor
) -> float:
    """Mean absolute weight change per candidate slot, accounting for
    candidate *identity* rather than pool position (a receiver's pool
    membership can differ entirely between steps, since the LSH hash
    depends on `z`, which changes every step). Symmetrized: candidates
    present at both `t` and `t+1` are compared directly; candidates only
    present at one of the two times are compared against an implicit `0`
    weight at the other time. Candidates present in both pools are
    counted from both directions -- a minor double-count, not corrected
    for, since this is a descriptive statistic, not a metric feeding any
    decision."""
    w_prev_at_curr = lookup_value(idx_prev, w_prev, idx_curr)
    diff_a = (w_curr - w_prev_at_curr).abs()
    w_curr_at_prev = lookup_value(idx_curr, w_curr, idx_prev)
    diff_b = (w_prev - w_curr_at_prev).abs()
    return float(torch.cat([diff_a.reshape(-1), diff_b.reshape(-1)]).mean().item())


def _padded_group_id(group_id: torch.Tensor, n_cells: int) -> torch.Tensor:
    """`group_id`: `(batch, n_objects)`. Returns `(batch, n_cells)`, with
    `-1` for filler cells beyond `n_objects` (no true group)."""
    batch, n_objects = group_id.shape
    if n_cells == n_objects:
        return group_id
    pad = group_id.new_full((batch, n_cells - n_objects), -1)
    return torch.cat([group_id, pad], dim=-1)


def _sparse_group_agreement_auroc(
    weights: torch.Tensor, candidate_idx: torch.Tensor, group_id: torch.Tensor, n_cells: int
) -> float:
    """AUROC of local-association strength predicting "receiver and
    candidate belong to the same true hidden group," restricted to pairs
    where both sides are real objects (filler cells have no true group)."""
    padded = _padded_group_id(group_id, n_cells)
    receiver_group = padded.unsqueeze(-1).expand_as(candidate_idx)
    candidate_group = gather_scalar(padded, candidate_idx)

    valid = _off_self_mask(candidate_idx) & (receiver_group >= 0) & (candidate_group >= 0)
    scores = weights[valid].detach().cpu().numpy()
    labels = (receiver_group[valid] == candidate_group[valid]).cpu().numpy().astype(int)
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _sparse_global_cross_group_fraction(
    weights: torch.Tensor, candidate_idx: torch.Tensor, group_id: torch.Tensor, n_cells: int
) -> float:
    padded = _padded_group_id(group_id, n_cells)
    receiver_group = padded.unsqueeze(-1).expand_as(candidate_idx)
    candidate_group = gather_scalar(padded, candidate_idx)

    valid = _off_self_mask(candidate_idx) & (receiver_group >= 0) & (candidate_group >= 0)
    edges = valid & (weights > 0)
    cross = edges & (receiver_group != candidate_group)
    total = int(edges.sum().item())
    if total == 0:
        return float("nan")
    return float(cross.sum().item() / total)


def _graph_metrics(
    model: SparseDynamicBeliefGraph,
    x: torch.Tensor,
    group_id: torch.Tensor | None,
    n_cells: int,
    is_full: bool,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _, _, graphs = model.forward_with_graphs(x)
    local_idx_steps = [g[0] for g in graphs]
    a_local_steps = [g[1] for g in graphs]

    metrics: dict[str, float] = {
        "local_sparsity_t0": _sparse_sparsity(a_local_steps[0], local_idx_steps[0]),
        "local_sparsity_tT": _sparse_sparsity(a_local_steps[-1], local_idx_steps[-1]),
    }
    changes = [
        _sparse_change(local_idx_steps[t - 1], a_local_steps[t - 1], local_idx_steps[t], a_local_steps[t])
        for t in range(1, len(graphs))
    ]
    metrics["local_graph_mean_change_per_step"] = float(np.mean(changes)) if changes else float("nan")

    if group_id is not None:
        metrics["local_group_agreement_auroc_t0"] = _sparse_group_agreement_auroc(
            a_local_steps[0], local_idx_steps[0], group_id, n_cells
        )
        metrics["local_group_agreement_auroc_tT"] = _sparse_group_agreement_auroc(
            a_local_steps[-1], local_idx_steps[-1], group_id, n_cells
        )

    if is_full:
        global_idx_final, a_global_final = graphs[-1][2], graphs[-1][3]
        metrics["global_sparsity_tT"] = _sparse_sparsity(a_global_final, global_idx_final)
        if group_id is not None:
            metrics["global_cross_group_fraction_tT"] = _sparse_global_cross_group_fraction(
                a_global_final, global_idx_final, group_id, n_cells
            )
    return metrics


def _save_graph_evolution(
    model: SparseDynamicBeliefGraph,
    x: torch.Tensor,
    y: torch.Tensor,
    group_id: torch.Tensor,
    is_full: bool,
    out_path: Path,
) -> None:
    """Per-step gathered `(candidate_idx, weights)` for up to `x.shape[0]`
    held-out examples, plus the true grouping/target/prediction. Unlike
    the dense version, `a_local[t, b, i, p]` refers to `local_candidate_idx
    [t, b, i, p]`, not a fixed column -- load both together."""
    model.eval()
    with torch.no_grad():
        pred, _, graphs = model.forward_with_graphs(x)
    payload = {
        "local_candidate_idx": np.stack([g[0].cpu().numpy() for g in graphs]),  # (T, batch, N, pool_local)
        "a_local": np.stack([g[1].cpu().numpy() for g in graphs]),
        "group_id": group_id.cpu().numpy(),
        "y_true": y.cpu().numpy(),
        "y_pred": pred.cpu().numpy(),
    }
    if is_full:
        payload["global_candidate_idx"] = np.stack([g[2].cpu().numpy() for g in graphs])
        payload["a_global"] = np.stack([g[3].cpu().numpy() for g in graphs])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def run_task(
    task: str,
    seed: int,
    steps: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 2_000,
    n_val: int = 500,
    n_test: int = 500,
    graph_eval_n: int = 100,
    results_dir: str | Path = "results/raw",
    n_cells: int = N_CELLS,
    association_dim: int = ASSOCIATION_DIM,
    num_steps: int = NUM_STEPS,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    num_hashes_local: int = NUM_HASHES_LOCAL,
    bits_local: int = BITS_LOCAL,
    chunk_size_local: int = CHUNK_SIZE_LOCAL,
    window_local: int = WINDOW_LOCAL,
    num_hashes_global: int = NUM_HASHES_GLOBAL,
    bits_global: int = BITS_GLOBAL,
    chunk_size_global: int = CHUNK_SIZE_GLOBAL,
    window_global: int = WINDOW_GLOBAL,
) -> dict:
    """Trains all four architectures (CellV1 arms via `SparseDynamicBeliefGraph`)
    on `task` with identical optimizer/steps/batch size, evaluates standard
    prediction metrics for all four, and (for `cellv1_local`/`cellv1_full`)
    graph-analysis metrics -- `dynamic_groups` additionally saves per-step
    graphs for up to `graph_eval_n` held-out examples. Writes one
    `RunRecord` per architecture. Returns a flat dict, results prefixed
    `<arch>__`.

    `n_cells` is the main scale lever (`V1_SCALES`: 128/256/512).
    `n_objects`/`k_min`/`k_max` control `dynamic_groups`' complexity (not
    used yet for anything but the default 24-object/K-in-{2..5} setting,
    but threaded through so a harder run doesn't need code changes).
    `n_cells` must be `>= n_objects` for the `dynamic_groups` task
    (`ObjectSeededEncoder` needs one cell per object)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(task)
    is_dynamic_groups = task in OBJECT_SEEDED_TASKS

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_dynamic_groups:
        x_train, y_train, _group_train = train_split
        x_val, y_val, _group_val = val_split
        x_test, y_test, group_test = test_split
        # dynamic_groups features are already well-scaled ([-1,1]-ish);
        # standardizing would distort ObjectSeededEncoder's mu = v_j.
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        group_test = None
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    models, sizing = _build_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        num_steps=num_steps,
        hidden_dim=hidden_dim,
        num_hashes_local=num_hashes_local,
        bits_local=bits_local,
        chunk_size_local=chunk_size_local,
        window_local=window_local,
        num_hashes_global=num_hashes_global,
        bits_global=bits_global,
        chunk_size_global=chunk_size_global,
        window_global=window_global,
    )

    results: dict[str, float | int | str] = {"task": task, "seed": seed, "n_cells": n_cells, "headline": headline, **sizing}

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)
    n_graph = min(graph_eval_n, x_test.shape[0])
    x_graph_dev = x_test_dev[:n_graph]
    group_graph = group_test[:n_graph].to(device) if group_test is not None else None

    for arch_name, model in models.items():
        model.to(device)
        train_wall_clock = _train(model, loss_fn, x_train, y_train, device, seed, steps, batch_size, lr)

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        is_cellv1 = arch_name in ("cellv1_local", "cellv1_full")
        if is_cellv1:
            graph_metrics = _graph_metrics(
                model, x_graph_dev, group_graph, n_cells, is_full=(arch_name == "cellv1_full")
            )
            test_metrics.update(graph_metrics)
            if is_dynamic_groups:
                run_dir = Path(results_dir) / "graph_evolution"
                out_path = run_dir / f"v1_001_{arch_name}_{task}_n{n_cells}_seed{seed}.npz"
                _save_graph_evolution(
                    model, x_graph_dev, y_test_dev[:n_graph], group_graph, arch_name == "cellv1_full", out_path
                )
                test_metrics["graph_evolution_path"] = str(out_path)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        config = ExperimentConfig(
            experiment_id="v1_001_dynamic_groups",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={**sizing, "n_cells": n_cells, "association_dim": association_dim, "num_steps": num_steps},
        )
        run_id = make_run_id(config)
        record = RunRecord(
            run_id=run_id,
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=num_steps if is_cellv1 else None,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={},
            test_metrics={k: v for k, v in test_metrics.items() if k != "graph_evolution_path"},
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results


def _build_field_models(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    association_dim: int = ASSOCIATION_DIM,
    routing_dim: int | None = None,
    num_features: int = FIELD_NUM_FEATURES,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
) -> tuple[dict[str, torch.nn.Module], dict[str, int | str]]:
    """`mlp`/`cellv0.1`, parameter-matched to `field_t2` -- `field_t1` and
    `field_t2` share the *same* parameter count (they differ only in how
    many times the one shared step is applied, `docs/architecture_v1.md`
    §13's "same shared rules" convention -- `num_steps` never adds
    parameters), so either could serve as the matching target; `field_t2`
    is used for no reason beyond being the one built first. No change to
    `field_dynamics.py`/`field_fusion.py`/`field_model.py` math -- this
    only constructs and sizes the existing model."""
    routing_dim = routing_dim if routing_dim is not None else association_dim
    set_seed(seed)

    encoder_kind = task in OBJECT_SEEDED_TASKS

    def make_encoder() -> torch.nn.Module | None:
        if not encoder_kind:
            return None
        return ObjectSeededEncoder(n_objects=n_objects, n_cells=n_cells, association_dim=association_dim)

    field_t2 = SelfOrganizingRefinementField(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_steps=2,
        encoder=make_encoder(),
    )
    target_params = count_parameters(field_t2)

    field_t1 = SelfOrganizingRefinementField(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_steps=1,
        encoder=make_encoder(),
    )

    hidden_cells = _match_belief_hidden_cells(target_params, in_features, out_features)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, out_features, aggregation="scale_stable_precision")

    mlp_hidden_dim = match_hidden_dim(target_params, in_features, out_features, num_hidden_layers=2)
    mlp = MLPBaseline(in_features, mlp_hidden_dim, out_features, num_hidden_layers=2)

    models = {"mlp": mlp, "cellv0.1": cellv0_1, "field_t1": field_t1, "field_t2": field_t2}
    sizing = {
        "field_t2__params": target_params,
        "field_t1__params": count_parameters(field_t1),
        "cellv0.1__params": count_parameters(cellv0_1),
        "cellv0.1__hidden_cells": hidden_cells,
        "mlp__params": count_parameters(mlp),
        "mlp__hidden_dim": mlp_hidden_dim,
    }
    return models, sizing


def run_field_task(
    task: str,
    seed: int,
    steps: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 2_000,
    n_val: int = 500,
    n_test: int = 500,
    results_dir: str | Path = "results/raw",
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    association_dim: int = ASSOCIATION_DIM,
    routing_dim: int | None = None,
    num_features: int = FIELD_NUM_FEATURES,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
) -> dict:
    """`mlp`/`cellv0.1`/`field_t1`/`field_t2` on `task`, identical
    optimizer/steps/batch size, standard prediction metrics only (no
    graph analysis -- see `FIELD_ARCHITECTURES`'s docstring note above).
    Writes one `RunRecord` per architecture. Returns a flat dict, results
    prefixed `<arch>__`."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(task)
    is_dynamic_groups = task in OBJECT_SEEDED_TASKS

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_dynamic_groups:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    models, sizing = _build_field_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
    )

    results: dict[str, float | int | str] = {
        "task": task,
        "seed": seed,
        "n_cells": n_cells,
        "num_features": num_features,
        "headline": headline,
        **sizing,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    for arch_name, model in models.items():
        model.to(device)
        train_wall_clock = _train(model, loss_fn, x_train, y_train, device, seed, steps, batch_size, lr)

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        refinement_iterations = {"field_t1": 1, "field_t2": 2}.get(arch_name)
        config = ExperimentConfig(
            experiment_id="v1_001_dynamic_groups_field",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={**sizing, "n_cells": n_cells, "association_dim": association_dim, "num_features": num_features},
        )
        run_id = make_run_id(config)
        record = RunRecord(
            run_id=run_id,
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=refinement_iterations,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results


def _build_global_comparison_models(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    association_dim: int = ASSOCIATION_DIM,
    routing_dim: int | None = None,
    num_features: int = FIELD_NUM_FEATURES,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    global_dim: int = GLOBAL_DIM,
    include_field_t1: bool = True,
    include_mlp: bool = False,
) -> tuple[dict[str, torch.nn.Module], dict[str, int | str]]:
    """`cellv0.1`/`field_t1`/`field_t2`/`field_local_global_t2`[/`mlp`] --
    point 18's second comparison. `field_local_global_t2` (not
    `field_t2`) is now the size-defining/matching target: global
    communication's ~900 new parameters (`GlobalSendFunction`/
    `GlobalNeedFunction`/`GlobalQueryKey`/the 3-way fuse's own
    bias+gate, `field_dynamics.py`) are a much bigger fraction of this
    task's ~3.5k-param field model than of a from-scratch large model, so
    reusing `field_t2`'s target would leave `field_local_global_t2`
    meaningfully larger than everything it's compared against.
    `field_t1`/`field_t2` are reported at their own frozen, unchanged
    size (smaller by that ~900) -- a real capacity difference, not
    equalized away.

    `include_field_t1=False` skips building/training `field_t1` entirely
    (not just omitting it from the returned dict) -- for the complexity-
    scaling sweep (`run_complexity_scaling.py`), where T1 vs local-T2 was
    already shown to be ~identical on this task and training a 4th model
    at every (level, seed) is a real (~25%) chunk of a much larger sweep's
    wall-clock time. `include_mlp=True` adds a plain `MLPBaseline`
    (`match_hidden_dim`-sized to the same `target_params`) -- off by
    default since it wasn't part of the user's original point-18 listing,
    on for the convergence-based sweep's MLP-vs-CellV0.1 curiosity check
    (does a conventional feed-forward baseline converge faster *and*
    score higher than CellV0.1 under the same protocol?)."""
    routing_dim = routing_dim if routing_dim is not None else association_dim
    set_seed(seed)

    encoder_kind = task in OBJECT_SEEDED_TASKS

    def make_encoder() -> torch.nn.Module | None:
        if not encoder_kind:
            return None
        return ObjectSeededEncoder(n_objects=n_objects, n_cells=n_cells, association_dim=association_dim)

    field_local_global_t2 = SelfOrganizingRefinementField(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_steps=2,
        encoder=make_encoder(),
        use_global=True,
        global_dim=global_dim,
    )
    target_params = count_parameters(field_local_global_t2)

    field_t2 = SelfOrganizingRefinementField(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_steps=2,
        encoder=make_encoder(),
    )
    field_t1 = (
        SelfOrganizingRefinementField(
            in_features=in_features,
            out_features=out_features,
            n_cells=n_cells,
            association_dim=association_dim,
            routing_dim=routing_dim,
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_steps=1,
            encoder=make_encoder(),
        )
        if include_field_t1
        else None
    )

    hidden_cells = _match_belief_hidden_cells(target_params, in_features, out_features)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, out_features, aggregation="scale_stable_precision")

    models = {"cellv0.1": cellv0_1, "field_t2": field_t2, "field_local_global_t2": field_local_global_t2}
    sizing = {
        "field_local_global_t2__params": target_params,
        "field_t2__params": count_parameters(field_t2),
        "cellv0.1__params": count_parameters(cellv0_1),
        "cellv0.1__hidden_cells": hidden_cells,
    }
    if field_t1 is not None:
        models["field_t1"] = field_t1
        sizing["field_t1__params"] = count_parameters(field_t1)
    if include_mlp:
        mlp_hidden_dim = match_hidden_dim(target_params, in_features, out_features, num_hidden_layers=2)
        mlp = MLPBaseline(in_features, mlp_hidden_dim, out_features, num_hidden_layers=2)
        models["mlp"] = mlp
        sizing["mlp__params"] = count_parameters(mlp)
        sizing["mlp__hidden_dim"] = mlp_hidden_dim
    return models, sizing


def run_global_field_comparison(
    task: str,
    seed: int,
    steps: int = 1500,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 3_000,
    n_val: int = 500,
    n_test: int = 500,
    results_dir: str | Path = "results/raw",
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    association_dim: int = ASSOCIATION_DIM,
    routing_dim: int | None = None,
    num_features: int = FIELD_NUM_FEATURES,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    global_dim: int = GLOBAL_DIM,
    include_field_t1: bool = True,
) -> dict:
    """`cellv0.1`/`field_t1`/`field_t2`/`field_local_global_t2` on `task`
    -- point 18's question: "does learned non-local communication improve
    beyond the already-successful local self-organizing refinement?"
    Mirrors `run_field_task`'s structure (same optimizer/steps/batch,
    standard prediction metrics only, one `RunRecord` per architecture),
    kept as a separate function so stage 1's already-recorded
    `run_field_task` 3-seed result stays untouched. `include_field_t1`:
    see `_build_global_comparison_models`'s docstring."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(task)
    is_dynamic_groups = task in OBJECT_SEEDED_TASKS

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_dynamic_groups:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    models, sizing = _build_global_comparison_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        global_dim=global_dim,
        include_field_t1=include_field_t1,
    )

    results: dict[str, float | int | str] = {
        "task": task,
        "seed": seed,
        "n_cells": n_cells,
        "num_features": num_features,
        "global_dim": global_dim,
        "headline": headline,
        **sizing,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    for arch_name, model in models.items():
        model.to(device)
        train_wall_clock = _train(model, loss_fn, x_train, y_train, device, seed, steps, batch_size, lr)

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        refinement_iterations = {"field_t1": 1, "field_t2": 2, "field_local_global_t2": 2}.get(arch_name)
        config = ExperimentConfig(
            experiment_id="v1_001_dynamic_groups_field_global",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={
                **sizing,
                "n_cells": n_cells,
                "association_dim": association_dim,
                "num_features": num_features,
                "global_dim": global_dim,
            },
        )
        run_id = make_run_id(config)
        record = RunRecord(
            run_id=run_id,
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=refinement_iterations,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results


def run_convergence_field_comparison(
    task: str,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 3_000,
    n_val: int = 500,
    n_test: int = 500,
    val_every: int = 100,
    patience_steps: int = 500,
    max_steps: int = 5_000,
    results_dir: str | Path = "results/raw",
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    association_dim: int = ASSOCIATION_DIM,
    routing_dim: int | None = None,
    num_features: int = FIELD_NUM_FEATURES,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    global_dim: int = GLOBAL_DIM,
    include_mlp: bool = False,
) -> dict:
    """`cellv0.1`/`field_t2`/`field_local_global_t2`[/`mlp`] on `task`,
    trained with early stopping (`_train_until_convergence`) instead of a
    fixed step count -- the user's protocol for the convergence-based
    complexity sweep (`run_complexity_scaling_convergence.py`): "stop
    comparing models at a fixed arbitrary step count... a 24-object
    problem and a 192-object problem do not converge on the same
    optimization timescale." No `field_t1` (matching
    `run_complexity_scaling.py`'s reasoning -- already-shown ~identical
    to local-T2 on this task). `include_mlp`: see
    `_build_global_comparison_models`'s docstring. Records, per architecture: best test
    metric (at the best-validation checkpoint), steps/wall-clock *to*
    that checkpoint (the efficiency question -- "how much compute did it
    need to get there"), and total steps/wall-clock actually run
    (checkpoint time plus the patience tail)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(task)
    is_dynamic_groups = task in OBJECT_SEEDED_TASKS

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_dynamic_groups:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    models, sizing = _build_global_comparison_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        routing_dim=routing_dim,
        num_features=num_features,
        hidden_dim=hidden_dim,
        global_dim=global_dim,
        include_field_t1=False,
        include_mlp=include_mlp,
    )

    results: dict[str, float | int | str] = {
        "task": task,
        "seed": seed,
        "n_cells": n_cells,
        "num_features": num_features,
        "global_dim": global_dim,
        "headline": headline,
        **sizing,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    for arch_name, model in models.items():
        model.to(device)
        convergence = _train_until_convergence(
            model, loss_fn, x_train, y_train, x_val, y_val, device, seed, regression,
            batch_size, lr, val_every=val_every, patience_steps=patience_steps, max_steps=max_steps,
        )

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock
        for name, value in convergence.items():
            results[f"{arch_name}__{name}"] = value

        config = ExperimentConfig(
            experiment_id="v1_001_dynamic_groups_field_global_convergence",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            extra={
                **sizing,
                "n_cells": n_cells,
                "association_dim": association_dim,
                "num_features": num_features,
                "global_dim": global_dim,
                "val_every": val_every,
                "patience_steps": patience_steps,
                **convergence,
            },
        )
        run_id = make_run_id(config)
        record = RunRecord(
            run_id=run_id,
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=convergence["total_steps_run"],
            examples_or_tokens_seen=convergence["total_steps_run"] * batch_size,
            refinement_iterations={"field_t2": 2, "field_local_global_t2": 2}.get(arch_name),
            approximate_flops=None,
            train_wall_clock_seconds=convergence["total_wall_clock_seconds"],
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"best": convergence["best_val_metric"]},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results


def _build_association_comparison_models(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    association_dim: int = ASSOCIATION_DIM,
    assoc_dim: int = ASSOC_DIM,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    global_dim: int = GLOBAL_DIM,
) -> tuple[dict[str, torch.nn.Module], dict[str, int | str]]:
    """`cellv0.1`/`association_local_global_t2` -- the user's "leading
    architecture" comparison (`docs/research_log.md`'s learned-
    association redesign). `association_local_global_t2` is the size-
    defining/matching target for `cellv0.1`, same convention as
    `_build_global_comparison_models`. No `field_t1`/`field_t2`/
    `field_local_global_t2`/`mlp` here -- the ORFF field is retired from
    active development (kept elsewhere as the research baseline that
    motivated this redesign, not part of this comparison)."""
    set_seed(seed)

    encoder_kind = task in OBJECT_SEEDED_TASKS

    def make_encoder() -> torch.nn.Module | None:
        if not encoder_kind:
            return None
        return ObjectSeededEncoder(n_objects=n_objects, n_cells=n_cells, association_dim=association_dim)

    association_local_global_t2 = LearnedAssociationField(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        assoc_dim=assoc_dim,
        hidden_dim=hidden_dim,
        num_steps=2,
        encoder=make_encoder(),
        use_global=True,
        global_dim=global_dim,
    )
    target_params = count_parameters(association_local_global_t2)

    hidden_cells = _match_belief_hidden_cells(target_params, in_features, out_features)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, out_features, aggregation="scale_stable_precision")

    models = {"cellv0.1": cellv0_1, "association_local_global_t2": association_local_global_t2}
    sizing = {
        "association_local_global_t2__params": target_params,
        "cellv0.1__params": count_parameters(cellv0_1),
        "cellv0.1__hidden_cells": hidden_cells,
    }
    return models, sizing


def run_convergence_association_comparison(
    task: str,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 3_000,
    n_val: int = 500,
    n_test: int = 500,
    val_every: int = 100,
    patience_steps: int = 500,
    max_steps: int = 5_000,
    results_dir: str | Path = "results/raw",
    n_cells: int = N_CELLS,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    association_dim: int = ASSOCIATION_DIM,
    assoc_dim: int = ASSOC_DIM,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    global_dim: int = GLOBAL_DIM,
) -> dict:
    """`cellv0.1`/`association_local_global_t2` on `task`, trained with
    early stopping (`_train_until_convergence`) -- the user's "does
    dynamically learned organization give something beyond the already-
    strong BeliefCell baseline" question, over the same
    easy/medium/hard/very_hard sweep as `run_complexity_scaling_convergence.py`
    (`run_complexity_scaling_association.py`). Mirrors
    `run_convergence_field_comparison`'s structure exactly."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(task)
    is_dynamic_groups = task in OBJECT_SEEDED_TASKS

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_dynamic_groups:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    models, sizing = _build_association_comparison_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        assoc_dim=assoc_dim,
        hidden_dim=hidden_dim,
        global_dim=global_dim,
    )

    results: dict[str, float | int | str] = {
        "task": task,
        "seed": seed,
        "n_cells": n_cells,
        "assoc_dim": assoc_dim,
        "global_dim": global_dim,
        "headline": headline,
        **sizing,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    for arch_name, model in models.items():
        model.to(device)
        convergence = _train_until_convergence(
            model, loss_fn, x_train, y_train, x_val, y_val, device, seed, regression,
            batch_size, lr, val_every=val_every, patience_steps=patience_steps, max_steps=max_steps,
        )

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock
        for name, value in convergence.items():
            results[f"{arch_name}__{name}"] = value

        config = ExperimentConfig(
            experiment_id="v1_001_dynamic_groups_association_convergence",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            extra={
                **sizing,
                "n_cells": n_cells,
                "association_dim": association_dim,
                "assoc_dim": assoc_dim,
                "global_dim": global_dim,
                "val_every": val_every,
                "patience_steps": patience_steps,
                **convergence,
            },
        )
        run_id = make_run_id(config)
        record = RunRecord(
            run_id=run_id,
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=convergence["total_steps_run"],
            examples_or_tokens_seen=convergence["total_steps_run"] * batch_size,
            refinement_iterations={"association_local_global_t2": 2}.get(arch_name),
            approximate_flops=None,
            train_wall_clock_seconds=convergence["total_wall_clock_seconds"],
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"best": convergence["best_val_metric"]},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results
