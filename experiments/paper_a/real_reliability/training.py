"""Shared training protocol for Paper A Phase 3 Part B.

Phase 1 showed that forcing a single learning rate can handicap ordinary MLP
baselines on some regression tasks, so **every** neural model here sweeps the
same predeclared grid ``{1e-3, 3e-3, 1e-2}`` (NeuMiss additionally sweeps its
depth grid ``{1, 3, 5}``). For each ``(dataset, model, seed)``: train every
configuration on train/validation only, pick the one with the best validation
primary metric, and evaluate the test set exactly once.

* Primary validation metric: **PR-AUC / average precision** for APS,
  **R^2** for Air Quality (never accuracy).
* Loss: the shared ``BCEWithLogitsLoss(pos_weight)`` for APS (``pos_weight``
  computed once from the training labels), ``MSELoss`` for Air Quality.
* Optimizer: ``AdamW`` (``weight_decay = 0``), batch 128, best-checkpoint
  restore, early stopping, a generous step cap. Nothing tuned per model beyond
  the shared LR / depth grids.
* NeuMiss consumes the NaN-preserving inputs; every other model consumes the
  zero-imputed inputs. Both receive ``c`` (NeuMiss ignores it).
* APS only: after training, the ``10*FP + 500*FN`` cost threshold is chosen on
  the **validation** set, frozen, and applied once to the test set.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from experiments.paper_a.real_reliability.datasets import RealPreparedDataset  # noqa: E402
from experiments.paper_a.real_reliability.models import (  # noqa: E402
    NEUMISS,
    NEUMISS_DEPTHS,
    build_model,
)
from src.evaluation.regression import r_squared  # noqa: E402

LEARNING_RATES: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_STEPS = 8_000
DEFAULT_MIN_DELTA = 1e-4
EVAL_CHUNK = 4096

# APS official cost weights.
FP_COST = 10.0
FN_COST = 500.0

_LARGE_VAL_THRESHOLD = 2_000
_SMALL_VAL_CADENCE = (50, 30)    # (val_every, patience_checks) -> 1500-step patience
_LARGE_VAL_CADENCE = (200, 8)    # -> 1600-step patience


def cadence_for(n_val: int) -> tuple[int, int]:
    return _LARGE_VAL_CADENCE if n_val > _LARGE_VAL_THRESHOLD else _SMALL_VAL_CADENCE


def model_inputs(
    prepared: RealPreparedDataset, split: str, family: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x, c)` for one split -- NaN-preserving x for NeuMiss, zero-imputed x
    for everyone else."""
    imputed = family != NEUMISS
    x = {
        ("train", True): prepared.x_train, ("train", False): prepared.x_train_nan,
        ("val", True): prepared.x_val, ("val", False): prepared.x_val_nan,
        ("test", True): prepared.x_test, ("test", False): prepared.x_test_nan,
    }[(split, imputed)]
    c = {"train": prepared.c_train, "val": prepared.c_val, "test": prepared.c_test}[split]
    return x, c


@dataclass
class TrainOutcome:
    family: str
    lr: float
    neumiss_depth: int | None
    best_val_metric: float
    best_val_step: int
    total_steps: int
    diverged: bool
    cap_hit: bool
    train_wall_clock_seconds: float
    time_per_step_seconds: float
    history: list[dict] = field(default_factory=list)
    best_state: dict | None = None
    cost_threshold: float | None = None  # APS: frozen validation-cost-optimal threshold


def _loss_fn(prepared: RealPreparedDataset, device: torch.device) -> torch.nn.Module:
    if prepared.task_type == "classification":
        pw = torch.tensor(float(prepared.pos_weight), device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    return torch.nn.MSELoss()


@torch.no_grad()
def _forward(model, x, c, chunk=EVAL_CHUNK) -> torch.Tensor:
    model.eval()
    if x.shape[0] <= chunk:
        return model(x, c).reshape(x.shape[0], -1)
    return torch.cat(
        [model(x[i : i + chunk], c[i : i + chunk]).reshape(-1, 1)
         for i in range(0, x.shape[0], chunk)]
    )


def _primary_val_metric(
    prepared: RealPreparedDataset, logits_or_pred: torch.Tensor, y: torch.Tensor
) -> float:
    if prepared.task_type == "classification":
        prob = torch.sigmoid(logits_or_pred.reshape(-1)).cpu().numpy()
        yt = y.reshape(-1).cpu().numpy()
        if yt.sum() == 0 or yt.sum() == len(yt):
            return float("nan")
        return float(average_precision_score(yt, prob))
    return r_squared(logits_or_pred.reshape(-1), y.reshape(-1))


def _cost_optimal_threshold(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold on P(pos) that minimizes ``FP_COST*FP + FN_COST*FN``, and that
    minimum cost. Candidates are the sorted unique probabilities plus 0/1."""
    order = np.argsort(prob)
    p_sorted = prob[order].astype(np.float64)
    y_sorted = y[order].astype(np.float64)
    total_pos = y_sorted.sum()
    total_neg = len(y_sorted) - total_pos
    # cut at position k: indices >= k predicted pos, < k predicted neg.
    fn_at_cut = np.concatenate([[0.0], np.cumsum(y_sorted)])       # positives below the cut
    neg_below = np.concatenate([[0.0], np.cumsum(1.0 - y_sorted)])
    fp_at_cut = total_neg - neg_below                             # negatives at/above the cut
    cost = FP_COST * fp_at_cut + FN_COST * fn_at_cut
    k = int(np.argmin(cost))
    thr = 0.0 if k == 0 else float(p_sorted[k - 1] + 1e-12)
    return thr, float(cost[k])


def official_cost(prob: np.ndarray, y: np.ndarray, threshold: float) -> float:
    pred = (prob >= threshold).astype(np.int64)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return FP_COST * fp + FN_COST * fn


def train_one(
    model: torch.nn.Module,
    prepared: RealPreparedDataset,
    *,
    family: str,
    lr: float,
    neumiss_depth: int | None,
    device: torch.device,
    seed: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> TrainOutcome:
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    xtr, ctr = (t.to(device) for t in model_inputs(prepared, "train", family))
    ytr = prepared.y_train.to(device).float().reshape(xtr.shape[0], -1)
    xva, cva = (t.to(device) for t in model_inputs(prepared, "val", family))
    yva = prepared.y_val.to(device).float().reshape(xva.shape[0], -1)

    n = xtr.shape[0]
    bs = min(batch_size, n)
    val_every, patience_checks = cadence_for(yva.shape[0])
    loss_fn = _loss_fn(prepared, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed * 7919 + int(round(lr * 1e6)))

    out = TrainOutcome(
        family=family, lr=lr, neumiss_depth=neumiss_depth,
        best_val_metric=float("-inf"), best_val_step=0, total_steps=0,
        diverged=False, cap_hit=False, train_wall_clock_seconds=0.0,
        time_per_step_seconds=float("nan"),
    )
    best_metric = float("-inf")
    ref, since = float("-inf"), 0
    step, epoch, stop = 0, 0, False
    start = time.perf_counter()

    while step < max_steps and not stop:
        perm = torch.randperm(n, generator=gen)
        for b0 in range(0, n, bs):
            if step >= max_steps:
                break
            idx = perm[b0 : b0 + bs]
            opt.zero_grad()
            pred = model(xtr[idx], ctr[idx]).reshape(idx.shape[0], -1)
            loss = loss_fn(pred, ytr[idx])
            if not torch.isfinite(loss):
                out.diverged = True
                out.history.append({"step": step + 1, "event": "nan"})
                stop = True
                break
            loss.backward()
            opt.step()
            step += 1

            if step % val_every == 0 or step >= max_steps:
                vpred = _forward(model, xva, cva)
                vmetric = _primary_val_metric(prepared, vpred, yva)
                out.history.append(
                    {"step": step, "val_metric": vmetric, "train_loss": float(loss.detach())}
                )
                if np.isfinite(vmetric) and vmetric > best_metric + 1e-9:
                    best_metric = vmetric
                    out.best_val_metric = vmetric
                    out.best_val_step = step
                    out.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if np.isfinite(vmetric) and vmetric > ref + min_delta:
                    ref, since = vmetric, 0
                else:
                    since += val_every
                if since >= patience_checks * val_every:
                    stop = True
                    break
        epoch += 1

    out.total_steps = step
    out.train_wall_clock_seconds = time.perf_counter() - start
    out.time_per_step_seconds = out.train_wall_clock_seconds / max(step, 1)
    out.cap_hit = (step >= max_steps) and not out.diverged

    if out.best_state is not None:
        model.load_state_dict(out.best_state)
    model.eval()

    # APS: freeze the cost-optimal threshold from the validation set.
    if prepared.task_type == "classification" and out.best_state is not None:
        vprob = torch.sigmoid(_forward(model, xva, cva).reshape(-1)).cpu().numpy()
        thr, _ = _cost_optimal_threshold(vprob, yva.reshape(-1).cpu().numpy())
        out.cost_threshold = thr
    return out


def train_with_grid(
    prepared: RealPreparedDataset,
    family: str,
    *,
    in_features: int,
    param_budget: int,
    seed: int,
    device: torch.device,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    max_steps: int = DEFAULT_MAX_STEPS,
    build_kwargs: dict | None = None,
) -> tuple[TrainOutcome, torch.nn.Module, object]:
    """Sweep the shared LR grid (and, for NeuMiss, the depth grid), pick the
    configuration with the best validation primary metric, and return
    ``(winning_outcome, restored_model, built)``."""
    from src.utilities.seeding import set_seed

    depths = NEUMISS_DEPTHS if family == NEUMISS else (None,)
    configs = [(lr, d) for d in depths for lr in learning_rates]

    best: tuple[TrainOutcome, torch.nn.Module, object] | None = None
    for lr, depth in configs:
        set_seed(seed)
        kw = dict(build_kwargs or {})
        if depth is not None:
            kw["neumiss_depth"] = depth
        built = build_model(family, in_features, param_budget, **kw)
        outcome = train_one(
            built.model, prepared,
            family=family, lr=lr, neumiss_depth=depth,
            device=device, seed=seed, max_steps=max_steps,
        )
        if best is None or (
            np.isfinite(outcome.best_val_metric)
            and outcome.best_val_metric > best[0].best_val_metric
        ):
            best = (outcome, built.model, built)
    assert best is not None
    return best
