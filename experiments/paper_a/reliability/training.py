"""Phase-2 training protocol -- identical to the frozen Phase-1 protocol
(`experiments/paper_a/training.py`: AdamW, ``lr = 1e-2``, ``weight_decay = 0``,
batch 128, best-checkpoint restore, early stopping, 15000-step cap), with two
corruption-specific changes required by the Phase-2 task:

1. **Training data is corrupted per epoch.** At the start of every epoch the
   deterministic corruption for ``(seed, "train", epoch)`` is regenerated over
   the whole training pool, so a model sees a fresh corruption draw each pass
   but re-running the same seed reproduces every value. Batching is a seeded
   permutation of row indices; corruption is bound to the row index, never the
   batch position, so it is independent of shuffle order and model identity.

2. **Checkpoint selection uses the corrupted-validation primary metric,
   averaged over the in-distribution corruption severities** (Sec 9) --
   accuracy for classification, R^2 for California Housing -- *not* validation
   loss. The severe / OOD severities never touch training or selection.

Nothing is tuned per model or per dataset. Divergence / NaNs are recorded, not
worked around; the full validation curve is retained so a cap-censored run is
detectable after the fact.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.paper_a.datasets import PreparedDataset  # noqa: E402
from experiments.paper_a.reliability.corruption import corrupt, in_dist_severities  # noqa: E402
from experiments.paper_a.training import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
    DEFAULT_MIN_DELTA,
    DEFAULT_WEIGHT_DECAY,
)
from src.evaluation.regression import r_squared  # noqa: E402

# Same early-stopping cadence rule as the Phase-1 harness: a large validation
# set is not re-scored every 50 steps, but the patience window stays ~1500-1600
# steps either way.
_LARGE_VAL_THRESHOLD = 2000
_SMALL_VAL_CADENCE = (50, 30)   # (val_every, patience_checks) -> 1500-step patience
_LARGE_VAL_CADENCE = (200, 8)   # -> 1600-step patience
EVAL_CHUNK = 1024


def cadence_for(n_val: int) -> tuple[int, int]:
    return _LARGE_VAL_CADENCE if n_val > _LARGE_VAL_THRESHOLD else _SMALL_VAL_CADENCE


@dataclass
class TrainOutcome:
    history: list[dict] = field(default_factory=list)
    best_val_step: int = 0
    best_val_metric: float = float("nan")   # avg primary metric over in-dist severities
    best_val_loss: float = float("inf")
    total_steps: int = 0
    epochs_completed: float = 0.0
    diverged: bool = False
    cap_hit: bool = False
    train_wall_clock_seconds: float = 0.0
    time_per_step_seconds: float = float("nan")
    examples_seen: int = 0
    effective_batch_size: int = 0
    in_dist_severities: tuple[float, ...] = ()


def _loss_fn(task_type: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()


def _primary_metric(task_type: str, pred: torch.Tensor, y: torch.Tensor) -> float:
    if task_type == "classification":
        return (pred.argmax(dim=-1) == y).float().mean().item()
    return r_squared(pred, y)


@torch.no_grad()
def forward_xc(
    model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> torch.Tensor:
    """Chunked ``model(x_corrupted, c)`` -- the belief layers broadcast to
    ``(batch, out, in)`` internally, so a full 10k-row image split forward must
    be chunked."""
    model.eval()
    if x.shape[0] <= chunk:
        return model(x, c)
    return torch.cat(
        [model(x[i : i + chunk], c[i : i + chunk]) for i in range(0, x.shape[0], chunk)]
    )


def train_model(
    model: torch.nn.Module,
    prepared: PreparedDataset,
    *,
    corruption_family: str,
    device: torch.device,
    seed: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_every: int | None = None,
    patience_checks: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    min_delta: float = DEFAULT_MIN_DELTA,
    eval_chunk: int = EVAL_CHUNK,
) -> TrainOutcome:
    """Train ``model`` on ``prepared``'s split under per-epoch ``corruption_family``
    corruption, restoring the checkpoint with the best averaged in-distribution
    corrupted-validation primary metric. ``model`` is left on ``device`` in eval
    mode holding that checkpoint."""
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    x_train_clean = prepared.x_train                      # kept on CPU; corrupted per epoch
    y_train = prepared.y_train.to(device)
    x_val_clean = prepared.x_val
    y_val = prepared.y_val.to(device)
    task_type = prepared.task_type

    n_train = x_train_clean.shape[0]
    effective_bs = min(batch_size, n_train)
    steps_per_epoch = max(1, math.ceil(n_train / effective_bs))

    default_val_every, default_patience = cadence_for(prepared.meta["n_val"])
    val_every = default_val_every if val_every is None else val_every
    patience_checks = default_patience if patience_checks is None else patience_checks

    severities = in_dist_severities(corruption_family)
    # Fixed corrupted-validation batches (epoch pinned to 0), materialized once.
    val_batches = []
    for sev in severities:
        cor = corrupt(
            corruption_family, x_val_clean,
            experiment_seed=seed, split="val", epoch=0, severity=sev,
        )
        val_batches.append((cor.x.to(device), cor.c.to(device)))

    loss_fn = _loss_fn(task_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    outcome = TrainOutcome(
        effective_batch_size=effective_bs, in_dist_severities=tuple(severities)
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("-inf")
    patience_ref = float("-inf")
    steps_since_improvement = 0
    step = 0
    epoch = 0
    stop = False
    start = time.perf_counter()

    def _validate() -> tuple[float, float]:
        model.eval()
        losses, metrics = [], []
        for xv, cv in val_batches:
            pred = forward_xc(model, xv, cv, chunk=eval_chunk)
            losses.append(loss_fn(pred, y_val).item())
            metrics.append(_primary_metric(task_type, pred, y_val))
        model.train()
        return float(np.mean(losses)), float(np.mean(metrics))

    while step < max_steps and not stop:
        cor = corrupt(
            corruption_family, x_train_clean,
            experiment_seed=seed, split="train", epoch=epoch,
        )
        xc_epoch, c_epoch = cor.x, cor.c  # CPU tensors
        g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
        perm = torch.randperm(n_train, generator=g)

        for b0 in range(0, n_train, effective_bs):
            if step >= max_steps:
                break
            idx = perm[b0 : b0 + effective_bs]
            xb = xc_epoch[idx].to(device)
            cb = c_epoch[idx].to(device)
            yb = y_train[idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            if not torch.isfinite(loss):
                outcome.diverged = True
                outcome.history.append(
                    {"step": step + 1, "train_loss": float("nan"), "event": "nan"}
                )
                stop = True
                break
            loss.backward()
            optimizer.step()
            step += 1

            if step % val_every == 0 or step >= max_steps:
                val_loss, val_metric = _validate()
                outcome.history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": float(loss.item()),
                        "val_loss": val_loss,
                        "val_metric": val_metric,
                    }
                )
                if val_metric > best_metric + 1e-9:
                    best_metric = val_metric
                    outcome.best_val_metric = val_metric
                    outcome.best_val_loss = val_loss
                    outcome.best_val_step = step
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                if val_metric > patience_ref + min_delta:
                    patience_ref = val_metric
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += val_every
                if steps_since_improvement >= patience_checks * val_every:
                    stop = True
                    break
        epoch += 1

    outcome.total_steps = step
    outcome.train_wall_clock_seconds = time.perf_counter() - start
    outcome.time_per_step_seconds = (
        outcome.train_wall_clock_seconds / step if step else float("nan")
    )
    outcome.examples_seen = step * effective_bs
    outcome.epochs_completed = step / steps_per_epoch
    outcome.cap_hit = (step >= max_steps) and not outcome.diverged

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return outcome
