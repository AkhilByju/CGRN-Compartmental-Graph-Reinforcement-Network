"""Training protocol for the strong-baselines benchmark (spec Sec 7).

AdamW; a common predeclared LR grid (`{1e-3, 3e-3}`) tried for *every*
family, never tuned per architecture; weight decay `1e-4`; a common batch
size across all seven families (`pick_batch_size` below); best-validation
checkpoint restore with `max_epochs=100`, `patience=15` epochs; checkpoint/LR
selection uses validation accuracy averaged over the in-distribution patch
severities `{0, 4, 8, 12}` only -- the test split is touched exactly once,
after LR selection (spec Sec 7's "Test once after selection").

Per epoch (mirroring `experiments/belief_dendrite/training.py`'s frozen
convention for the MNIST benchmark): a fresh horizontal flip + missing-patch
corruption draw over the *whole* resident-on-device training tensor,
deterministic in `(seed, epoch)` and independent of model identity, then
ordinary mini-batching. This keeps corruption vectorized (one draw per
epoch, not per step) and keeps training data resident on the training
device throughout (spec Sec 1: "Keep batches resident on MPS only for each
step; avoid unnecessary host/device transfers" -- the one-time
CPU→device copy happens once per run, not once per step).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.belief_dendrite.strong_baselines.corruption import (
    corrupt_missing_patch,
    horizontal_flip,
    in_dist_severities,
)
from experiments.belief_dendrite.strong_baselines.datasets import PreparedImageDataset
from experiments.belief_dendrite.strong_baselines.device_utils import DTYPE, StepTimer

LR_GRID: tuple[float, ...] = (1e-3, 3e-3)
WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZES: tuple[int, ...] = (256, 128, 64)
MAX_EPOCHS = 100
PATIENCE_EPOCHS = 15
MIN_DELTA = 1e-4
EVAL_CHUNK = 512


@dataclass
class TrainOutcome:
    history: list[dict] = field(default_factory=list)
    best_val_epoch: int = 0
    best_val_metric: float = float("-inf")
    total_steps: int = 0
    epochs_completed: int = 0
    diverged: bool = False
    cap_hit: bool = False
    train_wall_clock_seconds: float = 0.0
    time_per_step_seconds: float = float("nan")
    examples_seen: int = 0
    batch_size: int = 0
    lr: float = float("nan")


@torch.no_grad()
def forward_xc(
    model: torch.nn.Module, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> torch.Tensor:
    """Chunked `model(x, c)` -- bounds peak activation memory for a full
    validation/test split forward pass."""
    model.eval()
    if x.shape[0] <= chunk:
        return model(x, c)
    return torch.cat(
        [model(x[i : i + chunk], c[i : i + chunk]) for i in range(0, x.shape[0], chunk)]
    )


def _val_accuracy(
    model: torch.nn.Module,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    y_val: torch.Tensor,
    chunk: int,
) -> float:
    accs = []
    for xv, cv in val_batches:
        pred = forward_xc(model, xv, cv, chunk=chunk)
        accs.append((pred.argmax(dim=-1) == y_val).float().mean().item())
    return float(np.mean(accs))


def train_one_lr(
    model: torch.nn.Module,
    prepared: PreparedImageDataset,
    *,
    device: torch.device,
    seed: int,
    lr: float,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = 256,
    max_epochs: int = MAX_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
    min_delta: float = MIN_DELTA,
    eval_chunk: int = EVAL_CHUNK,
) -> TrainOutcome:
    """Trains `model` at one fixed learning rate under the full Sec 7
    protocol and returns the best-validation-checkpoint outcome (`model` is
    left holding the best-checkpoint weights on return)."""
    torch.manual_seed(seed)
    model.to(device=device, dtype=DTYPE)
    model.train()

    x_train_clean = prepared.x_train.to(device=device, dtype=DTYPE)
    y_train = prepared.y_train.to(device)
    x_val_clean = prepared.x_val.to(device=device, dtype=DTYPE)
    y_val = prepared.y_val.to(device)

    n_train = x_train_clean.shape[0]
    effective_bs = min(batch_size, n_train)

    severities = in_dist_severities()
    val_batches = []
    for sev in severities:
        cor = corrupt_missing_patch(
            x_val_clean, experiment_seed=seed, split="val", epoch=0, severity=sev
        )
        val_batches.append((cor.x, cor.c))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    timer = StepTimer(device)

    outcome = TrainOutcome(batch_size=effective_bs, lr=lr)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0
    step = 0
    start = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        x_flipped = horizontal_flip(x_train_clean, experiment_seed=seed, split="train", epoch=epoch)
        cor = corrupt_missing_patch(x_flipped, experiment_seed=seed, split="train", epoch=epoch)
        xc_epoch, c_epoch = cor.x, cor.c

        gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + epoch)
        perm = torch.randperm(n_train, generator=gen).to(device)

        for b0 in range(0, n_train, effective_bs):
            idx = perm[b0 : b0 + effective_bs]
            xb, cb, yb = xc_epoch[idx], c_epoch[idx], y_train[idx]

            with timer.step():
                optimizer.zero_grad()
                loss = loss_fn(model(xb, cb), yb)
                if not torch.isfinite(loss):
                    outcome.diverged = True
                    outcome.history.append({"epoch": epoch, "step": step, "event": "nan"})
                    break
                loss.backward()
                optimizer.step()
            step += 1
        if outcome.diverged:
            break

        val_metric = _val_accuracy(model, val_batches, y_val, eval_chunk)
        outcome.history.append(
            {
                "epoch": epoch,
                "step": step,
                "train_loss": float(loss.item()),
                "val_metric": val_metric,
            }
        )
        if val_metric > outcome.best_val_metric + min_delta:
            outcome.best_val_metric = val_metric
            outcome.best_val_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience_epochs:
            break

    outcome.total_steps = step
    outcome.epochs_completed = len(outcome.history)
    outcome.train_wall_clock_seconds = time.perf_counter() - start
    outcome.time_per_step_seconds = (
        outcome.train_wall_clock_seconds / step if step else float("nan")
    )
    outcome.examples_seen = step * effective_bs
    outcome.cap_hit = (outcome.epochs_completed >= max_epochs) and not outcome.diverged

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return outcome


@dataclass
class LRSelectionResult:
    chosen_lr: float
    chosen_outcome: TrainOutcome
    all_outcomes: dict[float, TrainOutcome]


def select_lr_and_train(
    model_factory,
    prepared: PreparedImageDataset,
    *,
    device: torch.device,
    seed: int,
    lr_grid: tuple[float, ...] = LR_GRID,
    batch_size: int = 256,
    **train_kwargs,
) -> tuple[torch.nn.Module, LRSelectionResult]:
    """Trains a fresh model (from `model_factory()`, so each LR starts from
    an independent initialization) at every LR in `lr_grid`, selects the one
    with the best validation metric (spec Sec 7: "Select LR/checkpoint using
    validation only"), and returns that model (best-checkpoint weights
    already loaded) plus a record of every LR's outcome."""
    outcomes: dict[float, TrainOutcome] = {}
    models: dict[float, torch.nn.Module] = {}
    for lr in lr_grid:
        m = model_factory()
        outcomes[lr] = train_one_lr(
            m, prepared, device=device, seed=seed, lr=lr, batch_size=batch_size, **train_kwargs
        )
        models[lr] = m

    best_lr = max(outcomes, key=lambda lr: outcomes[lr].best_val_metric)
    return models[best_lr], LRSelectionResult(best_lr, outcomes[best_lr], outcomes)
