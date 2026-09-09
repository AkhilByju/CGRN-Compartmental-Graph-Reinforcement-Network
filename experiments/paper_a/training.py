"""One shared training protocol for all three model families (Paper-A task
Sec 7): AdamW, one learning rate (the repo's CellV0.1 protocol, `lr=1e-2`),
one weight-decay convention (`0.0`, explicit), the same batch size wherever
the dataset is large enough for it, best-validation checkpoint restoration,
and early stopping with a generous step cap.

Nothing here is tuned per model or per dataset. Divergence / NaNs are
reported (`TrainOutcome.diverged`), never silently worked around by changing
the learning rate. The full validation curve is retained so a censored run
(one that hit the step cap before converging) can be detected after the fact
(`TrainOutcome.cap_hit`).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from experiments.paper_a.datasets import PreparedDataset  # noqa: E402
from experiments.paper_a.models import batched_forward  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402

# Shared protocol constants (Paper-A task Sec 7).
DEFAULT_LR = 1e-2
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_VAL_EVERY = 50
DEFAULT_PATIENCE_CHECKS = 30  # 30 * 50 = 1500 steps with no meaningful val improvement
DEFAULT_MAX_STEPS = 15_000
# A val-loss decrease smaller than this doesn't reset the early-stopping
# patience counter (it still updates the restored checkpoint). Keeps tiny
# noise fluctuations from holding a run open until the step cap.
DEFAULT_MIN_DELTA = 1e-4


@dataclass
class TrainOutcome:
    history: list[dict] = field(default_factory=list)
    best_val_step: int = 0
    best_val_loss: float = float("inf")
    best_val_metric: float = float("nan")
    total_steps: int = 0
    epochs_completed: float = 0.0
    diverged: bool = False
    cap_hit: bool = False
    train_wall_clock_seconds: float = 0.0
    time_per_step_seconds: float = float("nan")
    examples_seen: int = 0
    effective_batch_size: int = 0


def _loss_fn(task_type: str) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()


@torch.no_grad()
def _val_loss_and_metric(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    task_type: str,
) -> tuple[float, float]:
    model.eval()
    pred = batched_forward(model, x_val)
    loss = loss_fn(pred, y_val).item()
    if task_type == "classification":
        metric = (pred.argmax(dim=-1) == y_val).float().mean().item()
    else:
        metric = r_squared(pred, y_val)  # affine-invariant -> same on raw scale
    model.train()
    return loss, metric


def train_model(
    model: torch.nn.Module,
    prepared: PreparedDataset,
    *,
    device: torch.device,
    seed: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_every: int = DEFAULT_VAL_EVERY,
    patience_checks: int = DEFAULT_PATIENCE_CHECKS,
    max_steps: int = DEFAULT_MAX_STEPS,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> TrainOutcome:
    """Trains `model` on `prepared`'s train split, checkpointing the best
    validation state and restoring it before returning. `model` is left on
    `device` in eval mode holding the best-validation weights."""
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    x_train = prepared.x_train.to(device)
    y_train = prepared.y_train.to(device)
    x_val = prepared.x_val.to(device)
    y_val = prepared.y_val.to(device)

    n_train = x_train.shape[0]
    effective_bs = min(batch_size, n_train)
    steps_per_epoch = max(1, (n_train + effective_bs - 1) // effective_bs)

    loss_fn = _loss_fn(prepared.task_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=effective_bs,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    outcome = TrainOutcome(effective_batch_size=effective_bs)
    best_state: dict[str, torch.Tensor] | None = None
    patience_ref_loss = float("inf")  # last loss that meaningfully improved
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

        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            outcome.diverged = True
            outcome.history.append({"step": step + 1, "train_loss": float("nan"), "event": "nan"})
            break
        loss.backward()
        optimizer.step()
        step += 1

        if step % val_every == 0 or step >= max_steps:
            val_loss, val_metric = _val_loss_and_metric(
                model, loss_fn, x_val, y_val, prepared.task_type
            )
            outcome.history.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "val_metric": val_metric,
                }
            )
            if val_loss < outcome.best_val_loss - 1e-9:
                outcome.best_val_loss = val_loss
                outcome.best_val_metric = val_metric
                outcome.best_val_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if val_loss < patience_ref_loss - min_delta:
                patience_ref_loss = val_loss
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every

            if steps_since_improvement >= patience_checks * val_every:
                break

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
