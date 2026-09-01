"""Generic, architecture-agnostic supervised train/eval loops.

Takes any `nn.Module` -- a baseline or (once specified) ArchitectureV0 --
plus a dataloader, loss function, and optimizer. Contains no
architecture-specific logic and must stay that way; task-specific training
scripts (per experiment) should compose these functions rather than fork
them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    steps_completed: int
    examples_seen: int
    train_wall_clock_seconds: float
    final_loss: float


def train_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    max_steps: int | None = None,
    max_epochs: int | None = None,
) -> TrainResult:
    """Ordinary forward/loss/backward/step loop over `(inputs, targets)`
    batches. Stops at `max_steps` if given, else after `max_epochs` full
    passes over `dataloader`. Exactly one of the two must be provided (both
    may be provided; whichever is hit first wins)."""
    if max_steps is None and max_epochs is None:
        raise ValueError("Specify at least one of max_steps or max_epochs.")

    model.to(device)
    model.train()

    start = time.perf_counter()
    steps = 0
    examples_seen = 0
    final_loss = float("nan")
    epoch = 0

    while True:
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            steps += 1
            examples_seen += inputs.shape[0]
            final_loss = loss.item()

            if max_steps is not None and steps >= max_steps:
                return TrainResult(steps, examples_seen, time.perf_counter() - start, final_loss)

        epoch += 1
        if max_epochs is not None and epoch >= max_epochs:
            return TrainResult(steps, examples_seen, time.perf_counter() - start, final_loss)


@torch.no_grad()
def evaluate_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    metric_fns: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
    device: torch.device,
) -> dict[str, float]:
    """Runs `model` over `dataloader` once and applies each metric function
    to the concatenated (predictions, targets)."""
    model.to(device)
    model.eval()

    all_outputs = []
    all_targets = []
    for inputs, targets in dataloader:
        outputs = model(inputs.to(device))
        all_outputs.append(outputs.cpu())
        all_targets.append(targets)

    predictions = torch.cat(all_outputs)
    targets = torch.cat(all_targets)
    return {name: fn(predictions, targets) for name, fn in metric_fns.items()}
