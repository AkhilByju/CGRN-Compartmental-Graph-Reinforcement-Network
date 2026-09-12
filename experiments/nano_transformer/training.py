"""Token-budget training loop, LR grid, seeds, and run-record logging
(task Sec 10). Reuses this project's generic run-provenance conventions
(`src/utilities/config.py`, `src/utilities/seeding.py`, `src/utilities/
device.py`, `src/training/logging.py`, `src/training/checkpointing.py`)
rather than reinventing them; the AdamW betas/warmup/cosine-decay schedule
below is this experiment's own config since the shared `src/training/
optimization.py` helper doesn't expose custom betas.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from experiments.nano_transformer.data import DatasetBundle, iterate_batches, load_token_ids
from experiments.nano_transformer.diagnostics import collect_belief_diagnostics
from experiments.nano_transformer.models import BUILDERS, NanoTransformerSpec
from src.training.checkpointing import save_checkpoint
from src.training.logging import RunRecord, get_git_commit, write_run_record
from src.utilities.config import ExperimentConfig, make_run_id
from src.utilities.device import get_device
from src.utilities.seeding import set_seed

LR_GRID: tuple[float, ...] = (3e-4, 1e-3)
SEEDS: tuple[int, ...] = (0, 1, 2)
TOKEN_BUDGET = 30_000_000
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
LEARNING_EFFICIENCY_CHECKPOINTS = (1_000_000, 3_000_000, 10_000_000, 20_000_000, 30_000_000)
# Fixed-size subset of the validation stream used for the frequent
# learning-efficiency curve (Sec 14); the *final* validation/test NLL used
# for LR selection and the single test evaluation (Sec 10) always uses the
# full split -- see evaluate_nll's `max_tokens=None` default.
LEARNING_CURVE_VAL_TOKENS = 500_000


def tokens_per_step() -> int:
    return BATCH_SIZE * CONTEXT_LENGTH


def total_optimizer_steps(token_budget: int = TOKEN_BUDGET) -> int:
    return math.ceil(token_budget / tokens_per_step())


@dataclass
class RunConfig:
    model_kind: str  # one of models.BUILDERS keys
    seed: int
    learning_rate: float
    token_budget: int = TOKEN_BUDGET
    batch_size: int = BATCH_SIZE
    context_length: int = CONTEXT_LENGTH
    target_params: int = 1_000_000
    device_override: str | None = None
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("experiments/nano_transformer/results/raw/checkpoints")
    )
    run_record_dir: Path = field(
        default_factory=lambda: Path("experiments/nano_transformer/results/raw/run_records")
    )
    experiment_id: str = "nano_transformer_1m"


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_nll(
    model: torch.nn.Module,
    ids,
    batch_size: int,
    context_length: int,
    device: torch.device,
    max_tokens: int | None = None,
    **ffn_kwargs: Any,
) -> dict[str, float]:
    """Full (or `max_tokens`-truncated) deterministic pass over `ids`,
    non-overlapping blocks, no shuffling. Returns mean per-token NLL (nats),
    perplexity, and bits-per-token."""
    was_training = model.training
    model.eval()
    stride = context_length + 1
    n_blocks_total = (len(ids) - stride) // context_length + 1
    if max_tokens is not None:
        n_blocks = min(n_blocks_total, max(1, max_tokens // context_length))
    else:
        n_blocks = n_blocks_total

    total_nll = 0.0
    total_tokens = 0
    block = 0
    while block < n_blocks:
        this_batch = min(batch_size, n_blocks - block)
        batch_x = torch.empty((this_batch, context_length), dtype=torch.long)
        batch_y = torch.empty((this_batch, context_length), dtype=torch.long)
        for row in range(this_batch):
            start = block * context_length
            window = ids[start : start + stride]
            batch_x[row] = torch.from_numpy(window[:-1].astype("int64"))
            batch_y[row] = torch.from_numpy(window[1:].astype("int64"))
            block += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x, **ffn_kwargs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), batch_y.reshape(-1), reduction="sum"
        )
        total_nll += loss.item()
        total_tokens += batch_y.numel()

    if was_training:
        model.train()

    mean_nll = total_nll / total_tokens
    return {
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_token": mean_nll / math.log(2),
        "tokens_evaluated": total_tokens,
    }


def build_model_and_spec(
    model_kind: str,
    vocab_size: int,
    context_length: int = CONTEXT_LENGTH,
    target_params: int = 1_000_000,
) -> tuple[torch.nn.Module, dict]:
    spec = NanoTransformerSpec(vocab_size=vocab_size, context_length=context_length)
    built = BUILDERS[model_kind](spec, target=target_params)
    return built.model, built.param_report()


def train_one_run(
    config: RunConfig,
    dataset: DatasetBundle,
    log_every_steps: int = 100,
) -> dict[str, Any]:
    """Trains one (model_kind, seed, learning_rate) run for exactly
    `config.token_budget` non-padding tokens (Sec 10: "compare by
    processed-token budget", never by epoch count). Returns a dict with the
    trained model, its RunRecord, the learning-efficiency curve (Sec 14),
    and belief diagnostics snapshots (Sec 11, cellv0.3 only)."""
    set_seed(config.seed)
    device = get_device(config.device_override)

    model, param_report = build_model_and_spec(
        config.model_kind, dataset.tokenizer.vocab_size, config.context_length,
        config.target_params,
    )
    model.to(device)

    train_ids = load_token_ids(dataset.train_ids_path)
    val_ids = load_token_ids(dataset.val_ids_path)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    total_steps = total_optimizer_steps(config.token_budget)
    warmup_steps = max(1, round(WARMUP_FRACTION * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_lambda(step, warmup_steps, total_steps)
    )

    batches = iterate_batches(train_ids, config.batch_size, config.context_length)

    learning_curve: list[dict[str, Any]] = []
    diagnostics_snapshots: list[dict[str, Any]] = []
    checkpoint_fractions = {0.0, 0.1, 0.5, 1.0}
    tokens_per_batch = config.batch_size * config.context_length
    next_curve_idx = 0

    def _maybe_record_curve_point(tokens_seen: int) -> None:
        nonlocal next_curve_idx
        while (
            next_curve_idx < len(LEARNING_EFFICIENCY_CHECKPOINTS)
            and tokens_seen >= LEARNING_EFFICIENCY_CHECKPOINTS[next_curve_idx]
        ):
            metrics = evaluate_nll(
                model, val_ids, config.batch_size, config.context_length, device,
                max_tokens=LEARNING_CURVE_VAL_TOKENS,
            )
            learning_curve.append(
                {"tokens_seen": LEARNING_EFFICIENCY_CHECKPOINTS[next_curve_idx], **metrics}
            )
            next_curve_idx += 1

    def _maybe_record_diagnostics(fraction: float) -> None:
        diag = collect_belief_diagnostics(model, val_ids, device, config.context_length)
        if diag is not None:
            diagnostics_snapshots.append({"training_fraction": fraction, "layers": diag})

    start_time = time.perf_counter()
    tokens_seen = 0
    _maybe_record_diagnostics(0.0)
    remaining_fractions = sorted(checkpoint_fractions - {0.0})

    model.train()
    step = 0
    while tokens_seen < config.token_budget:
        batch_x_np, batch_y_np = next(batches)
        batch_x = torch.from_numpy(batch_x_np).to(device)
        batch_y = torch.from_numpy(batch_y_np).to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch_y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        tokens_seen += tokens_per_batch
        step += 1
        _maybe_record_curve_point(tokens_seen)

        progress_fraction = tokens_seen / config.token_budget
        while remaining_fractions and progress_fraction >= remaining_fractions[0]:
            _maybe_record_diagnostics(remaining_fractions.pop(0))

        if log_every_steps and step % log_every_steps == 0:
            print(
                f"[{config.model_kind} seed={config.seed} lr={config.learning_rate:.0e}] "
                f"step {step}/{total_steps} tokens={tokens_seen} loss={loss.item():.4f}"
            )

    train_wall_clock = time.perf_counter() - start_time

    final_val_metrics = evaluate_nll(
        model, val_ids, config.batch_size, config.context_length, device
    )

    exp_config = ExperimentConfig(
        experiment_id=config.experiment_id,
        architecture=config.model_kind,
        dataset="tinystories",
        seed=config.seed,
        optimizer="adamw",
        learning_rate=config.learning_rate,
        weight_decay=WEIGHT_DECAY,
        batch_size=config.batch_size,
        max_steps=total_steps,
        extra={
            "context_length": config.context_length,
            "token_budget": config.token_budget,
            "betas": list(BETAS),
            "grad_clip": GRAD_CLIP,
            "warmup_fraction": WARMUP_FRACTION,
            "param_report": param_report,
            "vocab_size": dataset.tokenizer.vocab_size,
        },
    )
    run_id = make_run_id(exp_config)
    checkpoint = save_checkpoint(model, optimizer, run_id, config.checkpoint_dir)

    record = RunRecord(
        run_id=run_id,
        experiment_id=config.experiment_id,
        architecture=config.model_kind,
        config=exp_config.to_dict(),
        parameter_count=param_report["total"],
        dataset="tinystories",
        seed=config.seed,
        optimizer="adamw",
        learning_rate=config.learning_rate,
        steps_completed=step,
        examples_or_tokens_seen=tokens_seen,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=train_wall_clock,
        inference_wall_clock_seconds=None,
        validation_metrics=final_val_metrics,
        test_metrics={},
        git_commit=get_git_commit(),
        checkpoint_path=str(checkpoint),
    )
    write_run_record(record, config.run_record_dir)

    return {
        "model": model,
        "record": record,
        "learning_curve": learning_curve,
        "diagnostics": diagnostics_snapshots,
        "param_report": param_report,
    }


def select_best_lr(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Sec 10: 'Select the LR using validation NLL only.'"""
    return min(run_results, key=lambda r: r["record"].validation_metrics["nll"])
