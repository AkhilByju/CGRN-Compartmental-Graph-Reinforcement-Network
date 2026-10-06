"""MPS speed preflight (spec Sec 8): before the full sweep, every one of the
seven model families gets 100 warm-up steps and 300 measured forward +
backward + optimizer steps at the common batch size, reporting ms/step,
examples/sec, and MPS memory. This changes nothing about model math --
"Implementation-level optimization is allowed... Do not change mathematical
behavior" (spec Sec 8) -- it is a read-only measurement pass, and also picks
the one batch size every family trains at (spec Sec 7: "Do not give
different batch sizes to different models unless absolutely required").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from experiments.belief_dendrite.strong_baselines.device_utils import (
    DTYPE,
    StepTimer,
    mps_memory_snapshot,
)
from experiments.belief_dendrite.strong_baselines.models import MODEL_FAMILIES, build_model
from experiments.belief_dendrite.strong_baselines.training import WEIGHT_DECAY

DEFAULT_WARMUP_STEPS = 100
DEFAULT_MEASURED_STEPS = 300
CANDIDATE_BATCH_SIZES: tuple[int, ...] = (256, 128, 64)
SLOWDOWN_FLAG_RATIO = 5.0  # spec Sec 8: flag if belief_dendrite > 5x tiny_vit/small_cnn


@dataclass
class PreflightResult:
    family: str
    batch_size: int
    parameter_count: int
    ms_per_step: float
    examples_per_sec: float
    mps_memory_mb: dict[str, float] | None
    oom: bool = False
    error: str | None = None


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def _random_batch(
    batch_size: int, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch_size, 3, 32, 32, generator=g).to(device=device, dtype=DTYPE)
    c = torch.rand(batch_size, 3, 32, 32, generator=g).to(device=device, dtype=DTYPE)
    y = torch.randint(0, 10, (batch_size,), generator=g).to(device)
    return x, c, y


def preflight_one(
    family: str,
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    lr: float = 1e-3,
) -> PreflightResult:
    built = build_model(family, seed)
    model = built.model.to(device=device, dtype=DTYPE)
    x, c, y = _random_batch(batch_size, device, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _step() -> None:
        optimizer.zero_grad()
        loss = loss_fn(model(x, c), y)
        loss.backward()
        optimizer.step()

    try:
        for _ in range(warmup_steps):
            _step()
        timer = StepTimer(device)
        for _ in range(measured_steps):
            with timer.step():
                _step()
    except RuntimeError as exc:
        if _is_oom(exc):
            return PreflightResult(
                family,
                batch_size,
                built.parameter_count,
                float("nan"),
                float("nan"),
                None,
                oom=True,
                error=str(exc),
            )
        raise

    summary = timer.summary()
    ms_per_step = summary["ms_per_step"]
    examples_per_sec = (
        batch_size / (ms_per_step / 1000.0) if ms_per_step == ms_per_step else float("nan")
    )
    return PreflightResult(
        family=family,
        batch_size=batch_size,
        parameter_count=built.parameter_count,
        ms_per_step=ms_per_step,
        examples_per_sec=examples_per_sec,
        mps_memory_mb=mps_memory_snapshot(),
    )


def pick_batch_size(
    device: torch.device,
    *,
    seed: int = 0,
    candidates: tuple[int, ...] = CANDIDATE_BATCH_SIZES,
    families: tuple[str, ...] = MODEL_FAMILIES,
) -> int:
    """The largest batch size (spec Sec 7: "256 -> 128 -> 64" on OOM) at
    which every family survives a short (5 warmup + 5 measured) probe."""
    for bs in candidates:
        ok = True
        for fam in families:
            result = preflight_one(
                fam, seed=seed, device=device, batch_size=bs, warmup_steps=5, measured_steps=5
            )
            if result.oom:
                ok = False
                break
        if ok:
            return bs
    raise RuntimeError(f"no candidate batch size in {candidates} fit every family on {device}")


@dataclass
class PreflightReport:
    device: str
    batch_size: int
    results: list[PreflightResult] = field(default_factory=list)
    slowdown_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "results": [
                {
                    "family": r.family,
                    "parameter_count": r.parameter_count,
                    "ms_per_step": r.ms_per_step,
                    "examples_per_sec": r.examples_per_sec,
                    "mps_memory_mb": r.mps_memory_mb,
                    "oom": r.oom,
                }
                for r in self.results
            ],
            "slowdown_flags": self.slowdown_flags,
        }


def run_preflight(
    device: torch.device,
    *,
    seed: int = 0,
    batch_size: int | None = None,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    measured_steps: int = DEFAULT_MEASURED_STEPS,
    families: tuple[str, ...] = MODEL_FAMILIES,
) -> PreflightReport:
    bs = (
        batch_size
        if batch_size is not None
        else pick_batch_size(device, seed=seed, families=families)
    )
    results = [
        preflight_one(
            fam,
            seed=seed,
            device=device,
            batch_size=bs,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        for fam in families
    ]
    report = PreflightReport(device=str(device), batch_size=bs, results=results)

    baseline_candidates = [
        r.ms_per_step for r in results if r.family in ("small_cnn", "tiny_vit") and not r.oom
    ]
    if baseline_candidates:
        baseline = min(baseline_candidates)
        for r in results:
            if not r.oom and r.ms_per_step > SLOWDOWN_FLAG_RATIO * baseline:
                report.slowdown_flags.append(
                    f"{r.family}: {r.ms_per_step:.2f}ms/step vs baseline {baseline:.2f}ms/step "
                    f"({r.ms_per_step / baseline:.1f}x) -- profile implementation, "
                    "do not change math"
                )
    return report
