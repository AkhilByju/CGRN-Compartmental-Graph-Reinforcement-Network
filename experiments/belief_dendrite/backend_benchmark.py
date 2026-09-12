"""Speed benchmark for Architecture V2's optimized backends
(`src/models/architecture_v2/belief_dendrite_fast.py`) against the frozen
reference implementation, at the exact shapes the CIFAR-10 strong-baselines
benchmark uses (spec: `experiments/belief_dendrite/strong_baselines/
models.py`'s `dendrite_target_hidden()`, `DENDRITE_BRANCHES`,
`DENDRITE_PATCH`, `DENDRITE_K1`/`K2`).

Correctness is NOT this script's job -- see
`tests/test_architecture_v2_belief_dendrite_fast.py` for the parity tests
that must pass before any number here is meaningful. This script only
measures forward-only and forward+backward wall-clock time (mirroring
`experiments/belief_dendrite/strong_baselines/preflight.py`'s
warmup+measured pattern), MPS memory, and reports the speedup each backend
gets over the frozen reference -- run once per session, not part of the
pytest suite (real timing measurements, not deterministic assertions).

Usage: python -m experiments.belief_dendrite.backend_benchmark
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from experiments.belief_dendrite.strong_baselines.device_utils import (  # noqa: E402
    DTYPE,
    StepTimer,
    mps_memory_snapshot,
    resolve_device,
)
from experiments.belief_dendrite.strong_baselines.models import (  # noqa: E402
    DENDRITE_BRANCHES,
    DENDRITE_K1,
    DENDRITE_K2,
    DENDRITE_PATCH,
    IMAGE_SHAPE,
    OUT_FEATURES,
    build_dendrite_connectivity,
    dendrite_target_hidden,
)
from src.models.architecture_v2.belief_dendrite import BeliefDendriteNetwork  # noqa: E402
from src.models.architecture_v2.belief_dendrite_fast import (  # noqa: E402
    BeliefDendriteNetworkDenseFast,
    BeliefDendriteNetworkSparseFast,
)

BATCH_SIZE = 256
WARMUP_STEPS = 30
MEASURED_STEPS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

BACKENDS: dict[str, type] = {
    "frozen_reference": BeliefDendriteNetwork,
    "sparse_fast": BeliefDendriteNetworkSparseFast,
    "dense_fast": BeliefDendriteNetworkDenseFast,
}


@dataclass
class BenchResult:
    name: str
    forward_ms: float
    forward_backward_ms: float
    speedup_vs_reference: float
    mps_memory_mb: dict[str, float] | None


def _build(name: str, conn1, conn2, device: torch.device) -> torch.nn.Module:
    cls = BACKENDS[name]
    if name == "frozen_reference":
        model = BeliefDendriteNetwork(conn1, conn2, OUT_FEATURES)
    else:
        model = cls(conn1, conn2, OUT_FEATURES)
    return model.to(device=device, dtype=DTYPE)


def _random_batch(device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_dim = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
    x = torch.randn(BATCH_SIZE, input_dim, generator=g).to(device=device, dtype=DTYPE)
    c = torch.rand(BATCH_SIZE, input_dim, generator=g).clamp_min(1e-3)
    c = c.to(device=device, dtype=DTYPE)
    y = torch.randint(0, OUT_FEATURES, (BATCH_SIZE,), generator=g).to(device)
    return x, c, y


def _time_forward_only(model, x, c, device, warmup, measured) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, c)
        timer = StepTimer(device)
        for _ in range(measured):
            with timer.step():
                model(x, c)
    return timer.summary()["ms_per_step"]


def _time_forward_backward(model, x, c, y, device, warmup, measured) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _step() -> None:
        optimizer.zero_grad()
        loss = loss_fn(model(x, c), y)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        _step()
    timer = StepTimer(device)
    for _ in range(measured):
        with timer.step():
            _step()
    return timer.summary()["ms_per_step"]


def run_benchmark() -> list[BenchResult]:
    report = resolve_device()
    device = report.device
    print(f"device: {report.to_dict()}")

    hidden = dendrite_target_hidden()
    conn1, conn2 = build_dendrite_connectivity(hidden, seed=0)
    print(
        f"shapes: H={hidden}, B={DENDRITE_BRANCHES}, K1={DENDRITE_K1} "
        f"(patch={DENDRITE_PATCH}), K2={DENDRITE_K2}, batch={BATCH_SIZE}"
    )

    x, c, y = _random_batch(device)
    results: dict[str, BenchResult] = {}
    reference_fwdbwd: float | None = None

    for name in BACKENDS:
        torch.manual_seed(0)
        model = _build(name, conn1, conn2, device)

        fwd_ms = _time_forward_only(model, x, c, device, WARMUP_STEPS, MEASURED_STEPS)
        fwdbwd_ms = _time_forward_backward(model, x, c, y, device, WARMUP_STEPS, MEASURED_STEPS)
        mem = mps_memory_snapshot()

        if name == "frozen_reference":
            reference_fwdbwd = fwdbwd_ms
        speedup = (reference_fwdbwd / fwdbwd_ms) if reference_fwdbwd else 1.0

        results[name] = BenchResult(name, fwd_ms, fwdbwd_ms, speedup, mem)
        print(
            f"  {name:18s} forward={fwd_ms:8.3f}ms  forward+backward={fwdbwd_ms:8.3f}ms  "
            f"speedup={speedup:.2f}x"
        )
        del model

    return list(results.values())


if __name__ == "__main__":
    run_benchmark()
