"""Device policy for the strong-baselines benchmark (spec Sec 1).

Resolution order is always `mps -> cpu`; CUDA is never selected on this
machine, even if present, because the spec pins this benchmark to Apple
Silicon. `src.utilities.device.get_device` already does `mps -> cuda -> cpu`
generically for the rest of the project -- this module wraps it with the
CUDA-excluded policy this benchmark specifically requires, plus the
device/dtype/fallback logging spec Sec 1 asks every run to record.

CPU-fallback detection: this module deliberately never sets
`PYTORCH_ENABLE_MPS_FALLBACK=1`. With that flag unset, an op with no MPS
kernel raises `NotImplementedError` immediately instead of silently running
on CPU -- "detect/report rather than silently pretend the whole run is MPS"
(spec Sec 1) is satisfied by failing loud at the exact op, not by auditing
after the fact. `warn_if_cpu_fallback_enabled` additionally flags the case
where something *else* (the shell, a parent process) has set that variable,
since that would defeat the guarantee.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

DTYPE = torch.float32
_FALLBACK_ENV_VAR = "PYTORCH_ENABLE_MPS_FALLBACK"


@dataclass
class DeviceReport:
    """Sec 1's required per-run device metadata."""

    device: torch.device
    device_type: str
    mps_available: bool
    mps_built: bool
    dtype: torch.dtype
    cpu_fallback_env_set: bool

    def to_dict(self) -> dict:
        return {
            "device": str(self.device),
            "device_type": self.device_type,
            "mps_available": self.mps_available,
            "mps_built": self.mps_built,
            "dtype": str(self.dtype),
            "cpu_fallback_env_set": self.cpu_fallback_env_set,
        }


def warn_if_cpu_fallback_enabled() -> bool:
    """Returns True (and prints a warning) if `PYTORCH_ENABLE_MPS_FALLBACK`
    is set -- that setting would let unsupported ops silently run on CPU
    instead of raising, defeating this benchmark's "require MPS actually
    runs" guarantee (spec Sec 1)."""
    is_set = os.environ.get(_FALLBACK_ENV_VAR, "0") not in ("0", "", None)
    if is_set:
        print(
            f"WARNING: {_FALLBACK_ENV_VAR} is set -- unsupported ops will silently "
            "fall back to CPU instead of raising. Unset it for this benchmark."
        )
    return is_set


def resolve_device(prefer: str | None = None) -> DeviceReport:
    """`mps -> cpu`, never `cuda` (spec Sec 1), unless `prefer` overrides
    explicitly (e.g. `"cpu"` for a CI/debug run without MPS)."""
    cpu_fallback_env_set = warn_if_cpu_fallback_enabled()
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    if prefer is not None:
        device = torch.device(prefer)
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    report = DeviceReport(
        device=device,
        device_type=device.type,
        mps_available=mps_available,
        mps_built=mps_built,
        dtype=DTYPE,
        cpu_fallback_env_set=cpu_fallback_env_set,
    )
    return report


def require_mps(report: DeviceReport) -> None:
    """Raises if the resolved device is not MPS -- used to gate the primary
    benchmark run (spec Sec 1: "Require the main benchmark to actually run
    on MPS"). Preflight/smoke-test callers that explicitly pass
    `prefer="cpu"` never call this."""
    if report.device_type != "mps":
        raise RuntimeError(
            f"The primary benchmark requires MPS; resolved device is "
            f"{report.device_type!r} (mps_available={report.mps_available}). "
            "Pass an explicit prefer='cpu' only for a deliberate CPU smoke test."
        )


@dataclass
class StepTimer:
    """Accumulates per-step wall-clock time (spec Sec 1's "per-step wall
    time" logging requirement). `mps_synchronize` forces a device sync
    before stopping the clock -- MPS (like CUDA) dispatches asynchronously,
    so an un-synchronized timer would measure launch overhead, not actual
    compute time."""

    device: torch.device
    step_times_seconds: list[float] = field(default_factory=list)

    def _sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def step(self) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.step_times_seconds.append(time.perf_counter() - start)

    def summary(self) -> dict[str, float]:
        if not self.step_times_seconds:
            return {"n_steps": 0, "ms_per_step": float("nan"), "examples_per_sec": float("nan")}
        times = self.step_times_seconds
        mean_s = sum(times) / len(times)
        return {
            "n_steps": len(times),
            "ms_per_step": mean_s * 1000.0,
            "seconds_per_step_min": min(times),
            "seconds_per_step_max": max(times),
        }


def mps_memory_snapshot() -> dict[str, float] | None:
    """Current/peak MPS allocator memory in MB, or None off-MPS / on a torch
    build without the `mps` memory API."""
    if not torch.backends.mps.is_available():
        return None
    try:
        return {
            "current_allocated_mb": torch.mps.current_allocated_memory() / 1e6,
            "driver_allocated_mb": torch.mps.driver_allocated_memory() / 1e6,
        }
    except AttributeError:
        return None
