"""Standard regression metrics (Track A, docs/benchmark_plan.md). Generic --
works for any model's predictions, not just the novel architecture."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mae(predictions, targets) -> float:
    p, t = _to_numpy(predictions), _to_numpy(targets)
    return float(np.mean(np.abs(p - t)))


def rmse(predictions, targets) -> float:
    p, t = _to_numpy(predictions), _to_numpy(targets)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def r_squared(predictions, targets) -> float:
    p, t = _to_numpy(predictions), _to_numpy(targets)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
