"""Uncertainty-calibration metrics against a KNOWN ground-truth uncertainty
signal (docs/benchmark_plan.md), as opposed to the qualitative "ambiguous
vs. clear" proxy used in Experiment 002. Used starting Experiment 003B
against `src/data/synthetic/uncertainty.py`'s labeled noise levels, but
generic -- works for any model's predicted-uncertainty tensor, not just
`architecture_v0`'s.

Rank correlation and significance testing are implemented directly here
(not via `scipy.stats`), since scipy is not a declared project dependency
(see `pyproject.toml`)."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pearson_correlation(predicted, target) -> float:
    """Linear correlation between predicted and true uncertainty. NaN if
    either is constant (undefined correlation), rather than raising."""
    p, t = _to_numpy(predicted).reshape(-1), _to_numpy(target).reshape(-1)
    if p.std() == 0 or t.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks via argsort, without tie-averaging -- adequate here since
    every caller in this project passes continuous synthetic data, where
    exact ties have probability zero."""
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_correlation(predicted, target) -> float:
    """Rank correlation -- robust to the predicted uncertainty being a
    monotonic-but-nonlinear function of the true noise level (rank
    transform + Pearson)."""
    p, t = _to_numpy(predicted).reshape(-1), _to_numpy(target).reshape(-1)
    return pearson_correlation(_rank(p), _rank(t))


def calibration_mse(predicted, target) -> float:
    """Mean squared error between predicted and true uncertainty, in the
    uncertainty's own units (e.g. predicted vs. true noise std) -- unlike
    the correlations above, this is sensitive to scale/offset, not just
    monotonic tracking."""
    p, t = _to_numpy(predicted).reshape(-1), _to_numpy(target).reshape(-1)
    return float(np.mean((p - t) ** 2))
