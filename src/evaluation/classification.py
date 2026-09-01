"""Standard classification metrics (Track B, docs/benchmark_plan.md).
Generic -- works for any model's predictions, not just the novel
architecture."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy(predictions, targets) -> float:
    """`predictions` may be class indices or per-class scores/logits (in
    which case argmax is taken)."""
    p, t = _to_numpy(predictions), _to_numpy(targets)
    if p.ndim > 1 and p.shape[-1] > 1:
        p = p.argmax(axis=-1)
    return float(np.mean(p == t))


def f1(predictions, targets, average: str = "binary") -> float:
    p, t = _to_numpy(predictions), _to_numpy(targets)
    if p.ndim > 1 and p.shape[-1] > 1:
        p = p.argmax(axis=-1)
    return float(f1_score(t, p, average=average))


def auroc(scores, targets) -> float:
    """`scores` must be continuous (probabilities/logits for the positive
    class), not hard class predictions."""
    s, t = _to_numpy(scores), _to_numpy(targets)
    return float(roc_auc_score(t, s))
