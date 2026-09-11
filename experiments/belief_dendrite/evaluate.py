"""Per-severity evaluation, robustness-curve summaries, and the Sec O
mechanism diagnostics (branch/soma precision, branch-routing effective
count, the corruption-localization correlation, somatic conflict) for the
Architecture V2 frozen benchmark.

Each best checkpoint is evaluated **once** across the whole fixed severity
grid for its corruption family (never a separate model per severity); at
every severity, `N_TEST_REPLICAS` deterministic corruption replicas
(identical across models) are averaged before any other statistic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.belief_dendrite.corruption import (  # noqa: E402
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
    N_TEST_REPLICAS,
    corrupt,
    eval_severities,
    in_dist_severities,
)
from experiments.belief_dendrite.models import BELIEF_DENDRITE, CELLV03, DenseCellV03  # noqa: E402
from experiments.belief_dendrite.training import EVAL_CHUNK, forward_xc  # noqa: E402
from experiments.paper_a.datasets import PreparedDataset  # noqa: E402
from src.evaluation.classification import accuracy, f1  # noqa: E402
from src.models.architecture_v0.precision_gain import effective_precision  # noqa: E402
from src.models.architecture_v2.belief_dendrite import BeliefDendriteNetwork  # noqa: E402

PRIMARY_METRIC = "accuracy"
METRIC_NAMES: tuple[str, ...] = ("accuracy", "macro_f1")


@dataclass
class SeverityResult:
    severity: float
    in_distribution: bool
    metrics: dict[str, float]
    replica_metrics: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, float] | None = None  # cellv0.3 / belief_dendrite only

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "in_distribution": self.in_distribution,
            "metrics": self.metrics,
            "replica_metrics": self.replica_metrics,
            "diagnostics": self.diagnostics,
        }


@dataclass
class SweepResult:
    dataset: str
    corruption_family: str
    family: str
    severities: list[SeverityResult]
    corruption_auc: dict[str, float]
    ood_drop: dict[str, float]

    def metric_curve(self, metric: str) -> list[tuple[float, float]]:
        return [(s.severity, s.metrics[metric]) for s in self.severities]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "corruption_family": self.corruption_family,
            "family": self.family,
            "severities": [s.to_dict() for s in self.severities],
            "corruption_auc": self.corruption_auc,
            "ood_drop": self.ood_drop,
        }


def _score(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu()
    y = y.detach().cpu()
    preds = pred.argmax(dim=-1)
    return {"accuracy": accuracy(preds, y), "macro_f1": f1(preds, y, average="macro")}


def _rs() -> dict[str, float]:
    return {"n": 0.0, "sum": 0.0, "sq": 0.0, "min": float("inf"), "max": float("-inf")}


def _rs_update(slot: dict[str, float], t: torch.Tensor) -> None:
    slot["n"] += t.numel()
    slot["sum"] += float(t.sum())
    slot["sq"] += float((t * t).sum())
    slot["min"] = min(slot["min"], float(t.min()))
    slot["max"] = max(slot["max"], float(t.max()))


def _rs_final(slot: dict[str, float]) -> tuple[float, float, float, float]:
    n = slot["n"]
    mean = slot["sum"] / n
    var = max(slot["sq"] / n - mean * mean, 0.0)
    return mean, var**0.5, slot["min"], slot["max"]


def _corr_accum() -> dict[str, float]:
    return {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxy": 0.0, "sx2": 0.0, "sy2": 0.0}


def _corr_update(slot: dict[str, float], x: np.ndarray, y: np.ndarray) -> None:
    slot["n"] += x.size
    slot["sx"] += float(x.sum())
    slot["sy"] += float(y.sum())
    slot["sxy"] += float((x * y).sum())
    slot["sx2"] += float((x * x).sum())
    slot["sy2"] += float((y * y).sum())


def _corr_final(slot: dict[str, float]) -> float:
    n = slot["n"]
    if n < 2:
        return float("nan")
    cov = slot["sxy"] / n - (slot["sx"] / n) * (slot["sy"] / n)
    var_x = slot["sx2"] / n - (slot["sx"] / n) ** 2
    var_y = slot["sy2"] / n - (slot["sy"] / n) ** 2
    denom = (var_x * var_y) ** 0.5
    return float(cov / denom) if denom > 1e-12 else float("nan")


@torch.no_grad()
def cellv03_diagnostics(
    model: DenseCellV03, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> dict[str, float]:
    """Per-layer `e`/`u`/`pi` mean/std/min/max, chunked over `x`'s rows so a
    full 10k-row test split stays memory-flat -- the flat-fusion comparison
    point for `belief_dendrite_diagnostics` below."""
    model.eval()
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for i in range(0, x.shape[0], chunk):
        (b1, _), (b2, _) = model.belief_states_verbose(x[i : i + chunk], c[i : i + chunk])
        for tag, belief in (("layer1", b1), ("layer2", b2)):
            pi = effective_precision(belief.evidence, belief.uncertainty)
            for q, t in (("e", belief.evidence), ("u", belief.uncertainty), ("pi", pi)):
                _rs_update(acc.setdefault((tag, q), _rs()), t)

    out: dict[str, float] = {}
    for (tag, q), slot in acc.items():
        mean, std, lo, hi = _rs_final(slot)
        out[f"diag_{tag}_{q}_mean"] = mean
        out[f"diag_{tag}_{q}_std"] = std
        out[f"diag_{tag}_{q}_min"] = lo
        out[f"diag_{tag}_{q}_max"] = hi
    return out


@torch.no_grad()
def belief_dendrite_diagnostics(
    model: BeliefDendriteNetwork,
    x: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor | None,
    chunk: int = EVAL_CHUNK,
) -> dict[str, float]:
    """Sec O: branch/soma precision distributions (both layers), the
    branch-routing effective count, somatic conflict, and -- when a
    corruption `mask` is given -- the localization correlation between a
    layer-1 branch's corruption overlap and its own precision. Chunked over
    `x`'s rows: an unchunked full-10k-row `[batch, H*B, K]` gather is both
    slow and memory-heavy (measured ~3GB/tensor at this experiment's sizing)
    -- this is the same chunk-and-accumulate pattern `forward_xc` already
    uses for the forward pass itself."""
    model.eval()
    acc: dict[tuple[str, str], dict[str, float]] = {}
    eff_acc: dict[str, dict[str, float]] = {}
    corr_acc = _corr_accum()

    for i in range(0, x.shape[0], chunk):
        xb, cb = x[i : i + chunk], c[i : i + chunk]
        belief_in = model.input_belief(xb, cb)
        (b1, d1), (b2, d2) = model.layer_states_verbose(belief_in)

        for tag, belief, diag in (("layer1", b1, d1), ("layer2", b2, d2)):
            for q, t in (
                ("pi_branch", diag.pi_branch),
                ("e_branch", diag.e_branch),
                ("u_branch", diag.u_branch),
                ("pi_soma", effective_precision(belief.evidence, belief.uncertainty)),
                ("u_soma", belief.uncertainty),
            ):
                _rs_update(acc.setdefault((tag, q), _rs()), t)
            _rs_update(eff_acc.setdefault(tag, _rs()), diag.effective_branches())

        if mask is not None:
            mb = mask[i : i + chunk]
            overlap = model.layer1.connectivity.corruption_overlap(mb)  # [chunk, M1]
            pi_flat = d1.pi_branch.reshape(overlap.shape[0], -1)  # same M1 ordering
            _corr_update(
                corr_acc,
                overlap.reshape(-1).detach().cpu().numpy(),
                pi_flat.reshape(-1).detach().cpu().numpy(),
            )

    out: dict[str, float] = {}
    for (tag, q), slot in acc.items():
        mean, std, lo, hi = _rs_final(slot)
        out[f"diag_{tag}_{q}_mean"] = mean
        out[f"diag_{tag}_{q}_std"] = std
        out[f"diag_{tag}_{q}_min"] = lo
        out[f"diag_{tag}_{q}_max"] = hi
    for tag, slot in eff_acc.items():
        mean, std, _, _ = _rs_final(slot)
        out[f"diag_{tag}_effective_branches_mean"] = mean
        out[f"diag_{tag}_effective_branches_std"] = std

    if mask is not None:
        out["localization_corr_overlap_vs_branch_pi"] = _corr_final(corr_acc)
        out["localization_n"] = corr_acc["n"]

    return out


@torch.no_grad()
def evaluate_severity(
    model: torch.nn.Module,
    family: str,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    severity: float,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> SeverityResult:
    model.eval()
    x_test_clean = prepared.x_test
    y_test = prepared.y_test.to(device)

    replica_metrics: list[dict[str, float]] = []
    diag_accum: list[dict[str, float]] = []
    for replica in range(n_replicas):
        cor = corrupt(
            corruption_family,
            x_test_clean,
            experiment_seed=seed,
            split="test",
            epoch=0,
            replica=replica,
            severity=severity,
        )
        xc, cc = cor.x.to(device), cor.c.to(device)
        pred = forward_xc(model, xc, cc, chunk=chunk)
        replica_metrics.append(_score(pred, y_test))
        if family == CELLV03:
            diag_accum.append(cellv03_diagnostics(model, xc, cc, chunk=chunk))
        elif family == BELIEF_DENDRITE:
            diag_accum.append(
                belief_dendrite_diagnostics(model, xc, cc, cor.mask.to(device), chunk=chunk)
            )

    avg = {m: float(np.mean([rm[m] for rm in replica_metrics])) for m in METRIC_NAMES}
    diagnostics = None
    if diag_accum:
        diagnostics = {k: float(np.nanmean([d[k] for d in diag_accum])) for k in diag_accum[0]}

    return SeverityResult(
        severity=float(severity),
        in_distribution=severity in in_dist_severities(corruption_family),
        metrics=avg,
        replica_metrics=replica_metrics,
        diagnostics=diagnostics,
    )


def corruption_auc(points: list[tuple[float, float]]) -> float:
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    order = np.argsort(xs)
    return float(np.trapezoid(ys[order], xs[order]))


def ood_drop(corruption_family: str, severities: list[SeverityResult], metric: str) -> float:
    lo, hi = MAX_TRAIN_SEVERITY[corruption_family], MAX_OOD_SEVERITY[corruption_family]
    at = {s.severity: s.metrics[metric] for s in severities}
    return float(at[lo] - at[hi])


def sweep(
    model: torch.nn.Module,
    family: str,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> SweepResult:
    results = [
        evaluate_severity(
            model,
            family,
            prepared,
            corruption_family,
            seed,
            sev,
            device=device,
            n_replicas=n_replicas,
            chunk=chunk,
        )
        for sev in eval_severities(corruption_family)
    ]
    auc = {m: corruption_auc([(r.severity, r.metrics[m]) for r in results]) for m in METRIC_NAMES}
    drop = {m: ood_drop(corruption_family, results, m) for m in METRIC_NAMES}
    return SweepResult(
        dataset=prepared.name,
        corruption_family=corruption_family,
        family=family,
        severities=results,
        corruption_auc=auc,
        ood_drop=drop,
    )
