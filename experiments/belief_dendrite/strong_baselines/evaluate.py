"""Per-severity evaluation, corruption-robustness summaries, and the Sec 10
(BeliefDendrite mechanism) / Sec 11 (Transformer) diagnostics for the
strong-baselines benchmark.

Every best checkpoint is evaluated **once** across the full fixed severity
grid (never a separate model per severity, spec Sec 3); at each severity,
`N_TEST_REPLICAS` deterministic corruption replicas (identical across every
model) are averaged before any other statistic, mirroring
`experiments/belief_dendrite/evaluate.py`'s convention for the frozen
MNIST/Fashion-MNIST benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.belief_dendrite.strong_baselines.corruption import (
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
    N_TEST_REPLICAS,
    corrupt_missing_patch,
    eval_severities,
    in_dist_severities,
)
from experiments.belief_dendrite.strong_baselines.models import (
    BELIEF_DENDRITE,
    CONFIDENCE_TINY_VIT,
    RELIABILITY_GATED_TINY_VIT,
    BeliefDendriteModel,
    ConfidenceTinyViT,
    ReliabilityGatedTinyViT,
)
from experiments.belief_dendrite.strong_baselines.training import EVAL_CHUNK, forward_xc
from src.evaluation.classification import accuracy, f1
from src.models.architecture_v0.precision_gain import effective_precision

PRIMARY_METRIC = "accuracy"
METRIC_NAMES: tuple[str, ...] = ("accuracy", "macro_f1")


@dataclass
class SeverityResult:
    severity: float
    in_distribution: bool
    metrics: dict[str, float]
    replica_metrics: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, float] | None = None

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
    family: str
    severities: list[SeverityResult]
    corruption_auc: dict[str, float]
    ood_drop: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "severities": [s.to_dict() for s in self.severities],
            "corruption_auc": self.corruption_auc,
            "ood_drop": self.ood_drop,
        }


def _score(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred, y = pred.detach().cpu(), y.detach().cpu()
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
def _belief_dendrite_forward_and_diagnostics(
    model: BeliefDendriteModel,
    x: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor | None,
    chunk: int = EVAL_CHUNK,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sec 10 diagnostics *and* classification logits from one
    `layer_states_verbose` pass per chunk -- computing them separately (one
    `model(x,c)` call for logits, a second full `layer_states_verbose` call
    for diagnostics) would forward the same input through the whole network
    twice for nothing; `b2.mu` already has everything `model.forward` needs
    (`net.readout(b2.mu)`).

    Per-layer branch/soma precision distributions, the branch-routing
    effective count, somatic conflict, and -- when a corruption `mask` is
    given -- both localization correlations: branch overlap vs. the
    branch's own precision, and (new vs. the frozen MNIST benchmark) branch
    overlap vs. the branch's somatic contribution `r_b`, closing the causal
    chain from localized reliability to somatic routing. `mask`: `(N, H, W)`
    spatial-only (shared across channels, as `corruption.py` produces it)."""
    net = model.net
    net.eval()
    n = x.shape[0]
    acc: dict[tuple[str, str], dict[str, float]] = {}
    eff_acc: dict[str, dict[str, float]] = {}
    corr_pi = _corr_accum()
    corr_rb = _corr_accum()
    logits_chunks: list[torch.Tensor] = []

    for i in range(0, n, chunk):
        xb = x[i : i + chunk].reshape(x[i : i + chunk].shape[0], -1)
        cb = c[i : i + chunk].reshape(xb.shape[0], -1)
        belief_in = net.input_belief(xb, cb)
        (b1, d1), (b2, d2) = net.layer_states_verbose(belief_in)
        logits_chunks.append(net.readout(b2.mu))

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
            mb = mask[i : i + chunk]  # (chunk, H, W)
            mb_c = mb.unsqueeze(1).expand(-1, 3, -1, -1).reshape(mb.shape[0], -1).to(torch.float32)
            overlap = net.layer1.connectivity.corruption_overlap(mb_c)  # (chunk, M1)
            pi_flat = d1.pi_branch.reshape(overlap.shape[0], -1)
            rb_flat = d1.branch_contribution().reshape(overlap.shape[0], -1)
            overlap_np = overlap.reshape(-1).detach().cpu().numpy()
            _corr_update(corr_pi, overlap_np, pi_flat.reshape(-1).detach().cpu().numpy())
            _corr_update(corr_rb, overlap_np, rb_flat.reshape(-1).detach().cpu().numpy())

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
        out["localization_corr_overlap_vs_branch_pi"] = _corr_final(corr_pi)
        out["localization_corr_overlap_vs_somatic_contribution"] = _corr_final(corr_rb)
        out["localization_n"] = corr_pi["n"]
    return torch.cat(logits_chunks, dim=0), out


def belief_dendrite_diagnostics(
    model: BeliefDendriteModel,
    x: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor | None,
    chunk: int = EVAL_CHUNK,
) -> dict[str, float]:
    """Diagnostics-only entry point (kept for direct testing/inspection);
    `evaluate_severity` calls `_belief_dendrite_forward_and_diagnostics`
    directly instead, so a real evaluation run never forwards the model
    twice per replica."""
    return _belief_dendrite_forward_and_diagnostics(model, x, c, mask, chunk)[1]


@torch.no_grad()
def _confidence_vit_forward_and_diagnostics(
    model: ConfidenceTinyViT, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sec 11 diagnostic (`||reliability_embedding|| / ||image_patch_
    embedding||`) *and* classification logits from one `forward_verbose`
    pass per chunk -- `forward_verbose`'s first return value already is
    `model(x,c)`'s logits."""
    model.eval()
    ratios: list[torch.Tensor] = []
    logits_chunks: list[torch.Tensor] = []
    n = x.shape[0]
    for i in range(0, n, chunk):
        logits, img_tok, rel_emb = model.forward_verbose(x[i : i + chunk], c[i : i + chunk])
        logits_chunks.append(logits)
        ratio = rel_emb.norm(dim=-1) / img_tok.norm(dim=-1).clamp_min(1e-8)
        ratios.append(ratio.reshape(-1))
    all_ratios = torch.cat(ratios)
    diag = {
        "reliability_to_image_embedding_norm_ratio_mean": float(all_ratios.mean()),
        "reliability_to_image_embedding_norm_ratio_std": float(all_ratios.std()),
    }
    return torch.cat(logits_chunks, dim=0), diag


def confidence_vit_diagnostics(
    model: ConfidenceTinyViT, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> dict[str, float]:
    """Diagnostics-only entry point; see `belief_dendrite_diagnostics`."""
    return _confidence_vit_forward_and_diagnostics(model, x, c, chunk)[1]


@torch.no_grad()
def _gated_vit_forward_and_diagnostics(
    model: ReliabilityGatedTinyViT, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sec 11 diagnostic (mean patch reliability) *and* classification
    logits from one `forward_verbose` pass per chunk."""
    model.eval()
    means: list[torch.Tensor] = []
    logits_chunks: list[torch.Tensor] = []
    n = x.shape[0]
    for i in range(0, n, chunk):
        logits, rel = model.forward_verbose(x[i : i + chunk], c[i : i + chunk])
        logits_chunks.append(logits)
        means.append(rel.reshape(rel.shape[0], -1).mean(dim=-1))
    all_means = torch.cat(means)
    diag = {
        "mean_patch_reliability_mean": float(all_means.mean()),
        "mean_patch_reliability_std": float(all_means.std()),
    }
    return torch.cat(logits_chunks, dim=0), diag


def gated_vit_diagnostics(
    model: ReliabilityGatedTinyViT, x: torch.Tensor, c: torch.Tensor, chunk: int = EVAL_CHUNK
) -> dict[str, float]:
    """Diagnostics-only entry point; see `belief_dendrite_diagnostics`."""
    return _gated_vit_forward_and_diagnostics(model, x, c, chunk)[1]


@torch.no_grad()
def _forward_and_diagnostics_for(
    family: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    """One forward pass, dispatched by family: `(logits, diagnostics)`,
    `diagnostics=None` for families with no Sec 10/11 diagnostic."""
    if family == BELIEF_DENDRITE:
        return _belief_dendrite_forward_and_diagnostics(model, x, c, mask, chunk=chunk)
    if family == CONFIDENCE_TINY_VIT:
        return _confidence_vit_forward_and_diagnostics(model, x, c, chunk=chunk)
    if family == RELIABILITY_GATED_TINY_VIT:
        return _gated_vit_forward_and_diagnostics(model, x, c, chunk=chunk)
    return forward_xc(model, x, c, chunk=chunk), None


@torch.no_grad()
def evaluate_severity(
    model: torch.nn.Module,
    family: str,
    x_test_clean: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
    severity: float,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> SeverityResult:
    model.eval()
    replica_metrics: list[dict[str, float]] = []
    diag_accum: list[dict[str, float]] = []
    for replica in range(n_replicas):
        cor = corrupt_missing_patch(
            x_test_clean,
            experiment_seed=seed,
            split="test",
            epoch=0,
            replica=replica,
            severity=severity,
        )
        pred, diag = _forward_and_diagnostics_for(family, model, cor.x, cor.c, cor.mask, chunk)
        replica_metrics.append(_score(pred, y_test))
        if diag is not None:
            diag_accum.append(diag)

    avg = {m: float(np.mean([rm[m] for rm in replica_metrics])) for m in METRIC_NAMES}
    diagnostics = None
    if diag_accum:
        diagnostics = {k: float(np.nanmean([d[k] for d in diag_accum])) for k in diag_accum[0]}

    return SeverityResult(
        severity=float(severity),
        in_distribution=severity in in_dist_severities(),
        metrics=avg,
        replica_metrics=replica_metrics,
        diagnostics=diagnostics,
    )


def corruption_auc(points: list[tuple[float, float]]) -> float:
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    order = np.argsort(xs)
    return float(np.trapezoid(ys[order], xs[order]))


def ood_drop(severities: list[SeverityResult], metric: str) -> float:
    at = {s.severity: s.metrics[metric] for s in severities}
    return float(at[MAX_TRAIN_SEVERITY] - at[MAX_OOD_SEVERITY])


def sweep(
    model: torch.nn.Module,
    family: str,
    x_test_clean: torch.Tensor,
    y_test: torch.Tensor,
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
            x_test_clean,
            y_test,
            seed,
            sev,
            device=device,
            n_replicas=n_replicas,
            chunk=chunk,
        )
        for sev in eval_severities()
    ]
    auc = {m: corruption_auc([(r.severity, r.metrics[m]) for r in results]) for m in METRIC_NAMES}
    drop = {m: ood_drop(results, m) for m in METRIC_NAMES}
    return SweepResult(family=family, severities=results, corruption_auc=auc, ood_drop=drop)
