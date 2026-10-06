#!/usr/bin/env python3
"""Export versioned CellV0.3 inference traces for the visualizer.

This module is deliberately an inference adapter. It never imports a training
loop and never writes to experiment result records. A ``trained_trace`` must
name an existing compatible checkpoint; ``demo_trace`` explicitly creates a
seeded demonstration model and is marked as such in provenance.

Examples:
    python visualization/trace_exporter/export_trace.py --dataset mnist \
        --trace-kind demo_trace --out visualization/src/data/mnist-demo.json
    python visualization/trace_exporter/export_trace.py --dataset aps \
        --trace-kind trained_trace --checkpoint /path/to/model.pt \
        --input-json sample.json --out visualization/src/data/aps-trained.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TraceExportError(RuntimeError):
    """Raised when the export request cannot produce an honest trace."""


def top_k_connections(weights: torch.Tensor, k: int = 4) -> list[list[dict[str, float | int | str]]]:
    """Return top-k normalized absolute influence with signed weights.

    ``weights`` is ``[out_cells, in_cells]``. The visualizer only receives the
    selected paths, while the complete layer remains available through the
    model diagnostics during export. No batch-by-edge tensor is materialized.
    """
    if weights.ndim != 2:
        raise ValueError(f"expected a 2D connection matrix, got {tuple(weights.shape)}")
    normalized = weights / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    result: list[list[dict[str, float | int | str]]] = []
    limit = min(k, normalized.shape[1])
    for row in normalized:
        indices = row.abs().topk(limit).indices.tolist()
        result.append([
            {
                "source": int(index),
                "weight": round(float(row[index].item()), 7),
                "signed_influence": round(float(row[index].item()), 7),
                "kind": "support" if row[index].item() >= 0 else "conflict",
            }
            for index in indices
        ])
    return result


def neighborhood_links(
    weights: torch.Tensor, *, max_links_per_node: int = 2, profile_width: int = 8,
    minimum_similarity: float = 0.16,
) -> list[dict[str, float | int | str]]:
    """Derive a sparse same-layer graph from normalized upstream profiles."""
    if weights.ndim != 2:
        raise ValueError(f"expected a 2D connection matrix, got {tuple(weights.shape)}")
    normalized = weights / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    width = min(profile_width, normalized.shape[1])
    indices = normalized.abs().topk(width, dim=1).indices
    signed_values = normalized.gather(1, indices)
    inverted: dict[int, list[int]] = {}
    for row_index, row_sources in enumerate(indices.tolist()):
        for source in row_sources:
            inverted.setdefault(int(source), []).append(row_index)

    links: list[dict[str, float | int | str]] = []
    for row_index in range(normalized.shape[0]):
        candidates: set[int] = set()
        for source in indices[row_index].tolist():
            candidates.update(inverted.get(int(source), []))
        candidates.discard(row_index)
        row_profile = dict(zip(indices[row_index].tolist(), signed_values[row_index].tolist()))
        row_norm = max(sum(value * value for value in row_profile.values()) ** 0.5, 1e-8)
        scored: list[tuple[float, float, int]] = []
        for candidate in candidates:
            candidate_profile = dict(zip(indices[candidate].tolist(), signed_values[candidate].tolist()))
            dot = sum(row_profile.get(source, 0.0) * value for source, value in candidate_profile.items())
            candidate_norm = max(sum(value * value for value in candidate_profile.values()) ** 0.5, 1e-8)
            signed_similarity = dot / (row_norm * candidate_norm)
            similarity = abs(signed_similarity)
            if similarity >= minimum_similarity:
                scored.append((similarity, signed_similarity, candidate))
        for similarity, signed, candidate in sorted(scored, reverse=True)[:max_links_per_node]:
            links.append({
                "source": row_index,
                "target": candidate,
                "similarity": round(float(similarity), 7),
                "signed_influence": round(float(signed), 7),
                "kind": "shared_evidence" if signed >= 0 else "conflict",
            })
    return links


def node_groups(
    weights: torch.Tensor, *, group_count: int = 8, profile_width: int = 8,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Assign cells to stable explanatory groups using their strongest source."""
    normalized = weights / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    width = min(profile_width, normalized.shape[1])
    indices = normalized.abs().topk(width, dim=1).indices
    source_span = max(1, normalized.shape[1] // max(1, group_count))
    membership = [min(group_count - 1, int(indices[row, 0].item()) // source_span) for row in range(normalized.shape[0])]
    groups = [
        {
            "id": group,
            "label": f"evidence cluster {group + 1:02d}",
            "cells": [index for index, member in enumerate(membership) if member == group],
            "upstream_sources": sorted({int(source) for row, member in enumerate(membership) if member == group for source in indices[row].tolist()[:2]}),
        }
        for group in range(group_count)
    ]
    return membership, groups


def _array(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 7) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dimensions(dataset: str) -> tuple[int, int, str]:
    if dataset == "mnist":
        return 784, 10, "classification"
    if dataset == "aps":
        return 24, 1, "classification"
    if dataset == "air_quality":
        return 32, 1, "regression"
    raise TraceExportError(f"unknown dataset {dataset!r}; expected mnist, aps, or air_quality")


def _default_input(dataset: str, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    raw = torch.rand(1, size, generator=generator) * 0.6
    reliability = torch.ones_like(raw)
    if dataset == "mnist":
        reliability[:, 170:230] = 0.08
        observed = raw * reliability
    else:
        missing = torch.arange(size) % 7 == 0
        reliability[:, missing] = 0.1
        observed = raw * reliability
    return raw, observed, reliability


def _input_from_json(path: Path, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text())
    try:
        raw = torch.tensor(payload["raw"], dtype=torch.float32).reshape(1, size)
        observed = torch.tensor(payload.get("observed", payload["raw"]), dtype=torch.float32).reshape(1, size)
        reliability = torch.tensor(payload["reliability"], dtype=torch.float32).reshape(1, size)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise TraceExportError(f"input JSON must contain raw and reliability arrays of length {size}") from exc
    return raw, observed, reliability


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise TraceExportError(f"trained_trace requires an existing checkpoint; not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TraceExportError(f"checkpoint {checkpoint} does not contain a model state dict")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise TraceExportError(f"checkpoint is not compatible with CellV0.3: {exc}") from exc


def _layer_payload(
    name: str, belief: Any, consensus: torch.Tensor,
    connections: list[list[dict[str, float | int | str]]], weights: torch.Tensor,
) -> dict[str, Any]:
    mu = belief.mu[0]
    evidence = belief.evidence[0]
    uncertainty = belief.uncertainty[0]
    precision = evidence / (1 + evidence * uncertainty)
    membership, groups = node_groups(weights)
    return {
        "name": name,
        "mu": _array(mu),
        "evidence": _array(evidence),
        "uncertainty": _array(uncertainty),
        "precision": _array(precision),
        "consensus": _array(consensus[0]),
        "top_connections": connections,
        "neighborhood_links": neighborhood_links(weights),
        "node_groups": membership,
        "groups": groups,
    }


def _frames() -> list[dict[str, float | int | str]]:
    return [
        {"progress": 0.0, "active_stage": "observation", "visible_layers": 0, "energy": 0.14},
        {"progress": 0.34, "active_stage": "belief_layer_1", "visible_layers": 1, "energy": 0.54},
        {"progress": 0.68, "active_stage": "belief_layer_2", "visible_layers": 2, "energy": 0.78},
        {"progress": 1.0, "active_stage": "readout", "visible_layers": 2, "energy": 1.0},
    ]


def export_trace(
    *, dataset: str, trace_kind: str, out: Path, checkpoint: Path | None = None,
    input_json: Path | None = None, sample_id: str = "inference / 0000", seed: int = 0,
    severity: float = 0.2, top_k: int = 4,
) -> dict[str, Any]:
    """Run one forward pass and write a schema-versioned trace."""
    if trace_kind not in {"trained_trace", "demo_trace"}:
        raise TraceExportError("trace_kind must be trained_trace or demo_trace")
    if trace_kind == "trained_trace" and checkpoint is None:
        raise TraceExportError("trained_trace requested without --checkpoint; supply compatible model weights")
    if trace_kind == "trained_trace" and checkpoint is not None and not checkpoint.exists():
        raise TraceExportError(f"trained_trace requires an existing checkpoint; not found: {checkpoint}")

    in_features, out_features, task = _dimensions(dataset)
    from experiments.paper_a.reliability.models import ReliabilityCellV03
    from experiments.paper_a.real_reliability.models import cellv03_target

    torch.manual_seed(seed)
    hidden_cells, _ = cellv03_target(in_features, 100_000 if dataset == "aps" else 50_000)
    model = ReliabilityCellV03(in_features, hidden_cells, out_features).eval()
    if checkpoint is not None:
        _load_checkpoint(model, checkpoint)
    raw, observed, reliability = _input_from_json(input_json, in_features) if input_json else _default_input(dataset, in_features)

    with torch.no_grad():
        (belief1, consensus1), (belief2, consensus2) = model.belief_states_verbose(observed, reliability)
        logits = model(observed, reliability)
    precision2 = belief2.evidence / (1 + belief2.evidence * belief2.uncertainty)
    if task == "classification":
        prediction: int | float = int(logits.argmax(dim=-1).item())
    else:
        prediction = round(float(logits.item()), 7)

    layers = [
        _layer_payload(
            "belief_layer_1", belief1, consensus1,
            top_k_connections(model.net.layer1.V, top_k), model.net.layer1.V,
        ),
        _layer_payload(
            "belief_layer_2", belief2, consensus2,
            top_k_connections(model.net.layer2.V, top_k), model.net.layer2.V,
        ),
    ]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "dataset": dataset,
        "dataset_label": {"mnist": "MNIST / handwritten digits", "aps": "APS Failure at Scania Trucks", "air_quality": "UCI Air Quality"}[dataset],
        "task": task,
        "model_family": "cellv0.3",
        "sample_id": sample_id,
        "seed": seed,
        "corruption": {"family": "missing_pixel" if dataset == "mnist" else "missing_sensor", "severity": severity, "missing_count": int((reliability < 0.2).sum().item())},
        "input": {"raw": _array(raw[0]), "observed": _array(observed[0]), "reliability": _array(reliability[0])},
        "layers": layers,
        "readout": {
            "logits": _array(logits[0]),
            "prediction": prediction,
            "uncertainty": round(float(belief2.uncertainty.mean().item()), 7),
            "evidence": round(float(belief2.evidence.mean().item()), 7),
            "usable_precision": round(float(precision2.mean().item()), 7),
        },
        "frames": _frames(),
        "metrics": {},
        "provenance": {
            "checkpoint": str(checkpoint) if checkpoint else None,
            "git_commit": _git_commit(),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_kind": trace_kind,
            "source_note": "Inference from supplied compatible checkpoint." if trace_kind == "trained_trace" else "Seeded demonstration model. Not a trained Paper-A result.",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "aps", "air_quality"), required=True)
    parser.add_argument("--trace-kind", choices=("trained_trace", "demo_trace"), default="demo_trace")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--sample-id", default="inference / 0000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--severity", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = export_trace(**vars(args))
    except TraceExportError as exc:
        parser.error(str(exc))
    print(f"Wrote {args.out} ({payload['provenance']['trace_kind']}; schema {payload['schema_version']})")


if __name__ == "__main__":
    main()
