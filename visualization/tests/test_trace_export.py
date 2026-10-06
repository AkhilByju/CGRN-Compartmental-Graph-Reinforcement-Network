import json
from pathlib import Path

import pytest
import torch

from visualization.trace_exporter.export_trace import TraceExportError, export_trace, top_k_connections


def test_top_k_connections_are_signed_and_normalized():
    weights = torch.tensor([[1.0, -4.0, 2.0], [-3.0, 0.2, 1.0]])
    result = top_k_connections(weights, k=2)
    assert [item["source"] for item in result[0]] == [1, 2]
    assert result[0][0]["weight"] < 0
    assert len(result[1]) == 2


def test_trained_trace_requires_checkpoint(tmp_path: Path):
    with pytest.raises(TraceExportError, match="without --checkpoint"):
        export_trace(dataset="aps", trace_kind="trained_trace", out=tmp_path / "trace.json")


def test_demo_trace_schema_shapes_and_provenance(tmp_path: Path):
    out = tmp_path / "trace.json"
    payload = export_trace(dataset="aps", trace_kind="demo_trace", out=out, seed=4, top_k=3)
    assert out.exists()
    assert json.loads(out.read_text())["schema_version"] == 2
    assert payload["provenance"]["trace_kind"] == "demo_trace"
    assert payload["provenance"]["checkpoint"] is None
    assert len(payload["input"]["raw"]) == 24
    assert len(payload["layers"]) == 2
    assert len(payload["layers"][0]["mu"]) > 0
    assert len(payload["layers"][0]["top_connections"][0]) == 3
    assert payload["layers"][0]["top_connections"][0][0]["kind"] in {"support", "conflict"}
    assert payload["layers"][0]["neighborhood_links"]
    assert len(payload["layers"][0]["node_groups"]) == len(payload["layers"][0]["mu"])
    assert len(payload["frames"]) == 4
