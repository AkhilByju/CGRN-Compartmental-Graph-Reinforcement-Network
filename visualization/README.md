# Belief Cell Inference Observatory

Standalone React + TypeScript + Vite showcase for CellV0.3's precomputed
belief-flow diagnostics. The default bundle is local and deterministic: it does
not require a backend, model weights, or a network request.

The diagram has two explanatory network modes. `pathways` follows sparse,
signed evidence connections from observations through belief cells to the
readout. `neighborhoods` shows same-layer cells whose normalized upstream
profiles are similar; it is a structural explanatory view, not a claim of
learned recurrent edges. The 3D view uses Three.js with orbit, right-drag pan,
zoom, reset-camera, and raycast cell picking. PNG capture is available from
the inspector and uses the currently active renderer.

```bash
cd visualization
npm install
npm run dev
```

Production checks:

```bash
npm run build
pytest -q visualization/tests
```

The initial scenes are explicit `demo_trace` fixtures. They are not trained
Paper-A results. To export a trained trace, provide a compatible CellV0.3
checkpoint; the exporter refuses to create a `trained_trace` without one:

```bash
python visualization/trace_exporter/export_trace.py \
  --dataset aps --trace-kind trained_trace \
  --checkpoint /path/to/cellv03.pt \
  --input-json /path/to/sample.json \
  --out visualization/src/data/aps-trained.json
```

The exporter only performs inference. It uses the existing verbose belief
diagnostics (`mu`, evidence, uncertainty, effective precision, signed
consensus, and normalized connection weights), stores top-k edges per displayed
cell, and writes schema version 2's optional neighborhood links, node-group
membership, signed influence, and connection kind fields. Existing schema 1
traces remain valid in pathway mode because `top_connections` is preserved.
Experiment definitions, training code, and result records remain untouched.
