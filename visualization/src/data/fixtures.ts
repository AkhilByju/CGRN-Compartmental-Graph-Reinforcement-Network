import type { BeliefTrace, DatasetId, LayerState, TraceFrame } from '../types';

const round = (value: number) => Number(value.toFixed(4));

function mnistPixels(): { raw: number[]; observed: number[]; reliability: number[] } {
  const raw: number[] = [];
  const observed: number[] = [];
  const reliability: number[] = [];
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      const ring = Math.abs(Math.hypot(x - 13.5, y - 13.5) - 8.4) < 1.8;
      const stem = x > 11 && x < 16 && y > 11 && y < 23;
      const pixel = ring || stem ? 0.78 + 0.18 * Math.sin(x * 0.7 + y) ** 2 : 0.02;
      const masked = x > 16 && x < 23 && y > 4 && y < 12;
      raw.push(round(pixel));
      observed.push(round(masked ? pixel * 0.08 : pixel));
      reliability.push(round(masked ? 0.08 : 0.98 - ((x * 3 + y) % 7) * 0.018));
    }
  }
  return { raw, observed, reliability };
}

function sensorInput(dataset: DatasetId): { raw: number[]; observed: number[]; reliability: number[]; labels: string[] } {
  const count = dataset === 'aps' ? 24 : 32;
  const raw: number[] = [];
  const observed: number[] = [];
  const reliability: number[] = [];
  const labels: string[] = [];
  for (let i = 0; i < count; i += 1) {
    const value = 0.5 + 0.32 * Math.sin(i * 0.72) + 0.11 * Math.cos(i * 0.21);
    const missing = dataset === 'aps' ? [3, 4, 12, 13, 19].includes(i) : [6, 7, 8, 21, 26].includes(i);
    raw.push(round(value));
    observed.push(round(missing ? value * 0.12 : value));
    reliability.push(round(missing ? 0.1 : 0.91 - (i % 5) * 0.035));
    labels.push(dataset === 'aps' ? `S${String(i + 1).padStart(2, '0')}` : ['CO', 'NO2', 'O3', 'TEMP', 'RH'][i % 5]);
  }
  return { raw, observed, reliability, labels };
}

function makeLayer(name: string, count: number, inputCount: number, phase: number, severity: number): LayerState {
  const mu: number[] = [];
  const evidence: number[] = [];
  const uncertainty: number[] = [];
  const precision: number[] = [];
  const consensus: number[] = [];
  const top_connections: LayerState['top_connections'] = [];
  const node_groups = Array.from({ length: count }, (_, index) => Math.floor(index / Math.max(1, Math.ceil(count / 8))));
  const groupCount = Math.max(...node_groups) + 1;
  const groups = Array.from({ length: groupCount }, (_, id) => ({
    id,
    label: `evidence cluster ${String(id + 1).padStart(2, '0')}`,
    cells: node_groups.flatMap((group, index) => group === id ? [index] : []),
  }));
  for (let i = 0; i < count; i += 1) {
    const support = Math.max(0.2, 0.98 - severity * 0.42 + 0.08 * Math.sin(i * 0.41 + phase));
    const conflict = Math.max(0.02, 0.1 + severity * 0.18 + 0.08 * Math.abs(Math.cos(i * 0.37 + phase)));
    const c = 0.54 * Math.sin(i * 0.29 + phase) + 0.12 * Math.cos(i * 0.8);
    const m = Math.tanh(c * (0.9 + support * 0.6));
    mu.push(round(m));
    evidence.push(round(support));
    uncertainty.push(round(conflict));
    precision.push(round(support / (1 + support * conflict)));
    consensus.push(round(c));
    const connections: LayerState['top_connections'][number] = [];
    for (let k = 0; k < Math.min(4, inputCount); k += 1) {
      const source = (i * 7 + k * 11 + Math.floor(phase * 3)) % inputCount;
      const signed = Math.sin((source + 1) * (i + 2) * 0.23 + phase) * 0.5 + 0.5;
      const weight = round((0.96 - k * 0.16) * (signed > 0.5 ? signed : -signed));
      connections.push({ source, weight, signed_influence: weight, kind: weight >= 0 ? 'support' : 'conflict' });
    }
    top_connections.push(connections);
  }
  const neighborhood_links: LayerState['neighborhood_links'] = [];
  for (let source = 0; source < count; source += 1) {
    const peers = [1, 2].map((offset) => (source + offset) % count);
    peers.forEach((target, offset) => {
      const sameGroup = node_groups[source] === node_groups[target];
      const signedInfluence = round((sameGroup ? 0.72 : -0.28) + Math.sin((source + target) * 0.17 + phase) * 0.08);
      neighborhood_links.push({
        source,
        target,
        similarity: round((sameGroup ? 0.72 : 0.36) - offset * 0.08),
        signed_influence: signedInfluence,
        kind: signedInfluence >= 0 ? 'shared_evidence' : 'conflict',
      });
    });
  }
  return { name, mu, evidence, uncertainty, precision, consensus, top_connections, neighborhood_links, node_groups, groups };
}

function frames(layerCount: number): TraceFrame[] {
  const stages: TraceFrame[] = [
    { progress: 0, active_stage: 'observation', visible_layers: 0, energy: 0.14 },
  ];
  for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
    stages.push({
      progress: (layerIndex + 1) / (layerCount + 1),
      active_stage: `belief_layer_${layerIndex + 1}`,
      visible_layers: layerIndex + 1,
      energy: 0.32 + ((layerIndex + 1) / layerCount) * 0.52,
    });
  }
  stages.push({ progress: 1, active_stage: 'readout', visible_layers: layerCount, energy: 1 });
  return stages;
}

export function makeFixture(dataset: DatasetId): BeliefTrace {
  const isMnist = dataset === 'mnist';
  const input = isMnist ? mnistPixels() : sensorInput(dataset);
  const severity = dataset === 'mnist' ? 0.2 : dataset === 'aps' ? 0.31 : 0.24;
  // This is intentionally a synthetic visualization field, not a claim about
  // the trained model's width. Multiple widths make the staged network read
  // as a larger architecture while keeping edge density bounded elsewhere.
  const layerSizes = isMnist ? [80, 96, 112, 96] : [64, 80, 96, 80];
  const output = dataset === 'mnist' ? 10 : 1;
  const layers: LayerState[] = [];
  let inputCount = input.raw.length;
  layerSizes.forEach((size, layerIndex) => {
    layers.push(makeLayer(`belief_layer_${layerIndex + 1}`, size, inputCount, 0.72 + layerIndex * 0.56, severity));
    inputCount = size;
  });
  const logits = Array.from({ length: output }, (_, i) => round(0.2 + Math.sin(i * 1.2 + 2) * 0.18));
  if (isMnist) logits[3] = 2.84;
  const prediction = isMnist ? 3 : dataset === 'aps' ? 0.17 : 46.8;
  return {
    schema_version: 2,
    dataset,
    dataset_label: dataset === 'mnist' ? 'MNIST / handwritten digits' : dataset === 'aps' ? 'APS Failure at Scania Trucks' : 'UCI Air Quality',
    task: isMnist ? 'classification' : 'regression',
    model_family: 'cellv0.3',
    sample_id: isMnist ? 'test / 000417' : dataset === 'aps' ? 'holdout / 8129' : 'chronological / 2004-03-11 08:00',
    seed: 0,
    corruption: { family: isMnist ? 'missing_pixel' : 'missing_sensor', severity, missing_count: input.reliability.filter((v) => v < 0.2).length },
    input: { ...input, shape: isMnist ? [28, 28] : undefined },
    layers,
    readout: {
      logits,
      prediction,
      uncertainty: isMnist ? 0.081 : dataset === 'aps' ? 0.12 : 3.7,
      evidence: 0.83,
      usable_precision: 0.76,
      comparison: 'demonstration readout · research metric unavailable',
    },
    frames: frames(layerSizes.length),
    metrics: {
      'trace status': 'demo fixture',
      'research metrics': 'available only from an exported artifact',
    },
    provenance: {
      checkpoint: null,
      git_commit: 'fixture-local',
      export_timestamp: '2026-09-27T00:00:00.000Z',
      trace_kind: 'demo_trace',
      source_note: 'Deterministic synthetic visualization field. Not a trained Paper-A result.',
    },
  };
}

export const fixtures: Record<DatasetId, BeliefTrace> = {
  mnist: makeFixture('mnist'),
  aps: makeFixture('aps'),
  air_quality: makeFixture('air_quality'),
};
