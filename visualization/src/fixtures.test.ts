import { describe, expect, it } from 'vitest';
import { fixtures } from './data/fixtures';
import { buildSceneEdges } from './sceneModel';

describe('visualization scene fixtures', () => {
  it('loads all gallery scenes through the same versioned schema', () => {
    expect(Object.keys(fixtures)).toEqual(['mnist', 'aps', 'air_quality']);
    for (const trace of Object.values(fixtures)) {
      expect(trace.schema_version).toBe(2);
      expect(trace.layers[0].neighborhood_links?.length).toBeGreaterThan(0);
      expect(trace.layers[0].node_groups?.length).toBe(trace.layers[0].mu.length);
      expect(trace.layers).toHaveLength(4);
      expect(trace.frames.map((frame) => frame.active_stage)).toEqual([
        'observation', 'belief_layer_1', 'belief_layer_2', 'belief_layer_3', 'belief_layer_4', 'readout',
      ]);
      expect(trace.provenance.trace_kind).toBe('demo_trace');
      expect(trace.provenance.checkpoint).toBeNull();
      expect(trace.input.observed).toHaveLength(trace.input.reliability.length);
    }
  });

  it('keeps the MNIST hero at image resolution while sensor scenes stay grouped', () => {
    expect(fixtures.mnist.input.raw).toHaveLength(28 * 28);
    expect(fixtures.aps.input.raw.length).toBeLessThan(32);
    expect(fixtures.air_quality.input.raw.length).toBeGreaterThan(fixtures.aps.input.raw.length);
  });

  it('keeps legacy traces usable in pathway mode and makes density meaningful', () => {
    const legacy = { ...fixtures.mnist, schema_version: 1 as const, layers: fixtures.mnist.layers.map((layer) => ({
      ...layer, neighborhood_links: undefined, node_groups: undefined, groups: undefined,
    })) };
    const sparse = buildSceneEdges(legacy, 'pathways', 0.15, legacy.frames.length - 1, null, { layerIndex: 1, cellIndex: 0 });
    const dense = buildSceneEdges(legacy, 'pathways', 1, legacy.frames.length - 1, null, { layerIndex: 1, cellIndex: 0 });
    expect(sparse.length).toBeGreaterThan(0);
    expect(dense.length).toBeGreaterThan(sparse.length);
    const legacyNeighborhoods = buildSceneEdges(legacy, 'neighborhoods', 1, legacy.frames.length - 1, null, { layerIndex: 1, cellIndex: 0 });
    expect(legacyNeighborhoods.length).toBeGreaterThan(0);
    expect(legacyNeighborhoods.every((edge) => edge.from.kind === 'input' || edge.from.kind === 'node')).toBe(true);
  });

  it('uses exported neighborhood links as same-layer edges', () => {
    const edges = buildSceneEdges(fixtures.aps, 'neighborhoods', 1, fixtures.aps.frames.length - 1, null, { layerIndex: 1, cellIndex: 4 });
    expect(edges.length).toBeGreaterThan(0);
    const neighborhood = edges.filter((edge) => edge.to.kind !== 'output');
    expect(neighborhood.every((edge) => edge.from.kind === 'node' && edge.to.kind === 'node' && edge.from.layerIndex === edge.to.layerIndex)).toBe(true);
    expect(edges.some((edge) => edge.from.kind === 'node' && edge.to.kind === 'output')).toBe(true);
  });

  it('keeps pathway endpoints inside the exported trace and selection deterministic', () => {
    const trace = fixtures.mnist;
    const selected = { layerIndex: 1, cellIndex: 19 };
    const finalFrame = trace.frames.length - 1;
    const sparse = buildSceneEdges(trace, 'pathways', .15, finalFrame, null, selected);
    const dense = buildSceneEdges(trace, 'pathways', 1, finalFrame, null, selected);
    expect(buildSceneEdges(trace, 'pathways', .42, finalFrame, null, selected)).toEqual(buildSceneEdges(trace, 'pathways', .42, finalFrame, null, selected));
    expect(dense.length).toBeGreaterThan(sparse.length);
    expect(dense.some((edge) => edge.to.kind === 'node' && edge.to.layerIndex === selected.layerIndex && edge.to.cellIndex === selected.cellIndex)).toBe(true);
    expect(dense.some((edge) => edge.from.kind === 'node' && edge.to.kind === 'output')).toBe(true);
    [...sparse, ...dense].forEach((edge) => {
      if (edge.from.kind === 'input') expect(edge.from.index).toBeLessThan(trace.input.observed.length);
      if (edge.from.kind === 'node') expect(edge.from.cellIndex).toBeLessThan(trace.layers[edge.from.layerIndex].mu.length);
      if (edge.to.kind === 'node') expect(edge.to.cellIndex).toBeLessThan(trace.layers[edge.to.layerIndex].mu.length);
      expect(edge.from.kind).not.toBe('output');
      expect(edge.to.kind).not.toBe('input');
    });
  });
});
