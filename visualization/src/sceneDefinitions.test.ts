import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { fixtures } from './data/fixtures';
import { clampSelection } from './App';
import { fieldPosition, sceneDefinitions } from './sceneDefinitions';
import type { SceneRuntimeLike } from './sceneDefinitions';

function fakeRuntime(trace: typeof fixtures.mnist, dataset: keyof typeof sceneDefinitions): SceneRuntimeLike & { buckets: Record<string, THREE.Object3D[]> } {
  const definition = sceneDefinitions[dataset];
  const buckets: Record<string, THREE.Object3D[]> = { background: [], input: [], 'belief-field': [], readout: [] };
  const scene = new THREE.Scene();
  return {
    trace, scene, content: new THREE.Group(), palette: definition.palette, frameIndex: trace.frames.length - 1, networkMode: 'pathways', selectedCell: { layerIndex: 1, cellIndex: 0 }, isolatedLayer: null, edgeDensity: 0.58, reducedMotion: true,
    buckets, severity: trace.corruption.severity,
    add(object, group = 'belief-field') { buckets[group].push(object); },
    registerPick() {},
    addPulse() {},
    registerNodePulse() {},
    addTravelingPulse() {},
  };
}

describe('dataset-driven observatory scenes', () => {
  it('defines distinct compositions for all three datasets', () => {
    expect(Object.keys(sceneDefinitions)).toEqual(['mnist', 'aps', 'air_quality']);
    for (const dataset of Object.keys(sceneDefinitions) as (keyof typeof sceneDefinitions)[]) {
      const runtime = fakeRuntime(fixtures[dataset], dataset);
      const definition = sceneDefinitions[dataset];
      definition.buildBackground(runtime);
      definition.buildInput(runtime.trace, runtime);
      definition.buildBeliefField(runtime.trace, runtime);
      definition.buildReadout(runtime.trace, runtime);
      expect(runtime.buckets.background.length).toBeGreaterThan(0);
      expect(runtime.buckets.input.length).toBeGreaterThan(0);
      expect(runtime.buckets['belief-field'].length).toBeGreaterThan(0);
      expect(runtime.buckets.readout.length).toBeGreaterThan(0);
      const points = runtime.buckets['belief-field'].filter((object) => object.type === 'Points');
      const boundaries = runtime.buckets['belief-field'].filter((object) => object.userData.groupId !== undefined);
      const edges = runtime.buckets['belief-field'].filter((object) => object.userData.edge !== undefined);
      expect(points.length).toBe(runtime.trace.layers.length);
      expect(boundaries.length).toBe(0);
      expect(edges.length).toBeGreaterThan(0);
    }
  });

  it('clamps selection when a scene switch changes layer dimensions', () => {
    expect(clampSelection(fixtures.aps, { layerIndex: 99, cellIndex: 999 })).toEqual({ layerIndex: 3, cellIndex: 79 });
    expect(clampSelection(fixtures.mnist, { layerIndex: -2, cellIndex: -8 })).toEqual({ layerIndex: 0, cellIndex: 0 });
  });

  it('gives the belief network real depth instead of a mostly planar offset', () => {
    const trace = fixtures.mnist;
    const depth = trace.layers[0].mu.map((_, cellIndex) => fieldPosition(trace, 0, cellIndex).z);
    expect(Math.max(...depth) - Math.min(...depth)).toBeGreaterThan(0.5);
    const runtime = fakeRuntime(trace, 'mnist');
    sceneDefinitions.mnist.buildBeliefField(trace, runtime);
    const edgeLines = runtime.buckets['belief-field'].filter((object) => object.userData.edge !== undefined);
    const hasDepthArc = edgeLines.some((object) => {
      const values = Array.from((object as THREE.Line).geometry.getAttribute('position').array as ArrayLike<number>);
      const zValues = values.filter((_, index) => index % 3 === 2);
      return Math.max(...zValues) - Math.min(...zValues) > 0.12;
    });
    expect(hasDepthArc).toBe(true);
  });
});
