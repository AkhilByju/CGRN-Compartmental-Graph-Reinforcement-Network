import type {
  BeliefTrace, LayerState, NetworkMode, NeighborhoodLink, SelectedCell, TopConnection,
} from './types';

export interface Point2 { x: number; y: number }
export interface Point3 { x: number; y: number; z: number }

export interface SceneNode {
  layerIndex: number;
  cellIndex: number;
  point2: Point2;
  point3: Point3;
  group: number;
}

export type EdgeEndpoint =
  | { kind: 'input'; index: number }
  | { kind: 'node'; layerIndex: number; cellIndex: number }
  | { kind: 'output'; index?: number };

export interface SceneEdge {
  from: EdgeEndpoint;
  to: EdgeEndpoint;
  influence: number;
  similarity?: number;
  phase: number;
  kind: 'support' | 'conflict' | 'shared_evidence';
}

export interface SceneLayout {
  nodes: SceneNode[];
  input2: (index: number) => Point2;
  input3: (index: number) => Point3;
  output2: Point2;
  output3: Point3;
}

function fallbackGroupForCell(layer: LayerState, cellIndex: number) {
  return layer.node_groups?.[cellIndex]
    ?? layer.groups?.find((group) => group.cells.includes(cellIndex))?.id
    ?? Math.floor(cellIndex / Math.max(1, Math.ceil((layer.mu.length || 1) / 8)));
}

function groupIds(layer: LayerState) {
  const ids = new Set<number>();
  layer.groups?.forEach((group) => ids.add(group.id));
  layer.node_groups?.forEach((group) => ids.add(group));
  if (!ids.size) {
    for (let index = 0; index < layer.mu.length; index += 1) ids.add(fallbackGroupForCell(layer, index));
  }
  return [...ids].sort((a, b) => a - b);
}

function groupCells(layer: LayerState, id: number) {
  const exported = layer.groups?.find((group) => group.id === id)?.cells;
  if (exported?.length) return exported.filter((cell) => cell >= 0 && cell < layer.mu.length);
  return Array.from({ length: layer.mu.length }, (_, cellIndex) => cellIndex)
    .filter((cellIndex) => fallbackGroupForCell(layer, cellIndex) === id);
}

function constellationPosition(index: number, count: number, layerIndex: number, group: number, layer: LayerState, layerCount: number): Point3 {
  const ids = groupIds(layer);
  const slot = Math.max(0, ids.indexOf(group));
  const groupColumns = Math.max(1, Math.ceil(Math.sqrt(ids.length)));
  const groupRows = Math.ceil(ids.length / groupColumns);
  const groupColumn = slot % groupColumns;
  const groupRow = Math.floor(slot / groupColumns);
  const layerSpacing = layerCount > 2 ? 1.65 : 2.35;
  const groupX = -1.1 + layerIndex * layerSpacing;
  const groupY = 0.54 - (groupRow - (groupRows - 1) / 2) * 0.98;
  const groupZ = (groupColumn - (groupColumns - 1) / 2) * 0.92 + (groupRow - (groupRows - 1) / 2) * 0.2 + (layerIndex ? 0.12 : -0.12);
  const cells = groupCells(layer, group);
  const localIndex = Math.max(0, cells.indexOf(index));
  const localColumns = Math.max(1, Math.ceil(Math.sqrt(Math.max(1, cells.length))));
  const localRows = Math.ceil(Math.max(1, cells.length) / localColumns);
  const localColumn = localIndex % localColumns;
  const localRow = Math.floor(localIndex / localColumns);
  const localPhase = localIndex * 0.71 + slot * 0.43 + layerIndex * 0.9;
  const localColumnCenter = localColumn - (localColumns - 1) / 2;
  const localRowCenter = localRow - (localRows - 1) / 2;
  return {
    x: groupX + localRowCenter * 0.14 + Math.sin(localPhase * 0.7) * 0.08,
    y: groupY + localColumnCenter * 0.23 + Math.sin(localPhase) * 0.07,
    z: groupZ - localRowCenter * 0.34 + Math.cos(localPhase) * 0.1 + (count % 2 ? 0.015 : -0.015),
  };
}

function inputPosition(trace: BeliefTrace, index: number): Point3 {
  const angle = index / Math.max(1, trace.input.observed.length) * Math.PI * 2;
  const radius = trace.dataset === 'mnist' ? 0.24 : 0.34;
  const center = trace.dataset === 'mnist' ? { x: -2.62, y: 0.23, z: 0.16 } : { x: -2.8, y: 0.1, z: 0.05 };
  return { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius * 0.72, z: center.z + Math.sin(angle) * radius * 0.45 };
}

export function buildSceneLayout(trace: BeliefTrace, width: number, height: number): SceneLayout {
  const layerX = trace.layers.map((_, layerIndex) => width * (0.2 + ((layerIndex + 1) / (trace.layers.length + 1)) * 0.68));
  const nodes = trace.layers.flatMap((layer, layerIndex) => Array.from({ length: layer.mu.length }, (_, cellIndex) => {
    const group = fallbackGroupForCell(layer, cellIndex);
    const point = constellationPosition(cellIndex, layer.mu.length, layerIndex, group, layer, trace.layers.length);
    const ids = groupIds(layer);
    const slot = Math.max(0, ids.indexOf(group));
    const cells = groupCells(layer, group);
    const localIndex = Math.max(0, cells.indexOf(cellIndex));
    const groupColumns = Math.max(1, Math.ceil(Math.sqrt(ids.length)));
    const groupRows = Math.ceil(ids.length / groupColumns);
    const groupColumn = slot % groupColumns;
    const groupRow = Math.floor(slot / groupColumns);
    const localColumns = Math.max(1, Math.ceil(Math.sqrt(Math.max(1, cells.length))));
    const localRows = Math.ceil(Math.max(1, cells.length) / localColumns);
    return {
      layerIndex,
      cellIndex,
      point2: {
        x: layerX[layerIndex] + (groupColumn - (groupColumns - 1) / 2) * width * 0.018 + (localIndex % localColumns - (localColumns - 1) / 2) * width * 0.008,
        y: height * 0.5 - (groupRow - (groupRows - 1) / 2) * height * 0.16 - (Math.floor(localIndex / localColumns) - (localRows - 1) / 2) * height * 0.028,
      },
      point3: point,
      group,
    };
  }));
  const inputCount = trace.input.observed.length;
  const input2 = (index: number): Point2 => {
    if (trace.dataset === 'mnist') {
      const x = index % 28;
      const y = Math.floor(index / 28);
      return { x: width * 0.08 - 48 + x * 3.45, y: height * 0.5 - 48 + y * 3.45 };
    }
    const span = Math.min(168, width * 0.19);
    return { x: width * 0.12 - span / 2 + (index / Math.max(1, inputCount - 1)) * span, y: height * 0.5 + Math.sin(index * 0.55) * 28 };
  };
  return {
    nodes,
    input2,
    input3: (index: number) => inputPosition(trace, index),
    output2: { x: width * 0.9, y: height * 0.5 },
    output3: { x: 3.35, y: 0.18, z: 0.12 },
  };
}

function connectionKind(connection: TopConnection) {
  return connection.kind ?? ((connection.signed_influence ?? connection.weight) >= 0 ? 'support' : 'conflict');
}

function neighborhoodKind(connection: NeighborhoodLink) {
  return connection.kind ?? ((connection.signed_influence ?? connection.similarity) >= 0 ? 'shared_evidence' : 'conflict');
}

function strength(connection: TopConnection) {
  return Math.abs(connection.signed_influence ?? connection.weight);
}

function validPathwayConnections(trace: BeliefTrace, layerIndex: number, layer: LayerState, cellIndex: number) {
  const sourceCount = layerIndex === 0 ? trace.input.observed.length : trace.layers[layerIndex - 1]?.mu.length ?? 0;
  return (layer.top_connections[cellIndex] ?? [])
    .filter((connection) => Number.isInteger(connection.source) && connection.source >= 0 && connection.source < sourceCount)
    .map((connection, originalIndex) => ({ connection, originalIndex }))
    .sort((a, b) => strength(b.connection) - strength(a.connection)
      || a.connection.source - b.connection.source
      || a.originalIndex - b.originalIndex);
}

function selectedEndpoint(endpoint: EdgeEndpoint, selectedCell: SelectedCell) {
  return endpoint.kind === 'node'
    && endpoint.layerIndex === selectedCell.layerIndex
    && endpoint.cellIndex === selectedCell.cellIndex;
}

function selectPathwayEdges(trace: BeliefTrace, edgeDensity: number, selectedCell: SelectedCell, isolatedLayer: number | null) {
  const edges: SceneEdge[] = [];
  const perTarget = Math.max(1, Math.min(4, Math.ceil(edgeDensity * 4)));
  const perGroup = Math.max(2, Math.round(2 + edgeDensity * 5));
  trace.layers.forEach((layer, layerIndex) => {
    if (isolatedLayer !== null && isolatedLayer !== layerIndex) return;
    const candidates = layer.mu.flatMap((_, cellIndex) => validPathwayConnections(trace, layerIndex, layer, cellIndex)
      .map(({ connection, originalIndex }, rank) => ({ layerIndex, cellIndex, connection, originalIndex, rank, group: fallbackGroupForCell(layer, cellIndex) })));
    const byGroup = new Map<number, typeof candidates>();
    candidates.forEach((candidate) => { if (!byGroup.has(candidate.group)) byGroup.set(candidate.group, []); byGroup.get(candidate.group)?.push(candidate); });
    byGroup.forEach((groupCandidates) => {
      const forced = groupCandidates.filter((candidate) => (selectedCell.layerIndex === layerIndex && selectedCell.cellIndex === candidate.cellIndex)
        || (selectedCell.layerIndex === layerIndex - 1 && selectedCell.cellIndex === candidate.connection.source));
      const regular = groupCandidates
        .filter((candidate) => candidate.rank < perTarget && !forced.includes(candidate))
        .sort((a, b) => Math.abs(b.connection.signed_influence ?? b.connection.weight) - Math.abs(a.connection.signed_influence ?? a.connection.weight)
          || a.cellIndex - b.cellIndex || a.connection.source - b.connection.source || a.originalIndex - b.originalIndex);
      [...forced, ...regular.slice(0, Math.max(0, perGroup - forced.length))].forEach((candidate) => {
        const influence = candidate.connection.signed_influence ?? candidate.connection.weight;
        edges.push({
          from: layerIndex === 0
            ? { kind: 'input', index: candidate.connection.source }
            : { kind: 'node', layerIndex: layerIndex - 1, cellIndex: candidate.connection.source },
          to: { kind: 'node', layerIndex, cellIndex: candidate.cellIndex },
          influence,
          phase: candidate.cellIndex * 0.013 + candidate.originalIndex * 0.19 + layerIndex * 0.31,
          kind: connectionKind(candidate.connection),
        });
      });
    });
  });
  return edges;
}

function neighborhoodEdges(trace: BeliefTrace, edgeDensity: number, isolatedLayer: number | null) {
  const edges: SceneEdge[] = [];
  const perCell = Math.max(1, Math.min(4, Math.ceil(edgeDensity * 4)));
  trace.layers.forEach((layer, layerIndex) => {
    if (isolatedLayer !== null && isolatedLayer !== layerIndex) return;
    const sourceCounts = new Map<number, number>();
    layer.neighborhood_links
      ?.filter((link) => link.source >= 0 && link.source < layer.mu.length && link.target >= 0 && link.target < layer.mu.length)
      .sort((a, b) => b.similarity - a.similarity || a.source - b.source || a.target - b.target)
      .forEach((link) => {
        const sourceCount = sourceCounts.get(link.source) ?? 0;
        if (sourceCount >= perCell) return;
        sourceCounts.set(link.source, sourceCount + 1);
        edges.push({
          from: { kind: 'node', layerIndex, cellIndex: link.source },
          to: { kind: 'node', layerIndex, cellIndex: link.target },
          influence: link.signed_influence ?? link.similarity,
          similarity: link.similarity,
          phase: link.source * 0.017 + link.target * 0.007,
          kind: neighborhoodKind(link),
        });
      });
  });
  return edges;
}

function readoutEdges(trace: BeliefTrace, edgeDensity: number, isolatedLayer: number | null, selectedCell: SelectedCell) {
  const layerIndex = trace.layers.length - 1;
  const layer = trace.layers[layerIndex];
  if (!layer || (isolatedLayer !== null && isolatedLayer !== layerIndex)) return [];
  const groups = new Map<number, number[]>();
  layer.mu.forEach((_, cellIndex) => {
    const group = fallbackGroupForCell(layer, cellIndex);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)?.push(cellIndex);
  });
  const perGroup = Math.max(1, Math.min(3, Math.ceil(edgeDensity * 3)));
  const edges: SceneEdge[] = [];
  groups.forEach((cells, group) => {
    const ranked = [...cells].sort((a, b) => (layer.precision[b] ?? 0) - (layer.precision[a] ?? 0) || a - b);
    if (selectedCell.layerIndex === layerIndex && cells.includes(selectedCell.cellIndex)) {
      ranked.splice(ranked.indexOf(selectedCell.cellIndex), 1);
      ranked.unshift(selectedCell.cellIndex);
    }
    ranked.slice(0, perGroup).forEach((cellIndex, rank) => {
      edges.push({
        from: { kind: 'node', layerIndex, cellIndex },
        to: { kind: 'output', index: trace.task === 'classification' ? group % Math.max(1, trace.readout.logits.length) : 0 },
        // The trace does not export final readout weights. These neutral
        // aggregation links intentionally communicate precision, not a new
        // signed model connection.
        influence: layer.precision[cellIndex] ?? 0,
        phase: group * 0.23 + rank * 0.17,
        kind: 'shared_evidence',
      });
    });
  });
  return edges;
}

export function buildSceneEdges(
  trace: BeliefTrace, networkMode: NetworkMode, edgeDensity: number,
  frameIndex: number, isolatedLayer: number | null, selectedCell: SelectedCell,
): SceneEdge[] {
  const hasNeighborhoodData = trace.layers.some((layer) => (layer.neighborhood_links?.length ?? 0) > 0);
  const fieldEdges = networkMode === 'neighborhoods' && hasNeighborhoodData
    ? neighborhoodEdges(trace, edgeDensity, isolatedLayer)
    : selectPathwayEdges(trace, edgeDensity, selectedCell, isolatedLayer);
  const visibleLayers = Math.max(0, Math.min(trace.layers.length, trace.frames[frameIndex]?.visible_layers ?? trace.layers.length));
  const isVisible = (endpoint: EdgeEndpoint) => endpoint.kind !== 'node' || endpoint.layerIndex < visibleLayers;
  return [...fieldEdges, ...readoutEdges(trace, edgeDensity, isolatedLayer, selectedCell)]
    .filter((edge) => isVisible(edge.from) && isVisible(edge.to)
      && (edge.to.kind !== 'output' || visibleLayers >= trace.layers.length));
}

export function endpointPoint(layout: SceneLayout, endpoint: EdgeEndpoint, space: '2d' | '3d'): Point2 | Point3 {
  if (endpoint.kind === 'input') return space === '2d' ? layout.input2(endpoint.index) : layout.input3(endpoint.index);
  if (endpoint.kind === 'output') return space === '2d' ? layout.output2 : layout.output3;
  const node = layout.nodes.find((item) => item.layerIndex === endpoint.layerIndex && item.cellIndex === endpoint.cellIndex);
  return space === '2d' ? node?.point2 ?? layout.output2 : node?.point3 ?? layout.output3;
}

export function edgeTouchesSelected(edge: SceneEdge, selectedCell: SelectedCell) {
  return selectedEndpoint(edge.from, selectedCell) || selectedEndpoint(edge.to, selectedCell);
}
