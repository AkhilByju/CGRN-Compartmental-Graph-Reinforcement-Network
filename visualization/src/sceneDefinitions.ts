import * as THREE from 'three';
import { buildSceneEdges, edgeTouchesSelected, type EdgeEndpoint, type SceneEdge } from './sceneModel';
import type { BeliefTrace, CameraMode, DatasetId, LayerState, NetworkMode, SelectedCell } from './types';

export interface ScenePalette {
  background: string;
  blue: string;
  blueBright: string;
  ochre: string;
  vermilion: string;
  green: string;
  cloud: string;
  ink: string;
}

export interface CameraPreset {
  position: [number, number, number];
  target: [number, number, number];
  fov?: number;
}

export interface RuntimePick {
  kind: 'cell' | 'sensor' | 'group' | 'readout';
  cell?: SelectedCell;
  label?: string;
  groupId?: number;
}

export interface SceneRuntimeLike {
  readonly trace: BeliefTrace;
  readonly scene: THREE.Scene;
  readonly content: THREE.Group;
  readonly palette: ScenePalette;
  readonly frameIndex: number;
  readonly networkMode: NetworkMode;
  readonly selectedCell: SelectedCell;
  readonly isolatedLayer: number | null;
  readonly edgeDensity: number;
  readonly severity: number;
  readonly reducedMotion: boolean;
  add(object: THREE.Object3D, group?: 'background' | 'input' | 'belief-field' | 'readout'): void;
  registerPick(object: THREE.Object3D, pick: RuntimePick | ((instanceId?: number) => RuntimePick | undefined)): void;
  addPulse(object: THREE.Object3D, speed?: number, phase?: number): void;
  registerNodePulse(cell: SelectedCell, object: THREE.Object3D, color: THREE.Color, state?: { evidence: number; uncertainty: number; precision: number }, shell?: THREE.Object3D): void;
  addTravelingPulse(curve: THREE.QuadraticBezierCurve3, color: string, speed?: number, phase?: number, target?: SelectedCell): void;
}

export interface DatasetSceneDefinition {
  id: DatasetId;
  title: string;
  subtitle: string;
  palette: ScenePalette;
  camera: CameraPreset;
  buildBackground: (runtime: SceneRuntimeLike) => void;
  buildInput: (trace: BeliefTrace, runtime: SceneRuntimeLike) => void;
  buildBeliefField: (trace: BeliefTrace, runtime: SceneRuntimeLike) => void;
  buildReadout: (trace: BeliefTrace, runtime: SceneRuntimeLike) => void;
}

const basePalette: ScenePalette = {
  background: '#050913', blue: '#5eb8d8', blueBright: '#b8f2ff', ochre: '#d7ad68',
  vermilion: '#e57768', green: '#8fd1ae', cloud: '#dbe9e5', ink: '#07101c',
};

const palettes: Record<DatasetId, ScenePalette> = {
  mnist: { ...basePalette, background: '#060913', blue: '#65c4e2', ochre: '#d9ae69', vermilion: '#ed776a' },
  aps: { ...basePalette, background: '#080b0e', blue: '#74b9c9', ochre: '#d9a35f', vermilion: '#e06c5d', green: '#9dc8ae' },
  air_quality: { ...basePalette, background: '#08100f', blue: '#7bc9bd', ochre: '#d6b06f', vermilion: '#e58370', green: '#a8d5b2' },
};

function addGlowSprite(runtime: SceneRuntimeLike, position: THREE.Vector3, color: string, size: number, opacity = 0.65, group: 'background' | 'input' | 'belief-field' | 'readout' = 'background') {
  let texture: THREE.CanvasTexture | undefined;
  if (typeof document !== 'undefined') {
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = 128;
    glowCanvas.height = 128;
    const glowContext = glowCanvas.getContext('2d');
    if (glowContext) {
      const gradient = glowContext.createRadialGradient(64, 64, 0, 64, 64, 64);
      gradient.addColorStop(0, 'rgba(255,255,255,.92)');
      gradient.addColorStop(.18, 'rgba(255,255,255,.34)');
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      glowContext.fillStyle = gradient;
      glowContext.fillRect(0, 0, 128, 128);
      texture = new THREE.CanvasTexture(glowCanvas);
    }
  }
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  sprite.position.copy(position);
  sprite.scale.setScalar(size);
  runtime.add(sprite, group);
  return sprite;
}

function addStars(runtime: SceneRuntimeLike, count: number, spread: [number, number, number]) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  let seed = runtime.trace.seed + runtime.trace.dataset.length * 97;
  const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const color = new THREE.Color(runtime.palette.cloud);
  for (let index = 0; index < count; index += 1) {
    positions[index * 3] = (random() - 0.5) * spread[0];
    positions[index * 3 + 1] = (random() - 0.5) * spread[1];
    positions[index * 3 + 2] = (random() - 0.5) * spread[2] - 1.5;
    const intensity = 0.25 + random() * 0.75;
    colors[index * 3] = color.r * intensity;
    colors[index * 3 + 1] = color.g * intensity;
    colors[index * 3 + 2] = color.b * intensity;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  runtime.add(new THREE.Points(geometry, new THREE.PointsMaterial({ size: 0.025, vertexColors: true, transparent: true, opacity: 0.62, sizeAttenuation: true })), 'background');
}

function addObservatoryGrid(runtime: SceneRuntimeLike, radius = 5.5) {
  const group = new THREE.Group();
  const material = new THREE.LineBasicMaterial({ color: runtime.palette.blue, transparent: true, opacity: 0.055 });
  for (let ring = 1; ring <= 4; ring += 1) {
    const points: THREE.Vector3[] = [];
    const ringRadius = ring * radius / 4;
    for (let index = 0; index <= 64; index += 1) {
      const angle = (index / 64) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(angle) * ringRadius, -1.28, Math.sin(angle) * ringRadius));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material));
  }
  runtime.add(group, 'background');
}

function addBackground(runtime: SceneRuntimeLike, atmosphereColor: string) {
  runtime.scene.fog = new THREE.FogExp2(runtime.palette.background, 0.045);
  runtime.scene.add(new THREE.HemisphereLight(atmosphereColor, '#02040a', 1.05));
  const key = new THREE.PointLight(atmosphereColor, 10, 12, 2); key.position.set(1, 3.5, 3); runtime.scene.add(key);
  const rim = new THREE.PointLight(runtime.palette.vermilion, 3, 10, 2); rim.position.set(-4, 0.5, -3); runtime.scene.add(rim);
  addStars(runtime, 560, [15, 9, 12]);
  addObservatoryGrid(runtime);
  addGlowSprite(runtime, new THREE.Vector3(-3.6, 2.1, -3.1), atmosphereColor, 2.4, 0.1);
  addGlowSprite(runtime, new THREE.Vector3(3.4, -0.6, -4), runtime.palette.ochre, 2.2, 0.06);
}

function fallbackGroup(layer: LayerState, cellIndex: number) {
  return layer.node_groups?.[cellIndex]
    ?? layer.groups?.find((group) => group.cells.includes(cellIndex))?.id
    ?? Math.floor(cellIndex / Math.max(1, Math.ceil((layer.mu.length || 1) / 8)));
}

function layerGroups(layer: LayerState) {
  const groups = new Map<number, { cells: number[]; label?: string }>();
  layer.groups?.forEach((group) => groups.set(group.id, { cells: group.cells.filter((cell) => cell >= 0 && cell < layer.mu.length), label: group.label }));
  for (let cellIndex = 0; cellIndex < layer.mu.length; cellIndex += 1) {
    const id = fallbackGroup(layer, cellIndex);
    if (!groups.has(id)) groups.set(id, { cells: [] });
    if (!groups.get(id)?.cells.includes(cellIndex)) groups.get(id)?.cells.push(cellIndex);
  }
  return [...groups.entries()].sort(([a], [b]) => a - b).map(([id, value]) => ({ id, cells: value.cells, label: value.label ?? `evidence cluster ${String(id + 1).padStart(2, '0')}` }));
}

export function fieldPosition(trace: BeliefTrace, layerIndex: number, cellIndex: number): THREE.Vector3 {
  const layer = trace.layers[layerIndex];
  const groups = layerGroups(layer);
  const group = fallbackGroup(layer, cellIndex);
  const groupSlot = Math.max(0, groups.findIndex((item) => item.id === group));
  const groupColumns = Math.max(1, Math.ceil(Math.sqrt(groups.length)));
  const groupRows = Math.ceil(groups.length / groupColumns);
  const groupColumn = groupSlot % groupColumns;
  const groupRow = Math.floor(groupSlot / groupColumns);
  const cells = groups[groupSlot]?.cells ?? [cellIndex];
  const localIndex = Math.max(0, cells.indexOf(cellIndex));
  const localColumns = Math.max(1, Math.ceil(Math.sqrt(cells.length)));
  const localRows = Math.ceil(cells.length / localColumns);
  const localColumn = localIndex % localColumns;
  const localRow = Math.floor(localIndex / localColumns);
  const layerSpacing = trace.layers.length > 2 ? 1.65 : 2.35;
  const groupX = -1.1 + layerIndex * layerSpacing;
  const groupY = 0.54 - (groupRow - (groupRows - 1) / 2) * 0.98;
  const groupZ = (groupColumn - (groupColumns - 1) / 2) * 0.92 + (groupRow - (groupRows - 1) / 2) * 0.2 + (layerIndex ? 0.12 : -0.12);
  const localPhase = localIndex * 0.71 + groupSlot * 0.43 + layerIndex * 0.9;
  const localColumnCenter = localColumn - (localColumns - 1) / 2;
  const localRowCenter = localRow - (localRows - 1) / 2;
  // Each belief layer gets its own spatial grammar instead of repeating the
  // same vertical lattice four times. The exported topology still anchors
  // every edge to the correct cell; only the presentation changes.
  const layerMode = layerIndex % 4;
  const groupPhase = groupSlot * 0.78 + layerIndex * 1.17;
  const groupDrift = layerMode === 1
    ? Math.sin(groupPhase) * 0.24
    : layerMode === 2
      ? (groupRow - (groupRows - 1) / 2) * 0.26
      : layerMode === 3 ? Math.cos(groupPhase) * 0.28 : 0;
  const verticalScale = layerMode === 1 ? 0.82 : layerMode === 2 ? 0.94 : layerMode === 3 ? 1.08 : 1;
  const depthScale = layerMode === 1 ? 1.28 : layerMode === 2 ? 0.68 : layerMode === 3 ? 1.16 : 0.82;
  const braid = layerMode === 3 ? Math.sin(groupPhase) * 0.18 : 0;
  return new THREE.Vector3(
    groupX + groupDrift + localRowCenter * (layerMode === 2 ? 0.2 : 0.14) + Math.sin(localPhase * 0.7) * 0.08,
    groupY * verticalScale + braid + localColumnCenter * (layerMode === 1 ? 0.29 : 0.23) + Math.sin(localPhase) * 0.07,
    groupZ * depthScale - localRowCenter * (layerMode === 2 ? 0.46 : 0.34) + Math.cos(localPhase) * 0.1,
  );
}

function inputAnchor(trace: BeliefTrace, index: number): THREE.Vector3 {
  const center = trace.dataset === 'mnist' ? new THREE.Vector3(-2.62, 0.23, 0.16) : new THREE.Vector3(-2.8, 0.1, 0.05);
  if (trace.dataset === 'mnist') {
    const columns = trace.input.shape?.[1] ?? 28;
    const rows = trace.input.shape?.[0] ?? Math.ceil(trace.input.observed.length / columns);
    const x = index % columns;
    const y = Math.floor(index / columns);
    return center.add(new THREE.Vector3(
      (x / Math.max(1, columns - 1) - 0.5) * 1.5,
      (0.5 - y / Math.max(1, rows - 1)) * 1.5,
      ((index % 3) - 1) * 0.012,
    ));
  }
  const columns = trace.dataset === 'aps' ? 3 : 4;
  const rows = Math.ceil(trace.input.observed.length / columns);
  const column = index % columns;
  const row = Math.floor(index / columns);
  return center.add(new THREE.Vector3(
    (column - (columns - 1) / 2) * 0.28,
    (0.5 - row / Math.max(1, rows - 1)) * 1.42,
    Math.sin(index * 0.7) * 0.08,
  ));
}

function inputHub(trace: BeliefTrace) {
  return trace.dataset === 'mnist' ? new THREE.Vector3(-2.62, 0.23, 0.16) : new THREE.Vector3(-2.8, 0.1, 0.05);
}

function outputAnchor(trace: BeliefTrace, index = 0) {
  const layerSpacing = trace.layers.length > 2 ? 1.65 : 2.35;
  const lastLayerX = -1.1 + Math.max(0, trace.layers.length - 1) * layerSpacing;
  const count = Math.max(1, trace.readout.logits.length);
  if (count === 1) return new THREE.Vector3(lastLayerX + 1.55, 0.18, 0.12);
  return new THREE.Vector3(
    lastLayerX + 1.42,
    0.18 + (0.5 - index / Math.max(1, count - 1)) * 1.62,
    0.12 + Math.sin(index * 0.62) * 0.08,
  );
}

function outputHub(trace: BeliefTrace) {
  const layerSpacing = trace.layers.length > 2 ? 1.65 : 2.35;
  const lastLayerX = -1.1 + Math.max(0, trace.layers.length - 1) * layerSpacing;
  return new THREE.Vector3(lastLayerX + 2.22, 0.18, 0.12);
}

function endpointPosition(trace: BeliefTrace, endpoint: EdgeEndpoint) {
  if (endpoint.kind === 'input') return inputAnchor(trace, endpoint.index);
  if (endpoint.kind === 'output') return outputAnchor(trace, endpoint.index ?? 0);
  return fieldPosition(trace, endpoint.layerIndex, endpoint.cellIndex);
}

function edgeColor(runtime: SceneRuntimeLike, edge: SceneEdge) {
  return edge.kind === 'conflict' || edge.influence < 0 ? runtime.palette.vermilion : edge.kind === 'shared_evidence' ? runtime.palette.ochre : runtime.palette.blue;
}

function addDirectionalMarker(runtime: SceneRuntimeLike, curve: THREE.QuadraticBezierCurve3, color: string) {
  const point = curve.getPointAt(0.7);
  const tangent = curve.getTangentAt(0.7).normalize();
  const marker = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.12, 5), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.72, depthWrite: false }));
  marker.position.copy(point);
  marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
  marker.userData.directionalMarker = true;
  runtime.add(marker, 'belief-field');
}

function addEdge(runtime: SceneRuntimeLike, trace: BeliefTrace, edge: SceneEdge) {
  const from = endpointPosition(trace, edge.from);
  const to = endpointPosition(trace, edge.to);
  const middle = from.clone().lerp(to, 0.5);
  const phase = edge.phase * Math.PI;
  const bend = (edge.phase % 2 > 1 ? 1 : -1) * 0.16 + edge.influence * 0.07;
  middle.y += bend + Math.sin(phase) * 0.1;
  middle.z += Math.cos(phase) * 0.34 + (edge.from.kind === 'node' ? 0.08 : -0.06);
  const curve = new THREE.QuadraticBezierCurve3(from, middle, to);
  const selected = edgeTouchesSelected(edge, runtime.selectedCell);
  const color = edgeColor(runtime, edge);
  const baseOpacity = edge.kind === 'conflict' ? 0.17 : edge.kind === 'shared_evidence' ? 0.13 : 0.11;
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(curve.getPoints(selected ? 18 : 10)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: selected ? 0.68 : baseOpacity + Math.abs(edge.influence) * 0.09, depthWrite: false }),
  );
  line.userData.edge = edge;
  line.userData.edgeSource = edge.from;
  line.userData.edgeTarget = edge.to;
  runtime.add(line, 'belief-field');
  const target = edge.to.kind === 'node' ? { layerIndex: edge.to.layerIndex, cellIndex: edge.to.cellIndex } : undefined;
  runtime.addTravelingPulse(curve, color, 0.14 + Math.min(0.16, Math.abs(edge.influence) * 0.2), edge.phase * 0.37, target);
  if (selected) {
    runtime.addPulse(line, 0.15, edge.phase);
    addDirectionalMarker(runtime, curve, color);
  }
}

function labelSprite(text: string, color: string, position: THREE.Vector3) {
  if (typeof document === 'undefined') return undefined;
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) return undefined;
  context.font = '500 22px ui-monospace, monospace';
  context.fillStyle = color;
  context.globalAlpha = 0.72;
  context.fillText(text.toUpperCase(), 4, 40);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false, depthTest: false }));
  sprite.position.copy(position);
  sprite.scale.set(1.05, 0.13, 1);
  return sprite;
}

function groupBoundary(runtime: SceneRuntimeLike, trace: BeliefTrace, layerIndex: number, group: { id: number; cells: number[]; label?: string }) {
  const positions = group.cells.map((cell) => fieldPosition(trace, layerIndex, cell));
  if (!positions.length) return;
  const center = positions.reduce((sum, point) => sum.add(point), new THREE.Vector3()).multiplyScalar(1 / positions.length);
  const radiusX = Math.max(0.22, ...positions.map((point) => Math.abs(point.x - center.x) + 0.17));
  const radiusY = Math.max(0.19, ...positions.map((point) => Math.abs(point.y - center.y) + 0.16));
  const radiusZ = Math.max(0.19, ...positions.map((point) => Math.abs(point.z - center.z) + 0.16));
  const radius = Math.max(radiusX, radiusY, radiusZ);
  const boundary = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 8, 6),
    // This box is only a pick proxy for selecting a group. Keep it fully
    // invisible; visible rectangles compete with the belief field.
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
  );
  boundary.position.copy(center);
  boundary.userData.groupId = group.id;
  boundary.userData.layerIndex = layerIndex;
  boundary.userData.groupBoundaryProxy = true;
  runtime.registerPick(boundary, { kind: 'group', groupId: group.id, label: group.label ?? `group ${group.id + 1}` });
  runtime.add(boundary, 'belief-field');
}

function cellColor(runtime: SceneRuntimeLike, layer: LayerState, cellIndex: number, groupSize: number) {
  const consensus = layer.consensus[cellIndex] ?? 0;
  const uncertainty = layer.uncertainty[cellIndex] ?? 0;
  if (consensus < -0.18 || uncertainty > 0.72) return new THREE.Color(runtime.palette.vermilion);
  if (Math.abs(consensus) < 0.18) return new THREE.Color(runtime.palette.ochre);
  const color = new THREE.Color(runtime.palette.blueBright);
  if (groupSize > 1) color.lerp(new THREE.Color(runtime.palette.ochre), 0.28);
  return color;
}

function buildSharedField(trace: BeliefTrace, runtime: SceneRuntimeLike) {
  const visibleLayers = Math.max(0, Math.min(trace.layers.length, trace.frames[runtime.frameIndex]?.visible_layers ?? trace.layers.length));
  const detailedNodes = trace.layers.length <= 2;
  trace.layers.forEach((layer, layerIndex) => {
    if (runtime.isolatedLayer !== null && runtime.isolatedLayer !== layerIndex) return;
    if (layerIndex >= visibleLayers) return;
    const positions = new Float32Array(layer.mu.length * 3);
    const colors = new Float32Array(layer.mu.length * 3);
    const sizes = new Float32Array(layer.mu.length);
    const confidenceScale = Math.max(0.58, 1 - Math.max(0, runtime.severity - trace.corruption.severity) * 0.7);
    const groups = layerGroups(layer);
    const sizeByCell = new Map(groups.flatMap((group) => group.cells.map((cell) => [cell, group.cells.length] as [number, number])));
    // All cells use the same low-poly shape; sharing this buffer keeps a
    // denser fixture from multiplying geometry memory.
    const nodeGeometry = layerIndex % 4 === 0
      ? new THREE.OctahedronGeometry(0.065, 0)
      : layerIndex % 4 === 1
        ? new THREE.IcosahedronGeometry(0.06, 0)
        : layerIndex % 4 === 2
          ? new THREE.TetrahedronGeometry(0.075, 0)
          : new THREE.DodecahedronGeometry(0.062, 0);
    const shellGeometry = new THREE.IcosahedronGeometry(0.095, 0);
    for (let cellIndex = 0; cellIndex < layer.mu.length; cellIndex += 1) {
      const position = fieldPosition(trace, layerIndex, cellIndex);
      const precision = Math.max(0.08, Math.min(1, layer.precision[cellIndex] ?? 0.5));
      const active = runtime.selectedCell.layerIndex === layerIndex && runtime.selectedCell.cellIndex === cellIndex;
      const color = cellColor(runtime, layer, cellIndex, sizeByCell.get(cellIndex) ?? 1);
      positions[cellIndex * 3] = position.x; positions[cellIndex * 3 + 1] = position.y; positions[cellIndex * 3 + 2] = position.z;
      colors[cellIndex * 3] = color.r; colors[cellIndex * 3 + 1] = color.g; colors[cellIndex * 3 + 2] = color.b;
      // Keep cells readable at the default camera distance. These used to
      // resolve as 2–3px sprites, which made the belief field feel empty.
      sizes[cellIndex] = (0.11 + precision * 0.075 * confidenceScale) * (active ? 1.3 : 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { opacity: { value: 0.36 + confidenceScale * 0.12 } },
      vertexShader: 'attribute float size; attribute vec3 aColor; varying vec3 vColor; void main() { vColor = aColor; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_PointSize = max(6.0, size * (390.0 / max(1.0, -mvPosition.z))); gl_Position = projectionMatrix * mvPosition; }',
      fragmentShader: 'varying vec3 vColor; uniform float opacity; void main() { float distanceToCenter = distance(gl_PointCoord, vec2(0.5)); float halo = smoothstep(0.5, 0.02, distanceToCenter); float core = smoothstep(0.42, 0.06, distanceToCenter); float rim = smoothstep(0.49, 0.38, distanceToCenter) * smoothstep(0.5, 0.47, distanceToCenter); gl_FragColor = vec4(vColor * (0.72 + core * 1.1 + rim * 0.42), halo * opacity); }',
    });
    const cells = new THREE.Points(geometry, material);
    cells.userData.pickKind = 'cell'; cells.userData.layerIndex = layerIndex;
    const resolveCellPick = (instanceId?: number) => typeof instanceId === 'number' && instanceId >= 0 && instanceId < layer.mu.length
      ? { kind: 'cell' as const, cell: { layerIndex, cellIndex: instanceId }, label: `${layer.name} cell ${instanceId + 1}` } : undefined;
    runtime.registerPick(cells, resolveCellPick);
    runtime.add(cells, 'belief-field');
    if (!detailedNodes) {
      // The point layer carries the glow, but the readable node itself needs
      // volume. One instanced low-poly mesh per layer keeps this dense scene
      // inexpensive while making every belief cell visually substantial.
      const colorBuckets = new Map<number, { cellIndex: number; position: THREE.Vector3; scale: number }[]>();
      for (let cellIndex = 0; cellIndex < layer.mu.length; cellIndex += 1) {
        const position = fieldPosition(trace, layerIndex, cellIndex);
        const precision = Math.max(0.08, Math.min(1, layer.precision[cellIndex] ?? 0.5));
        const active = runtime.selectedCell.layerIndex === layerIndex && runtime.selectedCell.cellIndex === cellIndex;
        const scale = (0.76 + precision * 0.26) * (active ? 1.2 : 1);
        const colorKey = cellColor(runtime, layer, cellIndex, sizeByCell.get(cellIndex) ?? 1).getHex();
        const bucket = colorBuckets.get(colorKey) ?? [];
        bucket.push({ cellIndex, position, scale });
        colorBuckets.set(colorKey, bucket);
      }
      const nodeMatrix = new THREE.Matrix4();
      colorBuckets.forEach((bucket, colorKey) => {
        const nodeMesh = new THREE.InstancedMesh(
          nodeGeometry,
          new THREE.MeshStandardMaterial({ color: colorKey, emissive: colorKey, emissiveIntensity: 0.52, roughness: 0.26, metalness: 0.14, transparent: true, opacity: 0.86, depthWrite: true, flatShading: true }),
          bucket.length,
        );
        bucket.forEach((item, instanceIndex) => {
          nodeMatrix.compose(item.position, new THREE.Quaternion(), new THREE.Vector3(item.scale, item.scale, item.scale));
          nodeMesh.setMatrixAt(instanceIndex, nodeMatrix);
        });
        nodeMesh.instanceMatrix.needsUpdate = true;
        nodeMesh.userData.pickKind = 'cell';
        nodeMesh.userData.layerIndex = layerIndex;
        runtime.registerPick(nodeMesh, (instanceId) => {
          const item = typeof instanceId === 'number' ? bucket[instanceId] : undefined;
          return item ? { kind: 'cell' as const, cell: { layerIndex, cellIndex: item.cellIndex }, label: `${layer.name} cell ${item.cellIndex + 1}` } : undefined;
        });
        runtime.add(nodeMesh, 'belief-field');
        const shellMesh = new THREE.InstancedMesh(
          shellGeometry,
          new THREE.MeshBasicMaterial({ color: colorKey, transparent: true, opacity: 0.38, depthWrite: false, wireframe: true, blending: THREE.AdditiveBlending }),
          bucket.length,
        );
        bucket.forEach((item, instanceIndex) => {
          const uncertainty = Math.max(0, Math.min(1, layer.uncertainty[item.cellIndex] ?? 0.2));
          const shellScale = (0.78 + uncertainty * 0.24) * (runtime.selectedCell.layerIndex === layerIndex && runtime.selectedCell.cellIndex === item.cellIndex ? 1.18 : 1);
          nodeMatrix.compose(item.position, new THREE.Quaternion(), new THREE.Vector3(shellScale, shellScale, shellScale));
          shellMesh.setMatrixAt(instanceIndex, nodeMatrix);
        });
        shellMesh.instanceMatrix.needsUpdate = true;
        shellMesh.userData.layerIndex = layerIndex;
        runtime.add(shellMesh, 'belief-field');
      });
    }
    for (let cellIndex = 0; cellIndex < layer.mu.length; cellIndex += 1) {
      const position = fieldPosition(trace, layerIndex, cellIndex);
      const precision = Math.max(0.08, Math.min(1, layer.precision[cellIndex] ?? 0.5));
      const color = cellColor(runtime, layer, cellIndex, sizeByCell.get(cellIndex) ?? 1);
      if (!detailedNodes) {
        const anchor = new THREE.Object3D();
        anchor.position.copy(position);
        const active = runtime.selectedCell.layerIndex === layerIndex && runtime.selectedCell.cellIndex === cellIndex;
        anchor.scale.setScalar((0.88 + precision * 0.28) * (active ? 1.34 : 1));
        runtime.registerNodePulse(
          { layerIndex, cellIndex },
          anchor,
          color,
          {
            evidence: Math.max(0, Math.min(1, layer.evidence[cellIndex] ?? 0.5)),
            uncertainty: Math.max(0, Math.min(1, layer.uncertainty[cellIndex] ?? 0.2)),
            precision,
          },
        );
        continue;
      }
      const core = new THREE.Mesh(
        nodeGeometry,
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.2, metalness: 0.18, transparent: true, opacity: 0.82, depthWrite: true, flatShading: true }),
      );
      const uncertainty = Math.max(0, Math.min(1, layer.uncertainty[cellIndex] ?? 0.2));
      const shell = new THREE.Mesh(
        core.geometry,
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08 + uncertainty * 0.12, depthWrite: false, wireframe: true, blending: THREE.AdditiveBlending }),
      );
      shell.scale.setScalar(1.48 + uncertainty * 0.24);
      shell.rotation.set(cellIndex * 0.37, cellIndex * 0.19, cellIndex * 0.11);
      core.add(shell);
      core.position.copy(position);
      const scale = (0.88 + precision * 0.28) * (runtime.selectedCell.layerIndex === layerIndex && runtime.selectedCell.cellIndex === cellIndex ? 1.34 : 1);
      core.scale.setScalar(scale);
      core.userData.pickKind = 'cell';
      core.userData.layerIndex = layerIndex;
      core.userData.cellIndex = cellIndex;
      runtime.registerPick(core, { kind: 'cell', cell: { layerIndex, cellIndex }, label: `${layer.name} cell ${cellIndex + 1}` });
      runtime.registerNodePulse(
        { layerIndex, cellIndex },
        core,
        color,
        {
          evidence: Math.max(0, Math.min(1, layer.evidence[cellIndex] ?? 0.5)),
          uncertainty,
          precision,
        },
        shell,
      );
      runtime.add(core, 'belief-field');
    }
    if (detailedNodes) groups.forEach((group) => groupBoundary(runtime, trace, layerIndex, group));
  });
  buildSceneEdges(trace, runtime.networkMode, runtime.edgeDensity, runtime.frameIndex, runtime.isolatedLayer, runtime.selectedCell).forEach((edge) => addEdge(runtime, trace, edge));
}

function addLineLoop(runtime: SceneRuntimeLike, points: THREE.Vector3[], color: string, opacity: number, group: 'input' | 'readout' = 'input') {
  const geometry = new THREE.BufferGeometry().setFromPoints([...points, points[0]]);
  runtime.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false })), group);
}

function addInputFrame(runtime: SceneRuntimeLike, center: THREE.Vector3, width: number, height: number) {
  const frame = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, 0.05),
    new THREE.MeshBasicMaterial({ color: runtime.palette.blue, transparent: true, opacity: 0.045, depthWrite: false, side: THREE.DoubleSide }),
  );
  frame.position.copy(center);
  runtime.add(frame, 'input');
  const corners = [
    new THREE.Vector3(-width / 2, -height / 2, 0), new THREE.Vector3(width / 2, -height / 2, 0),
    new THREE.Vector3(width / 2, height / 2, 0), new THREE.Vector3(-width / 2, height / 2, 0),
  ].map((point) => point.add(center.clone().add(new THREE.Vector3(0, 0, 0.04))));
  addLineLoop(runtime, corners, runtime.palette.blueBright, 0.68);
}

function buildInputFreeField(trace: BeliefTrace, runtime: SceneRuntimeLike) {
  const center = inputHub(runtime.trace);
  const observed = trace.input.observed;
  const reliability = trace.input.reliability;
  const positions = observed.map((_, index) => inputAnchor(trace, index));
  const nodeGeometry = trace.dataset === 'mnist'
    ? new THREE.BoxGeometry(0.044, 0.044, 0.034)
    : new THREE.SphereGeometry(0.062, 10, 8);
  const nodeMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.92, depthWrite: false });
  const nodes = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, Math.max(1, observed.length));
  const matrix = new THREE.Matrix4();
  observed.forEach((value, index) => {
    const point = positions[index];
    const quality = Math.max(0.04, reliability[index] ?? 0);
    const missing = quality < 0.2;
    const scale = trace.dataset === 'mnist' ? 0.78 + Math.max(0, value) * 0.42 : 0.82 + Math.abs(value) * 0.22;
    matrix.compose(point, new THREE.Quaternion(), new THREE.Vector3(scale, scale, 1));
    nodes.setMatrixAt(index, matrix);
    const color = new THREE.Color(missing ? runtime.palette.vermilion : runtime.palette.blueBright);
    color.multiplyScalar(missing ? 0.72 : 0.34 + Math.min(0.72, Math.max(0, Math.abs(value)) * 0.62) + quality * 0.18);
    nodes.setColorAt(index, color);
  });
  nodes.instanceMatrix.needsUpdate = true;
  if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;
  nodes.userData.pickKind = 'sensor';
  runtime.registerPick(nodes, (instanceId) => typeof instanceId === 'number' && instanceId >= 0 && instanceId < observed.length
    ? { kind: 'sensor', label: trace.input.labels?.[instanceId] ?? `observation ${instanceId + 1}` }
    : undefined);
  runtime.add(nodes, 'input');

  if (trace.dataset === 'mnist') {
    addInputFrame(runtime, center, 1.68, 1.68);
    const grid = new THREE.Group();
    const lineMaterial = new THREE.LineBasicMaterial({ color: runtime.palette.blue, transparent: true, opacity: 0.12, depthWrite: false });
    const columns = trace.input.shape?.[1] ?? 28;
    const rows = trace.input.shape?.[0] ?? 28;
    for (let index = 1; index < columns; index += 1) {
      const x = (index / columns - 0.5) * 1.5;
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        center.clone().add(new THREE.Vector3(x, -0.75, 0.035)),
        center.clone().add(new THREE.Vector3(x, 0.75, 0.035)),
      ]), lineMaterial));
    }
    for (let index = 1; index < rows; index += 1) {
      const y = (0.5 - index / rows) * 1.5;
      grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        center.clone().add(new THREE.Vector3(-0.75, y, 0.035)),
        center.clone().add(new THREE.Vector3(0.75, y, 0.035)),
      ]), lineMaterial));
    }
    runtime.add(grid, 'input');
  } else {
    const railPoints = [
      center.clone().add(new THREE.Vector3(-0.42, -0.78, -0.02)),
      center.clone().add(new THREE.Vector3(-0.42, 0.78, -0.02)),
      center.clone().add(new THREE.Vector3(0, -0.78, -0.02)),
      center.clone().add(new THREE.Vector3(0, 0.78, -0.02)),
      center.clone().add(new THREE.Vector3(0.42, -0.78, -0.02)),
      center.clone().add(new THREE.Vector3(0.42, 0.78, -0.02)),
    ];
    for (let index = 0; index < railPoints.length; index += 2) runtime.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(railPoints.slice(index, index + 2)), new THREE.LineBasicMaterial({ color: runtime.palette.ochre, transparent: true, opacity: 0.34, depthWrite: false })), 'input');
    addInputFrame(runtime, center, 1.08, 1.72);
  }
  const ring = new THREE.Mesh(new THREE.TorusGeometry(trace.dataset === 'mnist' ? 0.92 : 0.74, 0.012, 8, 64), new THREE.MeshBasicMaterial({ color: runtime.palette.blueBright, transparent: true, opacity: 0.22, depthWrite: false }));
  ring.position.copy(center).add(new THREE.Vector3(0, 0, -0.06));
  runtime.add(ring, 'input'); runtime.addPulse(ring, 0.08);
  const label = labelSprite(trace.dataset === 'mnist' ? 'input · 28 × 28 pixels' : 'input · sensor array', runtime.palette.blueBright, center.clone().add(new THREE.Vector3(-0.68, -1.08, 0)));
  if (label) runtime.add(label, 'input');
}

function softmax(values: number[]) {
  const max = Math.max(...values, 0);
  const exponentials = values.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  return exponentials.map((value) => value / total);
}

function buildSharedReadout(trace: BeliefTrace, runtime: SceneRuntimeLike) {
  const hub = outputHub(trace);
  const outputCount = Math.max(1, trace.readout.logits.length);
  const probabilities = trace.task === 'classification' ? softmax(trace.readout.logits) : [1];
  const predictionIndex = trace.task === 'classification' && typeof trace.readout.prediction === 'number'
    ? Math.max(0, Math.min(outputCount - 1, trace.readout.prediction)) : 0;
  const rail = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(Array.from({ length: outputCount }, (_, index) => outputAnchor(trace, index))),
    new THREE.LineBasicMaterial({ color: runtime.palette.green, transparent: true, opacity: 0.2, depthWrite: false }),
  );
  runtime.add(rail, 'readout');
  for (let index = 0; index < outputCount; index += 1) {
    const probability = probabilities[index] ?? 0;
    const isPrediction = index === predictionIndex;
    const color = new THREE.Color(isPrediction ? runtime.palette.green : probability > 0.12 ? runtime.palette.ochre : runtime.palette.blue);
    const node = new THREE.Mesh(
      new THREE.SphereGeometry(0.072 + probability * 0.11, 12, 8),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: isPrediction ? 1.5 : 0.48 + probability, roughness: 0.22, metalness: 0.18 }),
    );
    node.position.copy(outputAnchor(trace, index));
    node.scale.z = 0.72;
    runtime.registerPick(node, { kind: 'readout', label: trace.task === 'classification' ? `class ${index}` : 'regression readout' });
    runtime.add(node, 'readout');
    addGlowSprite(runtime, node.position.clone().add(new THREE.Vector3(0, 0, 0.04)), `#${color.getHexString()}`, isPrediction ? 0.3 : 0.16 + probability * 0.12, isPrediction ? 0.18 : 0.08, 'readout');
    if (outputCount > 1) {
      const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([node.position, hub]), new THREE.LineBasicMaterial({ color, transparent: true, opacity: isPrediction ? 0.34 : 0.1 + probability * 0.2, depthWrite: false }));
      runtime.add(stem, 'readout');
    }
  }
  const aperture = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.018, 8, 64), new THREE.MeshBasicMaterial({ color: runtime.palette.green, transparent: true, opacity: 0.82, depthWrite: false }));
  aperture.position.copy(hub); runtime.registerPick(aperture, { kind: 'readout', label: 'stabilized readout' }); runtime.add(aperture, 'readout'); runtime.addPulse(aperture, 0.08);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.4, 0.006, 6, 64), new THREE.MeshBasicMaterial({ color: runtime.palette.ochre, transparent: true, opacity: 0.35, depthWrite: false }));
  ring.position.copy(hub); runtime.add(ring, 'readout');
  const core = new THREE.Mesh(new THREE.CircleGeometry(0.08, 32), new THREE.MeshBasicMaterial({ color: runtime.palette.green, transparent: true, opacity: 0.74, depthWrite: false }));
  core.position.copy(hub); runtime.add(core, 'readout');
  addGlowSprite(runtime, hub.clone().add(new THREE.Vector3(0, 0, 0.1)), runtime.palette.green, 0.58, 0.14, 'readout');
  const label = labelSprite(trace.task === 'classification' ? `digit ${trace.readout.prediction} · ${outputCount} classes` : String(trace.readout.prediction), runtime.palette.green, hub.clone().add(new THREE.Vector3(-0.62, -0.48, 0)));
  if (label) runtime.add(label, 'readout');
}

function makeDefinition(id: DatasetId): DatasetSceneDefinition {
  const title = id === 'mnist' ? 'The shape of a missing digit' : id === 'aps' ? 'The machine listens for failure' : 'A plume of uncertain air';
  const subtitle = id === 'mnist' ? 'Pixels become evidence, gaps become uncertainty, and competing glyphs gather in the field.' : id === 'aps' ? 'Mechanical sensor arcs route fault evidence through a living instrument core.' : 'Sensor channels become an atmospheric flow whose gaps remain visible in the final readout.';
  const camera = id === 'mnist'
    ? { position: [5.6, 3.55, 10.4] as [number, number, number], target: [1.25, 0.08, 0] as [number, number, number], fov: 47 }
    : id === 'aps'
      ? { position: [5.9, 3.25, 10.1] as [number, number, number], target: [1.25, 0.12, 0] as [number, number, number], fov: 48 }
      : { position: [5.35, 3.15, 10.35] as [number, number, number], target: [1.25, 0.1, 0] as [number, number, number], fov: 48 };
  return {
    id, title, subtitle, palette: palettes[id], camera,
    buildBackground: (runtime) => addBackground(runtime, palettes[id].blue),
    buildInput: buildInputFreeField,
    buildBeliefField: buildSharedField,
    buildReadout: buildSharedReadout,
  };
}

export const sceneDefinitions: Record<DatasetId, DatasetSceneDefinition> = { mnist: makeDefinition('mnist'), aps: makeDefinition('aps'), air_quality: makeDefinition('air_quality') };
export function getSceneDefinition(dataset: DatasetId) { return sceneDefinitions[dataset]; }

export function fieldPoint2(trace: BeliefTrace, layerIndex: number, cellIndex: number, width: number, height: number) {
  const layer = trace.layers[layerIndex];
  const groups = layerGroups(layer);
  const group = fallbackGroup(layer, cellIndex);
  const slot = Math.max(0, groups.findIndex((item) => item.id === group));
  const columns = Math.max(1, Math.ceil(Math.sqrt(groups.length)));
  const rows = Math.ceil(groups.length / columns);
  const groupColumn = slot % columns;
  const groupRow = Math.floor(slot / columns);
  const cells = groups[slot]?.cells ?? [cellIndex];
  const localIndex = Math.max(0, cells.indexOf(cellIndex));
  const localColumns = Math.max(1, Math.ceil(Math.sqrt(cells.length)));
  const localRows = Math.ceil(cells.length / localColumns);
  return {
    x: width * (0.2 + ((layerIndex + 1) / (trace.layers.length + 1)) * 0.68) + (groupColumn - (columns - 1) / 2) * width * 0.02 + (localIndex % localColumns - (localColumns - 1) / 2) * width * 0.009,
    y: height * 0.5 - (groupRow - (rows - 1) / 2) * height * 0.17 - (Math.floor(localIndex / localColumns) - (localRows - 1) / 2) * height * 0.03,
  };
}

export function sceneInputPoint2(trace: BeliefTrace, index: number, width: number, height: number) {
  if (trace.dataset === 'mnist') { const x = index % 28; const y = Math.floor(index / 28); return { x: width * 0.08 + x * Math.min(3.1, width * 0.006), y: height * 0.34 + y * Math.min(3.1, height * 0.005) }; }
  const columns = trace.dataset === 'aps' ? 3 : 4;
  const rows = Math.ceil(trace.input.observed.length / columns);
  const column = index % columns;
  const row = Math.floor(index / columns);
  const panelWidth = Math.min(width * 0.2, 176);
  return {
    x: width * 0.12 - panelWidth / 2 + (column / Math.max(1, columns - 1)) * panelWidth,
    y: height * 0.52 + (0.5 - row / Math.max(1, rows - 1)) * height * 0.3,
  };
}

export function outputPoint2(trace: BeliefTrace, index: number, width: number, height: number) {
  const count = Math.max(1, trace.readout.logits.length);
  if (count === 1) return { x: width * 0.86, y: height * 0.5 };
  return { x: width * 0.86, y: height * 0.5 + (0.5 - index / Math.max(1, count - 1)) * height * 0.3 };
}

export function outputHub2(_trace: BeliefTrace, width: number, height: number) {
  return { x: width * 0.94, y: height * 0.5 };
}

export function definitionForMode(definition: DatasetSceneDefinition, _mode: CameraMode) { return definition; }
