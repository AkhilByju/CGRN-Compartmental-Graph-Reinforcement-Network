import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent, ReactNode } from 'react';
import {
  Aperture, Camera, ChevronDown, ChevronRight, Cuboid, Eye, Layers3,
  Pause, Play, RotateCcw, ScanLine, Settings2, SlidersHorizontal, Sparkles, Waypoints, X,
} from 'lucide-react';
import { fixtures } from './data/fixtures';
import { buildSceneEdges, edgeTouchesSelected, type EdgeEndpoint, type SceneEdge } from './sceneModel';
import { getSceneDefinition, fieldPoint2, outputHub2, outputPoint2, sceneInputPoint2, type RuntimePick } from './sceneDefinitions';
import { SceneRuntime } from './sceneRuntime';
import type { BeliefTrace, CameraMode, DatasetId, NetworkMode, SelectedCell } from './types';
import './styles.css';

type CanvasHandle = { screenshot: () => void; resetCamera: () => void };

interface CanvasProps {
  trace: BeliefTrace;
  frameIndex: number;
  cameraMode: CameraMode;
  networkMode: NetworkMode;
  severity: number;
  edgeDensity: number;
  isolatedLayer: number | null;
  selectedCell: SelectedCell;
  reducedMotion: boolean;
  onSelectCell: (cell: SelectedCell) => void;
  onPick: (pick: RuntimePick) => void;
}

export function clampSelection(trace: BeliefTrace, selected: SelectedCell): SelectedCell {
  const layerIndex = Math.max(0, Math.min(trace.layers.length - 1, selected.layerIndex));
  const count = trace.layers[layerIndex]?.mu.length ?? 1;
  return { layerIndex, cellIndex: Math.max(0, Math.min(count - 1, selected.cellIndex)) };
}

export function validateTrace(trace: BeliefTrace): BeliefTrace {
  if (!trace.layers.length || !trace.frames.length) throw new Error(`Trace ${trace.dataset} has no renderable layers or frames`);
  if (trace.input.observed.length !== trace.input.reliability.length) throw new Error(`Trace ${trace.dataset} has mismatched observation reliability`);
  trace.layers.forEach((layer) => {
    const length = layer.mu.length;
    if ([layer.evidence, layer.uncertainty, layer.precision, layer.consensus, layer.top_connections].some((values) => values.length !== length)) throw new Error(`Trace ${trace.dataset} has an invalid ${layer.name} state vector`);
  });
  return trace;
}

function numberText(value: number, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }

type CanvasBackdropCache = { key: string; canvas: HTMLCanvasElement };
const canvasBackdropCache = new WeakMap<HTMLCanvasElement, CanvasBackdropCache>();

function draw2dBackdrop(ctx: CanvasRenderingContext2D, trace: BeliefTrace, severity: number, width: number, height: number) {
  const definition = getSceneDefinition(trace.dataset); const palette = definition.palette;
  const input = (index: number) => sceneInputPoint2(trace, index, width, height);
  const displayReliability = (index: number) => Math.max(0.03, (trace.input.reliability[index] ?? 0) * (1 - Math.max(0, severity - trace.corruption.severity)));
  if (trace.dataset === 'mnist') {
    trace.input.observed.forEach((value, index) => { const point = input(index); const reliability = displayReliability(index); ctx.fillStyle = reliability < 0.2 ? palette.vermilion : palette.blueBright; ctx.globalAlpha = reliability < 0.2 ? 0.45 : 0.08 + Math.max(0, value) * 0.8; ctx.fillRect(point.x, point.y, 2.5, 2.5); });
    ctx.globalAlpha = 1; ctx.strokeStyle = 'rgba(184,242,255,.28)'; ctx.strokeRect(width * 0.075, height * 0.32, Math.min(86, width * 0.22), Math.min(86, height * 0.34));
  } else if (trace.dataset === 'aps') {
    const left = width * 0.12 - Math.min(width * 0.2, 176) / 2; const right = width * 0.12 + Math.min(width * 0.2, 176) / 2;
    ctx.strokeStyle = 'rgba(215,173,104,.32)'; ctx.lineWidth = 1; ctx.strokeRect(left - 13, height * .34, right - left + 26, height * .36);
    trace.input.reliability.forEach((reliability, index) => { const point = input(index); const value = Math.abs(trace.input.observed[index] ?? 0); const color = reliability < 0.2 ? palette.vermilion : palette.blue; ctx.fillStyle = color; ctx.globalAlpha = .28 + Math.min(.65, value * .55); ctx.beginPath(); ctx.arc(point.x, point.y, reliability < .2 ? 4.3 : 3.2 + value * 2, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = .55; ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(point.x, point.y + 7); ctx.lineTo(point.x, point.y + 7 + value * 24); ctx.stroke(); }); ctx.globalAlpha = 1;
  } else {
    const colors = [palette.blue, palette.ochre, palette.green, palette.vermilion];
    colors.forEach((color, channel) => { ctx.strokeStyle = color; ctx.globalAlpha = .26; ctx.lineWidth = 1.4 + channel * .2; let drawing = false; ctx.beginPath(); trace.input.observed.forEach((value, index) => { if (displayReliability(index) < .2) { drawing = false; return; } const point = input(index); const x = point.x + (channel - 1.5) * 2; const y = point.y - value * 7 + Math.sin(index * .42 + channel) * 3; if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y); }); ctx.stroke(); });
    ctx.globalAlpha = .34; ctx.strokeStyle = palette.blueBright; ctx.lineWidth = 1; ctx.strokeRect(width * .12 - Math.min(width * .2, 176) / 2 - 13, height * .34, Math.min(width * .2, 176) + 26, height * .36); ctx.globalAlpha = 1;
  }
}

function draw2dScene(canvas: HTMLCanvasElement, trace: BeliefTrace, frameIndex: number, networkMode: NetworkMode, severity: number, edgeDensity: number, isolatedLayer: number | null, selectedCell: SelectedCell, elapsed: number, edges: SceneEdge[]) {
  const box = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  const width = Math.max(1, box.width); const height = Math.max(1, box.height);
  if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) { canvas.width = Math.floor(width * dpr); canvas.height = Math.floor(height * dpr); }
  let ctx: CanvasRenderingContext2D | null = null;
  try { ctx = canvas.getContext('2d'); } catch { return; }
  if (!ctx) return;
  const cacheKey = `${trace.dataset}|${trace.sample_id}|${trace.seed}|${severity}|${width}:${height}:${dpr}`;
  let cached = canvasBackdropCache.get(canvas);
  if (!cached || cached.key !== cacheKey || cached.canvas.width !== canvas.width || cached.canvas.height !== canvas.height) {
    const backdrop = document.createElement('canvas');
    backdrop.width = canvas.width; backdrop.height = canvas.height;
    const backdropContext = backdrop.getContext('2d');
    if (!backdropContext) return;
    backdropContext.save(); backdropContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw2dBackdrop(backdropContext, trace, severity, width, height);
    backdropContext.restore();
    cached = { key: cacheKey, canvas: backdrop };
    canvasBackdropCache.set(canvas, cached);
  }
  ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const definition = getSceneDefinition(trace.dataset); const palette = definition.palette;
  const gradient = ctx.createRadialGradient(width * 0.55, height * 0.46, 20, width * 0.55, height * 0.5, Math.max(width, height) * 0.78);
  gradient.addColorStop(0, palette.background); gradient.addColorStop(1, '#02040a'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  const random = (index: number) => { const value = Math.sin(index * 9283.41 + trace.seed * 17.3) * 43758.5453; return value - Math.floor(value); };
  for (let index = 0; index < 150; index += 1) { ctx.fillStyle = `rgba(209, 232, 229, ${0.12 + random(index) * 0.5})`; ctx.beginPath(); ctx.arc(random(index + 2) * width, random(index + 5) * height, 0.4 + random(index + 8) * 1.2, 0, Math.PI * 2); ctx.fill(); }
  ctx.strokeStyle = 'rgba(124, 184, 199, .08)'; ctx.lineWidth = 1;
  for (let index = 0; index < 13; index += 1) { const y = height * 0.2 + index * height * 0.052; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  // The observation layer is intentionally rendered over the atmosphere so
  // the actual sample remains legible in the 2D fallback as well.
  ctx.drawImage(cached.canvas, 0, 0, width, height);
  const frame = trace.frames[frameIndex] ?? trace.frames[trace.frames.length - 1] ?? trace.frames[0];
  const visibleLayers = Math.max(0, Math.min(trace.layers.length, frame?.visible_layers ?? trace.layers.length));
  const input = (index: number) => sceneInputPoint2(trace, index, width, height);
  const node = (layerIndex: number, cellIndex: number) => fieldPoint2(trace, layerIndex, cellIndex, width, height);

  const curveControls = (from: { x: number; y: number }, to: { x: number; y: number }, phase: number) => {
    const bend = (to.x - from.x) * .42;
    return [from, { x: from.x + bend, y: from.y - 18 + Math.sin(phase) * 18 }, { x: to.x - bend, y: to.y + 18 - Math.sin(phase) * 18 }, to] as const;
  };
  const curvePoint = (controls: ReturnType<typeof curveControls>, t: number) => {
    const [start, controlA, controlB, end] = controls;
    const inverse = 1 - t;
    return {
      x: inverse ** 3 * start.x + 3 * inverse ** 2 * t * controlA.x + 3 * inverse * t ** 2 * controlB.x + t ** 3 * end.x,
      y: inverse ** 3 * start.y + 3 * inverse ** 2 * t * controlA.y + 3 * inverse * t ** 2 * controlB.y + t ** 3 * end.y,
    };
  };
  const curve = (controls: ReturnType<typeof curveControls>, color: string, alpha: number, lineWidth: number) => {
    const [start, controlA, controlB, end] = controls;
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = lineWidth; ctx.beginPath(); ctx.moveTo(start.x, start.y);
    ctx.bezierCurveTo(controlA.x, controlA.y, controlB.x, controlB.y, end.x, end.y); ctx.stroke(); ctx.globalAlpha = 1;
  };
  const endpoint = (value: EdgeEndpoint) => value.kind === 'input' ? input(value.index) : value.kind === 'output' ? outputPoint2(trace, value.index ?? 0, width, height) : node(value.layerIndex, value.cellIndex);
  const motionAllowed = typeof window.matchMedia !== 'function' || !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (motionAllowed) ctx.globalCompositeOperation = 'lighter';
  edges.forEach((edge) => {
    const from = endpoint(edge.from); const to = endpoint(edge.to); const selected = edgeTouchesSelected(edge, selectedCell);
    const color = edge.kind === 'conflict' || edge.influence < 0 ? palette.vermilion : edge.kind === 'shared_evidence' ? palette.ochre : palette.blue;
    const controls = curveControls(from, to, edge.phase);
    curve(controls, color, selected ? .72 : .1 + Math.abs(edge.influence) * .12, selected ? 1.85 : .55);
    if (motionAllowed) {
      const speed = .14 + Math.min(.16, Math.abs(edge.influence) * .2);
      const start = ((elapsed * speed + edge.phase * .37) % 1 + 1) % 1;
      for (let packet = 0; packet < 2; packet += 1) {
        const packetStart = (start + packet * .5) % 1;
        for (let tail = 0; tail < 3; tail += 1) {
          const t = packetStart - tail * .045;
          const point = curvePoint(controls, (t + 1) % 1);
          ctx.fillStyle = color; ctx.globalAlpha = (selected ? .88 : .62) * (1 - tail * .26);
          ctx.beginPath(); ctx.arc(point.x, point.y, selected ? 2.1 - tail * .2 : 1.5 - tail * .16, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  });
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  trace.layers.forEach((layer, layerIndex) => { if (isolatedLayer !== null && isolatedLayer !== layerIndex || layerIndex >= visibleLayers) return; layer.mu.forEach((_, cellIndex) => { const point = node(layerIndex, cellIndex); const precision = Math.max(.08, Math.min(1, layer.precision[cellIndex] ?? .4)); const active = selectedCell.layerIndex === layerIndex && selectedCell.cellIndex === cellIndex; const consensus = layer.consensus[cellIndex] ?? 0; const unresolved = (layer.uncertainty[cellIndex] ?? 0) > .72; ctx.fillStyle = consensus < -.18 || unresolved ? palette.vermilion : Math.abs(consensus) < .18 ? palette.ochre : palette.blue; ctx.globalAlpha = .55 + precision * .45; ctx.beginPath(); ctx.arc(point.x, point.y, active ? 8 : 4.1 + precision * 3.3, 0, Math.PI * 2); ctx.fill(); if (active) { ctx.strokeStyle = palette.cloud; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(point.x, point.y, 14, 0, Math.PI * 2); ctx.stroke(); } }); });
  const outputCount = Math.max(1, trace.readout.logits.length);
  const maxLogit = Math.max(...trace.readout.logits, 0); const expLogits = trace.readout.logits.map((value) => Math.exp(value - maxLogit)); const expTotal = expLogits.reduce((sum, value) => sum + value, 0) || 1;
  const predictionIndex = trace.task === 'classification' && typeof trace.readout.prediction === 'number' ? Math.max(0, Math.min(outputCount - 1, trace.readout.prediction)) : 0;
  const outputNodes = Array.from({ length: outputCount }, (_, index) => outputPoint2(trace, index, width, height));
  ctx.strokeStyle = 'rgba(143,209,174,.22)'; ctx.lineWidth = 1; ctx.beginPath(); outputNodes.forEach((point, index) => { if (index === 0) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y); }); ctx.stroke();
  outputNodes.forEach((point, index) => { const probability = trace.task === 'classification' ? (expLogits[index] ?? 0) / expTotal : 1; const active = index === predictionIndex; const color = active ? palette.green : probability > .12 ? palette.ochre : palette.blue; ctx.fillStyle = color; ctx.globalAlpha = active ? .95 : .55 + probability * .3; ctx.beginPath(); ctx.arc(point.x, point.y, active ? 8 + probability * 6 : 4 + probability * 8, 0, Math.PI * 2); ctx.fill(); if (outputCount > 1) { ctx.fillStyle = palette.cloud; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.globalAlpha = active ? .9 : .5; ctx.fillText(String(index), point.x - 12, point.y + 3); } });
  const output = outputHub2(trace, width, height); ctx.strokeStyle = 'rgba(143,209,174,.28)'; ctx.lineWidth = 1.2; outputNodes.forEach((point) => { ctx.globalAlpha = .12; ctx.beginPath(); ctx.moveTo(point.x, point.y); ctx.quadraticCurveTo((point.x + output.x) / 2, point.y, output.x, output.y); ctx.stroke(); }); ctx.globalAlpha = .9; ctx.fillStyle = palette.green; ctx.beginPath(); ctx.arc(output.x, output.y, 10 + (frame?.energy ?? .5) * 5, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = 'rgba(143,209,174,.3)'; ctx.beginPath(); ctx.arc(output.x, output.y, 25, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = palette.cloud; ctx.font = '600 14px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(trace.task === 'classification' ? `digit ${trace.readout.prediction}` : String(trace.readout.prediction), output.x, output.y + 40); ctx.fillStyle = palette.green; ctx.font = '10px ui-monospace, monospace'; ctx.fillText(trace.task === 'classification' ? `READOUT LAYER · ${outputCount} CLASSES` : 'STABILIZED READOUT', output.x, output.y - 32); ctx.textAlign = 'left';
  ctx.fillStyle = palette.cloud; ctx.font = '500 11px Inter, sans-serif'; ctx.fillText(definition.title, 30, height - 48); ctx.fillStyle = 'rgba(219,233,229,.58)'; ctx.font = '10px ui-monospace, monospace'; ctx.fillText(networkMode === 'pathways' ? 'TRACE-GROUNDED PATHWAYS' : 'EXPORTED NEIGHBORHOODS', 30, height - 28); ctx.fillText('FULL INFERENCE FIELD', width - 194, 28); ctx.restore();
}

const ObservatoryCanvas = forwardRef<CanvasHandle, CanvasProps>(function ObservatoryCanvas({ trace, frameIndex, cameraMode, networkMode, severity, edgeDensity, isolatedLayer, selectedCell, reducedMotion, onSelectCell, onPick }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null); const mountRef = useRef<HTMLDivElement>(null); const runtimeRef = useRef<SceneRuntime | null>(null); const drawRef = useRef<() => void>(() => undefined); const elapsedRef = useRef(0); const [webglUnavailable, setWebglUnavailable] = useState(false);
  const edges = useMemo(() => buildSceneEdges(trace, networkMode, edgeDensity, frameIndex, isolatedLayer, selectedCell), [trace, networkMode, edgeDensity, frameIndex, isolatedLayer, selectedCell]);
  useEffect(() => {
    const mount = mountRef.current; if (!mount || cameraMode !== '3d') return undefined; let runtime: SceneRuntime | null = null;
    try { runtime = new SceneRuntime(mount, { trace, definition: getSceneDefinition(trace.dataset), frameIndex, cameraMode, networkMode, selectedCell, isolatedLayer, edgeDensity, severity, reducedMotion, onPick: (pick) => { onPick(pick); if (pick.cell) onSelectCell(pick.cell); } }); runtimeRef.current = runtime; setWebglUnavailable(false); } catch { setWebglUnavailable(true); }
    const resize = () => runtime?.resize(); const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null; observer?.observe(mount);
    return () => { observer?.disconnect(); runtime?.dispose(); runtimeRef.current = null; };
  }, [trace, frameIndex, cameraMode, networkMode, severity, selectedCell, isolatedLayer, edgeDensity, reducedMotion, onPick, onSelectCell]);
  useEffect(() => { drawRef.current = () => { const canvas = canvasRef.current; if (canvas && (cameraMode !== '3d' || webglUnavailable)) draw2dScene(canvas, trace, frameIndex, networkMode, severity, edgeDensity, isolatedLayer, selectedCell, elapsedRef.current, edges); }; drawRef.current(); }, [trace, frameIndex, networkMode, severity, edgeDensity, isolatedLayer, selectedCell, edges, cameraMode, webglUnavailable]);
  useEffect(() => { if (typeof requestAnimationFrame === 'undefined' || (cameraMode === '3d' && !webglUnavailable)) return undefined; let animation = 0; let lastDraw = 0; const frameBudget = 1000 / 45; const tick = (time: number) => { if (time - lastDraw >= frameBudget) { elapsedRef.current = time / 1000; drawRef.current(); lastDraw = time; } animation = requestAnimationFrame(tick); }; animation = requestAnimationFrame(tick); return () => cancelAnimationFrame(animation); }, [cameraMode, webglUnavailable]);
  useEffect(() => { const canvas = canvasRef.current; const mount = mountRef.current; if (!canvas || !mount) return undefined; const resize = () => drawRef.current(); const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null; observer?.observe(canvas); observer?.observe(mount); resize(); return () => observer?.disconnect(); }, []);
  const screenshot = () => { const runtime = runtimeRef.current; if (cameraMode === '3d' && runtime && !webglUnavailable) runtime.screenshot(); else { drawRef.current(); const canvas = canvasRef.current; if (!canvas) return; const link = document.createElement('a'); link.download = `belief-observatory-${trace.dataset}-2d.png`; link.href = canvas.toDataURL('image/png'); link.click(); } };
  useImperativeHandle(ref, () => ({ screenshot, resetCamera: () => runtimeRef.current?.resetCamera() }), [cameraMode, trace, webglUnavailable]);
  const on2dClick = (event: ReactMouseEvent<HTMLCanvasElement>) => { const box = event.currentTarget.getBoundingClientRect(); const point = { x: event.clientX - box.left, y: event.clientY - box.top }; const candidates: { distance: number; cell: SelectedCell }[] = []; trace.layers.forEach((layer, layerIndex) => layer.mu.forEach((_, cellIndex) => { const candidate = fieldPoint2(trace, layerIndex, cellIndex, box.width, box.height); candidates.push({ distance: Math.hypot(candidate.x - point.x, candidate.y - point.y), cell: { layerIndex, cellIndex } }); })); const best = candidates.sort((a, b) => a.distance - b.distance)[0]; if (best && best.distance < 24) onSelectCell(best.cell); };
  return <div className={`render-surface ${cameraMode === '3d' && !webglUnavailable ? 'is-3d' : 'is-2d'}`}><canvas ref={canvasRef} className="observatory-canvas" onClick={on2dClick} aria-label="Interactive 2D belief field overview" /><div ref={mountRef} className="three-mount" /></div>;
});

function DatasetButton({ id, active, label, onClick }: { id: DatasetId; active: boolean; label: string; onClick: () => void }) {
  return <button className={`scene-tab ${active ? 'is-active' : ''}`} onClick={onClick} aria-pressed={active} data-testid={`dataset-${id}`}><strong>{label}</strong><ChevronRight size={14} /></button>;
}

function App() {
  const [dataset, setDataset] = useState<DatasetId>('mnist');
  const [trace, setTrace] = useState<BeliefTrace>(fixtures.mnist);
  const [cameraMode, setCameraMode] = useState<CameraMode>('3d');
  const [networkMode, setNetworkMode] = useState<NetworkMode>('pathways');
  const [severity, setSeverity] = useState(trace.corruption.severity);
  const [edgeDensity, setEdgeDensity] = useState(.28);
  const [isolatedLayer, setIsolatedLayer] = useState<number | null>(null);
  const [selectedCell, setSelectedCell] = useState<SelectedCell>({ layerIndex: 1, cellIndex: 19 });
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [frameIndex, setFrameIndex] = useState(fixtures.mnist.frames.length - 1);
  const [isPlaying, setIsPlaying] = useState(false);
  const canvas = useRef<CanvasHandle>(null);
  const definition = getSceneDefinition(dataset);
  const safeSelection = useMemo(() => clampSelection(trace, selectedCell), [trace, selectedCell]);
  const layer = trace.layers[safeSelection.layerIndex]; const cellIndex = safeSelection.cellIndex;
  const selected = layer ? { mu: layer.mu[cellIndex] ?? 0, evidence: layer.evidence[cellIndex] ?? 0, uncertainty: layer.uncertainty[cellIndex] ?? 0, precision: layer.precision[cellIndex] ?? 0, consensus: layer.consensus[cellIndex] ?? 0 } : { mu: 0, evidence: 0, uncertainty: 0, precision: 0, consensus: 0 };
  const beliefNodeCount = trace.layers.reduce((total, item) => total + item.mu.length, 0);
  const missingness = Math.round((1 - trace.input.reliability.reduce((sum, value) => sum + value, 0) / Math.max(1, trace.input.reliability.length)) * 100);
  const frame = trace.frames[frameIndex] ?? trace.frames[0];
  useEffect(() => { if (typeof window.matchMedia !== 'function') return undefined; const media = window.matchMedia('(prefers-reduced-motion: reduce)'); const update = () => setReducedMotion(media.matches); update(); media.addEventListener?.('change', update); return () => media.removeEventListener?.('change', update); }, []);
  useEffect(() => { setSelectedCell(clampSelection(trace, selectedCell)); }, [trace]);
  useEffect(() => { if (!isPlaying || trace.frames.length < 2) return undefined; const timer = window.setInterval(() => setFrameIndex((current) => (current + 1) % trace.frames.length), 900); return () => window.clearInterval(timer); }, [isPlaying, trace.frames.length]);
  const changeDataset = (next: DatasetId) => { const nextTrace = validateTrace(fixtures[next]); setDataset(next); setTrace(nextTrace); setSeverity(nextTrace.corruption.severity); setCameraMode('3d'); setNetworkMode('pathways'); setIsolatedLayer(null); setInspectorOpen(false); setIsPlaying(false); setFrameIndex(Math.max(0, nextTrace.frames.length - 1)); setSelectedCell(clampSelection(nextTrace, { layerIndex: nextTrace.layers.length - 1, cellIndex: 0 })); };
  const selectCell = useCallback((cell: SelectedCell) => { setSelectedCell(clampSelection(trace, cell)); setInspectorOpen(true); }, [trace]);
  const handlePick = useCallback((pick: RuntimePick) => { if (pick.cell) selectCell(pick.cell); else setInspectorOpen(true); }, [selectCell]);
  const focusPath = () => { const index = trace.layers[1]?.precision.indexOf(Math.max(...(trace.layers[1]?.precision ?? [0]))); setNetworkMode('pathways'); setCameraMode('3d'); selectCell({ layerIndex: 1, cellIndex: Math.max(0, index ?? 0) }); };
  const confidence = trace.task === 'classification' ? Math.round((1 - trace.readout.uncertainty) * 100) : Math.round(Math.max(0, 100 - trace.readout.uncertainty * 10));
  return <main className={`hero-shell ${inspectorOpen ? 'inspector-is-open' : ''}`} style={{ '--scene-accent': definition.palette.ochre } as CSSProperties}>
    <section className="hero-stage">
      <ObservatoryCanvas ref={canvas} trace={trace} frameIndex={frameIndex} cameraMode={cameraMode} networkMode={networkMode} severity={severity} edgeDensity={edgeDensity} isolatedLayer={isolatedLayer} selectedCell={safeSelection} reducedMotion={reducedMotion} onSelectCell={selectCell} onPick={handlePick} />
      <div className="hero-vignette" />
      <div className="node-key" aria-label="Node color key"><span><i className="legend-dot blue" /> SUPPORT</span><span><i className="legend-dot vermilion" /> CONFLICT / UNCERTAINTY</span><span><i className="legend-dot ochre" /> NEUTRAL</span></div>
      <div className="trace-timeline" aria-label="Belief trace replay"><button className="timeline-play" onClick={() => setIsPlaying((playing) => !playing)} aria-label={isPlaying ? 'Pause trace replay' : 'Play trace replay'}>{isPlaying ? <Pause size={13} /> : <Play size={13} />}</button><div className="timeline-track"><input type="range" min="0" max={Math.max(0, trace.frames.length - 1)} value={frameIndex} style={{ '--timeline-progress': `${trace.frames.length > 1 ? (frameIndex / (trace.frames.length - 1)) * 100 : 100}%` } as CSSProperties} onChange={(event) => { setIsPlaying(false); setFrameIndex(Number(event.target.value)); }} aria-label="Trace state" /><div className="timeline-labels"><span>TRACE REPLAY</span><span>{String(frameIndex + 1).padStart(2, '0')} / {String(trace.frames.length).padStart(2, '0')} · {frame?.active_stage.replaceAll('_', ' ')}</span></div></div></div>
      <header className="observatory-nav"><div className="brand-lockup"><span className="brand-orbit"><Aperture size={18} /></span><span><strong>BELO</strong><small>INFERENCE OBSERVATORY</small></span></div><span className="nav-status"><i /> LOCAL TRACE RUNTIME / SCHEMA {trace.schema_version}.0</span><div className="nav-actions"><button className="inspect-button" onClick={() => setInspectorOpen((open) => !open)} aria-expanded={inspectorOpen}><SlidersHorizontal size={14} /> INSPECT <ChevronDown size={13} className={inspectorOpen ? 'rotate' : ''} /></button></div></header>
      <div className="hero-copy"><p className="eyebrow">CELLV0.3 / RELIABILITY-WEIGHTED BELIEF FLOW</p><h1>{definition.title}</h1><p>{definition.subtitle}</p></div>
      <div className="scene-dock"><div className="scene-dock-label"><span>SCENES</span><small>03 WORLDS / ACTIVE {dataset === 'air_quality' ? '03' : dataset === 'aps' ? '02' : '01'}</small></div><DatasetButton id="mnist" active={dataset === 'mnist'} label="MNIST" onClick={() => changeDataset('mnist')} /><DatasetButton id="aps" active={dataset === 'aps'} label="APS FAILURE" onClick={() => changeDataset('aps')} /><DatasetButton id="air_quality" active={dataset === 'air_quality'} label="AIR QUALITY" onClick={() => changeDataset('air_quality')} /></div>
      <div className="field-meta"><span><i className="status-dot" /> {trace.provenance.trace_kind.replace('_', ' ').toUpperCase()}</span><span>{trace.sample_id}</span><span>{beliefNodeCount} BELIEF CELLS</span><span>{frame?.active_stage.replaceAll('_', ' ')}</span></div>
      <div className="view-controls"><div className="segmented"><button className={cameraMode === '2d' ? 'active' : ''} onClick={() => setCameraMode('2d')}><ScanLine size={14} /> 2D OVERVIEW</button><button className={cameraMode === '3d' ? 'active' : ''} onClick={() => setCameraMode('3d')}><Cuboid size={14} /> 3D FIELD</button></div><button className="round-control" onClick={() => canvas.current?.resetCamera()} title="Reset camera" aria-label="Reset camera"><RotateCcw size={15} /></button></div>
    </section>
    {inspectorOpen && <aside className="inspector-drawer"><div className="drawer-header"><div><span className="eyebrow">OBSERVATORY INSPECTOR</span><h2>Evidence, made legible.</h2></div><div className="drawer-header-actions"><button className="drawer-capture" onClick={() => canvas.current?.screenshot()} title="Capture PNG"><Camera size={14} /> PNG</button><button className="quiet-button" onClick={() => setInspectorOpen(false)} aria-label="Close inspector"><X size={17} /></button></div></div><section className="drawer-readout"><div><span>STABILIZED READOUT</span><strong>{trace.task === 'classification' ? `digit ${trace.readout.prediction}` : trace.readout.prediction}</strong><small>{trace.readout.comparison ?? 'Readout derived from the final belief state.'}</small></div><div className="confidence-ring">{confidence}%</div></section><div className="metric-grid"><Metric label="EVIDENCE" value={numberText(trace.readout.evidence)} suffix="e" tone="ochre" /><Metric label="CONFLICT" value={numberText(trace.readout.uncertainty)} suffix="u" tone="vermilion" /><Metric label="PRECISION" value={numberText(trace.readout.usable_precision)} suffix="π" tone="green" /><Metric label="MISSINGNESS" value={String(missingness)} suffix="%" tone="blue" /></div><section className="drawer-section"><div className="drawer-section-title"><span>FIELD LENS</span><Waypoints size={14} /></div><div className="segmented full"><button className={networkMode === 'pathways' ? 'active' : ''} onClick={() => setNetworkMode('pathways')}><Waypoints size={13} /> PATHWAYS</button><button className={networkMode === 'neighborhoods' ? 'active' : ''} onClick={() => setNetworkMode('neighborhoods')}><Layers3 size={13} /> NEIGHBORHOODS</button></div><button className="focus-path" onClick={focusPath}><Sparkles size={14} /> FOCUS HIGHEST-PRECISION PATH <ChevronRight size={14} /></button></section><section className="drawer-section"><div className="drawer-section-title"><span>OBSERVATION CONTROL</span><Settings2 size={14} /></div><Control label="Corruption severity" value={`${Math.round(severity * 100)}%`}><input type="range" min="0" max=".75" step=".01" value={severity} onChange={(event) => setSeverity(Number(event.target.value))} /></Control><Control label="Evidence stream density" value={`${Math.round(edgeDensity * 100)}%`}><input type="range" min=".15" max="1" step=".01" value={edgeDensity} onChange={(event) => setEdgeDensity(Number(event.target.value))} /></Control><div className="drawer-note"><Eye size={14} /> Reduced motion {reducedMotion ? 'enabled' : 'follows system preference'}. Controls remain active.</div></section><section className="drawer-section"><div className="drawer-section-title"><span>SELECTED CELL</span><span className="cell-tag">cell_{String(cellIndex + 1).padStart(2, '0')}</span></div><div className="cell-location"><Layers3 size={14} /> {layer?.name?.replaceAll('_', ' ') ?? 'belief field'} <span>·</span> index {cellIndex}</div><div className="state-table"><StateRow label="μ / belief" value={numberText(selected.mu)} tone="blue" /><StateRow label="e / evidence" value={numberText(selected.evidence)} tone="ochre" /><StateRow label="u / conflict" value={numberText(selected.uncertainty)} tone="vermilion" /><StateRow label="π / precision" value={numberText(selected.precision)} tone="green" /><StateRow label="c / consensus" value={numberText(selected.consensus)} tone="white" /></div><div className="equation"><span>π = e / (1 + e · u)</span><small>confidence after disagreement</small></div></section><section className="drawer-section"><div className="drawer-section-title"><span>LAYER ISOLATION</span><Layers3 size={14} /></div><button className={`layer-select ${isolatedLayer === null ? 'active' : ''}`} onClick={() => setIsolatedLayer(null)}>all fields {isolatedLayer === null && '✓'}</button>{trace.layers.map((item, index) => <button key={item.name} className={`layer-select ${isolatedLayer === index ? 'active' : ''}`} onClick={() => setIsolatedLayer(index)}><i className={`legend-dot ${index === 0 ? 'blue' : 'ochre'}`} />{item.name.replaceAll('_', ' ')}{isolatedLayer === index && '✓'}</button>)}</section><footer className="drawer-footer"><span><i className="status-dot amber" /> {trace.provenance.source_note}</span><small>{trace.frames.length} states · active {cameraMode.toUpperCase()} renderer</small></footer></aside>}
  </main>;
}

function Metric({ label, value, suffix, tone }: { label: string; value: string; suffix: string; tone: string }) { return <div className="metric"><i className={`metric-pip ${tone}`} /><small>{label}</small><strong>{value}<em>{suffix}</em></strong></div>; }
function Control({ label, value, children }: { label: string; value: string; children: ReactNode }) { return <div className="control"><div><span>{label}</span><b>{value}</b></div>{children}</div>; }
function StateRow({ label, value, tone }: { label: string; value: string; tone: string }) { return <div className="state-row"><span><i className={`legend-dot ${tone}`} />{label}</span><strong>{value}</strong></div>; }

export default App;
