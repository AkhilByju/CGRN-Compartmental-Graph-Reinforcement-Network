import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { BeliefTrace, CameraMode, NetworkMode, SelectedCell } from './types';
import type { DatasetSceneDefinition, RuntimePick, ScenePalette, SceneRuntimeLike } from './sceneDefinitions';

export interface SceneRuntimeOptions {
  trace: BeliefTrace;
  definition: DatasetSceneDefinition;
  frameIndex: number;
  cameraMode: CameraMode;
  networkMode: NetworkMode;
  selectedCell: SelectedCell;
  isolatedLayer: number | null;
  edgeDensity: number;
  severity: number;
  reducedMotion: boolean;
  onPick: (pick: RuntimePick) => void;
}

type Pulse = { object: THREE.Object3D; speed: number; phase: number; baseOpacity?: number };
type TravelingPulse = {
  curve: THREE.QuadraticBezierCurve3;
  color: THREE.Color;
  speed: number;
  phase: number;
  wobble: number;
  orbitRadius: number;
  orbitSpeed: number;
  orbitPhase: number;
  orbitLean: number;
  index: number;
  previousNormalized: number;
  target?: NodePulseTarget;
};
type NodePulseTarget = {
  cell: string;
  object: THREE.Object3D;
  shell?: THREE.Object3D;
  color: THREE.Color;
  index: number;
  baseScale: THREE.Vector3;
  baseShellScale: THREE.Vector3;
  evidence: number;
  uncertainty: number;
  precision: number;
  strength: number;
  shockProgress: number;
  particleIndex: number;
};

export class SceneRuntime implements SceneRuntimeLike {
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
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly composer: EffectComposer;
  readonly controls: OrbitControls;
  readonly definition: DatasetSceneDefinition;
  private readonly mount: HTMLElement;
  private readonly groups: Record<'background' | 'input' | 'belief-field' | 'readout', THREE.Group>;
  private readonly pickables = new Map<THREE.Object3D, RuntimePick | ((instanceId?: number) => RuntimePick | undefined)>();
  private readonly pulses: Pulse[] = [];
  private readonly travelingPulses: TravelingPulse[] = [];
  private readonly nodePulseTargets = new Map<string, NodePulseTarget>();
  private readonly onPick: (pick: RuntimePick) => void;
  private pulsePoints: THREE.Points | null = null;
  private pulseTrails: THREE.LineSegments | null = null;
  private pulseCapacity = 0;
  private nodePulsePoints: THREE.Points | null = null;
  private nodePulseCapacity = 0;
  private nodeEvidencePoints: THREE.Points | null = null;
  private nodeEvidenceCapacity = 0;
  private readonly evidenceParticlesPerNode = 3;
  private readonly trailPointCount = 6;
  private animationFrame = 0;
  private disposed = false;
  private startedAt = performance.now();
  private lastAnimationTime = this.startedAt;
  private readonly pulseOrbitPoint = new THREE.Vector3();
  private readonly pulseOrbitTangent = new THREE.Vector3();
  private readonly pulseOrbitReference = new THREE.Vector3();
  private readonly pulseOrbitNormal = new THREE.Vector3();
  private readonly pulseOrbitBinormal = new THREE.Vector3();
  private pointerDown: { x: number; y: number } | null = null;

  constructor(mount: HTMLElement, options: SceneRuntimeOptions) {
    this.mount = mount;
    this.trace = options.trace;
    this.definition = options.definition;
    this.palette = options.definition.palette;
    this.frameIndex = options.frameIndex;
    this.networkMode = options.networkMode;
    this.selectedCell = options.selectedCell;
    this.isolatedLayer = options.isolatedLayer;
    this.edgeDensity = options.edgeDensity;
    this.severity = options.severity;
    this.reducedMotion = options.reducedMotion;
    this.onPick = options.onPick;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.palette.background);
    const preset = this.definition.camera;
    this.camera = new THREE.PerspectiveCamera(preset.fov ?? 42, 1, 0.1, 100);
    this.camera.position.set(...preset.position);
    this.camera.lookAt(...preset.target);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    // Keep high-DPI displays from multiplying the bloom/post-processing cost.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor(this.palette.background, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.className = 'three-canvas';
    this.renderer.domElement.setAttribute('aria-label', `${this.definition.title} interactive 3D observatory`);
    this.renderer.domElement.tabIndex = 0;
    mount.appendChild(this.renderer.domElement);
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.16, 0.34, 0.82));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.screenSpacePanning = true;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.minDistance = 3.4;
    this.controls.maxDistance = 12;
    this.controls.target.set(...preset.target);
    this.controls.update();
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.groups = {
      background: new THREE.Group(),
      input: new THREE.Group(),
      'belief-field': new THREE.Group(),
      readout: new THREE.Group(),
    };
    Object.values(this.groups).forEach((group) => this.content.add(group));
    this.definition.buildBackground(this);
    this.definition.buildInput(this.trace, this);
    this.definition.buildBeliefField(this.trace, this);
    this.definition.buildReadout(this.trace, this);
    this.bindPointerEvents();
    this.resize();
    this.animate = this.animate.bind(this);
    if (typeof requestAnimationFrame !== 'undefined') this.animationFrame = requestAnimationFrame(this.animate);
  }

  add(object: THREE.Object3D, group: 'background' | 'input' | 'belief-field' | 'readout' = 'belief-field') {
    this.groups[group].add(object);
  }

  registerPick(object: THREE.Object3D, pick: RuntimePick | ((instanceId?: number) => RuntimePick | undefined)) {
    this.pickables.set(object, pick);
  }

  addPulse(object: THREE.Object3D, speed = 0.1, phase = 0) {
    const material = (object as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
    const firstMaterial = Array.isArray(material) ? material[0] : material;
    this.pulses.push({ object, speed, phase, baseOpacity: firstMaterial && 'opacity' in firstMaterial ? firstMaterial.opacity : undefined });
  }

  registerNodePulse(cell: SelectedCell, object: THREE.Object3D, color: THREE.Color, state = { evidence: 0.5, uncertainty: 0.2, precision: 0.5 }, shell?: THREE.Object3D) {
    const key = `${cell.layerIndex}:${cell.cellIndex}`;
    if (this.nodePulseTargets.has(key)) return;
    this.ensureNodePulseCapacity(this.nodePulseTargets.size + 1);
    this.ensureNodeEvidenceCapacity((this.nodePulseTargets.size + 1) * this.evidenceParticlesPerNode);
    const index = this.nodePulseTargets.size;
    const target: NodePulseTarget = {
      cell: key,
      object,
      shell,
      color: color.clone(),
      index,
      baseScale: object.scale.clone(),
      baseShellScale: shell?.scale.clone() ?? new THREE.Vector3(1, 1, 1),
      evidence: state.evidence,
      uncertainty: state.uncertainty,
      precision: state.precision,
      strength: 0,
      shockProgress: 1,
      particleIndex: index * this.evidenceParticlesPerNode,
    };
    this.nodePulseTargets.set(key, target);
    this.updateNodePulsePoint(target);
    this.updateEvidenceParticles(target, 0);
    this.commitNodePulseBuffers();
  }

  addTravelingPulse(curve: THREE.QuadraticBezierCurve3, color: string, speed = 0.1, phase = 0, targetCell?: SelectedCell) {
    // Keep all packets in two shared GPU objects. The scene can therefore
    // show many pulses without creating one mesh, material, and draw call per
    // packet.
    [
      { offset: 0, speedScale: 1, wobble: 1 },
      { offset: 0.14, speedScale: 0.78, wobble: 1.23 },
      { offset: 0.27, speedScale: 1.18, wobble: 0.86 },
    ].forEach(({ offset, speedScale, wobble }) => {
      const packetPhase = phase + offset;
      const pulseSeed = this.travelingPulses.length + 1;
      const seededNoise = (salt: number) => {
        const value = Math.sin(pulseSeed * 12.9898 + packetPhase * 78.233 + salt * 37.719) * 43758.5453;
        return value - Math.floor(value);
      };
      this.ensurePulseCapacity(this.travelingPulses.length + 1);
      const target = targetCell ? this.nodePulseTargets.get(`${targetCell.layerIndex}:${targetCell.cellIndex}`) : undefined;
      const pulse: TravelingPulse = {
        curve,
        color: new THREE.Color(color),
        speed: speed * speedScale,
        phase: packetPhase,
        wobble,
        // Keep the orbit close to the edge: visible as motion, but still
        // visually attached to the connection it is traveling through.
        orbitRadius: 0.008 + seededNoise(1) * 0.018,
        // The orbit is intentionally much faster than the forward travel,
        // making each packet read as a quick helix around the connection.
        orbitSpeed: 5.5 + seededNoise(2) * 4.5 + wobble * 0.15,
        orbitPhase: seededNoise(3) * Math.PI * 2,
        orbitLean: seededNoise(4),
        index: this.travelingPulses.length,
        previousNormalized: packetPhase % 1,
      };
      if (target) pulse.target = target;
      this.travelingPulses.push(pulse);
      this.updateTravelingPulse(pulse, 0);
    });
    this.commitPulseBuffers();
  }

  resize() {
    const box = this.mount.getBoundingClientRect();
    const width = Math.max(1, box.width || this.mount.clientWidth || 1);
    const height = Math.max(1, box.height || this.mount.clientHeight || 1);
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  resetCamera() {
    const preset = this.definition.camera;
    this.camera.position.set(...preset.position);
    this.controls.target.set(...preset.target);
    this.controls.update();
    this.render();
  }

  render() {
    if (!this.disposed) this.composer.render();
  }

  screenshot(fileName = `belief-observatory-${this.trace.dataset}-3d.png`) {
    this.render();
    const link = document.createElement('a');
    link.download = fileName;
    link.href = this.renderer.domElement.toDataURL('image/png');
    link.click();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(this.animationFrame);
    this.renderer.domElement.removeEventListener('pointerdown', this.handlePointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this.handlePointerUp);
    this.renderer.domElement.removeEventListener('contextmenu', this.handleContextMenu);
    this.controls.dispose();
    this.disposeObject(this.scene);
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.pickables.clear();
    this.pulses.length = 0;
    this.travelingPulses.length = 0;
    this.pulsePoints = null;
    this.pulseTrails = null;
    this.pulseCapacity = 0;
    this.nodePulsePoints = null;
    this.nodePulseCapacity = 0;
    this.nodeEvidencePoints = null;
    this.nodeEvidenceCapacity = 0;
    this.nodePulseTargets.clear();
  }

  private animate(time: number) {
    if (this.disposed) return;
    const elapsed = (time - this.startedAt) / 1000;
    const delta = Math.min(0.05, Math.max(0, (time - this.lastAnimationTime) / 1000));
    this.lastAnimationTime = time;
    this.controls.update();
    if (!this.reducedMotion) {
      this.pulses.forEach(({ object, speed, phase, baseOpacity }) => {
        const value = 0.5 + 0.5 * Math.sin(elapsed * speed * 8 + phase * Math.PI * 2);
        const material = (object as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        const firstMaterial = Array.isArray(material) ? material[0] : material;
        if (firstMaterial && baseOpacity !== undefined && 'opacity' in firstMaterial) firstMaterial.opacity = baseOpacity * (0.76 + value * 0.35);
      });
      this.travelingPulses.forEach((pulse) => this.updateTravelingPulse(pulse, elapsed));
      this.updateNodePulses(elapsed, delta);
      this.commitPulseBuffers();
      this.commitNodePulseBuffers();
    }
    this.render();
    if (typeof requestAnimationFrame !== 'undefined') this.animationFrame = requestAnimationFrame(this.animate);
  }

  private orbitingPulsePoint(pulse: TravelingPulse, normalized: number, time: number) {
    const wrapped = ((normalized % 1) + 1) % 1;
    const tangentSample = Math.min(0.998, Math.max(0.002, wrapped));
    this.pulseOrbitPoint.copy(pulse.curve.getPointAt(wrapped));
    this.pulseOrbitTangent.copy(pulse.curve.getTangentAt(tangentSample)).normalize();

    // Build a local frame around the curve so each pulse can orbit its edge
    // instead of being locked exactly onto the centerline.
    if (Math.abs(this.pulseOrbitTangent.y) > 0.88) {
      this.pulseOrbitReference.set(1, 0, 0);
    } else {
      this.pulseOrbitReference.set(0, 1, 0);
    }
    this.pulseOrbitNormal.crossVectors(this.pulseOrbitTangent, this.pulseOrbitReference).normalize();
    this.pulseOrbitBinormal.crossVectors(this.pulseOrbitTangent, this.pulseOrbitNormal).normalize();

    // Keep both components constant-speed: the point advances uniformly on
    // the edge while its cross-section rotates at a separate, faster rate.
    // Per-pulse phase/radius/lean differences keep the packets from syncing
    // into one identical tube.
    const angle = time * pulse.orbitSpeed + pulse.orbitPhase;
    const radius = pulse.orbitRadius;
    const sideways = Math.cos(angle) * radius;
    const vertical = Math.sin(angle + pulse.wobble * 0.12) * radius * (0.74 + pulse.orbitLean * 0.24);
    this.pulseOrbitPoint.addScaledVector(this.pulseOrbitNormal, sideways);
    this.pulseOrbitPoint.addScaledVector(this.pulseOrbitBinormal, vertical);
    return this.pulseOrbitPoint;
  }

  private updateTravelingPulse(pulse: TravelingPulse, progress: number) {
    const travel = progress * pulse.speed + pulse.phase;
    const normalized = ((travel % 1) + 1) % 1;
    if (pulse.target && pulse.previousNormalized < 0.9 && normalized >= 0.9) {
      pulse.target.strength = Math.min(1.5, pulse.target.strength + 0.8);
      pulse.target.shockProgress = 0;
    }
    pulse.previousNormalized = normalized;
    if (!this.pulsePoints || !this.pulseTrails) return;
    const corePositions = this.pulsePoints.geometry.getAttribute('position') as THREE.BufferAttribute;
    const coreColors = this.pulsePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute;
    const coreSizes = this.pulsePoints.geometry.getAttribute('size') as THREE.BufferAttribute;
    const corePoint = this.orbitingPulsePoint(pulse, normalized, progress);
    corePositions.setXYZ(pulse.index, corePoint.x, corePoint.y, corePoint.z);
    coreColors.setXYZ(pulse.index, pulse.color.r, pulse.color.g, pulse.color.b);
    coreSizes.setX(pulse.index, 0.062 + Math.sin(progress * 8 + pulse.phase) * 0.006);

    const trailPositions = this.pulseTrails.geometry.getAttribute('position') as THREE.BufferAttribute;
    const trailColors = this.pulseTrails.geometry.getAttribute('color') as THREE.BufferAttribute;
    const tailLength = 0.12;
    const segmentVertexOffset = pulse.index * (this.trailPointCount - 1) * 2;
    for (let index = 0; index < this.trailPointCount - 1; index += 1) {
      const headProgress = normalized - (index / (this.trailPointCount - 1)) * tailLength;
      const tailProgress = normalized - ((index + 1) / (this.trailPointCount - 1)) * tailLength;
      const headTrailOffset = (index / (this.trailPointCount - 1)) * tailLength;
      const tailTrailOffset = ((index + 1) / (this.trailPointCount - 1)) * tailLength;
      const head = this.orbitingPulsePoint(pulse, headProgress, progress - headTrailOffset / Math.max(0.06, pulse.speed));
      const headX = head.x;
      const headY = head.y;
      const headZ = head.z;
      const tail = this.orbitingPulsePoint(pulse, tailProgress, progress - tailTrailOffset / Math.max(0.06, pulse.speed));
      const headVertex = segmentVertexOffset + index * 2;
      const tailVertex = headVertex + 1;
      const headFade = 1 - index / (this.trailPointCount - 1);
      const tailFade = 1 - (index + 1) / (this.trailPointCount - 1);
      trailPositions.setXYZ(headVertex, headX, headY, headZ);
      trailPositions.setXYZ(tailVertex, tail.x, tail.y, tail.z);
      trailColors.setXYZ(headVertex, pulse.color.r * headFade, pulse.color.g * headFade, pulse.color.b * headFade);
      trailColors.setXYZ(tailVertex, pulse.color.r * tailFade, pulse.color.g * tailFade, pulse.color.b * tailFade);
    }
  }

  private updateNodePulses(elapsed: number, delta: number) {
    this.nodePulseTargets.forEach((target) => {
      target.strength = Math.max(0, target.strength - delta * 2.8);
      target.shockProgress = Math.min(1, target.shockProgress + delta * (1.7 + target.precision * 0.8));
      const idle = 0.012 + (Math.sin(elapsed * 1.8 + target.index * 0.73) + 1) * 0.006;
      const wave = target.strength * (0.82 + Math.sin(elapsed * 12 + target.index) * 0.08);
      const impact = 1 - target.shockProgress;
      target.object.scale.copy(target.baseScale).multiplyScalar(1 + idle + wave * 0.28 + impact * 0.1);
      const material = (target.object as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
      if (material?.isMeshStandardMaterial) material.emissiveIntensity = 0.5 + target.uncertainty * 0.3 + idle * 2 + wave * 2.4 + impact * 1.8;
      if (target.shell) {
        target.shell.rotation.x += delta * (0.35 + target.precision * 0.4);
        target.shell.rotation.y += delta * (0.55 + target.uncertainty * 0.8);
        target.shell.scale.copy(target.baseShellScale).multiplyScalar(1 + target.uncertainty * 0.08 + impact * 0.55);
        const shellMaterial = (target.shell as THREE.Mesh).material as THREE.MeshBasicMaterial | undefined;
        if (shellMaterial?.isMeshBasicMaterial) shellMaterial.opacity = 0.065 + target.uncertainty * 0.14 + impact * 0.24;
      }
      this.updateNodePulsePoint(target);
      this.updateEvidenceParticles(target, elapsed);
    });
  }

  private updateNodePulsePoint(target: NodePulseTarget) {
    if (!this.nodePulsePoints) return;
    const positions = this.nodePulsePoints.geometry.getAttribute('position') as THREE.BufferAttribute;
    const colors = this.nodePulsePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute;
    const sizes = this.nodePulsePoints.geometry.getAttribute('size') as THREE.BufferAttribute;
    positions.setXYZ(target.index, target.object.position.x, target.object.position.y, target.object.position.z);
    const intensity = 0.17 + target.strength * 0.82 + (1 - target.shockProgress) * 0.4;
    colors.setXYZ(target.index, target.color.r * intensity, target.color.g * intensity, target.color.b * intensity);
    sizes.setX(target.index, 0.016 + target.strength * 0.12 + (1 - target.shockProgress) * 0.24);
  }

  private ensureNodePulseCapacity(required: number) {
    if (required <= this.nodePulseCapacity) return;
    const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(required)));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(capacity), 1));
    geometry.setDrawRange(0, this.nodePulseTargets.size);
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: 'attribute float size; attribute vec3 aColor; varying vec3 vColor; void main() { vColor = aColor; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_PointSize = max(5.0, size * (350.0 / max(1.0, -mvPosition.z))); gl_Position = projectionMatrix * mvPosition; }',
      fragmentShader: 'varying vec3 vColor; void main() { float distanceToCenter = distance(gl_PointCoord, vec2(0.5)); float halo = smoothstep(0.5, 0.02, distanceToCenter); float core = smoothstep(0.24, 0.02, distanceToCenter); float ring = smoothstep(0.47, 0.39, distanceToCenter) * smoothstep(0.5, 0.47, distanceToCenter); gl_FragColor = vec4(vColor * (0.62 + core * 0.9 + ring * 0.6), halo * 0.48 + ring * 0.92); }',
    });
    const nextPoints = new THREE.Points(geometry, material);
    nextPoints.frustumCulled = false;
    this.add(nextPoints, 'belief-field');
    this.nodePulsePoints?.removeFromParent();
    this.nodePulsePoints?.geometry.dispose();
    (this.nodePulsePoints?.material as THREE.Material | undefined)?.dispose();
    this.nodePulsePoints = nextPoints;
    this.nodePulseCapacity = capacity;
    this.nodePulseTargets.forEach((target) => this.updateNodePulsePoint(target));
  }

  private updateEvidenceParticles(target: NodePulseTarget, elapsed: number) {
    if (!this.nodeEvidencePoints) return;
    const positions = this.nodeEvidencePoints.geometry.getAttribute('position') as THREE.BufferAttribute;
    const colors = this.nodeEvidencePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute;
    const sizes = this.nodeEvidencePoints.geometry.getAttribute('size') as THREE.BufferAttribute;
    const spread = 0.026 + target.evidence * 0.055 + target.uncertainty * 0.018;
    const orbit = elapsed * (0.7 + target.precision * 1.5 + target.uncertainty * 0.45) + target.index * 1.91;
    for (let particle = 0; particle < this.evidenceParticlesPerNode; particle += 1) {
      const angle = orbit * (particle % 2 === 0 ? 1 : -1) + particle * (Math.PI * 2 / this.evidenceParticlesPerNode);
      const wobble = 1 + Math.sin(elapsed * 2.4 + target.index + particle * 2.1) * (0.12 + target.uncertainty * 0.1);
      const index = target.particleIndex + particle;
      positions.setXYZ(
        index,
        target.object.position.x + Math.cos(angle) * spread * wobble,
        target.object.position.y + Math.sin(angle) * spread * wobble,
        target.object.position.z + Math.sin(angle * 1.7) * spread * (0.55 + target.uncertainty * 0.8),
      );
      const intensity = 0.42 + target.evidence * 0.72 + (1 - target.shockProgress) * 0.35;
      colors.setXYZ(index, target.color.r * intensity, target.color.g * intensity, target.color.b * intensity);
      sizes.setX(index, 0.012 + target.precision * 0.015 + target.uncertainty * 0.004);
    }
  }

  private ensureNodeEvidenceCapacity(required: number) {
    if (required <= this.nodeEvidenceCapacity) return;
    const capacity = Math.max(192, 2 ** Math.ceil(Math.log2(required)));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(capacity), 1));
    geometry.setDrawRange(0, this.nodePulseTargets.size * this.evidenceParticlesPerNode);
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: 'attribute float size; attribute vec3 aColor; varying vec3 vColor; void main() { vColor = aColor; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_PointSize = max(4.0, size * (350.0 / max(1.0, -mvPosition.z))); gl_Position = projectionMatrix * mvPosition; }',
      fragmentShader: 'varying vec3 vColor; void main() { float d = distance(gl_PointCoord, vec2(0.5)); float sparkle = smoothstep(0.24, 0.02, d); float halo = smoothstep(0.5, 0.04, d); gl_FragColor = vec4(vColor * (0.72 + sparkle * 1.2), halo * 0.82); }',
    });
    const nextPoints = new THREE.Points(geometry, material);
    nextPoints.frustumCulled = false;
    this.add(nextPoints, 'belief-field');
    this.nodeEvidencePoints?.removeFromParent();
    this.nodeEvidencePoints?.geometry.dispose();
    (this.nodeEvidencePoints?.material as THREE.Material | undefined)?.dispose();
    this.nodeEvidencePoints = nextPoints;
    this.nodeEvidenceCapacity = capacity;
    this.nodePulseTargets.forEach((target) => this.updateEvidenceParticles(target, 0));
  }

  private commitNodePulseBuffers() {
    if (!this.nodePulsePoints) return;
    this.nodePulsePoints.geometry.setDrawRange(0, this.nodePulseTargets.size);
    (this.nodePulsePoints.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (this.nodePulsePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute).needsUpdate = true;
    (this.nodePulsePoints.geometry.getAttribute('size') as THREE.BufferAttribute).needsUpdate = true;
    if (this.nodeEvidencePoints) {
      this.nodeEvidencePoints.geometry.setDrawRange(0, this.nodePulseTargets.size * this.evidenceParticlesPerNode);
      (this.nodeEvidencePoints.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
      (this.nodeEvidencePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute).needsUpdate = true;
      (this.nodeEvidencePoints.geometry.getAttribute('size') as THREE.BufferAttribute).needsUpdate = true;
    }
  }

  private ensurePulseCapacity(required: number) {
    if (required <= this.pulseCapacity) return;
    const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(required)));
    const coreGeometry = new THREE.BufferGeometry();
    coreGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    coreGeometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(capacity * 3), 3));
    coreGeometry.setAttribute('size', new THREE.BufferAttribute(new Float32Array(capacity), 1));
    coreGeometry.setDrawRange(0, this.travelingPulses.length);
    const coreMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: 'attribute float size; attribute vec3 aColor; varying vec3 vColor; void main() { vColor = aColor; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_PointSize = max(4.0, size * (350.0 / max(1.0, -mvPosition.z))); gl_Position = projectionMatrix * mvPosition; }',
      fragmentShader: 'varying vec3 vColor; void main() { float distanceToCenter = distance(gl_PointCoord, vec2(0.5)); float halo = smoothstep(0.5, 0.02, distanceToCenter); float core = smoothstep(0.24, 0.02, distanceToCenter); gl_FragColor = vec4(vColor * (0.78 + core * 0.9), halo * 0.9); }',
    });
    const nextPoints = new THREE.Points(coreGeometry, coreMaterial);
    nextPoints.frustumCulled = false;
    this.add(nextPoints, 'belief-field');
    this.pulsePoints?.removeFromParent();
    this.pulsePoints?.geometry.dispose();
    (this.pulsePoints?.material as THREE.Material | undefined)?.dispose();
    this.pulsePoints = nextPoints;

    const trailVertexCount = (this.trailPointCount - 1) * 2;
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(capacity * trailVertexCount * 3), 3));
    trailGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(capacity * trailVertexCount * 3), 3));
    trailGeometry.setDrawRange(0, this.travelingPulses.length * trailVertexCount);
    const trailMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.52, blending: THREE.AdditiveBlending, depthWrite: false });
    const nextTrails = new THREE.LineSegments(trailGeometry, trailMaterial);
    nextTrails.frustumCulled = false;
    this.add(nextTrails, 'belief-field');
    this.pulseTrails?.removeFromParent();
    this.pulseTrails?.geometry.dispose();
    (this.pulseTrails?.material as THREE.Material | undefined)?.dispose();
    this.pulseTrails = nextTrails;
    this.pulseCapacity = capacity;
    this.travelingPulses.forEach((pulse) => this.updateTravelingPulse(pulse, 0));
  }

  private commitPulseBuffers() {
    if (!this.pulsePoints || !this.pulseTrails) return;
    const trailVertexCount = (this.trailPointCount - 1) * 2;
    this.pulsePoints.geometry.setDrawRange(0, this.travelingPulses.length);
    this.pulseTrails.geometry.setDrawRange(0, this.travelingPulses.length * trailVertexCount);
    (this.pulsePoints.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (this.pulsePoints.geometry.getAttribute('aColor') as THREE.BufferAttribute).needsUpdate = true;
    (this.pulsePoints.geometry.getAttribute('size') as THREE.BufferAttribute).needsUpdate = true;
    (this.pulseTrails.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (this.pulseTrails.geometry.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  }

  private readonly handlePointerDown = (event: PointerEvent) => { if (event.button === 0) this.pointerDown = { x: event.clientX, y: event.clientY }; };

  private readonly handlePointerUp = (event: PointerEvent) => {
    if (event.button !== 0 || !this.pointerDown || Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y) > 6) { this.pointerDown = null; return; }
    const box = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
    const raycaster = new THREE.Raycaster(); raycaster.setFromCamera(pointer, this.camera);
    const hits = raycaster.intersectObjects([...this.pickables.keys()], true);
    const hit = hits[0];
    if (hit) {
      let target: THREE.Object3D | undefined = hit.object;
      while (target && !this.pickables.has(target)) target = target.parent ?? undefined;
      const pick = target ? this.pickables.get(target) : undefined;
      const intersectionIndex = hit.object.type === 'Points' ? hit.index : hit.index ?? hit.instanceId;
      const resolved = typeof pick === 'function' ? pick(intersectionIndex) : pick;
      if (resolved) this.onPick(resolved);
    }
    this.pointerDown = null;
  };

  private readonly handleContextMenu = (event: MouseEvent) => event.preventDefault();

  private bindPointerEvents() {
    this.renderer.domElement.addEventListener('pointerdown', this.handlePointerDown);
    this.renderer.domElement.addEventListener('pointerup', this.handlePointerUp);
    this.renderer.domElement.addEventListener('contextmenu', this.handleContextMenu);
  }

  private disposeObject(object: THREE.Object3D) {
    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      mesh.geometry?.dispose();
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) material.forEach((item) => item.dispose()); else material?.dispose();
    });
  }
}
