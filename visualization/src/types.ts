export type DatasetId = 'mnist' | 'aps' | 'air_quality';
export type TraceKind = 'demo_trace' | 'trained_trace';
export type ViewMode = 'presentation' | 'mechanism';
export type CameraMode = '2d' | '3d';
export type NetworkMode = 'pathways' | 'neighborhoods';
export type ConnectionKind = 'support' | 'conflict' | 'shared_evidence';

export interface TopConnection {
  source: number;
  weight: number;
  signed_influence?: number;
  kind?: Exclude<ConnectionKind, 'shared_evidence'>;
}

export interface NeighborhoodLink {
  source: number;
  target: number;
  similarity: number;
  signed_influence?: number;
  kind?: ConnectionKind;
}

export interface NodeGroup {
  id: number;
  label: string;
  cells: number[];
  upstream_sources?: number[];
}

export interface CellState {
  mu: number;
  evidence: number;
  uncertainty: number;
  precision: number;
  consensus: number;
  top_connections: TopConnection[];
}

export interface LayerState {
  name: string;
  mu: number[];
  evidence: number[];
  uncertainty: number[];
  precision: number[];
  consensus: number[];
  top_connections: TopConnection[][];
  neighborhood_links?: NeighborhoodLink[];
  node_groups?: number[];
  groups?: NodeGroup[];
}

export interface TraceFrame {
  progress: number;
  active_stage: string;
  visible_layers: number;
  energy: number;
}

export interface TraceInput {
  raw: number[];
  observed: number[];
  reliability: number[];
  shape?: [number, number];
  labels?: string[];
}

export interface Readout {
  logits: number[];
  prediction: number | string;
  uncertainty: number;
  evidence: number;
  usable_precision: number;
  comparison?: string;
}

export interface TraceProvenance {
  checkpoint: string | null;
  git_commit: string;
  export_timestamp: string;
  trace_kind: TraceKind;
  source_note: string;
}

export interface BeliefTrace {
  schema_version: 1 | 2;
  dataset: DatasetId;
  dataset_label: string;
  task: 'classification' | 'regression';
  model_family: 'cellv0.3';
  sample_id: string;
  seed: number;
  corruption: {
    family: string;
    severity: number;
    missing_count: number;
  };
  input: TraceInput;
  layers: LayerState[];
  readout: Readout;
  frames: TraceFrame[];
  metrics: Record<string, string | number>;
  provenance: TraceProvenance;
}

export interface SelectedCell {
  layerIndex: number;
  cellIndex: number;
}
