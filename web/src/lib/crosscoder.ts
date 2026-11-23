// Crosscoder model diff reports (X6). The controller stores each report as an
// immutable artifact with a 64-hex ID. This module only reads reports.
import type { Tone } from './format';
import type { Parsed } from './weightspace';

export const CLASSES = ['parent_unique', 'child_unique', 'changed', 'shared', 'dead'] as const;
export type FeatureClass = (typeof CLASSES)[number];

/** Label, tone and plot shape: the shape repeats the class so colour is never the only signal. */
export const CLASS_INFO: Record<FeatureClass, { label: string; tone: Tone; shape: 'circle' | 'square' | 'triangle' | 'diamond' | 'cross' }> = {
  parent_unique: { label: 'Unique to parent', tone: 'warn', shape: 'triangle' },
  child_unique: { label: 'Unique to child', tone: 'info', shape: 'square' },
  changed: { label: 'Changed', tone: 'bad', shape: 'diamond' },
  shared: { label: 'Shared', tone: 'ok', shape: 'circle' },
  dead: { label: 'Dead', tone: 'muted', shape: 'cross' },
};

export interface SideMetrics {
  mse: number;
  explained_variance: number;
  l0: number;
  dead_features: number;
}
export interface CrosscoderFeature {
  feature: number;
  classification: FeatureClass;
  /** Child share of the decoder norm: child / (parent + child), 0 to 1. Null if both norms are zero. */
  child_norm_ratio: number | null;
  decoder_cosine: number | null;
  parent_norm: number;
  child_norm: number;
}
export interface CrosscoderPair {
  parent_job_id: string;
  child_job_id: string;
  split: 'train' | 'heldout';
}
export interface CrosscoderReport {
  id: string;
  schema_version: 1;
  qualification: string;
  parent: { id: string; sha256: string };
  child: { id: string; sha256: string };
  layer: number;
  role: string;
  metrics: { parent: SideMetrics; child: SideMetrics };
  features: CrosscoderFeature[];
  // Optional fields that the controller also reports.
  /** Activation dimension. The feature count is features.length. */
  width?: number;
  k?: number;
  method?: string;
  pairs?: CrosscoderPair[];
  weights_sha256?: string;
  weights_bytes?: number;
}

const ID_RE = /^[0-9a-f]{64}$/;
export const isCrosscoderId = (s: string) => ID_RE.test(s);
export const crosscoderHref = (id: string, rev?: string | null) =>
  `#/edits/crosscoder?${new URLSearchParams({ id, ...(rev ? { rev } : {}) })}`;
export const weightsUrl = (id: string) => `/api/crosscoders/${encodeURIComponent(id)}/safetensors`;

const rec = (v: unknown): Record<string, unknown> | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const numOrNull = (v: unknown) => v === null || num(v);
const ref = (v: unknown) => {
  const r = rec(v);
  return r && typeof r.id === 'string' && typeof r.sha256 === 'string' ? { id: r.id, sha256: r.sha256 } : null;
};
function side(v: unknown): SideMetrics | null {
  const r = rec(v);
  if (!r || !num(r.mse) || !num(r.explained_variance) || !num(r.l0) || !num(r.dead_features)) return null;
  return { mse: r.mse, explained_variance: r.explained_variance, l0: r.l0, dead_features: r.dead_features };
}
function feature(v: unknown): CrosscoderFeature | null {
  const r = rec(v);
  if (!r || !num(r.feature) || !CLASSES.includes(r.classification as FeatureClass)) return null;
  if (!num(r.parent_norm) || !num(r.child_norm) || !numOrNull(r.child_norm_ratio) || !numOrNull(r.decoder_cosine)) return null;
  return {
    feature: r.feature,
    classification: r.classification as FeatureClass,
    child_norm_ratio: r.child_norm_ratio as number | null,
    decoder_cosine: r.decoder_cosine as number | null,
    parent_norm: r.parent_norm,
    child_norm: r.child_norm,
  };
}

function isPair(v: unknown): boolean {
  const r = rec(v);
  return !!r && typeof r.parent_job_id === 'string' && typeof r.child_job_id === 'string' && (r.split === 'train' || r.split === 'heldout');
}
export function pairSplit(pairs: readonly CrosscoderPair[] | undefined): { total: number; train: number; heldout: number } | null {
  if (!pairs) return null;
  const train = pairs.filter((p) => p.split === 'train').length;
  return { total: pairs.length, train, heldout: pairs.length - train };
}

/** Check the report shape. A bad report gives one message for the operator. */
export function parseCrosscoder(raw: unknown): Parsed<CrosscoderReport> {
  const r = rec(raw);
  if (!r) return { ok: false, error: 'The controller did not return a crosscoder report.' };
  if (r.schema_version !== 1) return { ok: false, error: `This view reads crosscoder schema version 1. The report has version ${String(r.schema_version)}.` };
  const parent = ref(r.parent);
  const child = ref(r.child);
  const m = rec(r.metrics);
  const mp = side(m?.parent);
  const mc = side(m?.child);
  if (typeof r.id !== 'string' || !parent || !child || !num(r.layer) || typeof r.role !== 'string') {
    return { ok: false, error: 'The crosscoder report has no parent, child, layer or role.' };
  }
  if (!mp || !mc) return { ok: false, error: 'The crosscoder report has no reconstruction metrics for both sides.' };
  if (!Array.isArray(r.features)) return { ok: false, error: 'The crosscoder report has no feature list.' };
  const features: CrosscoderFeature[] = [];
  for (const f of r.features) {
    const p = feature(f);
    if (!p) return { ok: false, error: 'The crosscoder report has a feature that is not valid.' };
    features.push(p);
  }
  const opt = <T>(v: unknown, ok: (v: unknown) => boolean) => (ok(v) ? (v as T) : undefined);
  return {
    ok: true,
    value: {
      id: r.id,
      schema_version: 1,
      qualification: typeof r.qualification === 'string' ? r.qualification : 'not_release_qualified',
      parent,
      child,
      layer: r.layer,
      role: r.role,
      metrics: { parent: mp, child: mc },
      features,
      width: opt<number>(r.width, num),
      k: opt<number>(r.k, num),
      method: opt<string>(r.method, (v) => typeof v === 'string'),
      pairs: Array.isArray(r.pairs) && r.pairs.every(isPair) ? (r.pairs as CrosscoderPair[]) : undefined,
      weights_sha256: opt<string>(r.weights_sha256, (v) => typeof v === 'string'),
      weights_bytes: opt<number>(r.weights_bytes, num),
    },
  };
}

export function classCounts(features: readonly CrosscoderFeature[]): Record<FeatureClass, number> {
  const out = Object.fromEntries(CLASSES.map((c) => [c, 0])) as Record<FeatureClass, number>;
  for (const f of features) out[f.classification]++;
  return out;
}

export type FeatureSort = 'strength' | 'cosine' | 'ratio' | 'feature';
export const SORT_LABELS: Record<FeatureSort, string> = {
  strength: 'Largest decoder norm',
  ratio: 'Most one-sided norm',
  cosine: 'Lowest decoder cosine',
  feature: 'Feature number',
};


/** Features of one class (or all) whose number starts with the query, in the selected order. */
export function filterFeatures(features: readonly CrosscoderFeature[], cls: FeatureClass | 'all', query: string, sort: FeatureSort): CrosscoderFeature[] {
  const q = query.trim();
  const rows = features.filter((f) => (cls === 'all' || f.classification === cls) && (!q || String(f.feature).startsWith(q)));
  const nullsLast = (a: number | null, b: number | null, cmp: (a: number, b: number) => number) =>
    a === null ? (b === null ? 0 : 1) : b === null ? -1 : cmp(a, b);
  const by: Record<FeatureSort, (a: CrosscoderFeature, b: CrosscoderFeature) => number> = {
    strength: (a, b) => Math.max(b.parent_norm, b.child_norm) - Math.max(a.parent_norm, a.child_norm),
    cosine: (a, b) => nullsLast(a.decoder_cosine, b.decoder_cosine, (x, y) => x - y),
    ratio: (a, b) => nullsLast(a.child_norm_ratio, b.child_norm_ratio, (x, y) => Math.abs(y - 0.5) - Math.abs(x - 0.5)),
    feature: () => 0,
  };
  return rows.sort((a, b) => by[sort](a, b) || a.feature - b.feature);
}

export interface ScatterPoint {
  feature: number;
  cls: FeatureClass;
  /** Child share of the decoder norm, child / (parent + child). */
  x: number;
  y: number;
}
/** Points for features with both a child share and a cosine. */
export function scatterPoints(features: readonly CrosscoderFeature[]): { points: ScatterPoint[]; omitted: number } {
  const points: ScatterPoint[] = [];
  let omitted = 0;
  for (const f of features) {
    if (f.child_norm_ratio === null || f.decoder_cosine === null) omitted++;
    else points.push({ feature: f.feature, cls: f.classification, x: f.child_norm_ratio, y: f.decoder_cosine });
  }
  return { points, omitted };
}
