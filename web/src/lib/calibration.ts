// Read-only viewer input for two offline derived reports:
//   expert_usage.py    -> brewery-expert-usage-v1
//   profile_preview.py -> brewery.profile_preview.v1
//   Halogen timing export -> brewery-halogen-speculation-v1 (draft acceptance counts only)
// Files are read in the browser only. Nothing is uploaded and no model is
// loaded. Neither report is a calibration or quality result, and a preview
// is never a runnable profile. The two reports are never linked: a preview
// does not name telemetry, and a usage report does not name a plan.

import { obj } from './reports';
import { formatBytes } from './format';

export const USAGE_SCHEMA = 'brewery-expert-usage-v1';
export const PREVIEW_SCHEMA = 'brewery.profile_preview.v1';
export const SPECULATION_SCHEMA = 'brewery-halogen-speculation-v1';
export const SPECULATION_METHODS = ['head', 'lookup', 'grammar'] as const;
/** Derived reports are small; a larger file is most likely a raw trace. */
export const MAX_FILE_BYTES = 16 * 1024 * 1024;
/** Rendering caps. The report itself is never truncated. */
export const EXPERTS_SHOWN = 8;
export const CHANGES_PAGE = 100;

const n = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const s = (v: unknown): string | null => (typeof v === 'string' ? v : null);
const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);

/** Scalar fields of a flat map, in source order. Nested values are omitted. */
function scalars(v: unknown): [string, string][] {
  const o = obj(v);
  if (!o) return [];
  const out: [string, string][] = [];
  for (const [k, x] of Object.entries(o)) {
    if (typeof x === 'string' || typeof x === 'boolean' || (typeof x === 'number' && Number.isFinite(x))) out.push([k, String(x)]);
  }
  return out;
}

// ---- Usage ----

export interface ExpertRow {
  expert: number;
  selections: number | null;
  observed_positions: number | null;
  /** selections / observed routing positions in this layer and phase. */
  selection_fraction: number | null;
  weight_samples: number | null;
  weight_sum: number | null;
  /** Mean captured router coefficient of the selections. Not a probability, not a model weight. */
  weight_mean: number | null;
}

export interface Phase {
  observed_positions: number | null;
  missing_positions: number | null;
  selections: number | null;
  experts: ExpertRow[];
}

export interface UsageLayer {
  layer: number;
  prefill: Phase | null;
  decode: Phase | null;
}

export interface UsageReport {
  identity: { model: [string, string][]; runtime: [string, string][]; tokenizer: [string, string][] };
  traces: { sha256: string | null; prefill_tokens: number | null; decode_tokens: number | null }[];
  totals: Record<'trace_count' | 'prefill_tokens' | 'decode_tokens' | 'observed_positions' | 'missing_positions' | 'selections', number | null>;
  layers: UsageLayer[];
  caveats: string[];
}

function phase(v: unknown): Phase | null {
  const p = obj(v);
  if (!p) return null;
  const experts: ExpertRow[] = (Array.isArray(p.experts) ? p.experts : [])
    .map(obj)
    .filter((e): e is Record<string, unknown> => !!e && n(e.expert) !== null)
    .map((e) => ({
      expert: e.expert as number,
      selections: n(e.selections),
      observed_positions: n(e.observed_positions),
      selection_fraction: n(e.selection_fraction),
      weight_samples: n(e.weight_samples),
      weight_sum: n(e.weight_sum),
      weight_mean: n(e.weight_mean),
    }));
  return { observed_positions: n(p.observed_positions), missing_positions: n(p.missing_positions), selections: n(p.selections), experts };
}

function readUsage(r: Record<string, unknown>): UsageReport | string {
  if (!Array.isArray(r.layers)) return 'The usage report has no layer list.';
  const layers: UsageLayer[] = [];
  for (const x of r.layers) {
    const l = obj(x);
    if (!l || n(l.layer) === null) continue;
    layers.push({ layer: l.layer as number, prefill: phase(l.prefill), decode: phase(l.decode) });
  }
  layers.sort((a, b) => a.layer - b.layer);
  const t = obj(r.totals) ?? {};
  return {
    identity: { model: scalars(r.model), runtime: scalars(r.runtime), tokenizer: scalars(r.tokenizer) },
    traces: (Array.isArray(r.traces) ? r.traces : []).map(obj).filter((x): x is Record<string, unknown> => !!x).map((x) => ({ sha256: s(x.sha256), prefill_tokens: n(x.prefill_tokens), decode_tokens: n(x.decode_tokens) })),
    totals: {
      trace_count: n(t.trace_count),
      prefill_tokens: n(t.prefill_tokens),
      decode_tokens: n(t.decode_tokens),
      observed_positions: n(t.observed_positions),
      missing_positions: n(t.missing_positions),
      selections: n(t.selections),
    },
    layers,
    caveats: strings(r.caveats),
  };
}

/** Observed share of routing positions; null when unknown or no positions. */
export function usageCoverage(p: { observed_positions: number | null; missing_positions: number | null } | null): number | null {
  if (!p || p.observed_positions === null || p.missing_positions === null) return null;
  const all = p.observed_positions + p.missing_positions;
  return all > 0 ? p.observed_positions / all : null;
}

export function topExperts(experts: ExpertRow[], limit = EXPERTS_SHOWN): { shown: ExpertRow[]; hidden: number } {
  const sorted = [...experts].sort((a, b) => (b.selections ?? -1) - (a.selections ?? -1) || a.expert - b.expert);
  return { shown: sorted.slice(0, limit), hidden: Math.max(0, sorted.length - limit) };
}

// ---- Preview ----

export interface TensorSide {
  encoding: string | null;
  shape: number[] | null;
  payload_bytes: number | null;
}

export interface Change {
  filename: string;
  tensor: string;
  layer: number | null;
  role: string;
  kind: string;
  before: TensorSide | null;
  after: TensorSide | null;
  /** after - before payload bytes; null when either side is unknown. */
  delta: number | null;
  encodingChanged: boolean | null;
  shapeChanged: boolean | null;
}

export interface PreviewReport {
  /** Always false in the UI. */
  executable: false;
  release_eligible: false;
  /** The file claimed executable or release-eligible. Shown as a warning, never honoured. */
  claimedRunnable: boolean;
  /** source_compatibility scalars, plus one level of nested scalars as parent.key. */
  compatibility: [string, string][];
  limits: string[];
  recipeSha256: string | null;
  base: [string, string][];
  alternate: [string, string][];
  changes: Change[];
  size: { base: number | null; predicted: number | null; delta: number | null };
  files: { filename: string; base: number | null; predicted: number | null; delta: number | null; baseOverhead: number | null; predictedOverhead: number | null }[];
}

function side(v: unknown): TensorSide | null {
  const o = obj(v);
  if (!o) return null;
  const shape = Array.isArray(o.shape) && o.shape.every((d) => n(d) !== null) ? (o.shape as number[]) : null;
  return { encoding: s(o.encoding), shape, payload_bytes: n(o.payload_bytes) };
}

function readPreview(r: Record<string, unknown>): PreviewReport | string {
  const recipe = obj(r.resolved_recipe);
  if (!recipe || !Array.isArray(recipe.changes)) return 'The profile preview has no resolved recipe changes.';
  const changes: Change[] = recipe.changes
    .map(obj)
    .filter((c): c is Record<string, unknown> => !!c)
    .map((c) => {
      const before = side(c.before);
      const after = side(c.after);
      const bb = before?.payload_bytes ?? null;
      const ab = after?.payload_bytes ?? null;
      return {
        filename: s(c.filename) ?? '?',
        tensor: s(c.tensor) ?? '?',
        layer: n(c.layer),
        role: s(c.role) ?? '',
        kind: s(c.kind) ?? '?',
        before,
        after,
        delta: bb !== null && ab !== null ? ab - bb : null,
        encodingChanged: before?.encoding != null && after?.encoding != null ? before.encoding !== after.encoding : null,
        shapeChanged: before?.shape && after?.shape ? before.shape.join('x') !== after.shape.join('x') : null,
      };
    });
  const size = obj(r.size) ?? {};
  return {
    executable: false,
    release_eligible: false,
    claimedRunnable: r.executable === true || r.release_eligible === true,
    compatibility: [
      ...scalars(r.source_compatibility),
      ...Object.entries(obj(r.source_compatibility) ?? {}).flatMap(([k, v]) => scalars(v).map(([kk, vv]): [string, string] => [`${k}.${kk}`, vv])),
    ],
    limits: strings(r.limits),
    recipeSha256: s(r.resolved_recipe_sha256),
    base: scalars(recipe.base),
    alternate: scalars(recipe.alternate),
    changes,
    size: { base: n(size.base_hgn_bytes), predicted: n(size.predicted_hgn_bytes), delta: n(size.delta_hgn_bytes) },
    files: (Array.isArray(size.files) ? size.files : [])
      .map(obj)
      .filter((f): f is Record<string, unknown> => !!f)
      .map((f) => ({
        filename: s(f.filename) ?? '?',
        base: n(f.base_hgn_bytes),
        predicted: n(f.predicted_hgn_bytes),
        delta: n(f.delta_hgn_bytes),
        baseOverhead: n(f.base_overhead_bytes),
        predictedOverhead: n(f.predicted_overhead_bytes),
      })),
  };
}

// ---- Speculation ----

export interface MethodStats {
  rounds: number | null;
  drafted: number | null;
  accepted: number | null;
  /** As reported: accepted / drafted, null when there is no denominator. */
  acceptance_fraction: number | null;
  /** As reported: accepted / rounds, null when there is no denominator. */
  accepted_per_round: number | null;
}

export interface SpeculationReport {
  producerVersion: string | null;
  versionEvidence: string | null;
  model: string | null;
  inputSha256: string | null;
  checkpointSha256: string | null;
  /** The export has per-depth acceptance or expert routing data. v1 exports state null; the viewer does not show them. */
  hasByDepth: boolean;
  hasExpertRouting: boolean;
  requests: number | null;
  withSpeculation: number | null;
  missingSpeculation: number | null;
  /** Fixed order; stats null when the method was not observed. Unknown method keys are ignored. */
  methods: { name: (typeof SPECULATION_METHODS)[number]; stats: MethodStats | null }[];
}

function readSpeculation(r: Record<string, unknown>): SpeculationReport | string {
  const m = obj(r.methods);
  if (!m) return 'The speculation report has no methods object.';
  return {
    producerVersion: s(r.producer_version),
    versionEvidence: s(r.version_evidence),
    model: s(r.model),
    inputSha256: s(r.input_sha256),
    checkpointSha256: s(r.checkpoint_sha256),
    hasByDepth: r.acceptance_by_depth != null,
    hasExpertRouting: r.expert_routing != null,
    requests: n(r.requests),
    withSpeculation: n(r.total_with_speculation),
    missingSpeculation: n(r.missing_speculation),
    methods: SPECULATION_METHODS.map((name) => {
      const x = obj(m[name]);
      return {
        name,
        stats: x
          ? { rounds: n(x.rounds), drafted: n(x.drafted), accepted: n(x.accepted), acceptance_fraction: n(x.acceptance_fraction), accepted_per_round: n(x.accepted_per_round) }
          : null,
      };
    }),
  };
}

// ---- Detection ----

export type Parsed =
  | {kind: 'expertMap'; report: {source: string; method: string; layers: {layer: number; order: number[]}[]}; fileSha?: string}
  | { kind: 'usage'; report: UsageReport }
  | { kind: 'preview'; report: PreviewReport }
  | { kind: 'speculation'; report: SpeculationReport }
  | { kind: 'error'; message: string };

/** Short, content-free schema label for messages. */
function schemaLabel(v: unknown): string {
  if (typeof v !== 'string') return 'no schema field';
  return /^[A-Za-z0-9._-]{1,64}$/.test(v) ? `schema ${v}` : 'an unrecognised schema';
}

export function parseReportText(text: string): Parsed {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { kind: 'error', message: 'The file is not valid JSON.' };
  }
  const r = obj(data);
  if (!r) return { kind: 'error', message: 'The file is not a JSON object.' };
  if (r.schema === 'brewery.expert-map.v1') {
    const source = r.source_sha256;
    const entries = Object.entries(obj(r.layers) ?? {});
    const layers: {layer: number; order: number[]}[] = [];
    if (typeof source !== 'string' || !/^[0-9a-f]{64}$/.test(source) || !entries.length || entries.length > 4096)
      return {kind:'error', message:'Expected a source SHA-256 and a nonempty bounded layer map.'};
    for (const [id, order] of entries) {
      if (!/^(0|[1-9]\d{0,5})$/.test(id) || !Array.isArray(order) || !order.length || order.length > 65536 ||
          order.some(x => !Number.isSafeInteger(x) || x < 0 || x >= 65536) || new Set(order).size !== order.length)
        return {kind:'error', message:'Invalid layer ID or expert list. Each retained expert must appear exactly once in its layer.'};
      layers.push({layer: Number(id), order});
    }
    const method = obj(r.review)?.method;
    return {kind:'expertMap', report:{source, layers, method:method === 'frequency' ? 'Routing frequency' : method === 'reap' ? 'REAP saliency' : 'Not specified'}};
  }
  if (r.schema === USAGE_SCHEMA) {
    const u = readUsage(r);
    return typeof u === 'string' ? { kind: 'error', message: u } : { kind: 'usage', report: u };
  }
  if (r.schema === PREVIEW_SCHEMA) {
    const p = readPreview(r);
    return typeof p === 'string' ? { kind: 'error', message: p } : { kind: 'preview', report: p };
  }
  if (r.schema === SPECULATION_SCHEMA) {
    const x = readSpeculation(r);
    return typeof x === 'string' ? { kind: 'error', message: x } : { kind: 'speculation', report: x };
  }
  return { kind: 'error', message: `Not supported (${schemaLabel(r.schema)}). Open a usage, profile preview or speculation report.` };
}

/** Minimal File shape so tests do not need a DOM. */
export interface FileLike {
  name: string;
  size: number;
  text(): Promise<string>;
  arrayBuffer?(): Promise<ArrayBuffer>;
}

export async function readFileBounded(file: FileLike): Promise<Parsed> {
  if (file.size > MAX_FILE_BYTES)
    return { kind: 'error', message: `The file is larger than ${MAX_FILE_BYTES / 1024 / 1024} MiB. Derived reports are smaller; raw traces are not accepted.` };
  try {
    const parsed = parseReportText(await file.text());
    if (parsed.kind === 'expertMap' && file.arrayBuffer && globalThis.crypto?.subtle) {
      const hash = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
      parsed.fileSha = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
    }
    return parsed;
  } catch {
    return { kind: 'error', message: 'The file could not be read.' };
  }
}

// ---- Formatting ----

export function pct(v: number | null, digits = 1): string {
  return v === null ? '—' : `${(v * 100).toFixed(digits)}%`;
}

export function count(v: number | null): string {
  return v === null ? '—' : v.toLocaleString();
}

export function fixed(v: number | null, digits = 4): string {
  return v === null ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** Bytes with an explicit sign; '—' when unknown. */
export function signedBytes(v: number | null): string {
  if (v === null) return '—';
  if (v === 0) return '0 B';
  const mag = formatBytes(Math.abs(v));
  return `${v > 0 ? '+' : '−'}${mag}`;
}

/** Encoding transitions with change count and summed payload delta (bytes only; unknown deltas are counted apart). */
export function codecTransitions(changes: Change[]): { from: string; to: string; kind: string; count: number; delta: number; unknown: number }[] {
  const map = new Map<string, { from: string; to: string; kind: string; count: number; delta: number; unknown: number }>();
  for (const c of changes) {
    const from = c.before?.encoding ?? '?';
    const to = c.after?.encoding ?? '?';
    const key = `${c.kind}\u0000${from}\u0000${to}`;
    const row = map.get(key) ?? { from, to, kind: c.kind, count: 0, delta: 0, unknown: 0 };
    row.count += 1;
    if (c.delta === null) row.unknown += 1;
    else row.delta += c.delta;
    map.set(key, row);
  }
  return [...map.values()].sort((a, b) => b.count - a.count);
}
