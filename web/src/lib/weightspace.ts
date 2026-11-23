// Weight-space workbench logic: trace requests and slices, scratch lineage,
// check bundles and publish approval. Shapes follow api/openapi.json
// (TraceOptions, TraceTensor, TraceLens, ScratchRevision, diff reports).
// The UI shows what the controller sent and does not invent values.

import type { Tone } from './format';
import type { NewJob } from './types';

export const TRACE_ROLES = ['l_out', 'attn_out', 'ffn_inp', 'ffn_out', 'kq_soft_max'] as const;
export type TraceRole = (typeof TRACE_ROLES)[number];
export type PatchRole = Exclude<TraceRole, 'kq_soft_max'>;
export const ROLE_LABELS: Record<TraceRole, string> = {
  l_out: 'Block output (l_out)',
  attn_out: 'Attention write (attn_out)',
  ffn_inp: 'FFN input (ffn_inp)',
  ffn_out: 'FFN write (ffn_out)',
  kq_soft_max: 'Attention probabilities (kq_soft_max)',
};
export const MAX_PROMPTS = 4;
export const MAX_PATCHES = 8;
export const MAX_POSITION = 4096;
export const MAX_LAYER = 4096;
export const HARD_BYTE_CAP = 32 << 20;
const MIB = 1 << 20;

export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

/** "0-3, 8" to sorted unique integers in [0, limit). */
export function parseIndexList(text: string, limit: number, allowEmpty = false): Parsed<number[]> {
  const parts = text
    .split(/[\s,]+/)
    .map((p) => p.trim())
    .filter(Boolean);
  if (!parts.length) return allowEmpty ? { ok: true, value: [] } : { ok: false, error: 'enter at least one value' };
  const out = new Set<number>();
  for (const p of parts) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(p);
    if (!m) return { ok: false, error: `"${p}" is not a number or a range such as 0-3` };
    const a = Number(m[1]);
    const b = m[2] === undefined ? a : Number(m[2]);
    if (b < a) return { ok: false, error: `the range ${p} goes down` };
    if (b >= limit) return { ok: false, error: `use values below ${limit}` };
    for (let i = a; i <= b; i++) out.add(i);
  }
  return { ok: true, value: [...out].sort((x, y) => x - y) };
}

/** Same limits as the controller (crates/worker/src/trace.rs). */
export function needsFullDump(layers: number, positions: number, roles: number): boolean {
  return layers > 64 || positions > 128 || layers * positions * Math.max(1, roles) > 2048;
}

export interface EstimateInput {
  prompts: number;
  layers: number;
  positions: number;
  roles: readonly string[];
  topK: number;
  /** Hidden width of the model, when known. */
  hidden: number | null;
  /** Attention heads, when known. */
  heads: number | null;
  /** Keys per attention row; the token count, at least the last position + 1. */
  keys: number;
}

/** Bytes per stored top-k entry (token ID and logit) in the estimate. */
const LOGIT_ENTRY = 16;
const MANIFEST_BASE = 1024;
const MANIFEST_ENTRY = 256;

/**
 * Rough export size. The worker enforces the byte cap on the real files;
 * this only helps to pick a capture that can fit.
 */
export function estimateTraceBytes(e: EstimateInput): { bytes: number; payload: number; complete: boolean } {
  const plain = e.roles.filter((r) => r !== 'kq_soft_max').length;
  const attn = e.roles.includes('kq_soft_max');
  const rows = e.layers * e.positions;
  let complete = true;
  let row = 0;
  if (plain) {
    if (e.hidden && e.hidden > 0) row += plain * e.hidden * 4;
    else complete = false;
  }
  if (attn) {
    if (e.heads && e.heads > 0) row += e.heads * Math.max(1, e.keys) * 4;
    else complete = false;
  }
  const logits = e.topK > 0 ? (rows + e.positions) * e.topK * LOGIT_ENTRY : 0;
  const payload = e.prompts * (rows * row + logits);
  const entries = e.prompts * (rows * e.roles.length + (e.topK > 0 ? rows + e.positions : 0));
  return { bytes: payload + e.prompts * MANIFEST_BASE + entries * MANIFEST_ENTRY, payload, complete };
}

export interface PatchForm {
  source: string;
  prompt: number;
  layer: string;
  role: TraceRole;
  positions: string;
}

export interface TraceForm {
  model: string;
  host: string;
  prompts: string[];
  layers: string;
  roles: TraceRole[];
  positions: string;
  topK: number;
  /** Null: the controller uses the Settings default. */
  byteCapMiB: number | null;
  maxByteCap: number;
  fullDump: boolean;
  artifact: { id: string; version: number } | null;
  patches: PatchForm[];
}

export function buildTraceJob(f: TraceForm): { job: NewJob | null; blockers: string[] } {
  const out: string[] = [];
  if (!f.model) out.push('Select a GGUF model.');
  const prompts = f.prompts;
  if (!prompts.length || prompts.length > MAX_PROMPTS) out.push(`Enter 1 to ${MAX_PROMPTS} prompts.`);
  else if (prompts.some((p) => !p.trim())) out.push('Enter each prompt.');
  else if (prompts.some((p) => new TextEncoder().encode(p).length > 16384)) out.push('Each prompt must be at most 16384 bytes.');
  const layers = parseIndexList(f.layers, MAX_LAYER, true);
  if (!layers.ok) out.push(`Layers: ${layers.error}.`);
  const positions = parseIndexList(f.positions, MAX_POSITION);
  if (!positions.ok) out.push(`Positions: ${positions.error}.`);
  if (!Number.isInteger(f.topK) || f.topK < 0 || f.topK > 100) out.push('Top-k must be 0 to 100.');
  if (!f.roles.length && f.topK === 0) out.push('Select at least one role, or set top-k above 0.');
  if (layers.ok && !layers.value.length && f.roles.length) out.push('Select layers for the roles.');
  let byteCap: number | undefined;
  if (f.byteCapMiB !== null) {
    byteCap = Math.round(f.byteCapMiB * MIB);
    if (!Number.isFinite(f.byteCapMiB) || byteCap < 1) out.push('The byte cap must be more than 0.');
    else if (byteCap > Math.min(f.maxByteCap, HARD_BYTE_CAP)) out.push(`The byte cap must be at most ${Math.min(f.maxByteCap, HARD_BYTE_CAP) / MIB} MiB.`);
  }
  if (layers.ok && positions.ok && !f.fullDump && needsFullDump(layers.value.length, positions.value.length, f.roles.length))
    out.push('This selection is large. Select "Full dump" to send it.');
  if (f.patches.length > MAX_PATCHES) out.push(`Use at most ${MAX_PATCHES} activation patches.`);
  const patches = f.patches.map((p, i) => {
    const n = `Patch ${i + 1}`;
    if (!p.source) out.push(`${n}: select a source trace.`);
    if (p.role === 'kq_soft_max') out.push(`${n}: attention probabilities are read-only.`);
    const layer = parseIndexList(p.layer, MAX_LAYER);
    if (!layer.ok || layer.value.length !== 1) out.push(`${n}: enter one layer.`);
    const pos = parseIndexList(p.positions, MAX_POSITION);
    if (!pos.ok) out.push(`${n}: positions: ${pos.error}.`);
    else if (pos.value.length > 64) out.push(`${n}: use at most 64 positions.`);
    if (!Number.isInteger(p.prompt) || p.prompt < 0 || p.prompt > 3) out.push(`${n}: the source prompt must be 0 to 3.`);
    return {
      source_job_id: p.source,
      source_prompt: p.prompt,
      layer: layer.ok ? layer.value[0] : 0,
      role: p.role,
      positions: pos.ok ? pos.value : [],
    };
  });
  if (out.length || !layers.ok || !positions.ok) return { job: null, blockers: out };
  const options: Record<string, unknown> = { layers: layers.value, roles: [...f.roles], positions: positions.value, top_k: f.topK };
  if (byteCap !== undefined) options.byte_cap = byteCap;
  if (f.fullDump) options.full_dump = true;
  if (prompts.length > 1) options.prompts = [...prompts];
  if (f.artifact) options.intervention_artifact = { id: f.artifact.id, version: f.artifact.version };
  if (patches.length) options.activation_patches = patches;
  const job: NewJob = { kind: 'trace', input_ids: [f.model], options };
  if (f.host) job.host = f.host;
  if (prompts.length === 1) job.prompt = prompts[0];
  return { job, blockers: [] };
}

// ---- Trace slices ----

export interface TopLogit {
  token_id: number;
  logit: number;
}
export interface TraceTensorRow {
  position: number;
  shape?: number[];
  node?: string;
  sha256?: string;
  data_base64: string;
}
export interface TraceTensor {
  layer?: number;
  role?: string;
  rows: TraceTensorRow[];
  missing_positions?: number[];
}
export interface TraceSummary {
  prompt: number;
  model_id: string;
  model_sha256: string;
  lanternfish_commit: string;
  spec: { layers: number[]; roles: string[]; positions: number[]; top_k: number; byte_cap?: number };
  tokens: { position: number; token_id: number }[];
  tensor_count: number;
  missing_count: number;
  intervention_count: number;
  logit_lens_count: number;
  model_logits_count: number;
  bytes: number;
  manifest_sha256: string;
}

export function decodeF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length - (bin.length % 4));
  for (let i = 0; i < bytes.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

function l2(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

export interface Matrix {
  rows: number[];
  cols: number[];
  values: (number | null)[][];
  max: number;
}

/** Layer (rows) by position (columns) activation L2 norms. Null: not captured. */
export function normMatrix(layers: { layer: number; rows: TraceTensorRow[]; missing_positions?: number[] }[], positions: number[]): Matrix {
  const sorted = [...layers].sort((a, b) => a.layer - b.layer);
  let max = 0;
  const values = sorted.map((t) => {
    const by = new Map(t.rows.map((r) => [r.position, r]));
    return positions.map((p) => {
      const r = by.get(p);
      if (!r) return null;
      const n = l2(decodeF32(r.data_base64));
      if (Number.isFinite(n)) max = Math.max(max, n);
      return Number.isFinite(n) ? n : null;
    });
  });
  return { rows: sorted.map((t) => t.layer), cols: positions, values, max };
}

/** One head of a kq_soft_max capture: query position (rows) by key index. */
export function attentionMatrix(t: Pick<TraceTensor, 'rows'>, head: number): (Matrix & { heads: number }) | null {
  const rows = [...t.rows].sort((a, b) => a.position - b.position);
  const first = rows[0]?.shape;
  if (!first || first.length !== 2) return null;
  const [heads, keys] = first;
  if (head < 0 || head >= heads) return null;
  let max = 0;
  const values = rows.map((r) => {
    if (!r.shape || r.shape[0] !== heads || r.shape[1] !== keys) return Array.from({ length: keys }, () => null);
    const data = decodeF32(r.data_base64);
    return Array.from({ length: keys }, (_, k) => {
      const v = data[head * keys + k];
      if (v === undefined || !Number.isFinite(v)) return null;
      max = Math.max(max, v);
      return v;
    });
  });
  return { rows: rows.map((r) => r.position), cols: Array.from({ length: keys }, (_, k) => k), values, max, heads };
}

export function lensGrid(rows: { layer: number; position: number; top: TopLogit[] }[]) {
  const map = new Map<string, TopLogit[]>();
  const layers = new Set<number>();
  const positions = new Set<number>();
  for (const r of rows) {
    map.set(`${r.layer}:${r.position}`, r.top);
    layers.add(r.layer);
    positions.add(r.position);
  }
  return {
    layers: [...layers].sort((a, b) => a - b),
    positions: [...positions].sort((a, b) => a - b),
    at: (layer: number, position: number) => map.get(`${layer}:${position}`),
  };
}

// ---- Scratch revisions ----

export type CheckStatus = { status: 'passed' } | { status: 'failed' | 'not_run' | 'reported'; reason: string };
export interface CheckBundle {
  id: string;
  revision_sha256: string;
  parent_sha256: string;
  created_at_ms: number;
  qualification: string;
  integrity: CheckStatus | { status: string; reason?: string };
  smoke: CheckStatus | { status: string; reason?: string };
  behavior: CheckStatus | { status: string; reason?: string };
  behavior_direction?: import('./abliteration').BehaviorDirectionReport;
  program_correctness?: CheckStatus;
  program_cases?: number;
  offline_report?: unknown;
  measured_job_id?: string;
  acknowledged_at_ms: number | null;
  job_refs: { job_id: string; role: string; kind: string; verified_revision?: boolean }[];
}
export interface ScratchOperation {
  op: string;
  tensor: string;
  [extra: string]: unknown;
}
export interface ScratchRevision {
  id: string;
  kind: 'scratch' | (string & {});
  state: 'scratch' | 'publish_requested' | 'approved' | 'discarded' | (string & {});
  sha256: string;
  qualification: string;
  library_model_id: string | null;
  provenance: {
    parent_id: string;
    parent_sha256: string;
    revision_id: string;
    revision_sha256: string;
    worker: string;
    method: string;
    operations: ScratchOperation[];
    tensors: { name: string; shape: number[]; dtype: string }[];
    tool: string;
    tool_version: string;
    runtime_version: string;
    created_at_ms: number;
    compiled_program?: { program: string; config_sha256: string; runner_sha256: string };
    [extra: string]: unknown;
  };
  delta: { changed_tensors: number; changed_bytes: number; delta_bytes: number; delta_sha256: string; [extra: string]: unknown };
  check_bundle?: CheckBundle;
  gguf?: { sha256: string; revision_sha256: string; converter_version: string; bytes: number; job_id: string };
}

export const SCRATCH_STATE: Record<string, { label: string; tone: Tone }> = {
  scratch: { label: 'Scratch', tone: 'muted' },
  publish_requested: { label: 'Publish requested', tone: 'warn' },
  approved: { label: 'Approved', tone: 'ok' },
  discarded: { label: 'Discarded', tone: 'muted' },
};

export interface LineageNode {
  rev: ScratchRevision;
  depth: number;
  children: LineageNode[];
}

/** Forest of scratch revisions under each Library source, oldest first. */
export function lineage(list: readonly ScratchRevision[]): { source: string; nodes: LineageNode[] }[] {
  const ids = new Set(list.map((r) => r.id));
  const kids = new Map<string, ScratchRevision[]>();
  for (const r of list) {
    const p = r.provenance.parent_id;
    kids.set(p, [...(kids.get(p) ?? []), r]);
  }
  const order = (a: ScratchRevision, b: ScratchRevision) => a.provenance.created_at_ms - b.provenance.created_at_ms || a.id.localeCompare(b.id);
  const build = (parent: string, depth: number, seen: Set<string>): LineageNode[] =>
    (kids.get(parent) ?? [])
      .filter((r) => !seen.has(r.id))
      .sort(order)
      .map((rev) => ({ rev, depth, children: build(rev.id, depth + 1, new Set([...seen, rev.id])) }));
  const sources = [...new Set(list.filter((r) => !ids.has(r.provenance.parent_id)).map((r) => r.provenance.parent_id))].sort();
  return sources.map((source) => ({ source, nodes: build(source, 1, new Set()) }));
}

export function flatten(nodes: LineageNode[]): LineageNode[] {
  return nodes.flatMap((n) => [n, ...flatten(n.children)]);
}

const CHECK_NAMES: [keyof Pick<CheckBundle, 'integrity' | 'smoke' | 'behavior' | 'program_correctness'>, string][] = [
  ['integrity', 'Integrity'],
  ['smoke', 'Load and generate'],
  ['behavior', 'Behavior'],
];
const CHECK_STATUS: Record<string, { label: string; tone: Tone }> = {
  passed: { label: 'Passed', tone: 'ok' },
  failed: { label: 'Failed', tone: 'bad' },
  not_run: { label: 'Not run', tone: 'muted' },
  reported: { label: 'Reported', tone: 'info' },
};

export function checkRows(b: CheckBundle | undefined | null) {
  if (!b) return [];
  const names = b.program_correctness
    ? [...CHECK_NAMES, ['program_correctness', 'Program correctness'] as const]
    : CHECK_NAMES;
  return names.map(([k, name]) => {
    const c = (b[k] ?? {}) as { status?: string; reason?: string };
    const s = CHECK_STATUS[c.status ?? ''] ?? { label: c.status ?? 'Unknown', tone: 'warn' as Tone };
    return { name: k === 'smoke' && b.program_correctness ? 'Load and run' : name, status: c.status ?? 'unknown', label: s.label, tone: s.tone, reason: c.reason ?? '' };
  });
}

export function approvalState(r: ScratchRevision): { can: boolean; reason: string } {
  if (r.state !== 'publish_requested') return { can: false, reason: 'Only a revision with a publish request can be approved.' };
  if (!r.check_bundle) return { can: false, reason: 'The revision has no check bundle.' };
  if (r.check_bundle.revision_sha256 !== r.sha256) return { can: false, reason: 'The check bundle is for a different hash.' };
  return { can: true, reason: '' };
}

export function operationText(op: ScratchOperation): string {
  const o = op as Record<string, unknown>;
  const list = (v: unknown) => (Array.isArray(v) ? `[${v.join(', ')}]` : '?');
  const count = (v: unknown) => (Array.isArray(v) ? v.length : 0);
  switch (op.op) {
    case 'set':
    case 'add':
      return `${op.op} ${count(o.values)} values at ${list(o.origin)}, shape ${list(o.shape)}`;
    case 'low_rank':
      return `low rank, u ${count(o.u)}, v ${count(o.v)}, scale ${String(o.scale)}`;
    case 'orthogonalize':
      return `orthogonalize ${String(o.axis)}, direction ${count(o.direction)} values`;
    default:
      return op.op;
  }
}

export function traceDiffQuery(a: string, b: string, prompt: number, layer: string): string {
  const q = new URLSearchParams({ a, b, prompt: String(prompt) });
  if (layer.trim()) q.set('layer', layer.trim());
  return q.toString();
}

/** Short readable number for tables; exact value stays in titles. */
export function fmt(v: unknown, digits = 4): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return v === null || v === undefined ? '—' : String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(2) : Number(v.toPrecision(digits)).toString();
}
