// Visual weight editing: the tensor inventory, bounded 2-D tiles and the
// stored-type arithmetic for nudges. Ideas from Lanternfish (tensor roles,
// one cluster per layer, diverging value scale); data only from the worker.
//
// Contract (crates/controller/src/weights.rs, crates/engine/src/weight.rs):
//   weight_inspect {mode:'inventory', offset, limit≤128[, source_sha256]}
//     -> report {mode, source_sha256, offset, total, next_offset, tensors:[{name, shape, dtype, elements}]}
//   weight_inspect {mode:'tile', tensor, origin, shape, source_sha256}
//     origin/shape have the full tensor rank. Rank 1: ≤1024 values.
//     Ranks 2–4: ≤32 in each of the last two dimensions, 1 in the others.
//     -> report {mode, source_sha256, tensor, dtype, shape, origin, tile_shape, values:[[…]]}
//   Host needs weight_inspect and weight_visual.

import { obj } from './reports';
import type { JobSummary, NewJob } from './types';

export const INVENTORY_PAGE = 128;
export const TILE = 32;
export const TILE_1D = 1024;
/** Tensor request limits in bytes; the accepted characters are ASCII. */
const MAX_LEGACY_NAME = 63;
const MAX_SAFE_NAME = 95;
export const DTYPES = ['F32', 'F16', 'BF16'] as const;
export type Dtype = (typeof DTYPES)[number];

export type Packed = 'Q4_0' | 'Q8_0' | 'Q8G64';
export type Neighbors = { lower: number | null; upper: number | null };
export const isPacked = (s: string): s is Packed => s === 'Q4_0' || s === 'Q8_0' || s === 'Q8G64';
const SHA = /^[0-9a-f]{64}$/;
export const isDtype = (v: unknown): v is Dtype => typeof v === 'string' && (DTYPES as readonly string[]).includes(v);
const dims = (v: unknown, min = 1): v is number[] =>
  Array.isArray(v) && v.length >= 1 && v.length <= 4 && v.every((n) => Number.isSafeInteger(n) && n >= min);
// The SafeTensors reader inventories rank 0–8, including empty tensors.
// Only explicitly read-only entries may exceed the interactive shape contract.
function inventoryDims(v: unknown, readOnlySafe: boolean): v is number[] {
  return readOnlySafe
    ? Array.isArray(v) && v.length <= 8 && v.every(n => Number.isSafeInteger(n) && n >= 0 && n <= 2 ** 41)
    : dims(v);
}

// ---- Tensor roles (after Lanternfish tensors.ts). ----

export type Role = 'attn' | 'expert' | 'shared' | 'router' | 'ffn' | 'norm' | 'embed' | 'output' | 'other';
export const ROLES: { id: Role; label: string }[] = [
  { id: 'attn', label: 'Attention' },
  { id: 'ffn', label: 'Feed-forward' },
  { id: 'expert', label: 'Experts' },
  { id: 'shared', label: 'Shared expert' },
  { id: 'router', label: 'Router' },
  { id: 'norm', label: 'Norm and bias' },
  { id: 'embed', label: 'Embedding' },
  { id: 'output', label: 'Output' },
  { id: 'other', label: 'Other' },
];

/** Role from the tensor name only. It is a naming convention, not measured usage. */
export function roleOf(name: string): Role {
  const n = name.toLowerCase();
  if (/norm|\.bias$|exp_probs_b/.test(n)) return 'norm';
  if (/token_embd|tok_embeddings|embed_tokens/.test(n)) return 'embed';
  if (/(^|\.)output\.|lm_head/.test(n)) return 'output';
  if (/ffn_gate_inp|\.gate\.weight|router/.test(n)) return 'router';
  if (/_shexp|shared_expert/.test(n)) return 'shared';
  if (/_exps|\.experts\./.test(n)) return 'expert';
  if (/attn|self_attn|\.wq|\.wk|\.wv|\.wo/.test(n)) return 'attn';
  if (/ffn|mlp/.test(n)) return 'ffn';
  return 'other';
}

export function layerOf(name: string): number | null {
  const m = /(?:^|\.)(?:blk|layers|h)\.(\d+)\./.exec(name);
  return m ? Number(m[1]) : null;
}

// ---- Inventory. ----

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  reason?: string;
  sourceFormat?: string;
  elements: number;
  /** Position in the file (inventory order). */
  order: number;
  role: Role;
  layer: number | null;
}

export interface InventoryPage {
  source_sha256: string;
  offset: number;
  total: number;
  next_offset: number | null;
  tensors: TensorInfo[];
}

export function readInventory(result: unknown, expectedFormat?: string): InventoryPage | null {
  const r = obj(obj(result)?.report);
  if (!r || (expectedFormat && r.source_format !== expectedFormat) || r.mode !== 'inventory' || typeof r.source_sha256 !== 'string' || !SHA.test(r.source_sha256)) return null;
  const { offset, total, next_offset } = r;
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(total) || (next_offset !== null && !Number.isSafeInteger(next_offset))) return null;
  if (!Array.isArray(r.tensors)) return null;
  const tensors: TensorInfo[] = [];
  for (const [i, raw] of r.tensors.entries()) {
    const t = obj(raw);
    if (!t || typeof t.name !== 'string' || !t.name || (typeof t.dtype !== 'string' || !t.dtype || (!isDtype(t.dtype) && typeof t.editable !== 'boolean'))) return null;
    if (!inventoryDims(t.shape, t.editable === false && (r.source_format === 'hf' || r.source_format === 'safetensors'))) return null;
    const elements = t.shape.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(elements) || (t.elements !== undefined && t.elements !== elements)) return null;
    const locked = t.editable === false || (r.source_format === 'hgn' && t.editable !== true);
    const reason = locked
      ? (typeof t.reason === 'string' && t.reason) || 'The worker did not confirm that this tensor is editable.'
      : !isDtype(t.dtype) && !(isPacked(t.dtype) && t.editable === true) ? 'No verified code-step editor for this encoding.' : undefined;
    tensors.push({ name: t.name, shape: t.shape, dtype: t.dtype, reason, elements,
      sourceFormat: typeof r.source_format === 'string' ? r.source_format : undefined,
      order: (offset as number) + i, role: roleOf(t.name), layer: layerOf(t.name) });
  }
  return { source_sha256: r.source_sha256, offset: offset as number, total: total as number, next_offset: next_offset as number | null, tensors };
}

/** Why a tensor cannot be opened, or null. */
export function blockedReason(t: TensorInfo): string | null {
  if (t.reason) return t.reason;
  const native = t.sourceFormat === 'hf' || t.sourceFormat === 'safetensors';
  const valid = native
    ? t.name.length <= MAX_SAFE_NAME && t.name !== '__metadata__' && !/[^\x20-\x7e]/.test(t.name)
    : t.name.length <= MAX_LEGACY_NAME && !/[^A-Za-z0-9._-]/.test(t.name)
      && !t.name.startsWith('.') && !t.name.includes('..');
  if (!t.name || !valid)
    return 'The controller does not accept this tensor name in a request.';
  return null;
}

export interface Cluster {
  layer: number | null;
  members: TensorInfo[];
}

/** One cluster per layer, global tensors first; members in file order. */
export function clusters(tensors: TensorInfo[]): Cluster[] {
  const by = new Map<number | null, TensorInfo[]>();
  for (const t of tensors) {
    const list = by.get(t.layer) ?? [];
    list.push(t);
    by.set(t.layer, list);
  }
  return [...by.entries()]
    .sort(([a], [b]) => (a === null ? -1 : b === null ? 1 : a - b))
    .map(([layer, members]) => ({ layer, members }));
}

export function matches(t: TensorInfo, query: string, roles: ReadonlySet<Role>): boolean {
  if (roles.size && !roles.has(t.role)) return false;
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return terms.every((term) => {
    const m = /^l(\d+)$/.exec(term);
    if (m) return t.layer === Number(m[1]);
    return t.name.toLowerCase().includes(term);
  });
}

// ---- Tiles. ----

export interface Plan {
  rank: number;
  /** Sizes of the leading dimensions (empty for rank 1 and 2). */
  lead: number[];
  rows: number;
  cols: number;
  tileRows: number;
  tileCols: number;
  gridRows: number;
  gridCols: number;
}

export function plan(shape: number[]): Plan {
  const rank = shape.length;
  if (rank === 1) return { rank, lead: [], rows: 1, cols: shape[0], tileRows: 1, tileCols: TILE_1D, gridRows: 1, gridCols: Math.ceil(shape[0] / TILE_1D) };
  const rows = shape[rank - 2];
  const cols = shape[rank - 1];
  return { rank, lead: shape.slice(0, rank - 2), rows, cols, tileRows: TILE, tileCols: TILE, gridRows: Math.ceil(rows / TILE), gridCols: Math.ceil(cols / TILE) };
}

export interface TilePos {
  lead: number[];
  tr: number;
  tc: number;
}

/** Full-rank origin and shape for one tile, clipped at the tensor edge. */
export function tileRequest(p: Plan, pos: TilePos): { origin: number[]; shape: number[] } {
  const r0 = pos.tr * p.tileRows;
  const c0 = pos.tc * p.tileCols;
  const h = Math.min(p.tileRows, p.rows - r0);
  const w = Math.min(p.tileCols, p.cols - c0);
  if (p.rank === 1) return { origin: [c0], shape: [w] };
  return { origin: [...pos.lead, r0, c0], shape: [...pos.lead.map(() => 1), h, w] };
}

export const tileJob = (source: string, host: string, sha: string, tensor: string, req: { origin: number[]; shape: number[] }, packed = false): NewJob => ({
  kind: 'weight_inspect',
  host,
  input_ids: [source],
  options: { mode: 'tile', ...(packed ? {packed:true} : {}), tensor, origin: req.origin, shape: req.shape, source_sha256: sha },
});

export const inventoryJob = (source: string, host: string, offset: number, sha?: string, packed = false): NewJob => ({
  kind: 'weight_inspect',
  host,
  input_ids: [source],
  options: { mode: 'inventory', ...(packed ? {packed:true} : {}), offset, limit: INVENTORY_PAGE, ...(sha ? { source_sha256: sha } : {}) },
});

export interface Tile {
  tensor: string;
  dtype: Dtype | Packed;
  steps?: Neighbors[][];
  sha: string;
  origin: number[];
  shape: number[];
  values: number[][];
  absMax: number;
}

const same = (a: unknown, b: number[]) => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);

/** Validate a tile report against the request and the inventory snapshot. */
export function readTile(result: unknown, want: { tensor: string; origin: number[]; shape: number[]; sha: string; dtype: Dtype | Packed; full: number[] }): Tile | string {
  const r = obj(obj(result)?.report);
  if (!r || r.mode !== 'tile') return 'The tile report has an unknown shape.';
  if (r.source_sha256 !== want.sha) return 'The source file changed after the tensor list was read. Read the tensor list again.';
  if (r.tensor !== want.tensor || !same(r.origin, want.origin) || !same(r.tile_shape, want.shape) || !same(r.shape, want.full) || r.dtype !== want.dtype)
    return 'The tile report does not match the request.';
  const rows = want.shape.length === 1 ? 1 : want.shape[want.shape.length - 2];
  const cols = want.shape[want.shape.length - 1];
  const v = r.values;
  if (!Array.isArray(v) || v.length !== rows || !v.every((row) => Array.isArray(row) && row.length === cols && row.every((x) => typeof x === 'number' && Number.isFinite(x))))
    return 'The tile values do not match the tile shape.';
  let steps: Neighbors[][] | undefined;
  if (isPacked(want.dtype)) {
    const grid = r.packed_steps;
    if (!Array.isArray(grid) || grid.length !== rows || !grid.every((row, i) => Array.isArray(row) && row.length === cols && row.every((item, j) => {
      const limits = obj(item), value = v[i][j];
      return limits && (limits.lower === null || (typeof limits.lower === 'number' && Number.isFinite(limits.lower) && limits.lower < value))
        && (limits.upper === null || (typeof limits.upper === 'number' && Number.isFinite(limits.upper) && limits.upper > value));
    }))) return 'Packed editing unavailable: the worker did not return valid adjacent decoded values for this tile.';
    steps = grid;
  }
  let absMax = 0;
  for (const row of v as number[][]) for (const x of row) absMax = Math.max(absMax, Math.abs(x));
  return { tensor: want.tensor, dtype: want.dtype, steps, sha: want.sha, origin: want.origin, shape: want.shape, values: v as number[][], absMax };
}

export interface Cell {
  key: string;
  index: number[];
  value: number;
}

/** Logical index (weight_edit order) of the value at values[i][j]. */
export function cellIndex(t: Pick<Tile, 'origin' | 'shape'>, i: number, j: number): number[] {
  if (t.shape.length === 1) return [t.origin[0] + j];
  const n = t.origin.length;
  return [...t.origin.slice(0, n - 2), t.origin[n - 2] + i, t.origin[n - 1] + j];
}

export const keyOf = (index: number[]) => index.join(',');

// ---- Job lookup: reuse earlier reads instead of a new job. ----

const eq = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Newest job for this source whose options equal the request; succeeded or active first. */
export function findRead(jobs: JobSummary[], source: string, options: Record<string, unknown>): JobSummary | null {
  let best: JobSummary | null = null;
  const rank = (j: JobSummary) => (j.state === 'succeeded' ? 3 : j.state === 'queued' || j.state === 'running' ? 2 : 1);
  for (const j of jobs) {
    if (j.kind !== 'weight_inspect' || j.input_ids[0] !== source) continue;
    const o = obj(j.options);
    if (!o || Object.keys(o).length !== Object.keys(options).length || !Object.entries(options).every(([k, v]) => eq(o[k], v))) continue;
    if (!best || rank(j) > rank(best) || (rank(j) === rank(best) && (j.created_at ?? 0) > (best.created_at ?? 0))) best = j;
  }
  return best;
}

// ---- Stored-type arithmetic. Matches tools/weight_edit.py encode_value. ----

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function halfToNumber(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

function floorLog2(a: number): number {
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  if (2 ** (e + 1) <= a) e++;
  return e;
}

function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Round binary64 to a stored type with ties to even, or null when it is not finite there. */
export function roundTo(dtype: Dtype, v: number): number | null {
  if (!Number.isFinite(v)) return null;
  if (dtype === 'F32') {
    const r = Math.fround(v);
    return Number.isFinite(r) ? r : null;
  }
  const a = Math.abs(v);
  if (a === 0) return v;
  const [mant, minExp, maxFinite] = dtype === 'F16' ? [10, -14, 65504] : [7, -126, (2 - 2 ** -7) * 2 ** 127];
  const e = Math.max(floorLog2(a), minExp);
  const r = roundHalfEven(a * 2 ** (mant - e)) * 2 ** (e - mant);
  if (r > maxFinite) return null;
  return Math.sign(v) * r;
}

/** Magnitude bits of a representable value; sign returned separately. */
function bitsOf(dtype: Dtype, v: number): number {
  const a = Math.abs(v);
  if (dtype === 'F32') {
    f32[0] = a;
    return u32[0];
  }
  if (dtype === 'BF16') {
    f32[0] = a;
    return u32[0] >>> 16;
  }
  if (a === 0) return 0;
  if (a < 2 ** -14) return Math.round(a / 2 ** -24);
  const e = floorLog2(a);
  return ((e + 15) << 10) | Math.round((a / 2 ** e - 1) * 1024);
}

function fromBits(dtype: Dtype, bits: number): number {
  if (dtype === 'F16') return halfToNumber(bits);
  u32[0] = dtype === 'BF16' ? bits << 16 : bits;
  return f32[0];
}

const MAX_BITS: Record<Dtype, number> = { F32: 0x7f7fffff, F16: 0x7bff, BF16: 0x7f7f };

/**
 * Move a stored value by n representable steps (n < 0 goes down). The result
 * stays finite. -0 and +0 are the same step.
 */
export function step(dtype: Dtype, v: number, n: number): number {
  const base = roundTo(dtype, v) ?? v;
  const mag = bitsOf(dtype, base);
  let ord = base < 0 ? -mag : mag;
  ord = Math.max(-MAX_BITS[dtype], Math.min(MAX_BITS[dtype], ord + n));
  const r = fromBits(dtype, Math.abs(ord));
  return ord < 0 ? -r : r;
}

// ---- Colour. Diverging around 0, symmetric over the scale. ----

type RGB = [number, number, number];
const hex = (h: string): RGB => {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const SCALES: Record<'dark' | 'light', [RGB, RGB, RGB, RGB, RGB]> = {
  // Negative cyan, neutral centre, positive pink (Lanternfish DIV_STOPS).
  dark: [hex('#62d8f5'), hex('#2b7d96'), hex('#231c33'), hex('#9c4689'), hex('#f28fdc')],
  light: [hex('#0f5f8a'), hex('#5fa9cf'), hex('#f3f0f8'), hex('#cf6fae'), hex('#8a1d6b')],
};

/** Colour for v on a scale of ±max; t outside is clamped. */
export function divergent(v: number, max: number, theme: 'dark' | 'light'): string {
  const stops = SCALES[theme];
  const t = max > 0 ? Math.max(-1, Math.min(1, v / max)) : 0;
  const x = ((t + 1) / 2) * (stops.length - 1);
  const k = Math.min(stops.length - 2, Math.floor(x));
  const f = x - k;
  const c = stops[k].map((a, i) => Math.round(a + (stops[k + 1][i] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function legend(theme: 'dark' | 'light'): string {
  return `linear-gradient(to right, ${SCALES[theme].map((c) => `rgb(${c.join(',')})`).join(', ')})`;
}

/** Short readable form; exact value stays in titles and the inspector. */
export function short(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1e4 || a < 1e-4 ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
}

/** 1-2-5 values around a magnitude, for slider ranges. */
export function niceRanges(scale: number): number[] {
  const base = scale > 0 && Number.isFinite(scale) ? scale : 1e-3;
  const p = 10 ** Math.floor(Math.log10(base));
  const m = base / p;
  const nice = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  return [nice / 100, nice / 10, nice].map((x) => Number(x.toPrecision(2)));
}
