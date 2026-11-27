// Weight editing: manual coordinate edits and automatic tuning of selected
// weights in one unquantized GGUF. The controller and the worker check every
// rule again; these checks only give early, specific messages in the form.
//
// Contract (crates/controller/src/weights.rs):
//   weight_inspect  [gguf]                         {tensor, index}
//   weight_propose  [gguf]                         {edits:[{tensor,index,op,value}]}
//   weight_tune     [gguf, calibration, held-out]  {tensor, indices:[[row,col]], max_update}
//   weight_apply    [gguf]                         {proposal_job_id, reviewed:true}
// Each kind is an opt-in host capability with the same name.

import { obj } from './reports';
import { readInventory, type InventoryPage } from './weightmap';
import type { JobSummary, NewJob } from './types';

export const WEIGHT_KINDS = ['weight_inspect', 'weight_propose', 'weight_tune', 'weight_apply'] as const;
export type WeightKind = (typeof WEIGHT_KINDS)[number];
export const PROPOSAL_KINDS: ReadonlySet<string> = new Set(['weight_propose', 'weight_tune']);

export const MAX_EDITS = 256;
export const MAX_COORDINATES = 128;
export const MAX_RANK = 4;
export const MAX_TENSOR_NAME = 127;
export const PROPOSAL_SCHEMA = 'brewery.weight-edit-proposal.v1';
export const TUNE_SCHEMA = 'brewery.weight-tune-report.v1';
export const RESULT_SCHEMA = 'brewery.weight-edit-result.v1';

export type Op = 'set' | 'delta';
export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

const ok = <T>(value: T): Parsed<T> => ({ ok: true, value });
const bad = <T>(error: string): Parsed<T> => ({ ok: false, error });

export function parseTensor(text: string): Parsed<string> {
  const name = text.trim();
  if (!name) return bad('Enter a tensor name.');
  if (name.length > MAX_TENSOR_NAME) return bad(`Tensor names have at most ${MAX_TENSOR_NAME} characters.`);
  if (!/^[A-Za-z0-9._-]+$/.test(name)) return bad('Tensor names use only letters, digits, dot, underscore and hyphen.');
  return ok(name);
}

const UINT = /^\d+$/;
const MAX_INDEX = 0xffff_ffff;

/** Zero-based row-major coordinate, written as "3, 17" or "[3,17]". */
export function parseIndex(text: string, rank?: number): Parsed<number[]> {
  const body = text.trim().replace(/^\[(.*)\]$/s, '$1').trim();
  if (!body) return bad('Enter a coordinate, for example 0, 5.');
  const parts = body.split(/\s*[,\s]\s*/);
  if (!parts.every((p) => UINT.test(p))) return bad('Coordinates are whole numbers from 0, separated by commas.');
  const index = parts.map(Number);
  if (index.some((n) => !Number.isSafeInteger(n) || n > MAX_INDEX)) return bad('A coordinate is too large.');
  if (rank !== undefined && index.length !== rank) return bad(`Give exactly ${rank} coordinates.`);
  if (index.length > MAX_RANK) return bad(`Give at most ${MAX_RANK} coordinates (one per tensor dimension).`);
  return ok(index);
}

const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** A finite decimal number. Text, not <input type=number>, so exact digits and exponents survive. */
export function parseValue(text: string): Parsed<number> {
  const t = text.trim();
  if (!t) return bad('Enter a value.');
  if (!DECIMAL.test(t)) return bad('Enter a decimal number, for example -0.125 or 3e-4.');
  const n = Number(t);
  if (!Number.isFinite(n)) return bad('The value is outside the finite range.');
  return ok(n);
}

export interface EditRow {
  key: string;
  tensor: string;
  index: string;
  op: Op;
  value: string;
}

export interface Edit {
  tensor: string;
  index: number[];
  op: Op;
  value: number;
}

export interface RowCheck {
  tensor?: string;
  index?: string;
  value?: string;
}

/** Check manual rows. Returns per-row field errors and the edits when all rows are valid. */
export function checkEdits(rows: EditRow[]): { errors: RowCheck[]; edits: Edit[] | null; blocker: string | null } {
  const errors: RowCheck[] = [];
  const edits: Edit[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    const e: RowCheck = {};
    const tensor = parseTensor(row.tensor);
    const index = parseIndex(row.index);
    const value = parseValue(row.value);
    if (!tensor.ok) e.tensor = tensor.error;
    if (!index.ok) e.index = index.error;
    if (!value.ok) e.value = value.error;
    else if (row.op === 'delta' && value.value === 0) e.value = 'A delta of 0 changes nothing.';
    if (tensor.ok && index.ok) {
      const id = `${tensor.value}[${index.value.join(',')}]`;
      if (seen.has(id)) e.index = 'This coordinate is already in another row.';
      seen.add(id);
    }
    errors.push(e);
    if (tensor.ok && index.ok && value.ok && !e.index && !e.value) edits.push({ tensor: tensor.value, index: index.value, op: row.op, value: value.value });
  }
  let blocker: string | null = null;
  if (rows.length === 0) blocker = 'Add at least one edit.';
  else if (rows.length > MAX_EDITS) blocker = `Use at most ${MAX_EDITS} edits.`;
  else if (edits.length !== rows.length) blocker = 'Correct the marked fields.';
  return { errors, edits: blocker ? null : edits, blocker };
}

/**
 * Bulk entry: one edit per line, "<tensor> <coordinate> <set|delta> <value>",
 * for example "blk.0.attn_q.weight 3,17 delta -0.002". Blank lines and lines
 * that start with # are skipped. The rows are checked again by checkEdits.
 */
export function parseEditLines(text: string, nextKey: () => string): Parsed<EditRow[]> {
  const rows: EditRow[] = [];
  const lines = text.split('\n');
  for (const [i, raw] of lines.entries()) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const m = /^(\S+)\s+(\[[^\]]*\]|\S+)\s+(set|delta)\s+(\S+)$/i.exec(line);
    if (!m) return bad(`Line ${i + 1}: write <tensor> <coordinate> <set|delta> <value>.`);
    rows.push({ key: nextKey(), tensor: m[1], index: m[2].replace(/\s+/g, ''), op: m[3].toLowerCase() as Op, value: m[4] });
    if (rows.length > MAX_EDITS) return bad(`At most ${MAX_EDITS} edits.`);
  }
  if (rows.length === 0) return bad('No edit lines found.');
  return ok(rows);
}

/** One "row, col" pair per line (rank-2 tensors only). */
export function parseCoordinates(text: string): Parsed<[number, number][]> {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return bad('Enter at least one row, column pair.');
  if (lines.length > MAX_COORDINATES) return bad(`Select at most ${MAX_COORDINATES} weights (${lines.length} entered).`);
  const out: [number, number][] = [];
  const seen = new Set<string>();
  for (const [i, line] of lines.entries()) {
    const p = parseIndex(line, 2);
    if (!p.ok) return bad(`Line ${i + 1}: ${p.error}`);
    const id = p.value.join(',');
    if (seen.has(id)) return bad(`Line ${i + 1}: ${id} is selected twice.`);
    seen.add(id);
    out.push([p.value[0], p.value[1]]);
  }
  return ok(out);
}

export function parseMaxUpdate(text: string): Parsed<number> {
  const v = parseValue(text);
  if (!v.ok) return v;
  if (v.value <= 0) return bad('The maximum change must be greater than 0.');
  return v;
}

export const inspectJob = (source: string, host: string, tensor: string, index: number[]): NewJob => ({
  kind: 'weight_inspect',
  host,
  input_ids: [source],
  options: { tensor, index },
});

export const proposeJob = (source: string, host: string, edits: Edit[]): NewJob => ({
  kind: 'weight_propose',
  host,
  input_ids: [source],
  options: { edits: edits.map(({ tensor, index, op, value }) => ({ tensor, index, op, value })) },
});

export const tuneJob = (source: string, host: string, calibration: string, heldOut: string, tensor: string, indices: [number, number][], maxUpdate: number): NewJob => ({
  kind: 'weight_tune',
  host,
  input_ids: [source, calibration, heldOut],
  options: { tensor, indices, max_update: maxUpdate },
});

export const applyJob = (source: string, host: string, proposalJobId: string): NewJob => ({
  kind: 'weight_apply',
  host,
  input_ids: [source],
  options: { proposal_job_id: proposalJobId, reviewed: true },
});

// ---- Results. Every reader returns null for a shape it does not know. ----

export interface ProposedEdit extends Edit {
  dtype: string;
  before: number;
  after: number;
}

export interface Proposal {
  source_sha256: string;
  edits: ProposedEdit[];
}

const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const str = (v: unknown): v is string => typeof v === 'string' && v.length > 0;

function readEdit(v: unknown): ProposedEdit | null {
  const e = obj(v);
  if (!e || !str(e.tensor) || !Array.isArray(e.index) || !e.index.every((n) => Number.isSafeInteger(n) && (n as number) >= 0)) return null;
  if ((e.op !== 'set' && e.op !== 'delta') || !num(e.value) || !str(e.dtype) || !num(e.before) || !num(e.after)) return null;
  return { tensor: e.tensor, index: e.index as number[], op: e.op, value: e.value, dtype: e.dtype, before: e.before, after: e.after };
}

/** The proposal is in result.report.proposal; result.proposal is accepted as a fallback. */
export function readProposal(result: unknown): Proposal | null {
  const r = obj(result);
  const p = obj(obj(r?.report)?.proposal) ?? obj(r?.proposal);
  if (!p || p.schema !== PROPOSAL_SCHEMA || !str(p.source_sha256) || !Array.isArray(p.edits) || p.edits.length === 0) return null;
  const edits = p.edits.map(readEdit);
  if (edits.some((e) => e === null)) return null;
  return { source_sha256: p.source_sha256, edits: edits as ProposedEdit[] };
}

export interface MsePair {
  before: number;
  after: number;
}

export interface TuneReport {
  tensor: string;
  method: string;
  selected: number | null;
  proposed: number | null;
  max_update: number | null;
  max_sweeps: number | null;
  calibration_sha256: string | null;
  held_out_sha256: string | null;
  calibration: MsePair;
  held_out: MsePair & { regression: boolean };
}

function mse(v: unknown): MsePair | null {
  const m = obj(v);
  return m && num(m.before_mse) && num(m.after_mse) ? { before: m.before_mse, after: m.after_mse } : null;
}

export function readTuneReport(result: unknown): TuneReport | null {
  const t = obj(obj(obj(result)?.report)?.tune);
  if (!t || t.schema !== TUNE_SCHEMA) return null;
  const cal = mse(t.calibration);
  const held = mse(t.held_out);
  if (!cal || !held) return null;
  const optNum = (v: unknown) => (num(v) ? v : null);
  return {
    tensor: str(t.tensor) ? t.tensor : '',
    method: str(t.method) ? t.method : '',
    selected: optNum(t.selected),
    proposed: optNum(t.proposed),
    max_update: optNum(t.max_update),
    max_sweeps: optNum(t.max_sweeps),
    calibration_sha256: str(t.calibration_sha256) ? t.calibration_sha256 : null,
    held_out_sha256: str(t.held_out_sha256) ? t.held_out_sha256 : null,
    calibration: cal,
    // Recompute the regression flag from the numbers; show both if they disagree.
    held_out: { ...held, regression: held.after > held.before || obj(t.held_out)?.regression === true },
  };
}

export interface Inspected {
  mode: 'value';
  tensor: string;
  index: number[];
  dtype: string;
  value: number;
  shape: number[];
  source_sha256: string | null;
}

type InspectedTile = Omit<Inspected, 'mode' | 'index' | 'value'> & {
  mode: 'tile';
  origin: number[];
  tile_shape: number[];
  values: number[][];
};

type InspectReport = Inspected | InspectedTile | (InventoryPage & { mode: 'inventory' });

const coordinates = (list: unknown, min: number): list is number[] =>
  Array.isArray(list) && list.length > 0 && list.every((n) => Number.isSafeInteger(n) && n >= min);

/** The manual editor reads single values; the result view also reads inventory pages and tiles. */
export function readInspect(result: unknown, mode: 'value'): Inspected | null;
export function readInspect(result: unknown): InspectReport | null;
export function readInspect(result: unknown, mode?: 'value'): InspectReport | null {
  const r = obj(obj(result)?.report);
  if (!r) return null;
  if (r.mode === 'inventory') {
    const inventory = mode === 'value' ? null : readInventory(result);
    return inventory ? { mode: 'inventory', ...inventory } : null;
  }
  if (!str(r.tensor) || !str(r.dtype) || !coordinates(r.shape, 1)) return null;
  const base = { tensor: r.tensor, dtype: r.dtype, shape: r.shape, source_sha256: str(r.source_sha256) ? r.source_sha256 : null };
  if (r.mode === 'tile') {
    const { origin, tile_shape, values } = r;
    const rank = base.shape.length;
    if (mode === 'value' || !coordinates(origin, 0) || !coordinates(tile_shape, 1) ||
        origin.length !== rank || tile_shape.length !== rank ||
        !tile_shape.every((size, i) => size <= base.shape[i] - origin[i] && (i >= rank - 2 || size === 1))) return null;
    const rows = rank === 1 ? 1 : tile_shape[rank - 2];
    const cols = tile_shape[rank - 1];
    if (!Array.isArray(values) || values.length !== rows ||
        !values.every((row) => Array.isArray(row) && row.length === cols && row.every(num))) return null;
    return { mode: 'tile', ...base, origin, tile_shape, values };
  }
  if (r.mode !== undefined && r.mode !== 'value') return null;
  if (!coordinates(r.index, 0) || r.index.length !== r.shape.length || !num(r.value)) return null;
  return { mode: 'value', ...base, index: r.index, value: r.value };
}

export interface Applied {
  source_sha256: string;
  output_sha256: string;
  edit_count: number;
  verified: boolean;
  library_model_id: string | null;
}

export function readApplied(result: unknown): Applied | null {
  const res = obj(result);
  const r = obj(res?.report);
  if (!r || r.schema !== RESULT_SCHEMA || !str(r.source_sha256) || !str(r.output_sha256) || !num(r.edit_count)) return null;
  return {
    source_sha256: r.source_sha256,
    output_sha256: r.output_sha256,
    edit_count: r.edit_count,
    verified: r.verified === true,
    library_model_id: str(res?.library_model_id) ? res.library_model_id : null,
  };
}

// ---- Job lists for one source. ----

export function jobsFor(jobs: JobSummary[], source: string, kinds: ReadonlySet<string>): JobSummary[] {
  return jobs.filter((j) => kinds.has(String(j.kind)) && j.input_ids[0] === source);
}

/** Apply jobs that consumed a given proposal job. */
export function appliesOf(jobs: JobSummary[], proposalJobId: string): JobSummary[] {
  return jobs.filter((j) => j.kind === 'weight_apply' && obj(j.options)?.proposal_job_id === proposalJobId);
}

// ---- Display helpers. ----

/** Shortest text that reads back to the same binary64 value. */
export function exact(n: number): string {
  return Object.is(n, -0) ? '-0' : String(n);
}

export function signed(n: number): string {
  const s = exact(n);
  return n > 0 ? `+${s}` : s;
}

/** Controller reasons may lack a final period; blockers are joined as sentences. */
export function sentence(text: string): string {
  const t = text.trim();
  return !t || /[.!?]$/.test(t) ? t : `${t}.`;
}

export const coord = (index: number[]) => `[${index.join(', ')}]`;

/** Relative change in percent, or null when the baseline is 0. */
export function relChange(before: number, after: number): number | null {
  if (before === 0) return null;
  return ((after - before) / Math.abs(before)) * 100;
}

export function percent(p: number | null): string {
  if (p === null) return '—';
  const a = Math.abs(p);
  const digits = a >= 10 ? 1 : a >= 1 ? 2 : 3;
  return `${p > 0 ? '+' : ''}${p.toFixed(digits)} %`;
}

/** Edits whose stored change differs from the request after source dtype rounding. */
export function rounded(e: ProposedEdit): boolean {
  const requested = e.op === 'set' ? e.value : e.before + e.value;
  return requested !== e.after;
}
