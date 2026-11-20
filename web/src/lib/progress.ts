// Structured job progress (Job.progress_detail) and the legacy live
// observation file (/live-progress/current.json). Values are shown only as
// measured: no invented totals, no ETA, and a live sample never changes a
// job's state.

import { basename, formatBytes, isTerminal, toMillis } from './format';
import { phaseEta, type RateSource, type RateTracker, type RateView } from './rate';
import type { StepTracker } from './steps';
import { obj } from './reports';
import type { JobEvent, JobSummary, Progress } from './types';

export const LIVE_SCHEMA = 'brewery-live-progress-v1';
export const LIVE_URL = './live-progress/current.json';
/** A live sample older than this is hidden. */
export const STALE_MS = 60_000;
export const LIVE_POLL_MS = 3_000;
export const LIVE_POLL_IDLE_MS = 15_000;

export interface ProgressDetail {
  phase: string;
  unit: string;
  completed: number;
  /** null when the producer does not know it. Never estimated here. */
  total: number | null;
  current_file: string | null;
  files_completed: number | null;
  files_total: number | null;
  /** 1-based */
  input_index: number | null;
  input_count: number | null;
  updated_at_ms: number | null;
  estimated: boolean;
  /**
   * Activity snapshot by logical worker slot (not a CPU core ID), sorted by
   * slot. null when absent or malformed: a list with a bad entry, a repeated
   * slot or more than MAX_THREADS entries is dropped whole, so no row is
   * guessed. Slots without an entry are not shown.
   */
  threads: ThreadActivity[] | null;
}

export interface ThreadActivity {
  /** 1..64 */
  slot: number;
  task: string;
}

export const MAX_THREADS = 16;

export function readThreads(v: unknown): ThreadActivity[] | null {
  if (!Array.isArray(v) || v.length === 0 || v.length > MAX_THREADS) return null;
  const out: ThreadActivity[] = [];
  const seen = new Set<number>();
  for (const e of v) {
    const o = obj(e);
    const slot = o?.slot;
    const task = typeof o?.task === 'string' ? o.task.replace(/[\u0000-\u001f\u007f]/g, ' ').trim() : '';
    if (!Number.isInteger(slot) || (slot as number) < 1 || (slot as number) > 64 || !task || seen.has(slot as number)) return null;
    seen.add(slot as number);
    out.push({ slot: slot as number, task: task.slice(0, 120) });
  }
  return out.sort((a, b) => a.slot - b.slot);
}

/**
 * Threads to show for a job, or null. Only while the job runs without a
 * cancel request and the detail is fresh; a stale snapshot is not current
 * activity.
 */
export function threadsFor(job: JobSummary, d: ProgressDetail, now: number): ThreadActivity[] | null {
  if (!d.threads || job.state !== 'running' || job.cancel_requested) return null;
  if (d.updated_at_ms === null || now - d.updated_at_ms >= STALE_MS) return null;
  return d.threads;
}

const nn = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : null);
const int = (v: unknown): number | null => (Number.isInteger(v) && (v as number) >= 0 ? (v as number) : null);

export function readDetail(v: unknown): ProgressDetail | null {
  const o = obj(v);
  if (!o || typeof o.phase !== 'string' || !o.phase || typeof o.unit !== 'string') return null;
  const completed = nn(o.completed);
  if (completed === null) return null;
  const totalRaw = nn(o.total);
  const total = totalRaw !== null && totalRaw >= completed && totalRaw > 0 ? totalRaw : null;
  const fc = int(o.files_completed);
  const ftRaw = int(o.files_total);
  const ft = ftRaw !== null && ftRaw > 0 && (fc === null || fc <= ftRaw) ? ftRaw : null;
  const ic = int(o.input_count);
  const iiRaw = int(o.input_index);
  const ii = iiRaw !== null && iiRaw >= 1 && (ic === null || iiRaw <= ic) ? iiRaw : null;
  const file = typeof o.current_file === 'string' && o.current_file ? basename(o.current_file) : null;
  return {
    phase: o.phase.slice(0, 80),
    unit: o.unit.slice(0, 24),
    completed,
    total,
    current_file: file ? file.slice(0, 200) : null,
    files_completed: fc,
    files_total: ft,
    input_index: ii,
    input_count: ic !== null && ic > 0 ? ic : null,
    updated_at_ms: nn(o.updated_at_ms),
    estimated: o.estimated === true,
    threads: readThreads(o.threads),
  };
}

export function detailFraction(d: ProgressDetail): number | null {
  return d.total ? Math.min(1, d.completed / d.total) : null;
}

function amount(d: ProgressDetail, v: number): string {
  return d.unit === 'bytes' ? formatBytes(v) : v.toLocaleString('en-US');
}

const PHASES: Record<string, string> = {
  hy4_encode_gate: 'Encoding Hy4 gate rows',
  hy4_copy_spans: 'Materializing Hy4 output',
  hy4_verify_readback: 'Verifying Hy4 output',
  inventory_input: 'Inventorying input files',
  inventory_output: 'Inventorying engine output',
  hash_input: 'Hashing input',
  plan_transfer: 'Checking target cache and disk space',
  copy_input: 'Copying bundle',
  verify_cache: 'Verifying cached bundle',
  verify_staged: 'Verifying staged bundle checksums',
  verify_output: 'Verifying engine output',
  verify_input: 'Verifying input',
  pinned_manifest: 'Checking pinned manifest',
  verify_pinned_input: 'Verifying input against pinned checksums',
  identify_output: 'Identifying output',
  publish_output: 'Publishing output',
  compare_left: 'Comparing, left model',
  compare_right: 'Comparing, right model',
  load_dataset: 'Loading dataset',
  baseline_questions: 'Baseline questions',
  evaluate_questions: 'Evaluation questions',
  baseline_scoring: 'Scoring baseline',
  evaluate_scoring: 'Scoring evaluation',
  prepare_rotations_values: 'Preparing rotations',
  convert_passes_values: 'Converting passes',
  verify_payloads_values: 'Verifying payloads',
};

/** Readable phase name. Unknown phases keep their own words. */
export function phaseLabel(phase: string): string {
  const known = PHASES[phase];
  if (known) return known;
  const words = phase.replace(/_+/g, ' ').trim();
  return words ? words[0].toUpperCase() + words.slice(1) : phase;
}

/**
 * Display unit of the counter, plural. Native converter phases named
 * `*_values` count tensor values with the generic unit `items`.
 */
export function unitLabel(d: Pick<ProgressDetail, 'phase' | 'unit'>): string {
  if (d.phase === 'hy4_encode_gate' && d.unit === 'items') return 'rows';
  return d.unit === 'items' && d.phase.endsWith('_values') ? 'values' : d.unit;
}

function counted(n: number, plural: string): string {
  return `${n.toLocaleString('en-US')} ${n === 1 && plural.endsWith('s') ? plural.slice(0, -1) : plural}`;
}

/** Verb for the current file. Parallel readers report one of the open files. */
function fileVerb(phase: string): string {
  if (phase === 'copy_input') return 'Copying';
  if (phase === 'pinned_manifest') return 'Checking name and size';
  if (phase.startsWith('inventory_')) return 'Last listed';
  if (phase.startsWith('hash_') || phase.startsWith('verify_')) return 'Reading';
  return 'Current file';
}

export function ageText(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)} h ago`;
}

/** Text lines for a detail. The percent is phase-local: it covers this phase only. */
export function detailLines(d: ProgressDetail, now: number): { main: string; extra: string[]; age: string | null } {
  const f = detailFraction(d);
  const u = unitLabel(d);
  const unit = d.unit === 'bytes' ? '' : ` ${u}`;
  const amt = d.total !== null ? `${amount(d, d.completed)} of ${amount(d, d.total)}${unit}` : `${amount(d, d.completed)}${unit}`;
  const pct = f !== null ? ` (${Math.floor(f * 100)}%${d.estimated ? ', estimated' : ''})` : '';
  const extra: string[] = [];
  if (d.input_index !== null) extra.push(d.input_count !== null ? `Input ${d.input_index} of ${d.input_count}` : `Input ${d.input_index}`);
  // Files finish in parallel: the count of finished files and the file being
  // read are separate facts. The current file has no position in the list.
  if (d.files_total !== null && d.files_completed !== null) extra.push(`${d.files_completed.toLocaleString('en-US')} of ${counted(d.files_total, 'files')} complete`);
  else if (d.files_total !== null) extra.push(`${counted(d.files_total, 'files')} in this phase`);
  else if (d.files_completed !== null) extra.push(`${counted(d.files_completed, 'files')} complete`);
  if (d.current_file) extra.push(`${fileVerb(d.phase)}: ${d.current_file}`);
  return { main: `${phaseLabel(d.phase)}: ${amt}${pct}`, extra, age: d.updated_at_ms !== null ? `Updated ${ageText(now - d.updated_at_ms)}` : null };
}

/**
 * A legacy structured `progress` counter ({ done, total, unit, phase }) as a
 * detail, so its throughput can be measured. It has no producer timestamp:
 * callers time it with the controller's job.updated_at. Strings and bare
 * fractions are not counters and give null.
 */
export function legacyDetail(p: Progress): ProgressDetail | null {
  if (!p || typeof p !== 'object') return null;
  const done = nn(p.done);
  if (done === null) return null;
  const totalRaw = nn(p.total);
  const unit = p.unit === 'B' || p.unit === 'bytes' ? 'bytes' : typeof p.unit === 'string' && p.unit ? p.unit.slice(0, 24) : 'items';
  return {
    phase: typeof p.phase === 'string' && p.phase ? p.phase.slice(0, 80) : 'progress',
    unit,
    completed: done,
    total: totalRaw !== null && totalRaw >= done && totalRaw > 0 ? totalRaw : null,
    current_file: null,
    files_completed: null,
    files_total: null,
    input_index: null,
    input_count: null,
    updated_at_ms: null,
    estimated: false,
    threads: null,
  };
}

// ---- Legacy live observation ----

export interface LiveSample {
  jobId: string;
  observedAt: number;
  source: string | null;
  detail: ProgressDetail;
}

export function parseLive(v: unknown): LiveSample | null {
  const o = obj(v);
  if (!o || o.schema !== LIVE_SCHEMA || typeof o.job_id !== 'string' || !o.job_id) return null;
  const at = nn(o.observed_at_ms);
  const detail = readDetail(o.progress_detail);
  if (at === null || !detail) return null;
  return { jobId: o.job_id, observedAt: at, source: typeof o.source === 'string' ? o.source : null, detail };
}

/**
 * The live sample for this job, or null. It applies only while the job is
 * running in the controller state, has no controller progress_detail, and
 * the sample is fresh. It never changes or implies the job state.
 */
export function liveFor(job: JobSummary, live: LiveSample | null, now: number): LiveSample | null {
  if (!live || job.state !== 'running' || isTerminal(job.state) || job.cancel_requested || live.jobId !== job.id) return null;
  if (readDetail(job.progress_detail)) return null;
  // Freshness is the observation time only; a later detail timestamp inside the sample does not refresh it.
  const age = now - live.observedAt;
  return age >= -5_000 && age < STALE_MS ? live : null;
}

/**
 * What the live loop must do for these jobs. `tick` keeps the age clock
 * running while any running job shows progress; `fetch` requests the
 * legacy file only when a running job has no worker progress_detail.
 * A job with cancel_requested gets neither.
 */
export function pollPlan(jobs: JobSummary[]): { tick: boolean; fetch: boolean } {
  const active = jobs.filter((j) => j.state === 'running' && !j.cancel_requested);
  return { tick: active.length > 0, fetch: active.some((j) => !readDetail(j.progress_detail)) };
}

// ---- Throughput ----

/**
 * Feed the rate tracker from the current job list and the legacy live
 * sample. Only running jobs without a cancel request keep series; all
 * others are dropped, so a terminal, cancelled or removed job never shows an
 * old rate. Returns true when the tracker changed.
 */
export function feedRates(tracker: RateTracker, jobs: JobSummary[], sample: LiveSample | null, receivedAt: number, steps?: StepTracker): boolean {
  const keep = new Set<string>();
  let changed = false;
  for (const j of jobs) {
    if (j.state !== 'running' || j.cancel_requested) continue;
    keep.add(j.id);
    const own = readDetail(j.progress_detail);
    if (own) changed = tracker.observe(j.id, 'controller', own, own.updated_at_ms, receivedAt) || changed;
    const legacy = legacyDetail(j.progress);
    const at = toMillis(j.updated_at);
    if (legacy) changed = tracker.observe(j.id, 'legacy', legacy, Number.isFinite(at) ? at : null, receivedAt) || changed;
    if (sample && sample.jobId === j.id) changed = tracker.observe(j.id, 'live', sample.detail, sample.detail.updated_at_ms ?? sample.observedAt, receivedAt) || changed;
    // Step history follows the controller detail only: it is the detail with a producer timestamp per job.
    if (own && steps) changed = steps.observe(j.id, own) || changed;
  }
  if (steps) changed = steps.retain(keep) || changed;
  return tracker.retain(keep) || changed;
}

/**
 * Phase time left for a job, in milliseconds, or null. Only for a running job
 * without a cancel request and a detail that is not stale; see phaseEta for
 * the rate conditions.
 */
export function phaseEtaFor(job: JobSummary, view: RateView | null, d: ProgressDetail, now: number): number | null {
  if (job.state !== 'running' || job.cancel_requested) return null;
  if (d.updated_at_ms !== null && now - d.updated_at_ms >= STALE_MS) return null;
  return phaseEta(view, d);
}

/** The detail a job shows and the rate source that measures it, in display order. */
export function shownDetail(job: JobSummary, sample: LiveSample | null, now: number): { detail: ProgressDetail; source: RateSource; observed: boolean } | null {
  const own = readDetail(job.progress_detail);
  if (own) return { detail: own, source: 'controller', observed: false };
  const l = liveFor(job, sample, now);
  if (l) return { detail: l.detail, source: 'live', observed: true };
  return null;
}

// ---- History ----

function phaseOf(e: JobEvent): string | null {
  const d = readDetail((e as { progress_detail?: unknown }).progress_detail);
  if (d) return d.phase;
  const p = e.progress;
  if (p && typeof p === 'object' && typeof (p as { phase?: unknown }).phase === 'string') return (p as { phase: string }).phase;
  return typeof p === 'string' ? '' : null;
}

/**
 * Event log rows. A run of progress-only events with the same state and
 * phase keeps only its last event; `collapsed` counts the hidden ones.
 * State changes, phase changes and events with an error are always kept.
 */
export function summarizeHistory(events: JobEvent[]): { event: JobEvent; collapsed: number }[] {
  const out: { event: JobEvent; collapsed: number }[] = [];
  for (const e of events) {
    const prev = out[out.length - 1];
    const ph = phaseOf(e);
    if (prev && !prev.event.error && !e.error && prev.event.state === e.state && ph !== null && phaseOf(prev.event) === ph) {
      out[out.length - 1] = { event: e, collapsed: prev.collapsed + 1 };
      continue;
    }
    out.push({ event: e, collapsed: 0 });
  }
  return out;
}
