// Multi-benchmark endpoint runs (job kind 'benchmark') and the combined
// scorecard export. The UI reads only what the controller and the report
// state. It never averages metrics across benchmarks, never upgrades
// release qualification, and never treats two runs as comparable because
// their names match.

import { capability, isTerminal } from './format';
import type { Blocker } from './blockers';
import { referenceInfo, benchmarkLine, isFullSplit, obj, readScore } from './reports';
import type { EligibilityEntry, Host, JobDetail, JobSummary, NewJob } from './types';

export const BENCHMARK_SCHEMA = 'brewery-benchmark-report-v1';
export const SCORECARD_SCHEMA = 'brewery-workbench-scorecard-v1';
export const MAX_SUITES = 8;

export interface Choice {
  id: string;
  label?: string;
}

export type Direction = 'higher' | 'lower' | 'neutral';

export interface Metric {
  name: string;
  /** null when the report did not hold a finite number. Never replaced by 0. */
  value: number | null;
  unit: string;
  direction: Direction;
}

export interface Run {
  benchmark: string;
  version: string;
  protocol: string;
  status: 'succeeded' | 'failed' | (string & {});
  metrics: Metric[];
  samples?: number;
  error?: string;
}

export interface BenchmarkReport {
  target: string;
  model: string;
  status: 'complete' | 'partial' | 'failed' | (string & {});
  runs: Run[];
  /** Always false in the UI: the workbench does not qualify releases from endpoint runs. */
  release_qualified: false;
}

function choices(list: unknown): Choice[] {
  if (!Array.isArray(list)) return [];
  const out: Choice[] = [];
  const seen = new Set<string>();
  for (const x of list) {
    const o = obj(x);
    if (!o || typeof o.id !== 'string' || !o.id || seen.has(o.id)) continue;
    seen.add(o.id);
    out.push(typeof o.label === 'string' && o.label ? { id: o.id, label: o.label } : { id: o.id });
  }
  return out;
}

/** Endpoint targets and benchmark suites that a host advertises. */
export function benchmarkChoices(host: Host | undefined): { targets: Choice[]; suites: Choice[] } {
  return { targets: choices(host?.benchmark_targets), suites: choices(host?.benchmarks) };
}

export function choiceLabel(list: Choice[], id: string): string {
  return list.find((c) => c.id === id)?.label ?? id;
}

/** The worker executes suites in wire order. Keep the historical order for
 * selected historical suites, then append new selections in stable ID order. */
export function orderedSuites(selected: readonly string[], historical: readonly string[]): string[] {
  const rest = new Set(selected);
  return [...historical.filter(id => rest.delete(id)), ...[...rest].sort()];
}

/** One 'benchmark' job: no inputs, options { benchmark_target, benchmarks }. */
export function buildBenchmarkJob(host: Host | undefined, target: string, suites: string[], decision?: EligibilityEntry): { job: NewJob | null; blockers: Blocker[] } {
  const blockers: Blocker[] = [];
  const picked = [...new Set(suites)];
  if (!host) blockers.push({ field: 'host', text: 'Select a host.' });
  else {
    const legacy = decision ? undefined : capability(host, 'benchmark');
    if (decision?.eligible === false || legacy?.allowed === false) {
      const reason = host.capability_reasons?.benchmark;
      const reasons = decision?.reasons ?? (reason ? [reason] : []);
      if (reasons.length) blockers.push(...reasons.map((r) => ({ text: r.text, fixAnchor: r.fix_anchor })));
      else blockers.push({ host: host.name ?? host.id, text: legacy?.reason ?? 'This host cannot run benchmarks.', fixAnchor: `#/settings/evaluation?host=${encodeURIComponent(host.id)}` });
    }
    const { targets, suites: offered } = benchmarkChoices(host);
    if (!target) blockers.push({ field: 'target', text: targets.length ? 'Select an endpoint target.' : 'This host advertises no endpoint target.' });
    else if (!targets.some((t) => t.id === target)) blockers.push({ field: 'target', text: 'The selected endpoint target is not advertised by this host.' });
    const unknown = picked.filter((id) => !offered.some((s) => s.id === id));
    if (unknown.length) blockers.push({ field: 'suites', text: `Suites not advertised by this host: ${unknown.join(', ')}.` });
  }
  if (picked.length === 0) blockers.push({ field: 'suites', text: 'Select at least one benchmark suite.' });
  if (picked.length > MAX_SUITES) blockers.push({ field: 'suites', text: `Select at most ${MAX_SUITES} benchmark suites.` });
  if (blockers.length || !host) return { job: null, blockers };
  return { job: { kind: 'benchmark', host: host.id, input_ids: [], options: { benchmark_target: target, benchmarks: picked } }, blockers };
}

/** Preserve report values as text beside their field paths, not as a JSON blob. */
export function provenanceRows(value: unknown): [string, string][] {
  function rows(v: unknown, path: string, depth: number): [string, string][] {
    const fields = Array.isArray(v) ? Object.entries(v) : Object.entries(obj(v) ?? {});
    if (fields.length) {
      if (depth >= 8) return [[path, 'See exported JSON for nested fields.']];
      return fields.flatMap(([key, item]) => rows(item, path ? `${path}.${key}` : key, depth + 1));
    }
    return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
      ? [[path || 'Value', String(v)]]
      : [];
  }
  return rows(value, '', 0);
}

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);

/** Parse a v1 benchmark report. Returns null for any other shape. */
export function readBenchmarkReport(result: unknown): BenchmarkReport | null {
  const r = obj(result);
  if (!r || r.schema !== BENCHMARK_SCHEMA || !Array.isArray(r.runs)) return null;
  const runs: Run[] = [];
  for (const x of r.runs) {
    const o = obj(x);
    if (!o) continue;
    const metrics: Metric[] = (Array.isArray(o.metrics) ? o.metrics : [])
      .map((m) => obj(m))
      .filter((m): m is Record<string, unknown> => !!m && typeof m.name === 'string')
      .map((m) => ({
        name: m.name as string,
        value: typeof m.value === 'number' && Number.isFinite(m.value) ? m.value : null,
        unit: str(m.unit),
        direction: m.direction === 'higher' || m.direction === 'lower' ? m.direction : 'neutral',
      }));
    const run: Run = { benchmark: str(o.benchmark, '?'), version: str(o.version, '?'), protocol: str(o.protocol, '?'), status: str(o.status, 'unknown'), metrics };
    if (typeof o.samples === 'number' && Number.isInteger(o.samples)) run.samples = o.samples;
    if (typeof o.error === 'string') run.error = o.error;
    runs.push(run);
  }
  return { target: str(r.target, '?'), model: str(r.model, '?'), status: str(r.status, 'unknown'), runs, release_qualified: false };
}

export function requestedOf(job: JobSummary): { benchmark_target: string | null; benchmarks: string[] } {
  const o = obj(job.options);
  const list = Array.isArray(o?.benchmarks) ? (o!.benchmarks as unknown[]).filter((x): x is string => typeof x === 'string') : [];
  return { benchmark_target: typeof o?.benchmark_target === 'string' ? o.benchmark_target : null, benchmarks: list };
}

/** Requested suites that have no run in the report (or all of them when there is no report). */
export function missingSuites(job: JobSummary, report: BenchmarkReport | null): string[] {
  const ran = new Set(report?.runs.map((r) => r.benchmark) ?? []);
  return requestedOf(job).benchmarks.filter((id) => !ran.has(id));
}

export function failedSuites(report: BenchmarkReport | null): string[] {
  return report ? report.runs.filter((r) => r.status !== 'succeeded').map((r) => r.benchmark) : [];
}

export function statusTone(status: string): 'ok' | 'bad' | 'warn' | 'muted' {
  if (status === 'complete' || status === 'succeeded') return 'ok';
  if (status === 'failed') return 'bad';
  if (status === 'partial') return 'warn';
  return 'muted';
}

export function formatValue(v: number | null): string {
  if (v === null) return '—';
  return v.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

export function directionText(d: Direction): string {
  return d === 'higher' ? 'higher is better' : d === 'lower' ? 'lower is better' : 'no preferred direction';
}

// ---- Combined scorecard export ----

/**
 * The cached detail only when it describes the same state as the current
 * summary. A detail fetched while the job ran is stale once the summary is
 * terminal, and must be fetched again.
 */
export function freshDetail(summary: JobSummary, detail: JobDetail | undefined): JobDetail | undefined {
  return detail && detail.state === summary.state ? detail : undefined;
}

/** Reasons that block export. A terminal job must have its fresh detail loaded. */
export function exportBlockers(selected: JobSummary[], details: Record<string, JobDetail | undefined>, errors: Record<string, string | undefined>): string[] {
  if (selected.length === 0) return ['Select at least one job.'];
  const out: string[] = [];
  const terminal = selected.filter((j) => isTerminal(j.state));
  const failed = terminal.filter((j) => errors[j.id]);
  const loading = terminal.filter((j) => !errors[j.id] && !freshDetail(j, details[j.id]));
  if (failed.length) out.push(`${failed.length} selected result${failed.length === 1 ? '' : 's'} could not be loaded: ${failed.map((j) => j.id).join(', ')}. Retry or deselect.`);
  if (loading.length) out.push(`${loading.length} selected result${loading.length === 1 ? ' is' : 's are'} still loading.`);
  return out;
}

export interface ScorecardPair {
  summary: JobSummary;
  /** Fresh detail, or undefined when it was not loaded. */
  detail: JobDetail | undefined;
}

export interface ScorecardBenchmarkEntry {
  job_id: string;
  host: string | null;
  requested_host: string | null;
  host_name: string | null;
  job_state: string;
  job_error: string | null;
  created_at: number | null;
  finished_at: number | null;
  requested: { benchmark_target: string | null; benchmarks: string[] };
  /** False when the job detail was not loaded; then the outcome fields are null (unknown). */
  result_loaded: boolean;
  /** Verbatim controller result when it is a v1 benchmark report; null otherwise. */
  report: Record<string, unknown> | null;
  /** Verbatim result when present but not a v1 benchmark report. */
  unrecognized_result?: unknown;
  target_matches_request: boolean | null;
  failed_benchmarks: string[] | null;
  missing_benchmarks: string[] | null;
}

export interface ScorecardGsm8kEntry {
  job_id: string;
  kind: string;
  host: string | null;
  job_state: string;
  job_error: string | null;
  created_at: number | null;
  finished_at: number | null;
  input_ids: string[];
  result_loaded: boolean;
  protocol_id: string | null;
  /** Verbatim result.dataset, or null. */
  dataset: unknown;
  full_split: boolean | null;
  /** From result.identity.synthetic; null when the result does not state it. */
  synthetic: boolean | null;
  /** Verbatim result.identity (source_sha256, synthetic, provenance), or null. */
  identity: unknown;
  score: unknown;
  comparison: unknown;
  /** Verbatim baseline fields of the result, or null when the result has no baseline. */
  reference: ReturnType<typeof referenceInfo>;
  baseline: { identity: unknown; score: unknown; artifact: unknown } | null;
  artifact: unknown;
  quality: string;
  provenance_note: string;
}

export interface Scorecard {
  schema: typeof SCORECARD_SCHEMA;
  generated_at: string;
  note: string;
  release_qualified: false;
  benchmark_jobs: ScorecardBenchmarkEntry[];
  gsm8k_jobs: ScorecardGsm8kEntry[];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const verbatim = (v: unknown): unknown => (v === undefined ? null : v);

export function buildScorecard(benchJobs: ScorecardPair[], gsmJobs: ScorecardPair[], hosts: Host[], generatedAt: string): Scorecard {
  return {
    schema: SCORECARD_SCHEMA,
    generated_at: generatedAt,
    note:
      'Each entry is a separate job with its own provenance. No cross-benchmark or cross-job aggregate is computed. Entries are not asserted to share a model, endpoint or protocol. GSM8K entries are separate evaluations and are not matched to benchmark jobs. When result_loaded is false, the outcome is unknown, not missing.',
    release_qualified: false,
    benchmark_jobs: benchJobs.map(({ summary: j, detail: d }) => {
      const loaded = d !== undefined;
      const parsed = loaded ? readBenchmarkReport(d.result) : null;
      const req = requestedOf(j);
      const entry: ScorecardBenchmarkEntry = {
        job_id: j.id,
        host: j.host ?? null,
        requested_host: j.requested_host ?? null,
        host_name: hosts.find((h) => h.id === j.host)?.name ?? null,
        job_state: String(j.state),
        job_error: (d?.error ?? j.error) ?? null,
        created_at: num(j.created_at),
        finished_at: num(d?.finished_at ?? j.finished_at),
        requested: req,
        result_loaded: loaded,
        report: parsed ? (obj(d!.result) as Record<string, unknown>) : null,
        target_matches_request: parsed && req.benchmark_target ? parsed.target === req.benchmark_target : null,
        failed_benchmarks: loaded ? failedSuites(parsed) : null,
        missing_benchmarks: loaded ? missingSuites(j, parsed) : null,
      };
      if (loaded && !parsed && d.result !== undefined && d.result !== null) entry.unrecognized_result = d.result;
      return entry;
    }),
    gsm8k_jobs: gsmJobs.map(({ summary: j, detail: d }) => {
      const r = obj(d?.result);
      const id = obj(r?.identity);
      const hasBaseline = !!r && (r.baseline_identity != null || r.baseline_score != null || r.baseline_artifact != null);
      return {
        job_id: j.id,
        kind: String(j.kind),
        host: j.host ?? null,
        job_state: String(j.state),
        job_error: (d?.error ?? j.error) ?? null,
        created_at: num(j.created_at),
        finished_at: num(d?.finished_at ?? j.finished_at),
        input_ids: j.input_ids,
        result_loaded: d !== undefined,
        protocol_id: typeof r?.protocol_id === 'string' ? r.protocol_id : null,
        dataset: verbatim(r?.dataset),
        full_split: r ? isFullSplit(r) : null,
        synthetic: typeof id?.synthetic === 'boolean' ? id.synthetic : null,
        identity: verbatim(r?.identity),
        score: r ? (readScore(r) ? r.score : null) : null,
        comparison: verbatim(r?.comparison),
        reference: referenceInfo(r),
        baseline: hasBaseline ? { identity: verbatim(r!.baseline_identity), score: verbatim(r!.baseline_score), artifact: verbatim(r!.baseline_artifact) } : null,
        artifact: verbatim(r?.artifact),
        quality: d ? benchmarkLine(j, d).status : 'not-loaded',
        provenance_note: 'Separate GSM8K job. Its model, endpoint and protocol are not matched to any benchmark job in this export.',
      };
    }),
  };
}

// ---- Side-by-side rows ----

export interface CompareRow {
  key: string;
  benchmark: string;
  version: string;
  protocol: string;
  metric: string | null;
  unit: string;
  direction: Direction;
  /** Per column: number, null (reported without a value), or undefined (no such metric in that job). */
  cells: (number | null | undefined)[];
  statuses: (string | undefined)[];
}

/**
 * Rows keyed by the exact benchmark, version, protocol, metric name, unit and direction.
 * A difference in any of these gives separate rows. A shared row does not
 * claim that the runs are comparable: targets and models can differ.
 */
export function compareRows(columns: { jobId: string; report: BenchmarkReport }[]): CompareRow[] {
  const rows = new Map<string, CompareRow>();
  const blank = () => columns.map(() => undefined);
  columns.forEach(({ report }, col) => {
    for (const run of report.runs) {
      const entries = run.metrics.length ? run.metrics : [null];
      for (const m of entries) {
        const key = JSON.stringify([run.benchmark, run.version, run.protocol, m?.name ?? null, m?.unit ?? '', m?.direction ?? null]);
        let row = rows.get(key);
        if (!row) {
          row = { key, benchmark: run.benchmark, version: run.version, protocol: run.protocol, metric: m?.name ?? null, unit: m?.unit ?? '', direction: m?.direction ?? 'neutral', cells: blank(), statuses: blank() };
          rows.set(key, row);
        }
        if (m) row.cells[col] = m.value;
        row.statuses[col] = run.status;
      }
    }
  });
  return [...rows.values()];
}
