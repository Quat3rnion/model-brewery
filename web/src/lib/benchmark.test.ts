import { describe, expect, it } from 'vitest';
import { benchmarkChoices, buildBenchmarkJob, buildScorecard, compareRows, exportBlockers, freshDetail, readBenchmarkReport, MAX_SUITES } from './benchmark';
import type { Host, JobDetail, JobSummary } from './types';
const pair = (j: JobDetail) => ({ summary: j as JobSummary, detail: j });

const host: Host = {
  id: 'frame',
  name: 'Frame',
  capabilities: { benchmark: true },
  benchmark_targets: [{ id: 'local-vllm', label: 'Local vLLM' }, { id: 'remote' }, { bad: 1 }, 'x'],
  benchmarks: [{ id: 'mmlu', label: 'MMLU' }, { id: 'arc' }, { id: 'hellaswag' }, null],
};

const report = (over: Record<string, unknown> = {}) => ({
  kind: 'benchmark',
  schema: 'brewery-benchmark-report-v1',
  target: 'local-vllm',
  model: 'qwen3-8b',
  status: 'partial',
  runs: [
    { benchmark: 'mmlu', version: '1.0', protocol: '5-shot', status: 'succeeded', samples: 14042, metrics: [{ name: 'accuracy', value: 0.71, unit: 'fraction', direction: 'higher' }] },
    { benchmark: 'arc', version: '2', protocol: '25-shot', status: 'failed', metrics: [], error: 'endpoint timeout' },
  ],
  release_qualified: false,
  ...over,
});
const bjob = (id: string, result: unknown, over: Partial<JobDetail> = {}): JobDetail => ({
  id,
  kind: 'benchmark',
  host: 'frame',
  state: 'succeeded',
  input_ids: [],
  options: { benchmark_target: 'local-vllm', benchmarks: ['mmlu', 'arc', 'hellaswag'] },
  result,
  ...over,
});

describe('benchmark launcher', () => {
  it('reads only well-formed target and suite entries', () => {
    const c = benchmarkChoices(host);
    expect(c.targets).toEqual([{ id: 'local-vllm', label: 'Local vLLM' }, { id: 'remote' }]);
    expect(c.suites.map((s) => s.id)).toEqual(['mmlu', 'arc', 'hellaswag']);
    expect(benchmarkChoices(undefined)).toEqual({ targets: [], suites: [] });
  });
  it('builds one benchmark job with no inputs', () => {
    const r = buildBenchmarkJob(host, 'local-vllm', ['mmlu', 'arc']);
    expect(r.problems).toEqual([]);
    expect(r.job).toEqual({ kind: 'benchmark', host: 'frame', input_ids: [], options: { benchmark_target: 'local-vllm', benchmarks: ['mmlu', 'arc'] } });
  });
  it('requires a capable host, an advertised target and 1..8 advertised suites', () => {
    expect(buildBenchmarkJob(undefined, 'local-vllm', ['mmlu']).job).toBeNull();
    expect(buildBenchmarkJob({ ...host, capabilities: { benchmark: false }, reasons: { benchmark: 'No endpoint' } }, 'local-vllm', ['mmlu']).problems).toContain('No endpoint');
    expect(buildBenchmarkJob(host, 'other', ['mmlu']).problems.join()).toMatch(/target/i);
    expect(buildBenchmarkJob(host, 'local-vllm', []).problems.join()).toMatch(/at least one/i);
    expect(buildBenchmarkJob(host, 'local-vllm', ['nope']).problems.join()).toMatch(/not advertised/);
    const many = Array.from({ length: MAX_SUITES + 1 }, (_, i) => `s${i}`);
    const big: Host = { ...host, benchmarks: many.map((id) => ({ id })) };
    expect(buildBenchmarkJob(big, 'local-vllm', many).problems.join()).toMatch(/at most 8/);
    expect(buildBenchmarkJob(host, 'local-vllm', ['mmlu', 'mmlu']).job?.options).toEqual({ benchmark_target: 'local-vllm', benchmarks: ['mmlu'] });
  });
});

describe('benchmark report reading', () => {
  it('accepts the v1 schema and keeps the reported status', () => {
    const r = readBenchmarkReport(report())!;
    expect(r.status).toBe('partial');
    expect(r.release_qualified).toBe(false);
    expect(r.runs[0].metrics[0]).toEqual({ name: 'accuracy', value: 0.71, unit: 'fraction', direction: 'higher' });
    expect(r.runs[1].error).toBe('endpoint timeout');
  });
  it('rejects other schemas and never upgrades release qualification', () => {
    expect(readBenchmarkReport({ ...report(), schema: 'v2' })).toBeNull();
    expect(readBenchmarkReport(null)).toBeNull();
    expect(readBenchmarkReport(report({ release_qualified: true }))!.release_qualified).toBe(false);
  });
  it('keeps a non-numeric metric value as missing, not zero', () => {
    const r = readBenchmarkReport(report({ runs: [{ benchmark: 'mmlu', version: '1', protocol: 'p', status: 'succeeded', metrics: [{ name: 'acc', value: 'NaN', unit: '', direction: 'sideways' }] }] }))!;
    expect(r.runs[0].metrics[0].value).toBeNull();
    expect(r.runs[0].metrics[0].direction).toBe('neutral');
  });
});

describe('combined scorecard', () => {
  const hosts: Host[] = [host];
  it('lists requested suites without a run as missing and failed runs as failed', () => {
    const card = buildScorecard([pair(bjob('b1', report()))], [], hosts, '2026-11-18T00:00:00.000Z');
    const e = card.benchmark_jobs[0];
    expect(e.job_id).toBe('b1');
    expect(e.host).toBe('frame');
    expect(e.requested).toEqual({ benchmark_target: 'local-vllm', benchmarks: ['mmlu', 'arc', 'hellaswag'] });
    expect(e.failed_benchmarks).toEqual(['arc']);
    expect(e.missing_benchmarks).toEqual(['hellaswag']);
    expect(card.release_qualified).toBe(false);
    expect(JSON.stringify(card)).not.toMatch(/average|mean|overall/i);
  });
  it('records a failed job without a report as all requested suites missing', () => {
    const card = buildScorecard([pair(bjob('b2', undefined, { state: 'failed', error: 'boom' }))], [], hosts, 't');
    expect(card.benchmark_jobs[0].report).toBeNull();
    expect(card.benchmark_jobs[0].job_error).toBe('boom');
    expect(card.benchmark_jobs[0].missing_benchmarks).toEqual(['mmlu', 'arc', 'hellaswag']);
  });
  it('keeps GSM8K results separate with their own provenance', () => {
    const g: JobDetail = {
      id: 'e1', kind: 'evaluate', host: 'frame', state: 'succeeded', input_ids: ['bundle'],
      result: { score: { correct: 1000, total: 1319 }, protocol_id: 'gsm8k-v1', dataset: { full_split: true }, identity: { synthetic: false } },
    };
    const card = buildScorecard([], [pair(g)], hosts, 't');
    expect(card.gsm8k_jobs[0]).toMatchObject({ job_id: 'e1', host: 'frame', input_ids: ['bundle'], protocol_id: 'gsm8k-v1', full_split: true, quality: 'completed', reference: null, result_loaded: true });
    expect(card.gsm8k_jobs[0].provenance_note).toMatch(/not matched/i);
  });
  it('copies the real GSM8K identity, dataset and baseline provenance verbatim', () => {
    const identity = { source_sha256: 'a'.repeat(64), synthetic: false, provenance: { method: 'halogen', bundle_sha256: 'b'.repeat(64) } };
    const baseline_identity = { source_sha256: 'a'.repeat(64), synthetic: false, provenance: { method: 'imported-predictions', precision: 'Q4_0', import_sha256: 'c'.repeat(64) } };
    const dataset = { name: 'openai/grade-school-math', revision: 'r1', split: 'test', sha256: 'd'.repeat(64), rows: 1319, full_split: true };
    const result = { kind: 'evaluate', reference_scope:'diagnostic', dataset, protocol_id: 'p', score: { correct: 1, total: 2 }, identity, baseline_identity, baseline_score: { correct: 1, total: 2 }, baseline_artifact: { sha256: 'e' }, artifact: { sha256: 'f' }, comparison: { within_two_percentage_points: true } };
    const g: JobDetail = { id: 'e2', kind: 'evaluate', host: 'frame', state: 'succeeded', input_ids: ['bundle', 'base'], result };
    const e = buildScorecard([], [pair(g)], hosts, 't').gsm8k_jobs[0];
    expect(e.identity).toEqual(identity);
    expect(e.dataset).toEqual(dataset);
    expect(e.reference).toMatchObject({scope:'diagnostic', precision:'Q4_0', identity:baseline_identity});
    expect(e.quality).toBe('completed');
    expect(e.baseline).toEqual({ identity: baseline_identity, score: { correct: 1, total: 2 }, artifact: { sha256: 'e' } });
    expect(e.artifact).toEqual({ sha256: 'f' });
    expect(e.synthetic).toBe(false);
  });
  it('a synthetic or identity-less GSM8K result is not reported as non-synthetic', () => {
    const syn: JobDetail = { id: 'e3', kind: 'evaluate', state: 'succeeded', input_ids: [], result: { score: { correct: 1, total: 2 }, identity: { synthetic: true } } };
    const none: JobDetail = { id: 'e4', kind: 'evaluate', state: 'succeeded', input_ids: [], result: { score: { correct: 1, total: 2 } } };
    const [a, b] = buildScorecard([], [pair(syn), pair(none)], hosts, 't').gsm8k_jobs;
    expect(a.synthetic).toBe(true);
    expect(b.synthetic).toBeNull();
    expect(b.identity).toBeNull();
    expect(b.full_split).toBeNull();
  });
  it('an unloaded GSM8K detail has unknown provenance, not a verified run', () => {
    const s: JobSummary = { id: 'e5', kind: 'evaluate', state: 'succeeded', input_ids: ['bundle'], has_result: true };
    const e = buildScorecard([], [{ summary: s, detail: undefined }], hosts, 't').gsm8k_jobs[0];
    expect(e.result_loaded).toBe(false);
    expect(e.synthetic).toBeNull();
    expect(e.identity).toBeNull();
    expect(e.full_split).toBeNull();
    expect(e.score).toBeNull();
  });
  it('an unloaded benchmark detail does not list suites as missing', () => {
    const s: JobSummary = { id: 'b9', kind: 'benchmark', state: 'succeeded', input_ids: [], options: { benchmark_target: 't', benchmarks: ['mmlu'] }, has_result: true };
    const e = buildScorecard([{ summary: s, detail: undefined }], [], hosts, 't').benchmark_jobs[0];
    expect(e.result_loaded).toBe(false);
    expect(e.missing_benchmarks).toBeNull();
    expect(e.failed_benchmarks).toBeNull();
  });
});

describe('side by side rows', () => {
  it('aligns only exact benchmark, version, protocol, metric and unit; differing keys get separate rows', () => {
    const a = readBenchmarkReport(report())!;
    const b = readBenchmarkReport(report({ target: 'remote', runs: [{ benchmark: 'mmlu', version: '1.1', protocol: '5-shot', status: 'succeeded', metrics: [{ name: 'accuracy', value: 0.7, unit: 'fraction', direction: 'higher' }] }] }))!;
    const rows = compareRows([{ jobId: 'b1', report: a }, { jobId: 'b2', report: b }]);
    const mmlu = rows.filter((r) => r.benchmark === 'mmlu');
    expect(mmlu).toHaveLength(2);
    expect(mmlu[0].cells).toEqual([0.71, undefined]);
    expect(mmlu[1].cells).toEqual([undefined, 0.7]);
  });
  it('keeps metrics with a different direction in separate rows', () => {
    const run = (direction: string) => ({ benchmark: 'x', version: '1', protocol: 'p', status: 'succeeded', metrics: [{ name: 'score', value: 1, unit: '', direction }] });
    const a = readBenchmarkReport(report({ runs: [run('higher')] }))!;
    const b = readBenchmarkReport(report({ runs: [run('lower')] }))!;
    const rows = compareRows([{ jobId: 'a', report: a }, { jobId: 'b', report: b }]);
    expect(rows.map((r) => [r.direction, r.cells])).toEqual([['higher', [1, undefined]], ['lower', [undefined, 1]]]);
  });
});

describe('export readiness and stale details', () => {
  const done: JobSummary = { id: 'b1', kind: 'benchmark', state: 'succeeded', input_ids: [] };
  const running: JobSummary = { ...done, id: 'b2', state: 'running' };
  it('blocks export while a selected terminal detail is not loaded or failed to load', () => {
    expect(exportBlockers([done], {}, {})).toEqual(['1 selected result is still loading.']);
    expect(exportBlockers([done], {}, { b1: 'HTTP 500' }).join()).toMatch(/could not be loaded/);
    expect(exportBlockers([done], { b1: { ...done, result: {} } }, {})).toEqual([]);
    expect(exportBlockers([], {}, {})).toEqual(['Select at least one job.']);
  });
  it('a non-terminal selected job does not block, it exports as not finished', () => {
    expect(exportBlockers([running], {}, {})).toEqual([]);
  });
  it('a stale running detail is not used once the summary is terminal', () => {
    const stale: JobDetail = { ...done, state: 'running' };
    expect(freshDetail(done, stale)).toBeUndefined();
    expect(exportBlockers([done], { b1: stale }, {})).toEqual(['1 selected result is still loading.']);
    const ok: JobDetail = { ...done, result: {} };
    expect(freshDetail(done, ok)).toBe(ok);
    expect(freshDetail(done, undefined)).toBeUndefined();
  });
});
