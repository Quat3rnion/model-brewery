import { describe, expect, it } from 'vitest';
import { detailFraction, detailLines, feedRates, legacyDetail, liveFor, parseLive, phaseLabel, pollPlan, readDetail, readThreads, shownDetail, threadsFor, STALE_MS, summarizeHistory, unitLabel } from './progress';
import { rateText, RateTracker } from './rate';
import type { JobEvent, JobSummary } from './types';

const NOW = 1_800_000_000_000;
const detail = (over: Record<string, unknown> = {}) => ({ phase: 'resolve', unit: 'bytes', completed: 512 * 1024 * 1024, total: 2 * 1024 * 1024 * 1024, current_file: 'model-00003-of-00131.safetensors', files_completed: 2, files_total: 131, input_index: 1, input_count: 2, updated_at_ms: NOW - 2000, ...over });

describe('progress_detail', () => {
  it('reads a full detail and formats bytes, fraction, shard and input', () => {
    const d = readDetail(detail())!;
    expect(detailFraction(d)).toBe(0.25);
    const l = detailLines(d, NOW);
    expect(l.main).toBe('Resolve: 512 MiB of 2.00 GiB (25%)');
    expect(l.extra).toEqual(['Input 1 of 2', '2 of 131 files complete', 'Current file: model-00003-of-00131.safetensors']);
    expect(l.age).toBe('Updated 2 s ago');
  });
  it('without a total there is no fraction or percent and no invented total', () => {
    const d = readDetail(detail({ total: null, files_total: undefined, files_completed: undefined, input_index: undefined }))!;
    expect(detailFraction(d)).toBeNull();
    const l = detailLines(d, NOW);
    expect(l.main).toBe('Resolve: 512 MiB');
    expect(l.extra).toEqual(['Current file: model-00003-of-00131.safetensors']);
    expect(JSON.stringify(l)).not.toMatch(/ETA|remaining|%/);
  });
  it('counts in non-byte units', () => {
    const d = readDetail(detail({ unit: 'tensors', completed: 1200, total: 1658, current_file: null, files_total: null, input_count: null }))!;
    expect(detailLines(d, NOW).main).toBe('Resolve: 1,200 of 1,658 tensors (72%)');
  });
  it('rejects malformed detail and clamps nothing silently', () => {
    expect(readDetail(null)).toBeNull();
    expect(readDetail({ phase: 'x', unit: 'bytes' })).toBeNull();
    expect(readDetail(detail({ completed: -1 }))).toBeNull();
    expect(readDetail(detail({ completed: 10, total: 5 }))!.total).toBeNull();
    expect(readDetail(detail({ current_file: '/abs/path/to/shard.bin' }))!.current_file).toBe('shard.bin');
    expect(readDetail(detail({ input_index: 3, input_count: 2 }))!.input_index).toBeNull();
  });
});

describe('live observation', () => {
  const live = (over: Record<string, unknown> = {}) => ({ schema: 'brewery-live-progress-v1', job_id: 'j1', observed_at_ms: NOW - 5000, source: 'observe_progress', progress_detail: detail({ phase: 'legacy file read', files_total: undefined, files_completed: undefined, input_index: undefined, input_count: undefined }), ...over });
  const running: JobSummary = { id: 'j1', kind: 'inspect', state: 'running', input_ids: [] };
  it('parses only the v1 schema with a job id and detail', () => {
    expect(parseLive(live())?.jobId).toBe('j1');
    expect(parseLive({ ...live(), schema: 'v2' })).toBeNull();
    expect(parseLive({ ...live(), job_id: '' })).toBeNull();
    expect(parseLive({ ...live(), progress_detail: null })).toBeNull();
  });
  it('applies only to the same running job and only while fresh', () => {
    const l = parseLive(live())!;
    expect(liveFor(running, l, NOW)).not.toBeNull();
    expect(liveFor({ ...running, id: 'j2' }, l, NOW)).toBeNull();
    expect(liveFor({ ...running, state: 'succeeded' }, l, NOW)).toBeNull();
    expect(liveFor(running, l, NOW + STALE_MS)).toBeNull();
    expect(liveFor(running, null, NOW)).toBeNull();
  });
  it('controller progress_detail wins over the live file', () => {
    const l = parseLive(live())!;
    expect(liveFor({ ...running, progress_detail: detail() }, l, NOW)).toBeNull();
  });
});

describe('history', () => {
  it('collapses consecutive progress-only events of the same state and phase', () => {
    const ev = (at: number, state: string, phase?: string): JobEvent => ({ at, state, progress: phase ? { phase, done: at, total: 100 } : undefined });
    const events = [ev(1, 'queued'), ev(2, 'running', 'plan'), ev(3, 'running', 'resolve'), ev(4, 'running', 'resolve'), ev(5, 'running', 'resolve'), ev(6, 'running', 'write'), ev(7, 'succeeded')];
    const h = summarizeHistory(events);
    expect(h.map((x) => [x.event.at, x.collapsed])).toEqual([[1, 0], [2, 0], [5, 2], [6, 0], [7, 0]]);
  });
  it('keeps error events and state changes', () => {
    const events: JobEvent[] = [{ at: 1, state: 'running', progress: 'a' }, { at: 2, state: 'running', progress: 'a', error: 'retry' }, { at: 3, state: 'running', progress: 'a' }];
    expect(summarizeHistory(events).map((x) => x.event.at)).toEqual([1, 2, 3]);
  });
});

describe('integration fixes (t_9c199438)', () => {
  const running: JobSummary = { id: 'j1', kind: 'inspect', state: 'running', input_ids: [] };
  const live = (observedAgo: number, updatedAgo: number) =>
    parseLive({ schema: 'brewery-live-progress-v1', job_id: 'j1', observed_at_ms: NOW - observedAgo, progress_detail: detail({ updated_at_ms: NOW - updatedAgo }) })!;
  it('labels an estimated fraction "estimated", not "estimated total"', () => {
    const l = detailLines(readDetail(detail({ estimated: true }))!, NOW);
    expect(l.main).toBe('Resolve: 512 MiB of 2.00 GiB (25%, estimated)');
    expect(l.main).not.toMatch(/estimated total/);
  });
  it('a later detail timestamp does not make a stale observation fresh', () => {
    expect(liveFor(running, live(STALE_MS + 1000, 1000), NOW)).toBeNull();
    expect(liveFor(running, live(1000, 1000), NOW)).not.toBeNull();
  });
  it('no observer fallback once cancellation is requested', () => {
    expect(liveFor({ ...running, cancel_requested: true }, live(1000, 1000), NOW)).toBeNull();
  });
  it('the clock ticks while a worker detail is shown, without fetching the legacy file', () => {
    const worker: JobSummary = { ...running, progress_detail: detail() };
    expect(pollPlan([worker])).toEqual({ tick: true, fetch: false });
    expect(pollPlan([running])).toEqual({ tick: true, fetch: true });
    expect(pollPlan([{ ...running, cancel_requested: true }])).toEqual({ tick: false, fetch: false });
    expect(pollPlan([{ ...running, state: 'succeeded', progress_detail: detail() }])).toEqual({ tick: false, fetch: false });
    expect(pollPlan([])).toEqual({ tick: false, fetch: false });
  });
});

describe('phase-specific counters (t_4f3f0376)', () => {
  it('parallel hashing: finished-file count and the file being read are separate lines', () => {
    const d = readDetail(detail({ phase: 'hash_input', files_completed: 7, files_total: 131, current_file: 'model-00042-of-00131.safetensors' }))!;
    const l = detailLines(d, NOW);
    expect(l.main).toBe('Hashing input: 512 MiB of 2.00 GiB (25%)');
    expect(l.extra).toEqual(['Input 1 of 2', '7 of 131 files complete', 'Reading: model-00042-of-00131.safetensors']);
    expect(l.extra.join()).not.toMatch(/File 8 of/);
  });
  it('native *_values phases count values, not generic items', () => {
    const d = readDetail(detail({ phase: 'convert_passes_values', unit: 'items', completed: 1_500_000, total: 3_000_000, current_file: null, files_total: null, files_completed: null, input_index: null }))!;
    expect(unitLabel(d)).toBe('values');
    expect(detailLines(d, NOW).main).toBe('Converting passes: 1,500,000 of 3,000,000 values (50%)');
    expect(unitLabel({ phase: 'plan', unit: 'items' })).toBe('items');
  });
  it('copy and inventory phases use their own verb; files without a total are counted', () => {
    expect(detailLines(readDetail(detail({ phase: 'copy_input' }))!, NOW).extra).toContain('Copying: model-00003-of-00131.safetensors');
    const inv = readDetail(detail({ phase: 'inventory_input', unit: 'files', completed: 12, total: null, files_total: null, files_completed: null, input_index: null }))!;
    expect(detailLines(inv, NOW).main).toBe('Inventorying input files: 12 files');
    expect(detailLines(inv, NOW).extra).toEqual(['Last listed: model-00003-of-00131.safetensors']);
    expect(detailLines(readDetail(detail({ files_completed: 1, files_total: 1, current_file: null }))!, NOW).extra).toContain('1 of 1 file complete');
  });
  it('unknown phases keep their words', () => {
    expect(phaseLabel('some_new_phase')).toBe('Some new phase');
    expect(phaseLabel('hash_input')).toBe('Hashing input');
  });
  it('legacy structured counters become measurable details; strings and fractions do not', () => {
    expect(legacyDetail({ phase: 'copy', done: 10, total: 100, unit: 'B' })).toMatchObject({ phase: 'copy', unit: 'bytes', completed: 10, total: 100, updated_at_ms: null });
    expect(legacyDetail({ done: 3, unit: 'questions' })).toMatchObject({ unit: 'questions', completed: 3, total: null });
    expect(legacyDetail('50%')).toBeNull();
    expect(legacyDetail(0.5)).toBeNull();
    expect(legacyDetail({ fraction: 0.5 })).toBeNull();
  });
});

describe('feedRates and shownDetail', () => {
  const run = (over: Partial<JobSummary> = {}): JobSummary => ({ id: 'j1', kind: 'plan', state: 'running', input_ids: [], ...over });
  const at = (ms: number, completed: number) => detail({ phase: 'hash_input', completed, updated_at_ms: NOW + ms });
  it('measures decimal MB/s from controller samples and shares it across views', () => {
    const t = new RateTracker();
    feedRates(t, [run({ progress_detail: at(0, 0) })], null, NOW);
    feedRates(t, [run({ progress_detail: at(0, 0) })], null, NOW + 500); // a second view or poll of the same sample
    feedRates(t, [run({ progress_detail: at(2000, 2_000_000_000) })], null, NOW + 2000);
    const shown = shownDetail(run({ progress_detail: at(2000, 2_000_000_000) }), null, NOW)!;
    expect(shown.source).toBe('controller');
    const v = t.view('j1', shown.source, shown.detail, NOW + 2100)!;
    expect(rateText(v, unitLabel(shown.detail))).toBe('Rate: 1.00 GB/s, measured over 2 s');
  });
  it('drops series for terminal, cancelling and removed jobs', () => {
    const t = new RateTracker();
    feedRates(t, [run({ progress_detail: at(0, 0) })], null, NOW);
    feedRates(t, [run({ progress_detail: at(2000, 10) })], null, NOW);
    expect(t.view('j1', 'controller', readDetail(at(2000, 10))!, NOW)?.kind).toBe('rate');
    feedRates(t, [run({ cancel_requested: true, progress_detail: at(2000, 10) })], null, NOW);
    expect(t.view('j1', 'controller', readDetail(at(2000, 10))!, NOW)).toBeNull();
    feedRates(t, [run({ progress_detail: at(2000, 10) })], null, NOW);
    feedRates(t, [run({ state: 'succeeded', progress_detail: at(2000, 10) })], null, NOW);
    expect(t.view('j1', 'controller', readDetail(at(2000, 10))!, NOW)).toBeNull();
  });
  it('times legacy counters with job.updated_at and the live file with its own timestamps', () => {
    const t = new RateTracker();
    // The browser polls every few seconds; the unchanged counter repeats its old timestamp.
    for (let s = 0; s < 60; s += 5) feedRates(t, [run({ progress: { done: 0, total: 100, unit: 'questions' }, updated_at: NOW })], null, NOW + s * 1000);
    feedRates(t, [run({ progress: { done: 30, total: 100, unit: 'questions' }, updated_at: NOW + 60_000 })], null, NOW + 60_000);
    const v = t.view('j1', 'legacy', legacyDetail({ done: 30, total: 100, unit: 'questions' })!, NOW + 60_000);
    expect(rateText(v, 'questions')).toBe('Rate: 30.0 questions/min, measured over 1 min 0 s');
    const ls = (ms: number, c: number) => parseLive({ schema: 'brewery-live-progress-v1', job_id: 'j1', observed_at_ms: NOW + ms, progress_detail: detail({ phase: 'read', completed: c, updated_at_ms: undefined }) });
    feedRates(t, [run()], ls(0, 0), NOW);
    feedRates(t, [run()], ls(3000, 3_000_000), NOW + 3000);
    const s = shownDetail(run(), ls(3000, 3_000_000), NOW + 3000)!;
    expect(s.source).toBe('live');
    expect(rateText(t.view('j1', 'live', s.detail, NOW + 3000), 'bytes')).toBe('Rate: 1.00 MB/s, measured over 3 s');
  });
});

describe('CPU thread activity', () => {
  const run: JobSummary = { id: 'j1', kind: 'plan', state: 'running', input_ids: [] };
  const two = [{ slot: 2, task: 'hash model-00002.safetensors' }, { slot: 1, task: 'hash model-00001.safetensors' }];
  it('reads slots 1..64 with tasks, sorted, and keeps it optional', () => {
    expect(readDetail(detail({ threads: two }))!.threads).toEqual([{ slot: 1, task: 'hash model-00001.safetensors' }, { slot: 2, task: 'hash model-00002.safetensors' }]);
    expect(readDetail(detail())!.threads).toBeNull();
    expect(readThreads([])).toBeNull();
  });
  it('drops the whole list when malformed: no guessed or partial rows', () => {
    expect(readThreads([{ slot: 1, task: 'a' }, { slot: 1, task: 'b' }])).toBeNull();
    expect(readThreads([{ slot: 0, task: 'a' }])).toBeNull();
    expect(readThreads([{ slot: 65, task: 'a' }])).toBeNull();
    expect(readThreads([{ slot: 1.5, task: 'a' }])).toBeNull();
    expect(readThreads([{ slot: 1, task: '  ' }])).toBeNull();
    expect(readThreads(Array.from({ length: 17 }, (_, i) => ({ slot: i + 1, task: 't' })))).toBeNull();
    expect(readThreads(Array.from({ length: 16 }, (_, i) => ({ slot: i + 1, task: 't' })))).toHaveLength(16);
    expect(readThreads([{ slot: 3, task: 'a\nb' }])).toEqual([{ slot: 3, task: 'a b' }]);
  });
  it('shows only for a fresh running job without a cancel request', () => {
    const d = readDetail(detail({ threads: two }))!;
    expect(threadsFor(run, d, NOW)).toHaveLength(2);
    expect(threadsFor({ ...run, cancel_requested: true }, d, NOW)).toBeNull();
    expect(threadsFor({ ...run, state: 'succeeded' }, d, NOW)).toBeNull();
    expect(threadsFor(run, d, NOW + STALE_MS)).toBeNull();
    expect(threadsFor(run, readDetail(detail({ threads: two, updated_at_ms: undefined }))!, NOW)).toBeNull();
  });
});
