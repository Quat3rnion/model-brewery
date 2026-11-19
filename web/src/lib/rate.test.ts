import { describe, expect, it } from 'vitest';
import { readDetail, type ProgressDetail } from './progress';
import { formatByteRate, formatCountRate, RATE_IDLE_MS, RATE_WINDOW_MS, rateText, RateTracker } from './rate';

const T0 = 1_800_000_000_000;
const d = (over: Record<string, unknown> = {}): ProgressDetail =>
  readDetail({ phase: 'hash_input', unit: 'bytes', completed: 0, total: 10_000_000_000, files_completed: 0, files_total: 4, current_file: 'a.safetensors', input_index: 1, input_count: 1, updated_at_ms: T0, ...over })!;

/** Feed (producer ms offset, completed, extra fields); browser receipt equals producer time here. */
function feed(t: RateTracker, rows: [number, number, Record<string, unknown>?][], job = 'j1') {
  for (const [ms, completed, over] of rows) t.observe(job, 'controller', d({ completed, ...over }), T0 + ms, T0 + ms);
}

describe('RateTracker', () => {
  it('warms up with one sample and keeps the counter, then measures', () => {
    const t = new RateTracker();
    feed(t, [[0, 0]]);
    expect(t.view('j1', 'controller', d(), T0)).toEqual({ kind: 'warming', samples: 1 });
    expect(rateText(t.view('j1', 'controller', d(), T0), 'bytes')).toBe('Rate: measuring, needs a second sample');
    feed(t, [[500, 400_000_000]]);
    expect(t.view('j1', 'controller', d(), T0 + 500)?.kind).toBe('warming'); // span below the minimum
    feed(t, [[2000, 1_982_000_000]]);
    const v = t.view('j1', 'controller', d(), T0 + 2000)!;
    expect(v).toMatchObject({ kind: 'rate', samples: 3, spanMs: 2000 });
    expect(rateText(v, 'bytes')).toBe('Rate: 991 MB/s, measured over 2 s');
  });

  it('ignores duplicate and out-of-order samples', () => {
    const t = new RateTracker();
    feed(t, [[0, 0], [1000, 100], [1000, 100], [500, 50], [2000, 200]]);
    const v = t.view('j1', 'controller', d(), T0 + 2000);
    expect(v).toMatchObject({ kind: 'rate', perSecond: 100, samples: 3 });
    expect(t.observe('j1', 'controller', d({ completed: 200 }), T0 + 2000, T0 + 9000)).toBe(false);
  });

  it('does not reset when parallel readers change the current shard of an aggregate counter', () => {
    const t = new RateTracker();
    feed(t, [[0, 0, { current_file: 'a' }], [1000, 1_000_000, { current_file: 'b', files_completed: 1 }], [2000, 2_000_000, { current_file: 'a', files_completed: 1 }]]);
    expect(t.view('j1', 'controller', d(), T0 + 2000)).toMatchObject({ kind: 'rate', perSecond: 1_000_000, samples: 3 });
  });

  it.each([
    ['phase', { phase: 'verify_cache' }],
    ['unit', { unit: 'files' }],
    ['total', { total: 20_000_000_000 }],
    ['input', { input_index: 2, input_count: 2 }],
    ['file set', { files_total: 5 }],
  ])('resets on a %s change', (_name, over) => {
    const t = new RateTracker();
    feed(t, [[0, 0], [2000, 2_000_000], [3000, 3_000_000, over]]);
    expect(t.view('j1', 'controller', d(over), T0 + 3000)).toEqual({ kind: 'warming', samples: 1 });
    expect(t.view('j1', 'controller', d(), T0 + 3000)).toBeNull(); // the old counter no longer exists
  });

  it('resets when the counter or the finished-file count goes back (per-file counter, restart)', () => {
    const t = new RateTracker();
    feed(t, [[0, 0], [2000, 5_000], [3000, 100]]);
    expect(t.view('j1', 'controller', d(), T0 + 3000)).toEqual({ kind: 'warming', samples: 1 });
    feed(t, [[4000, 200, { files_completed: 2 }], [5000, 300, { files_completed: 1 }]]);
    expect(t.view('j1', 'controller', d(), T0 + 5000)).toEqual({ kind: 'warming', samples: 1 });
  });

  it('keeps jobs and sources apart', () => {
    const t = new RateTracker();
    feed(t, [[0, 0], [1000, 1000]], 'a');
    feed(t, [[0, 0], [1000, 5000]], 'b');
    expect(t.view('a', 'controller', d(), T0 + 1000)).toMatchObject({ perSecond: 1000 });
    expect(t.view('b', 'controller', d(), T0 + 1000)).toMatchObject({ perSecond: 5000 });
    expect(t.view('a', 'live', d(), T0 + 1000)).toBeNull();
  });

  it('uses only the recent window, and hides the rate when no new sample arrives', () => {
    const t = new RateTracker();
    feed(t, [[0, 0], [1000, 10_000_000]]);
    for (let s = 2; s <= 30; s++) feed(t, [[s * 1000, 10_000_000 + (s - 1) * 1_000_000]]);
    const v = t.view('j1', 'controller', d(), T0 + 30_000)!;
    expect(v).toMatchObject({ kind: 'rate', perSecond: 1_000_000 });
    expect(v.kind === 'rate' && v.spanMs).toBeLessThanOrEqual(RATE_WINDOW_MS);
    const idle = t.view('j1', 'controller', d(), T0 + 30_000 + RATE_IDLE_MS + 1000)!;
    expect(idle.kind).toBe('idle');
    expect(rateText(idle, 'bytes')).toBe('Rate: no new measurement for 21 s');
  });

  it('a stalled counter shows a measured zero, not an old rate', () => {
    const t = new RateTracker();
    feed(t, [[0, 0], [1000, 1000], [20_000, 1000], [30_000, 1000]]);
    expect(t.view('j1', 'controller', d(), T0 + 30_000)).toMatchObject({ kind: 'rate', perSecond: 0 });
    expect(rateText(t.view('j1', 'controller', d(), T0 + 30_000), 'bytes')).toBe('Rate: 0 B/s, measured over 10 s');
  });

  it('labels a rate of an estimated counter', () => {
    const t = new RateTracker();
    t.observe('j', 'live', d({ estimated: true }), T0, T0);
    t.observe('j', 'live', d({ estimated: true, completed: 2000 }), T0 + 2000, T0 + 2000);
    expect(rateText(t.view('j', 'live', d({ estimated: true }), T0 + 2000), 'bytes')).toBe('Rate: 1.00 kB/s (estimated counter), measured over 2 s');
  });

  it('retain and forget drop series', () => {
    const t = new RateTracker();
    feed(t, [[0, 0]], 'a');
    feed(t, [[0, 0]], 'b');
    expect(t.retain(new Set(['b']))).toBe(true);
    expect(t.view('a', 'controller', d(), T0)).toBeNull();
    expect(t.forget('b', 'controller')).toBe(true);
    expect(t.view('b', 'controller', d(), T0)).toBeNull();
  });

  it('ignores samples without a timestamp', () => {
    const t = new RateTracker();
    expect(t.observe('j', 'legacy', d(), null, T0)).toBe(false);
  });
});

describe('rate formatting', () => {
  it('bytes use decimal units', () => {
    expect(formatByteRate(0)).toBe('0 B/s');
    expect(formatByteRate(999)).toBe('999 B/s');
    expect(formatByteRate(70_500_000)).toBe('70.5 MB/s');
    expect(formatByteRate(991_000_000)).toBe('991 MB/s');
    expect(formatByteRate(1_234_000_000)).toBe('1.23 GB/s');
  });
  it('counts use their unit, per minute below one per second', () => {
    expect(formatCountRate(12.34, 'files')).toBe('12.3 files/s');
    expect(formatCountRate(2_500_000, 'values')).toBe('2,500,000 values/s');
    expect(formatCountRate(0.25, 'questions')).toBe('15.0 questions/min');
    expect(formatCountRate(0, 'questions')).toBe('0 questions/s');
  });
  it('never mentions remaining time', () => {
    expect(rateText({ kind: 'rate', perSecond: 1e9, spanMs: 5000, samples: 4, estimated: false, maxGapMs: 1000, latest: 0, latestAt: 0 }, 'bytes')).not.toMatch(/ETA|remaining|left/i);
  });
});
