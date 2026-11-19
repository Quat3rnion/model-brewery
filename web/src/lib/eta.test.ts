import { describe, expect, it } from 'vitest';
import { feedRates, phaseEtaFor, readDetail, STALE_MS, type ProgressDetail } from './progress';
import { etaText, formatDuration, RATE_SUSPEND_MS, RateTracker } from './rate';
import type { JobSummary } from './types';

// t_e10cff80: phase time left = known remaining units / recent measured positive rate.
const T0 = 1_800_000_000_000;
const TOTAL = 10_000_000_000;
const d = (completed: number, ms: number, over: Record<string, unknown> = {}): ProgressDetail =>
  readDetail({ phase: 'hash_input', unit: 'bytes', completed, total: TOTAL, files_completed: 0, files_total: 4, updated_at_ms: T0 + ms, ...over })!;
const run: JobSummary = { id: 'j1', kind: 'plan', state: 'running', input_ids: [] };

/** Feed samples at 1 GB/s every second; producer time equals browser time. */
function steady(t: RateTracker, seconds: number, over: Record<string, unknown> = {}) {
  for (let s = 0; s <= seconds; s++) t.observe('j1', 'controller', d(s * 1e9, s * 1000, over), T0 + s * 1000, T0 + s * 1000);
}
const eta = (t: RateTracker, detail: ProgressDetail, now: number, job: JobSummary = run) =>
  phaseEtaFor(job, t.view('j1', 'controller', detail, now), detail, now);

describe('phase ETA', () => {
  it('remaining known units divided by the measured rate', () => {
    const t = new RateTracker();
    steady(t, 6);
    // 6 GB done, 4 GB left, 1 GB/s -> 4 s
    expect(eta(t, d(6e9, 6000), T0 + 6000)).toBe(4000);
  });

  it('no ETA with insufficient samples or span', () => {
    const t = new RateTracker();
    steady(t, 1);
    expect(eta(t, d(1e9, 1000), T0 + 1000)).toBeNull(); // 2 samples, 1 s
    steady(t, 3);
    expect(eta(t, d(3e9, 3000), T0 + 3000)).toBeNull(); // span below 5 s
  });

  it('no ETA without a total, at zero rate, or when the phase is complete', () => {
    const t = new RateTracker();
    steady(t, 6, { total: null });
    expect(eta(t, d(6e9, 6000, { total: null }), T0 + 6000)).toBeNull();
    const z = new RateTracker();
    for (let s = 0; s <= 6; s++) z.observe('j1', 'controller', d(5e9, s * 1000), T0 + s * 1000, T0 + s * 1000);
    expect(eta(z, d(5e9, 6000), T0 + 6000)).toBeNull();
  });

  it('no ETA for an estimated counter', () => {
    const t = new RateTracker();
    steady(t, 6, { estimated: true });
    expect(eta(t, d(6e9, 6000, { estimated: true }), T0 + 6000)).toBeNull();
  });

  it('no ETA for cancelled, cancelling, terminal or non-running jobs', () => {
    const t = new RateTracker();
    steady(t, 6);
    const x = d(6e9, 6000);
    expect(eta(t, x, T0 + 6000, { ...run, cancel_requested: true })).toBeNull();
    expect(eta(t, x, T0 + 6000, { ...run, state: 'succeeded' })).toBeNull();
    expect(eta(t, x, T0 + 6000, { ...run, state: 'cancelling' })).toBeNull();
    expect(eta(t, x, T0 + 6000, { ...run, state: 'queued' })).toBeNull();
  });

  it('no ETA for a stale detail or when no new sample arrived recently', () => {
    const t = new RateTracker();
    steady(t, 6);
    expect(eta(t, d(6e9, 6000), T0 + 6000 + STALE_MS + 1)).toBeNull();
    expect(eta(t, d(6e9, 6000), T0 + 6000 + 21_000)).toBeNull(); // rate idle
  });

  it('no ETA when the shown detail does not match the newest tracked sample', () => {
    const t = new RateTracker();
    steady(t, 6);
    expect(eta(t, d(5e9, 5000), T0 + 6000)).toBeNull(); // an older detail on screen
  });

  it('no ETA right after a reset (phase, total or counter going back)', () => {
    const t = new RateTracker();
    steady(t, 6);
    t.observe('j1', 'controller', d(1e9, 7000), T0 + 7000, T0 + 7000);
    expect(eta(t, d(1e9, 7000), T0 + 7000)).toBeNull();
    const p = new RateTracker();
    steady(p, 6);
    const next = d(0, 7000, { phase: 'verify_cache' });
    p.observe('j1', 'controller', next, T0 + 7000, T0 + 7000);
    expect(eta(p, next, T0 + 7000)).toBeNull();
  });

  it('does not average across a browser suspension: the series restarts', () => {
    const t = new RateTracker();
    steady(t, 6);
    // Tab suspended: the next feed arrives long after the last one.
    const back = T0 + 6000 + RATE_SUSPEND_MS + 1000;
    const y = d(9e9, back - T0);
    t.observe('j1', 'controller', y, back, back);
    expect(t.view('j1', 'controller', y, back)).toEqual({ kind: 'warming', samples: 1 });
    expect(eta(t, y, back)).toBeNull();
  });

  it('duplicate feeds keep the series alive while the tab is visible', () => {
    const t = new RateTracker();
    steady(t, 6);
    // the same sample seen again by later polls does not count as suspension
    for (let s = 7; s < 40; s += 5) t.observe('j1', 'controller', d(6e9, 6000), T0 + 6000, T0 + s * 1000);
    t.observe('j1', 'controller', d(7e9, 41_000), T0 + 41_000, T0 + 41_000);
    expect(t.view('j1', 'controller', d(7e9, 41_000), T0 + 41_000)?.kind).toBe('rate');
  });

  it('no ETA when the measured window spans a long producer gap', () => {
    const t = new RateTracker();
    // samples 0..6 s, then nothing for 30 s producer time, then one more (browser kept polling duplicates)
    steady(t, 6);
    for (let s = 10; s <= 36; s += 5) t.observe('j1', 'controller', d(6e9, 6000), T0 + 6000, T0 + s * 1000);
    const late = d(7e9, 37_000);
    t.observe('j1', 'controller', late, T0 + 37_000, T0 + 37_000);
    expect(t.view('j1', 'controller', late, T0 + 37_000)?.kind).toBe('rate'); // rate may still show, honestly labelled
    expect(eta(t, late, T0 + 37_000)).toBeNull();
  });

  it('feedRates path: legacy counter with job.updated_at gets an ETA from comparable samples', () => {
    const t = new RateTracker();
    for (let s = 0; s <= 8; s += 2) feedRates(t, [{ ...run, progress: { done: s, total: 100, unit: 'questions' }, updated_at: T0 + s * 1000 }], null, T0 + s * 1000);
    const legacy = readDetail({ phase: 'progress', unit: 'questions', completed: 8, total: 100 })!;
    // 1 question/s, 92 left
    expect(phaseEtaFor(run, t.view('j1', 'legacy', legacy, T0 + 8000), legacy, T0 + 8000)).toBe(92_000);
  });
});

describe('ETA text', () => {
  it('rounds to a readable duration', () => {
    expect(formatDuration(4_000)).toBe('less than 1 min');
    expect(formatDuration(90_000)).toBe('about 2 min');
    expect(formatDuration(7 * 60_000 + 10_000)).toBe('about 7 min');
    expect(formatDuration(23 * 60_000)).toBe('about 25 min');
    expect(formatDuration(58 * 60_000)).toBe('about 1 h');
    expect(formatDuration(83 * 60_000)).toBe('about 1 h 25 min');
    expect(formatDuration(12.4 * 3_600_000)).toBe('about 12 h');
    expect(formatDuration(3.2 * 86_400_000)).toBe('about 3 days');
    expect(formatDuration(9 * 86_400_000)).toBe('more than 7 days');
  });
  it('names the phase, not the whole job', () => {
    expect(etaText(23 * 60_000, false)).toBe('Phase time left: about 25 min (this phase only, at the measured rate)');
    expect(etaText(23 * 60_000, true)).toBe('about 25 min left in this phase');
    expect(etaText(null, false)).toBeNull();
  });
});
