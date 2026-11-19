import { describe, expect, it } from 'vitest';
import { detailLines, feedRates, readDetail, type ProgressDetail } from './progress';
import { RateTracker } from './rate';
import { describeStep, StepTracker } from './steps';
import type { JobSummary } from './types';

// t_0c3a5555: tell a new check in the same job apart from the next job.
const T0 = 1_800_000_000_000;
const UUID = '282d971f-2a0c-4e95-a028-54e69801b455';
const NEXT = 'cf995a44-7909-4bbe-b7d0-1829e7500988';
const d = (phase: string, completed: number, ms: number, over: Record<string, unknown> = {}): ProgressDetail =>
  readDetail({ phase, unit: 'bytes', completed, total: 1e12, input_index: 1, input_count: 1, updated_at_ms: T0 + ms, ...over })!;
const plan = (id = UUID, over: Partial<JobSummary> = {}): JobSummary => ({ id, kind: 'plan', state: 'running', input_ids: ['m1'], options: { profile: 'ht43' }, ...over });

describe('StepTracker', () => {
  it('records a phase change in the same job as a new check', () => {
    const t = new StepTracker();
    t.observe(UUID, d('hash_input', 0, 0));
    t.observe(UUID, d('hash_input', 5e11, 1000));
    t.observe(UUID, d('verify_input', 0, 2000));
    expect(t.history(UUID).map((s) => [s.phase, s.reason])).toEqual([
      ['hash_input', 'first'],
      ['verify_input', 'phase'],
    ]);
  });
  it('ignores duplicate and out-of-order samples', () => {
    const t = new StepTracker();
    t.observe(UUID, d('hash_input', 10, 1000));
    expect(t.observe(UUID, d('verify_input', 0, 1000))).toBe(false);
    expect(t.observe(UUID, d('verify_input', 0, 500))).toBe(false);
    expect(t.history(UUID)).toHaveLength(1);
  });
  it('records the next input and a counter that goes back', () => {
    const t = new StepTracker();
    t.observe(UUID, d('hash_input', 100, 0, { input_count: 2 }));
    t.observe(UUID, d('hash_input', 0, 1000, { input_index: 2, input_count: 2 }));
    t.observe(UUID, d('hash_input', 50, 2000, { input_index: 2, input_count: 2 }));
    t.observe(UUID, d('hash_input', 10, 3000, { input_index: 2, input_count: 2 }));
    expect(t.history(UUID).map((s) => s.reason)).toEqual(['first', 'input', 'restart']);
  });
  it('keeps jobs apart by ID only: the next job starts its own history', () => {
    const t = new StepTracker();
    t.observe(UUID, d('hash_input', 0, 0));
    t.observe(UUID, d('verify_input', 0, 1000));
    t.observe(NEXT, d('hash_input', 0, 2000));
    expect(t.history(NEXT).map((s) => s.reason)).toEqual(['first']);
    expect(t.retain(new Set([NEXT]))).toBe(true);
    expect(t.history(UUID)).toEqual([]);
  });
});

describe('describeStep', () => {
  const run = (job: JobSummary, rows: ProgressDetail[]) => {
    const t = new StepTracker();
    for (const r of rows) t.observe(job.id, r);
    return describeStep(job, rows[rows.length - 1], t.history(job.id));
  };

  it('hash_input is the worker fingerprint of this job, with kind, profile and job ID', () => {
    const s = run(plan(), [d('hash_input', 0, 0)]);
    expect(s.role).toBe('Worker fingerprint · Converter plan, HT43 profile · job 282d971f');
    expect(s.compactRole).toBe('Worker fingerprint, HT43 profile');
    expect(s.purpose).toBe('The worker reads every input file and records its SHA-256 for this job.');
    expect(s.transition).toBeNull();
    expect(s.compactTransition).toBeNull();
    expect(s.sequence).toBeNull();
  });

  it('hash_input then verify_input with the same UUID: same job, new check, compared with the fingerprint', () => {
    const s = run(plan(), [d('hash_input', 0, 0), d('hash_input', 1e12, 60_000), d('verify_input', 0, 61_000)]);
    expect(s.role).toBe('Engine reread · Converter plan, HT43 profile · job 282d971f');
    expect(s.compactRole).toBe('Engine reread, HT43 profile');
    expect(s.purpose).toBe('The engine reads the input again and compares it with the worker fingerprint from earlier in this job.');
    expect(s.transition).toBe('Same job, new check: the counter restarted because this job moved from Hashing input to Verifying input.');
    expect(s.compactTransition).toBe('Same job, new check');
    expect(s.sequence).toBe('Checks seen on this page for this job: 1. Worker fingerprint → 2. Engine reread (now)');
  });

  it('verify_input without an observed fingerprint does not claim a comparison', () => {
    const s = run(plan(), [d('verify_input', 0, 0)]);
    expect(s.purpose).toBe('The engine reads the input again and checks its SHA-256.');
    expect(s.purpose).not.toMatch(/fingerprint|manifest/);
    expect(s.transition).toBeNull();
  });

  it('a fingerprint of another input is not used for the comparison claim', () => {
    const s = run(plan(), [d('hash_input', 0, 0, { input_index: 1, input_count: 2 }), d('verify_input', 0, 1000, { input_index: 2, input_count: 2 })]);
    expect(s.purpose).toBe('The engine reads the input again and checks its SHA-256.');
  });

  it('the next plan with a different ID is a separate job: no same-job note, own ID', () => {
    const t = new StepTracker();
    t.observe(UUID, d('hash_input', 0, 0));
    t.observe(UUID, d('verify_input', 0, 1000));
    t.observe(NEXT, d('hash_input', 0, 2000));
    const s = describeStep(plan(NEXT), d('hash_input', 0, 2000), t.history(NEXT));
    expect(s.role).toBe('Worker fingerprint · Converter plan, HT43 profile · job cf995a44');
    expect(s.transition).toBeNull();
    expect(s.compactTransition).toBeNull();
  });

  it('inspect and plan are named apart; a missing profile is not invented', () => {
    expect(run(plan(UUID, { kind: 'inspect', options: {} }), [d('hash_input', 0, 0)]).role).toBe('Worker fingerprint · Converter inspect · job 282d971f');
    expect(run(plan(UUID, { kind: 'inspect', options: {} }), [d('hash_input', 0, 0)]).compactRole).toBe('Worker fingerprint');
    expect(run(plan(UUID, { options: { profile: 'v2' } }), [d('hash_input', 0, 0)]).role).toBe('Worker fingerprint · Converter plan, HGN v2 profile · job 282d971f');
  });

  it('evaluator and transfer roles; next input and restart notes', () => {
    expect(run(plan(UUID, { kind: 'evaluate', options: {} }), [d('verify_input', 0, 0)]).role).toBe('Evaluator input check · GSM8K evaluation · job 282d971f');
    expect(run(plan(UUID, { kind: 'transfer', options: {} }), [d('hash_input', 0, 0)]).purpose).toBe('The source host reads every file and records its SHA-256 before the copy.');
    const next = run(plan(), [d('hash_input', 9, 0, { input_count: 2 }), d('hash_input', 0, 1000, { input_index: 2, input_count: 2 })]);
    expect(next.transition).toBe('Same job, next input: the counter restarted for input 2 of 2.');
    expect(next.compactTransition).toBe('Same job, next input');
    const back = run(plan(), [d('hash_input', 9, 0), d('hash_input', 1, 1000)]);
    expect(back.transition).toBe('Same job: the counter restarted because the job reported a lower value.');
  });

  it('an unknown phase keeps its own name and gets no invented purpose', () => {
    const s = run(plan(), [d('some_new_phase', 0, 0)]);
    expect(s.role).toBe('Some new phase · Converter plan, HT43 profile · job 282d971f');
    expect(s.purpose).toBeNull();
  });
});

describe('feedRates also feeds steps', () => {
  it('records controller phases per job and drops terminal jobs', () => {
    const r = new RateTracker();
    const s = new StepTracker();
    feedRates(r, [plan(UUID, { progress_detail: d('hash_input', 0, 0) })], null, T0, s);
    feedRates(r, [plan(UUID, { progress_detail: d('verify_input', 0, 1000) })], null, T0 + 1000, s);
    expect(s.history(UUID).map((x) => x.phase)).toEqual(['hash_input', 'verify_input']);
    feedRates(r, [plan(UUID, { state: 'succeeded', progress_detail: d('verify_input', 0, 1000) })], null, T0 + 2000, s);
    expect(s.history(UUID)).toEqual([]);
  });
});

describe('pinned manifest reuse (t_d69dfe35)', () => {
  const run = (job: JobSummary, rows: ProgressDetail[]) => {
    const t = new StepTracker();
    for (const r of rows) t.observe(job.id, r);
    return describeStep(job, rows[rows.length - 1], t.history(job.id));
  };
  const files = (phase: string, completed: number, ms: number) => d(phase, completed, ms, { unit: 'files', total: 131 });

  it('pinned_manifest: operator-pinned checksums, names and sizes checked, contents not read', () => {
    const s = run(plan(), [files('pinned_manifest', 10, 0)]);
    expect(s.role).toBe('Pinned manifest check · Converter plan, HT43 profile · job 282d971f');
    expect(s.compactRole).toBe('Pinned manifest check, HT43 profile');
    expect(s.purpose).toBe('The worker uses the operator-pinned checksums. It checks file names and sizes only; it does not read file contents.');
    expect(detailLines(readDetail(files('pinned_manifest', 10, 0))!, T0).main).toBe('Checking pinned manifest: 10 of 131 files (7%)');
  });

  it('verify_pinned_input: one full read against pinned checksums, never a reread or a worker comparison', () => {
    const s = run(plan(), [files('pinned_manifest', 131, 0), d('verify_pinned_input', 0, 1000)]);
    expect(s.role).toBe('Engine full read · Converter plan, HT43 profile · job 282d971f');
    expect(s.purpose).toBe('The engine reads every input byte once and compares it with the operator-pinned checksums.');
    expect(s.transition).toBe('Same job, new check: the counter restarted because this job moved from Checking pinned manifest to Verifying input against pinned checksums.');
    expect(s.compactTransition).toBe('Same job, new check');
    expect(s.sequence).toBe('Checks seen on this page for this job: 1. Pinned manifest check → 2. Engine full read (now)');
    for (const text of [s.role, s.purpose, s.transition, s.sequence]) expect(text).not.toMatch(/\breread\b|\bagain\b|worker fingerprint/i);
  });

  it('verify_pinned_input after a worker hash still does not claim a worker comparison', () => {
    const s = run(plan(), [d('hash_input', 0, 0), d('verify_pinned_input', 0, 1000)]);
    expect(s.purpose).toBe('The engine reads every input byte once and compares it with the operator-pinned checksums.');
  });

  it('the next job with a different ID has no same-job note', () => {
    const t = new StepTracker();
    t.observe(UUID, files('pinned_manifest', 0, 0));
    t.observe(UUID, d('verify_pinned_input', 0, 1000));
    t.observe(NEXT, files('pinned_manifest', 0, 2000));
    const s = describeStep(plan(NEXT), files('pinned_manifest', 0, 2000), t.history(NEXT));
    expect(s.role).toBe('Pinned manifest check · Converter plan, HT43 profile · job cf995a44');
    expect(s.transition).toBeNull();
    expect(s.sequence).toBeNull();
  });

  it('file lines: pinned check names the file it checks; the full read says Reading', () => {
    expect(detailLines(readDetail({ ...files('pinned_manifest', 1, 0), current_file: 'a.safetensors' })!, T0).extra).toContain('Checking name and size: a.safetensors');
    expect(detailLines(readDetail({ ...d('verify_pinned_input', 1, 0), current_file: 'a.safetensors' })!, T0).extra).toContain('Reading: a.safetensors');
  });
});
