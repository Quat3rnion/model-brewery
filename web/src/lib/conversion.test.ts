import { describe, expect, it } from 'vitest';
import { buildConversion, parseConversion, parseConversions, stageRows, isActiveConversion, newerConversion, ConversionSubmitter, STAGES } from './conversion';

it('uses a new key for an identical plain conversion after retry query removal', async () => {
  const sent: string[] = [];
  const s = new ConversionSubmitter('convert.managed', async (_body, key) => {
    sent.push(key);
    return { data: { id: 'started', state: 'running' }, status: 201 };
  });
  await s.submit({ input_ids: ['source'] }, { blocked: false, failedId: 'retry-run' });
  expect((await s.submit({ input_ids: ['source'] }))?.id).toBe('started');
  expect(s.replayed).toBe(false);
  expect(sent).toHaveLength(2);
  expect(sent[0]).toMatch(/^r-retry-run-convert\.managed-/);
  expect(sent[1]).not.toBe(sent[0]);
});
import type { ConvertForm } from './convert';
import type { Model } from './types';

const models: Model[] = [
  { id: 'src', host: 'frame', path: '/m/ckpt', kind: 'hf' },
  { id: 'assets', host: 'frame', path: '/m/assets', kind: 'assets' },
];
const form = (over: Partial<ConvertForm> = {}): ConvertForm => ({
  source: 'src', bundle: '', extras: { assets: '', ngram: '', vision: '', mtp: '' }, host: '', anyHost: false,
  profile: 'ht43', ht_backend: '', i4r_backend: '', q4cp_backend: '', decode_backend: '', seed: null, threads: null, ht_beam: null, ...over,
});

const wf = (over: Record<string, unknown> = {}) => ({
  id: 'c1', host: 'frame', state: 'running', stage: 'plan', job_ids: { inspect: 'j-i', plan: 'j-p' }, cancel_requested: false,
  created_at: 1, updated_at: 2, input_ids: ['src'], options: { profile: 'ht43' }, ...over,
});

it('never replaces newer conversion progress with a delayed match', () => {
  const old = parseConversion(wf({ stage: 'inspect', updated_at: 10 }))!;
  const active = parseConversion(wf({ stage: 'verify', updated_at: 11 }))!;
  const done = parseConversion(wf({ state: 'succeeded', stage: 'verify', updated_at: 11 }))!;
  expect(newerConversion(active, old)).toBe(active);
  expect(newerConversion(done, { ...active, updatedAt: 11 })).toBe(done);
  expect(newerConversion(active, done)).toBe(done);
});

describe('conversion request', () => {
  it('builds one POST body: inputs, options with extra indexes, decode backend at top level', () => {
    const r = buildConversion(form({ extras: { assets: 'assets', ngram: '', vision: '', mtp: '' }, decode_backend: 'cuda', threads: 8 }), models);
    expect(r.problems).toEqual([]);
    expect(r.body).toEqual({ input_ids: ['src', 'assets'], options: { profile: 'ht43', assets: 1, threads: 8 }, decode_backend: 'cuda' });
  });
  it('requires an explicit profile and a source', () => {
    expect(buildConversion(form({ profile: '' }), models).problems).toContain('Choose the output profile.');
    expect(buildConversion(form({ source: '' }), models).problems).toContain('Select a source checkpoint.');
  });
  it('host: explicit, auto pool, or omitted for the source host', () => {
    expect(buildConversion(form({ host: 'gpu2' }), models).body?.host).toBe('gpu2');
    expect(buildConversion(form({ anyHost: true, host: 'gpu2' }), models).body?.host).toBe('auto');
    expect(buildConversion(form(), models).body).not.toHaveProperty('host');
  });
});

describe('workflow parsing', () => {
  it('reads the controller workflow and keeps unknown fields out', () => {
    const c = parseConversion(wf({ artifact_model_id: 'out1' }))!;
    expect(c).toMatchObject({ id: 'c1', state: 'running', stage: 'plan', jobIds: { inspect: 'j-i', plan: 'j-p' }, artifactModelId: 'out1', cancelRequested: false });
    expect(parseConversion({ id: 'x' })).toBeNull();
    expect(parseConversions([wf(), { bad: 1 }, wf({ id: 'c2' })]).map((x) => x.id)).toEqual(['c1', 'c2']);
    expect(parseConversions({ conversions: [] })).toEqual([]);
  });
  it('stage rows: done before the current stage, current from the job state, later pending; no invented results', () => {
    const rows = stageRows(parseConversion(wf())!, { 'j-i': 'succeeded', 'j-p': 'running' });
    expect(rows.map((r) => [r.stage, r.status, r.jobId])).toEqual([
      ['inspect', 'succeeded', 'j-i'],
      ['plan', 'running', 'j-p'],
      ['convert', 'pending', null],
      ['verify', 'pending', null],
    ]);
  });
  it('a failed or cancelled workflow marks later stages as not run', () => {
    const f = stageRows(parseConversion(wf({ state: 'failed', stage: 'convert', job_ids: { inspect: 'a', plan: 'b', convert: 'c' } }))!, { c: 'failed' });
    expect(f.map((r) => r.status)).toEqual(['succeeded', 'succeeded', 'failed', 'skipped']);
    const x = stageRows(parseConversion(wf({ state: 'cancelled', stage: 'plan' }))!, {});
    expect(x.map((r) => r.status)).toEqual(['succeeded', 'cancelled', 'skipped', 'skipped']);
  });
  it('active only while running', () => {
    expect(isActiveConversion(parseConversion(wf())!)).toBe(true);
    expect(isActiveConversion(parseConversion(wf({ state: 'succeeded', stage: 'verify' }))!)).toBe(false);
    expect(STAGES).toEqual(['inspect', 'plan', 'convert', 'verify']);
  });
});

describe('single submission', () => {
  it('sends exactly one POST per click even when clicked twice, and reuses the key on retry', async () => {
    const calls: { key: string; body: unknown }[] = [];
    let fail = true;
    const s = new ConversionSubmitter('convert.managed', async (body, key) => {
      calls.push({ key, body });
      await Promise.resolve();
      if (fail) {
        fail = false;
        throw new Error('network');
      }
      return { data: wf(), status: 201 };
    });
    const body = { input_ids: ['src'], options: { profile: 'ht43' } };
    const [a, b] = await Promise.all([s.submit(body), s.submit(body)]);
    expect(calls).toHaveLength(1);
    expect(a).toBeNull();
    expect(b).toBeNull();
    expect(s.error).toMatch(/network/);
    const c = await s.submit(body);
    expect(c?.id).toBe('c1');
    expect(calls).toHaveLength(2);
    expect(calls[1].key).toBe(calls[0].key);
    const d = await s.submit({ ...body, options: { profile: 'v2' } });
    expect(d?.id).toBe('c1');
    expect(calls[2].key).not.toBe(calls[0].key);
  });
});

describe('repeat start after success', () => {
  it('a second plain Start gets a fresh key even with the same accepted body', async () => {
    const keys: string[] = [];
    const s = new ConversionSubmitter('convert.managed', async (_b, key) => {
      keys.push(key);
      return { data: wf(), status: 201 };
    });
    const body = { input_ids: ['src'], options: { profile: 'ht43' } };
    await s.submit(body);
    expect((await s.submit(body))?.id).toBe('c1');
    expect(s.replayed).toBe(false);
    expect(keys).toHaveLength(2);
    expect(keys[1]).not.toBe(keys[0]);
  });
});

describe('saved profile revision', () => {
  const rev = 'd'.repeat(64);
  it('sends top-level profile_revision and omits options.profile', () => {
    const r = buildConversion(form({ profile: `rev:${rev}` }), models);
    expect(r.problems).toEqual([]);
    expect(r.body).toEqual({ input_ids: ['src'], options: {}, profile_revision: rev });
  });
  it('rejects a malformed revision value', () => {
    expect(buildConversion(form({ profile: 'rev:xyz' }), models).problems.join()).toMatch(/revision/);
  });
  it('requires HF and CPU backends for a custom profile', () => {
    const forced = buildConversion(form({ profile: `rev:${rev}`, q4cp_backend: 'hip' }), models);
    const gguf = buildConversion(form({ profile: `rev:${rev}` }), [{ ...models[0], kind: 'gguf' }]);
    expect(forced).toEqual({ body: null, problems: ['Select an HF source and CPU backends to use a custom profile.'] });
    expect(gguf).toEqual(forced);
  });
  it('does not build a request for an unavailable registered checkpoint', () => {
    expect(buildConversion(form(), [{ ...models[0], available: false }]).body).toBeNull();
  });
  it('workflow echoes profile identity', () => {
    const c = parseConversion(wf({ profile_revision: rev, profile_name: 'Fast attn', options: {} }))!;
    expect(c.profileRevision).toBe(rev);
    expect(c.profileName).toBe('Fast attn');
  });
});
