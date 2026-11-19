import { describe, expect, it } from 'vitest';
import { buildConvertJob, CONVERT_ENUMS, type ConvertForm } from './convert';
import { capability, progressFraction, hostName } from './format';
import { benchmarkLine, verificationLine, generationLine, isFullSplit } from './reports';
import { comparability, tokenRows } from './trace';
import { stripEmpty } from './submit.svelte';
import type { Host, JobDetail, JobSummary, Model } from './types';

const models: Model[] = [
  { id: 'src', host: 'frame', path: '/m/ckpt', kind: 'hf' },
  { id: 'assets', host: 'frame', path: '/m/assets', kind: 'assets' },
  { id: 'out', host: 'frame', path: '/m/out', kind: 'hgn' },
];
const form = (over: Partial<ConvertForm> = {}): ConvertForm => ({
  source: 'src', bundle: '', extras: { assets: '', ngram: '', vision: '', mtp: '' }, host: '', anyHost: false,
  profile: '', ht_backend: '', i4r_backend: '', q4cp_backend: '', decode_backend: '', seed: null, threads: null, ht_beam: null, ...over,
});
const job = (over: Partial<JobSummary>): JobSummary => ({ id: 'j1', kind: 'evaluate', state: 'succeeded', input_ids: ['out'], ...over });

describe('converter requests', () => {
  it('offers only the engine profiles v2 and ht43', () => {
    expect(CONVERT_ENUMS.profile).toEqual(['v2', 'ht43']);
  });
  it('maps extra inputs to zero-based input indexes', () => {
    const r = buildConvertJob('plan', form({ extras: { assets: 'assets', ngram: '', vision: '', mtp: '' }, profile: 'ht43' }), models);
    expect(r.job).toEqual({ kind: 'plan', host: undefined, input_ids: ['src', 'assets'], options: { assets: 1, profile: 'ht43' } });
  });
  it('sends host auto only for convert with any eligible worker', () => {
    expect(buildConvertJob('convert', form({ anyHost: true }), models).job?.host).toBe('auto');
    expect(buildConvertJob('plan', form({ anyHost: true }), models).job?.host).toBeUndefined();
  });
  it('verify takes original source then bundle, extras after', () => {
    const r = buildConvertJob('verify', form({ bundle: 'out', decode_backend: 'cuda', extras: { assets: 'assets', ngram: '', vision: '', mtp: '' } }), models);
    expect(r.job?.input_ids).toEqual(['src', 'out', 'assets']);
    expect(r.job?.options).toEqual({ decode_backend: 'cuda', assets: 2 });
  });
  it.each(['brewery-bundle', 'hgn-v2', 'ht43'])('does not submit a manual verify for output kind %s', (kind) => {
    const r = buildConvertJob('verify', form({ bundle: 'alias' }), [...models, { ...models[2], id: 'alias', kind }]);
    expect(r.job).toBeNull();
    expect(r.problems.join(' ')).toMatch(/hgn/i);
  });
  it('rejects out-of-range numbers and disallowed backends', () => {
    expect(buildConvertJob('convert', form({ threads: 65 }), models).problems).toContain('Threads must be 1 to 64.');
    expect(buildConvertJob('convert', form({ ht_beam: 257 }), models).problems).toContain('HT beam must be 1 to 256.');
    const hosts: Host[] = [{ id: 'frame', quantization_backends: { ht: ['cpu'] } }];
    expect(buildConvertJob('convert', form({ ht_backend: 'cuda' }), models, hosts).job).toBeNull();
    expect(buildConvertJob('convert', form({ i4r_backend: 'cuda' }), models, hosts).job).not.toBeNull();
  });
});

describe('capability gating', () => {
  it('treats unreported capability as unavailable with a reason', () => {
    const c = capability({ id: 'h' }, 'convert');
    expect(c.allowed).toBe(false);
    expect(c.reason).toMatch(/did not report/);
  });
  it('shows the controller reason', () => {
    expect(capability({ id: 'h', capabilities: { generate: false }, reasons: { generate: 'No Halogen' } }, 'generate')).toEqual({ allowed: false, reason: 'No Halogen' });
  });
  it('names pool placeholder host', () => {
    expect(hostName([], 'auto')).toMatch(/not yet assigned/);
  });
});

describe('report lines stay separate and never infer a pass', () => {
  it('finished verify without both verdicts is not a pass', () => {
    const d = { ...job({ kind: 'verify' }), result: { file_verification: { status: 'passed' } } } as JobDetail;
    expect(verificationLine(d, d).status).toBe('completed');
    const ok = { ...d, result: { file_verification: { status: 'passed' }, report: { passed: true } } } as JobDetail;
    expect(verificationLine(ok, ok).status).toBe('passed');
  });
  it('running verify is running, not qualified', () => {
    expect(verificationLine(job({ kind: 'verify', state: 'running' }), undefined).status).toBe('running');
  });
  it('generation text is never a quality pass', () => {
    const d = { ...job({ kind: 'generate' }), result: { text: 'hi' } } as JobDetail;
    expect(generationLine(d, d).status).toBe('completed');
  });
  const score = { correct: 1000, total: 1319, unparsed: 2 };
  it('requires dataset.full_split true and identity.synthetic false', () => {
    expect(isFullSplit({ dataset: { full_split: true }, identity: { synthetic: false } })).toBe(true);
    expect(isFullSplit({ dataset: { full_split: true }, identity: { synthetic: true } })).toBe(false);
    expect(isFullSplit({ score })).toBeNull();
  });
  it('full split is a standalone score; optional comparisons never become release verdicts', () => {
    const base = { score, dataset: { full_split: true }, identity: { synthetic: false } };
    const d = { ...job({}), result: base } as JobDetail;
    expect(benchmarkLine(d, d).label).toBe('Standalone GSM8K score');
    const p = { ...d, result: { ...base, comparison: { within_two_percentage_points: true, delta_correct: -3 } } } as JobDetail;
    expect(benchmarkLine(p, p).status).toBe('completed');
    const f = { ...d, result: { ...base, comparison: { within_two_percentage_points: false } } } as JobDetail;
    expect(benchmarkLine(f, f).status).toBe('completed');
  });
  it('synthetic or unconfirmed scores do not qualify', () => {
    const d = { ...job({}), result: { score, dataset: { full_split: true }, identity: { synthetic: true }, comparison: { within_two_percentage_points: true } } } as JobDetail;
    expect(benchmarkLine(d, d).status).toBe('not-qualifying');
  });
});

describe('trace view', () => {
  it('maps step candidates to the last token of each step', () => {
    const rows = tokenRows({
      tokens: [{ i: 0, id: 1, piece: 'a', phase: 'prefill' }, { i: 1, id: 2, piece: 'b', phase: 'prefill' }, { i: 2, id: 3, piece: 'c', phase: 'decode' }],
      steps: [{ first: 0, n: 2, phase: 'prefill' }, { first: 2, n: 1, phase: 'decode' }],
      next_candidates: [[{ id: 3, piece: 'c', p: 0.5 }], [{ id: 9, piece: 'z', p: 0.25 }]],
    });
    expect(rows.map((r) => r.candidates === '')).toEqual([true, false, false]);
    expect(rows[1].candidates).toContain('#3');
    expect(rows[2].candidates).toContain('#9');
  });
  it('prefix_only does not claim directly comparable values', () => {
    expect(comparability({ status: 'prefix_only', matched_prefix_tokens: 4 }).comparable).toBe(false);
    expect(comparability(undefined).comparable).toBe(false);
  });
});

describe('misc', () => {
  it('reads string progress', () => {
    expect(progressFraction('copying 50%')).toBe(0.5);
    expect(progressFraction('3/4 files')).toBe(0.75);
    expect(progressFraction('starting')).toBeNull();
  });
  it('strips empty options', () => {
    expect(stripEmpty({ kind: 'compare', input_ids: ['a', 'b'], prompt: 'x', options: { n: 16, threads: undefined, gpu_device: '' } })).toEqual({
      kind: 'compare', input_ids: ['a', 'b'], prompt: 'x', options: { n: 16 },
    });
  });
});

import { readSignals, fmtExperts, fmtSignal, MAX_EXPERTS_SHOWN } from './signals';
describe('layer signals', () => {
  const raw = {
    schema: 1,
    layers: [
      { layer: 1, kind: 'attention', cells: [{ token: 3, attn: 0.5, experts: [4, 9], weights: [0.6, 0.4] }] },
      { layer: 0, kind: 'linear', cells: [{ token: 3, attn: 1.25, ffn: 2 }, { token: 0, ffn: 0 }] },
    ],
    coverage: { total_layers: 24, total_tokens: 6, captured_cells: 3, included_cells: 3, layer_span: [0, 1], token_span: [0, 3], truncated: true },
    omissions: { missing_cells: 0, skipped_cells: 141, unavailable: { attn: 1, ffn: 1, experts: 2, weights: 2 }, inline_excluded: [] },
  };
  it('groups cells by position, sorted by layer, missing fields stay null', () => {
    const s = readSignals(raw)!;
    expect(s.positions).toEqual([0, 3]);
    expect(s.at(3).map((c) => c.layer)).toEqual([0, 1]);
    expect(s.at(3)[1].ffn).toBeNull();
    expect(s.at(0)[0].ffn).toBe(0);
    expect(s.at(0)[0].attn).toBeNull();
    expect(s.coverage.truncated).toBe(true);
    expect(s.rowsAt(0)).toEqual([{ layer: 0, cell: s.at(0)[0] }, { layer: 1, cell: null }]);
  });
  it('old receipts or unknown schema read as no signals', () => {
    expect(readSignals(undefined)).toBeNull();
    expect(readSignals({ schema: 2, layers: [] })).toBeNull();
  });
  it('caps listed experts explicitly', () => {
    const many = Array.from({ length: MAX_EXPERTS_SHOWN + 3 }, (_, i) => i);
    expect(fmtExperts(many, null)).toMatch(/\+3 more$/);
    expect(fmtExperts([4, 9], [0.6, 0.0745])).toBe('4 (0.6), 9 (0.0745)');
    expect(fmtSignal(1.2064)).toBe('1.2064');
    expect(fmtSignal(190.5358)).toBe('190.5358');
    expect(fmtSignal(0.1 + 0.2)).toBe('0.3');
    expect(fmtExperts([4, 9], [0.6])).toBe('4, 9');
  });
});
