import { describe, expect, it } from 'vitest';
import { parseBuildReport, readBuildFileBounded, MAX_BUILD_FILE_BYTES } from './buildreports';

const side = (job: string, profile: string, bundle: string) => ({
  job_id: job,
  policy: 'nf4-v1',
  source: { kind: 'hf', sha256: 'a'.repeat(64), alias: 'org/model', model: null, revision: null, branch: null },
  profile,
  options: { seed: 1 },
  tool: { name: 'brewery-convert', sha256: null, encoder_sha256: 'e'.repeat(64) },
  artifact: { bundle_sha256: bundle, bytes: 1000 },
  runtime: null,
});

const diff = (over: Record<string, unknown> = {}) => ({
  schema: 'brewery-build-diff-v1',
  left: side('job-ref', 'v2', 'b'.repeat(64)),
  right: side('job-cand', 'ht43', 'c'.repeat(64)),
  comparison: { checkpoint: 'same', source: 'same', profile: 'changed', options: 'same', tool: 'unknown', encoder: 'same', runtime: 'unknown', quantization_only: false },
  files: [
    { path: 'model.hgn', role: 'checkpoint', status: 'changed', left: { bytes: 900, sha256: 'd'.repeat(64) }, right: { bytes: 800, sha256: 'f'.repeat(64) }, delta_bytes: -100 },
    { path: 'model.hgn.rot', role: 'sidecar', status: 'added', left: null, right: { bytes: 40, sha256: '1'.repeat(64) }, delta_bytes: 40 },
    { path: 'manifest.json', role: 'manifest', status: 'unchanged', left: { bytes: 10, sha256: '2'.repeat(64) }, right: { bytes: 10, sha256: '2'.repeat(64) }, delta_bytes: 0 },
    { path: 'tok.json', role: 'assets', status: 'changed', left: { bytes: 5, sha256: null }, right: null, delta_bytes: null },
  ],
  bytes_by_role: {
    checkpoint: { left_bytes: 900, right_bytes: 800, delta_bytes: -100 },
    sidecar: { left_bytes: 0, right_bytes: 40, delta_bytes: 40 },
    assets: { left_bytes: null, right_bytes: null, delta_bytes: null },
  },
  limits: ['Byte sizes do not measure quality.'],
  ...over,
});

const quality = (over: Record<string, unknown> = {}) => ({
  schema: 'brewery-paired-quality-v1',
  dataset: { name: 'openai/grade-school-math', revision: 'r', split: 'test', sha256: 's'.repeat(64), rows: 1319, full_split: true },
  protocol_id: 'gsm8k-v1',
  synthetic: false,
  left: { job_id: 'e-ref', kind: 'evaluate', host: 'frame', source_sha256: 'a'.repeat(64), predictions: { sha256: 'p'.repeat(64), bytes: 10 }, score: { correct: 1000, total: 1319, unparsed: 2, accuracy_basis_points: 7581 }, provenance: { method: 'halogen', manifest_sha256: 'm'.repeat(64), files: null, tokenizer: null, runtime: null } },
  right: { job_id: 'e-cand', kind: 'evaluate', host: 'frame', source_sha256: 'a'.repeat(64), predictions: { sha256: 'q'.repeat(64), bytes: 10 }, score: { correct: 990, total: 1319, unparsed: 4, accuracy_basis_points: 7505 }, provenance: { method: 'halogen', manifest_sha256: 'n'.repeat(64), files: null, tokenizer: null, runtime: null } },
  transitions: { both_correct: 950, both_incorrect: 279, improvements: 40, regressions: 50 },
  improvement_ids: Array.from({ length: 40 }, (_, i) => i + 1),
  regression_ids: Array.from({ length: 50 }, (_, i) => 100 + i),
  parsed_answer_change_ids: [3, 7],
  delta_correct: -10,
  delta_percentage_points: -0.758,
  uncertainty: { method: 'exact-mcnemar-two-sided', discordant_pairs: 90, p_value: 0.343, p_value_exact: { numerator: '343', denominator: '1000' }, assumption: 'Questions are independent.' },
  warnings: ['Runtime identity unknown for both sides.'],
  ...over,
});

describe('build diff', () => {
  it('reads sides, files with sidecars, role totals and identity states', () => {
    const r = parseBuildReport(JSON.stringify(diff()));
    expect(r.kind).toBe('build-diff');
    if (r.kind !== 'build-diff') return;
    const d = r.report;
    expect(d.left.jobId).toBe('job-ref');
    expect(d.right.profile).toBe('ht43');
    expect(d.left.source.model).toBeNull();
    expect(d.left.policy).toBe('nf4-v1');
    expect(d.left.source.alias).toBe('org/model');
    expect(d.files.find((f) => f.role === 'sidecar')).toMatchObject({ status: 'added', left: null, deltaBytes: 40 });
    expect(d.files.find((f) => f.role === 'assets')?.deltaBytes).toBeNull();
    expect(d.roles.map((x) => x.role)).toEqual(['checkpoint', 'sidecar', 'assets']);
    expect(d.roles[2].deltaBytes).toBeNull();
    expect(d.comparison.find((c) => c.key === 'tool')?.state).toBe('unknown');
    expect(d.quantizationOnly).toBe(false);
  });
  it('quantization_only true only for a literal true; anything else is false', () => {
    const t = parseBuildReport(JSON.stringify(diff({ comparison: { ...diff().comparison, quantization_only: true } })));
    const s = parseBuildReport(JSON.stringify(diff({ comparison: { ...diff().comparison, quantization_only: 'true' } })));
    expect(t.kind === 'build-diff' && t.report.quantizationOnly).toBe(true);
    expect(s.kind === 'build-diff' && s.report.quantizationOnly).toBe(false);
  });
  it('an unknown identity state is shown as unknown, not same', () => {
    const r = parseBuildReport(JSON.stringify(diff({ comparison: { checkpoint: 'maybe' } })));
    if (r.kind !== 'build-diff') throw new Error('expected diff');
    expect(r.report.comparison.find((c) => c.key === 'checkpoint')?.state).toBe('unknown');
    expect(r.report.comparison.find((c) => c.key === 'runtime')?.state).toBe('unknown');
  });
  it('rejects a diff without left/right or files', () => {
    expect(parseBuildReport(JSON.stringify(diff({ files: 'x' }))).kind).toBe('error');
    expect(parseBuildReport(JSON.stringify(diff({ left: null }))).kind).toBe('error');
  });
});

describe('paired quality', () => {
  it('reads sides, transitions, IDs and exact McNemar uncertainty', () => {
    const r = parseBuildReport(JSON.stringify(quality()));
    expect(r.kind).toBe('paired-quality');
    if (r.kind !== 'paired-quality') return;
    const q = r.report;
    expect(q.left.jobId).toBe('e-ref');
    expect(q.left.score?.accuracyBasisPoints).toBe(7581);
    expect(q.transitions).toEqual({ bothCorrect: 950, bothIncorrect: 279, improvements: 40, regressions: 50 });
    expect(q.improvementIds).toHaveLength(40);
    expect(q.deltaCorrect).toBe(-10);
    expect(q.uncertainty).toMatchObject({ method: 'exact-mcnemar-two-sided', discordantPairs: 90, pValue: 0.343, pExact: '343/1000' });
    expect(q.datasetFull).toBe(true);
    expect(q.synthetic).toBe(false);
  });
  it('missing numbers stay null, never 0; synthetic stays unknown unless stated', () => {
    const r = parseBuildReport(JSON.stringify(quality({ delta_correct: null, transitions: { improvements: 3 }, synthetic: undefined, uncertainty: { method: 'exact-mcnemar-two-sided', p_value: null, p_value_exact: { numerator: '1', denominator: '1e400' } } })));
    if (r.kind !== 'paired-quality') throw new Error('expected quality');
    expect(r.report.deltaCorrect).toBeNull();
    expect(r.report.transitions.bothCorrect).toBeNull();
    expect(r.report.synthetic).toBeNull();
    expect(r.report.uncertainty.pValue).toBeNull();
    expect(r.report.uncertainty.pExact).toBe('1/1e400');
  });
  it('drops non-integer question IDs and keeps order', () => {
    const r = parseBuildReport(JSON.stringify(quality({ improvement_ids: [5, 'x', 2.5, 9, -1] })));
    if (r.kind !== 'paired-quality') throw new Error('expected quality');
    expect(r.report.improvementIds).toEqual([5, 9]);
  });
});

describe('detection and limits', () => {
  it('gives short errors without file content', () => {
    const r = parseBuildReport(JSON.stringify({ schema: 'brewery-expert-usage-v1', prompt: 'secret text' }));
    expect(r.kind).toBe('error');
    if (r.kind === 'error') {
      expect(r.message).toMatch(/not supported/i);
      expect(r.message).not.toMatch(/secret/);
    }
    expect(parseBuildReport('{bad').kind).toBe('error');
  });
  it('rejects oversize files before reading', async () => {
    let read = false;
    const r = await readBuildFileBounded({ name: 'x', size: MAX_BUILD_FILE_BYTES + 1, text: async () => ((read = true), '{}') });
    expect(r.kind).toBe('error');
    expect(read).toBe(false);
  });
});
