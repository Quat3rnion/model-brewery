import { expect, it } from 'vitest';
import { benchmarkLine, lineTone, verificationLine } from './reports';
import type { JobDetail } from './types';

const evaluation: JobDetail = { kind: 'evaluate', id: 'evaluation', state: 'succeeded', input_ids: ['bundle'] };
const score = { correct: 3, total: 4 };
const full = { score, dataset: { full_split: true }, identity: { synthetic: false } };

it.each([
  ['a synthetic score', { score, identity: { synthetic: true } }, 'Synthetic fixture'],
  ['a partial split', { score, dataset: { full_split: false } }, 'Not the full split'],
  ['an unconfirmed split', { score }, 'Split not confirmed'],
  ['a standalone full split', full, 'Standalone GSM8K score'],
  ['a full split with a reference', { ...full, comparison: { delta_correct: -1 } }, 'GSM8K score with reference'],
] as const)('GSM8K reports %s without a release claim', (_name, result, label) => {
  const line = benchmarkLine(evaluation, { ...evaluation, result });
  expect(line.label).toBe(label);
  expect(line.detail).toContain('3 / 4 correct (75.00%)');
  expect(line.detail).toMatch(/Not release-qualified\.$/);
  expect(line.detail).not.toContain('does not qualify a release');
});

it.each([
  [null, undefined],
  [{ ...evaluation, state: 'queued' }, undefined],
  [evaluation, undefined],
  [evaluation, { ...evaluation, result: {} }],
])('GSM8K shows the release disclaimer even without a score', (job, detail) => {
  expect(benchmarkLine(job, detail).detail).toMatch(/Not release-qualified\.$/);
});

it.each([
  ['not_performed', false, 'Not performed', 'muted'],
  ['unknown', undefined, 'No verdict', 'muted'],
  ['failed', undefined, 'Verification failed', 'bad'],
  ['passed', true, 'Files verified', 'ok'],
] as const)('shows file verification status %s with the correct label and tone', (status, passed, label, tone) => {
  const job = { ...evaluation, kind: 'verify' };
  const line = verificationLine(job, { ...job, result: { file_verification: { status }, report: { passed } } });
  expect(line.label).toBe(label);
  expect(lineTone(line.status)).toBe(tone);
});
