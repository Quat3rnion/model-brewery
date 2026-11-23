import { describe, expect, it } from 'vitest';
import { bodyKey, PlainAttempt, retryKey, resultOfPreviousSubmission } from './retry-keys';
import { stripEmpty } from './submit.svelte';
import { orderedSuites } from './benchmark';

it('canonicalizes only set-valued wire fields before hashing', async () => {
  const first = stripEmpty({ kind: 'transfer', input_ids: ['b', 'a'] });
  const second = stripEmpty({ kind: 'transfer', input_ids: ['a', 'b'] });
  expect(first).toEqual(second);
  expect(first.input_ids).toEqual(['a', 'b']);
  expect(await retryKey('job', 'transfers.transfer', first)).toBe(await retryKey('job', 'transfers.transfer', second));
  expect(stripEmpty({ kind: 'compare', input_ids: ['b', 'a'] }).input_ids).toEqual(['b', 'a']);
  expect(stripEmpty({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[3, 1], [1, 2], [1, 0]] } }))
    .toEqual({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[1, 0], [1, 2], [3, 1]] } });
  expect(orderedSuites(['a', 'c', 'b'], ['b', 'a'])).toEqual(['b', 'a', 'c']);
  expect(orderedSuites(['a', 'b'], ['b', 'a'])).toEqual(['b', 'a']);
});

it('labels a result as previous only when the current body differs from its accepted request', () => {
  const accepted = bodyKey({ host: 'a', input_ids: ['old'] });
  expect(resultOfPreviousSubmission(accepted, { input_ids: ['old'], host: 'a' })).toBe(false);
  expect(resultOfPreviousSubmission(accepted, { host: 'a', input_ids: ['edited'] })).toBe(true);
  expect(resultOfPreviousSubmission(accepted, null)).toBe(true);
});

describe('retryKey', () => {
  it('uses SHA-256 of JSON with recursively sorted keys and leaves array order intact', async () => {
    const body = { z: [{ b: 2, a: 1 }, 3], a: 'x' };
    const key = await retryKey('failed-job.1', 'convert.plan', body);
    expect(key).toBe('r-failed-job.1-convert.plan-97d79fc3d0baf958eaed57b68971facdffd8d93447596de801da5fb3e583fb3e');
    expect(await retryKey('failed-job.1', 'convert.plan', { a: 'x', z: [{ a: 1, b: 2 }, 3] })).toBe(key);
    expect(await retryKey('failed-job.1', 'convert.plan', { ...body, z: [3, ...body.z.slice(0, 1)] })).not.toBe(key);
    expect(await retryKey('failed-job.1', 'convert.inspect', body)).not.toBe(key);
    expect(await retryKey('failed-job.2', 'convert.plan', body)).not.toBe(key);
    expect(bodyKey({ 2: 'two', 10: 'ten' })).toBe('{"10":"ten","2":"two"}');
  });

  it('keeps a long valid job ID whole and hashes IDs that cannot fit or contain invalid characters', async () => {
    const id = 'a._-'.repeat(24);
    const key = await retryKey(id, 'convert.managed', {});
    expect(key).toMatch(new RegExp(`^r-${id.replaceAll('.', '\\.')}-convert\\.managed-[a-f0-9]+$`));
    expect(key).toHaveLength(128);
    for (const invalid of ['bad/id 🚀', 'a'.repeat(128)]) {
      const fallback = await retryKey(invalid, 'convert.managed', {});
      expect(fallback).toMatch(/^[A-Za-z0-9._-]{1,128}$/);
      expect(fallback).not.toContain(invalid);
      expect(await retryKey(invalid, 'convert.managed', { different: true })).not.toBe(fallback);
      expect(await retryKey(`${invalid}!`, 'convert.managed', {})).not.toBe(fallback);
    }
  });
});

it('reuses an unconfirmed plain key only for the same launcher and body in the same visit', () => {
  const plain = new PlainAttempt();
  const body = { host: 'worker-a', input_ids: ['input-a'] };
  let current = true;
  const first = plain.take('convert.plan', body, () => current);
  expect(plain.take('convert.plan', { input_ids: ['input-a'], host: 'worker-a' }, () => current).key).toBe(first.key);
  current = false;
  const next = plain.take('convert.plan', body, () => true);
  expect(next.key).not.toBe(first.key);
  next.complete();
  const afterSuccess = plain.take('convert.plan', body, () => true);
  expect(afterSuccess.key).not.toBe(next.key);
  expect(plain.take('convert.inspect', body, () => true).key).not.toBe(afterSuccess.key);
});
