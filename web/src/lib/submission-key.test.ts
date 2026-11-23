import { expect, it } from 'vitest';
import { PlainAttempt } from './submission-key';
import { stripEmpty, Submission } from './submit.svelte';

it('canonicalizes only set-valued wire fields', () => {
  const first = stripEmpty({ kind: 'transfer', input_ids: ['b', 'a'] });
  const second = stripEmpty({ kind: 'transfer', input_ids: ['a', 'b'] });
  expect(first).toEqual(second);
  expect(first.input_ids).toEqual(['a', 'b']);
  const plain = new PlainAttempt();
  expect(plain.take('transfers.transfer', first, () => true).key)
    .toBe(plain.take('transfers.transfer', second, () => true).key);
  expect(stripEmpty({ kind: 'compare', input_ids: ['b', 'a'] }).input_ids).toEqual(['b', 'a']);
  expect(stripEmpty({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[3, 1], [1, 2], [1, 0]] } }))
    .toEqual({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[1, 0], [1, 2], [3, 1]] } });
});

it('reuses an unconfirmed key only for the same launcher and body in the same visit', () => {
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

it('suppresses a multi-click only after a recent send with the same launcher and body', async () => {
  const submission = new Submission();
  const body = { input_ids: ['model-a'] };
  const changed = { input_ids: ['model-b'] };
  const post = async () => true;
  const click = (detail: number, timeStamp: number) => ({ detail, timeStamp });
  expect(await submission.send('preview', body, post, click(2, 10))).toBe(true);
  expect(await submission.send('preview', body, post, click(3, 20))).toBeNull();
  expect(await submission.send('preview', changed, post, click(4, 30))).toBe(true);
  expect(await submission.send('copy', changed, post, click(5, 40))).toBe(true);
  expect(await submission.send('copy', changed, post, click(6, 600))).toBe(true);
  expect(await submission.send('copy', changed, post, click(1, 610))).toBe(true);
});
