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

it('suppresses native repeat clicks across requests and launchers without a time cutoff', async () => {
  const submission = new Submission();
  const other = new Submission();
  const body = { input_ids: ['model-a'] };
  const changed = { input_ids: ['model-b'] };
  const post = async () => true;
  const click = (detail: number, timeStamp: number) => ({ detail, timeStamp });
  expect(await submission.send('preview', body, post, click(1, 10))).toBe(true);
  expect(await submission.send('preview', body, post, click(2, 610))).toBeNull();
  expect(await other.send('copy', changed, post, click(3, 620))).toBeNull();
  expect(await other.send('copy', changed, post, click(1, 630))).toBe(true);
});

it('allows a repeat click when the first was blocked, without mixing pointers', async () => {
  const submission = new Submission();
  let finish!: (value: boolean) => void;
  const body = { input_ids: ['model-a'] };
  const post = async () => true;
  const click = (detail: number, pointerId: number) => ({ detail, pointerId, pointerType: 'mouse' });
  const busy = submission.send('preview', body, () => new Promise<boolean>(resolve => { finish = resolve; }));
  expect(await submission.send('preview', body, post, click(1, 1))).toBeNull();
  finish(true);
  await busy;
  expect(await submission.send('preview', body, post, click(2, 1))).toBe(true);
  expect(await submission.send('preview', body, post, click(3, 2))).toBe(true);
  expect(await submission.send('preview', body, post, { detail: 0 })).toBe(true);
});

it('keeps each unconfirmed wire request key through other sends until acceptance', async () => {
  const submission = new Submission();
  const a = { options: { offset: 0, limit: 128 }, input_ids: ['source'] };
  const b = { ...a, options: { ...a.options, offset: 128 } };
  let original = '';
  expect(await submission.send('inventory', a, async (_, key) => {
    original = key;
    throw new Error('Reply lost.');
  })).toBeNull();
  expect(await submission.send('inventory', b, async (_, key) => key)).not.toBe(original);
  expect(await submission.send('recipe', a, async (_, key) => key)).not.toBe(original);
  expect(await submission.send('inventory', { input_ids: a.input_ids, options: { limit: 128, offset: 0 } },
    async (_, key) => key)).toBe(original);
  expect(await submission.send('inventory', a, async (_, key) => key)).not.toBe(original);
});
