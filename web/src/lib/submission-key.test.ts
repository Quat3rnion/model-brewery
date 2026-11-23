import { expect, it, vi } from 'vitest';
import { PlainAttempt } from './submission-key';
import { stripEmpty, Submission } from './submit.svelte';
import { ApiError } from './api';

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

it('uses the native sequence across touch contacts and changed bodies, but not across launchers', async () => {
  const submission = new Submission();
  const other = new Submission();
  const body = { input_ids: ['model-a'] };
  const changed = { input_ids: ['model-b'] };
  const post = async () => true;
  const click = (detail: number, pointerId: number) => ({ detail, pointerId, pointerType: 'touch' });
  expect(await submission.send('native.preview', body, post, click(1, 1))).toBe(true);
  expect(await submission.send('native.preview', changed, post, click(2, 2))).toBeNull();
  expect(await other.send('native.copy', changed, post, click(3, 3))).toBe(true);
});

it('allows a repeat click with a different body when the first click was blocked', async () => {
  const submission = new Submission();
  let finish!: (value: boolean) => void;
  const body = { input_ids: ['model-a'] };
  const post = async () => true;
  const click = (detail: number, pointerId: number) => ({ detail, pointerId, pointerType: 'mouse' });
  const busy = submission.send('preview', body, () => new Promise<boolean>(resolve => { finish = resolve; }));
  expect(await submission.send('preview', body, post, click(1, 1))).toBeNull();
  finish(true);
  await busy;
  expect(await submission.send('preview', { input_ids: ['model-b'] }, post, click(2, 1))).toBe(true);
});

it('suppresses a fresh click or keyboard activation of an accepted body until 800 ms have elapsed', async () => {
  const clock = vi.spyOn(performance, 'now').mockReturnValue(100);
  try {
    const first = new Submission();
    const second = new Submission();
    const body = { input_ids: ['repeat'] };
    const post = async () => true;
    expect(await first.send('timed', body, post, { detail: 1 })).toBe(true);
    clock.mockReturnValue(850);
    expect(await second.send('timed', body, post, { detail: 1 })).toBeNull();
    expect(await second.send('timed', body, post, { detail: 0 })).toBeNull();
    expect(await second.send('timed', { input_ids: ['changed'] }, post, { detail: 1 })).toBe(true);
    clock.mockReturnValue(901);
    expect(await second.send('timed', body, post, { detail: 1 })).toBe(true);
  } finally { clock.mockRestore(); }
});

it('lets the keyboard retry an unconfirmed request immediately with its original key', async () => {
  const submission = new Submission();
  const body = { input_ids: ['unconfirmed'] };
  let original = '';
  await submission.send('timed.retry', body, async (_, key) => {
    original = key;
    throw new ApiError('Reply lost.', 502);
  }, { detail: 1 });
  expect(await submission.send('timed.retry', body, async (_, key) => key, { detail: 1 })).toBeNull();
  expect(await submission.send('timed.retry', body, async (_, key) => key, { detail: 0 })).toBe(original);
});

it('suppresses a keyboard activation after an earlier send is accepted out of order', async () => {
  const clock = vi.spyOn(performance, 'now').mockReturnValue(100);
  try {
    const body = { input_ids: ['source'] };
    const click = { detail: 0 };
    let acceptFirst!: (value: boolean) => void;
    let acceptSecond!: (value: boolean) => void;
    const first = new Submission().send('timed.overlap', body, () => new Promise<boolean>(resolve => { acceptFirst = resolve; }), click);
    clock.mockReturnValue(1000);
    const second = new Submission().send('timed.overlap', body, () => new Promise<boolean>(resolve => { acceptSecond = resolve; }), click);
    clock.mockReturnValue(1100);
    acceptFirst(true);
    await first;
    const third = await new Submission().send('timed.overlap', body, async () => true, click);
    acceptSecond(true);
    await second;
    expect(third).toBeNull();
  } finally { clock.mockRestore(); }
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

it.each([400, 409, 413, 415, 422])('discards the key after a definitive HTTP %d rejection', async (status) => {
  const submission = new Submission();
  const body = { input_ids: ['model-a'] };
  let failed = '';
  await submission.send('preview', body, async (_, key) => {
    failed = key;
    throw new ApiError('Request rejected.', status);
  });
  expect(await submission.send('preview', body, async (_, key) => key)).not.toBe(failed);
});

it('evicts old unconfirmed keys instead of growing the history indefinitely', () => {
  const plain = new PlainAttempt();
  const current = () => true;
  const first = plain.take('preview', { offset: 0 }, current).key;
  for (let offset = 1; offset <= 64; offset++) plain.take('preview', { offset }, current);
  const last = plain.take('preview', { offset: 64 }, current).key;
  expect(plain.take('preview', { offset: 0 }, current).key).not.toBe(first);
  expect(plain.take('preview', { offset: 64 }, current).key).toBe(last);
});
