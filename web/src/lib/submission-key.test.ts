import { expect, it, vi } from 'vitest';
import { PlainAttempt } from './submission-key';
import { ReadVisit, stripEmpty, Submission } from './submit.svelte';
import { api, ApiError } from './api';

it('canonicalizes only set-valued wire fields', () => {
  const first = stripEmpty({ kind: 'transfer', input_ids: ['b', 'a'] });
  const second = stripEmpty({ kind: 'transfer', input_ids: ['a', 'b'] });
  expect(first).toEqual(second);
  expect(first.input_ids).toEqual(['a', 'b']);
  const plain = new PlainAttempt();
  expect(plain.take('transfers.transfer', first, () => true)!.key)
    .toBe(plain.take('transfers.transfer', second, () => true)!.key);
  expect(stripEmpty({ kind: 'compare', input_ids: ['b', 'a'] }).input_ids).toEqual(['b', 'a']);
  expect(stripEmpty({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[3, 1], [1, 2], [1, 0]] } }))
    .toEqual({ kind: 'weight_tune', input_ids: ['source', 'train', 'check'], options: { indices: [[1, 0], [1, 2], [3, 1]] } });
});

it('reuses an unconfirmed key only for the same launcher and body in the same visit', () => {
  const plain = new PlainAttempt();
  const body = { host: 'worker-a', input_ids: ['input-a'] };
  let current = true;
  const first = plain.take('convert.plan', body, () => current)!;
  expect(plain.take('convert.plan', { input_ids: ['input-a'], host: 'worker-a' }, () => current)?.key).toBe(first.key);
  current = false;
  const next = plain.take('convert.plan', body, () => true)!;
  expect(next.key).not.toBe(first.key);
  next.complete();
  const afterSuccess = plain.take('convert.plan', body, () => true)!;
  expect(afterSuccess.key).not.toBe(next.key);
  expect(plain.take('convert.inspect', body, () => true)?.key).not.toBe(afterSuccess.key);
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

it('guards a nearby repeat tap when the browser restarts the click count', async () => {
  const clock = vi.spyOn(performance, 'now').mockReturnValue(100);
  try {
    const read = new Submission();
    const post = async () => true;
    const first = { offset: 0 };
    const next = { offset: 128 };
    const tap = (clientX: number) => ({ detail: 1, pointerType: 'touch', clientX, clientY: 100 });
    expect(await read.send('paged-read', first, post, tap(42))).toBe(true);
    clock.mockReturnValue(500);
    expect(await read.send('paged-read', next, post, tap(43))).toBeNull();
    expect(await read.send('paged-read', next, post, tap(180))).toBe(true);
    clock.mockReturnValue(1301);
    expect(await read.send('paged-read', { offset: 256 }, post, tap(180))).toBe(true);
  } finally { clock.mockRestore(); }
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

it('notifies reads only after admission; a failed send does not replace the result', async () => {
  const read = new Submission();
  const body = { input_ids: ['source'] };
  const admitted = vi.fn();
  const clock = vi.spyOn(performance, 'now').mockReturnValue(100);
  try {
    let visible = await read.send('read.admission', body, async () => 'old inventory', { detail: 0 });
    const retry = async () => {
      const next = await read.send('read.admission', body, async () => { throw new Error('Read failed.'); }, { detail: 0 }, admitted);
      if (next !== null) visible = next;
    };
    await retry();
    expect(admitted).not.toHaveBeenCalled();
    expect(visible).toBe('old inventory');
    clock.mockReturnValue(901);
    await retry();
    expect(admitted).toHaveBeenCalledExactlyOnceWith(true);
    expect(read.error).toContain('Read failed.');
    expect(visible).toBe('old inventory');
    expect(await read.send('read.admission', body, async () => 'new inventory', { detail: 0 })).toBe('new inventory');
  } finally { clock.mockRestore(); }
});

it('lets read panes report a local rejection without admitting the send', async () => {
  const read = new Submission();
  const body = { prompt: 'x'.repeat(256 * 1024) };
  const attempted = vi.fn();
  const post = vi.fn();
  expect(await read.send('read.preflight', body, post, { detail: 0 }, attempted)).toBeNull();
  expect(attempted).toHaveBeenCalledExactlyOnceWith(false);
  expect(read.error).toContain('256 KiB');
  expect(post).not.toHaveBeenCalled();
});

it('shows a terminal read error only during the source visit that started it', () => {
  const visit = new ReadVisit();
  const queued = { id: 'read', state: 'queued' } as const;
  const failed = { ...queued, state: 'failed' } as const;
  expect(visit.shows(queued)).toBe(true);
  visit.remember(queued.id);
  expect(visit.shows(failed)).toBe(true);
  expect(new ReadVisit().shows(failed)).toBe(false);
  expect(new ReadVisit().shows(queued)).toBe(true);
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
  const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(Response.json({ error: 'Request rejected.' }, {
    status, headers: { 'Brewery-Error': '1' },
  }));
  try {
    await submission.send('preview', body, (_, key) => api.queueJob({ ...body, kind: 'inspect' }, key));
    expect(submission.error).toBe('Request rejected.');
    const failed = new Headers(fetch.mock.calls[0][1]?.headers).get('Idempotency-Key');
    expect(await submission.send('preview', body, async (_, key) => key)).not.toBe(failed);
  } finally { fetch.mockRestore(); }
});

it.each([
  [403, 'text/plain', 'Forbidden', ''],
  [403, 'text/plain', '{"error":"Forbidden"}', '1'],
  [502, 'text/html', '<h1>Bad gateway</h1>', ''],
  [503, 'application/json', '{"error":"Unavailable"}', '1'],
  [504, 'application/json', '{"error":"Timed out"}', ''],
  [422, 'application/json', '{"error":"Not the exact controller error shape","extra":true}', '1'],
  [403, 'application/json', '{"error":"Proxy rejection"}', ''],
] as const)('keeps the original key after an ambiguous HTTP %d reply', async (status, type, text, marker) => {
  const submission = new Submission();
  const job = { kind: 'generate', input_ids: ['model-a'], prompt: 'Fixture prompt' } as const;
  const body = { ...job, input_ids: [...job.input_ids] };
  const fetch = vi.spyOn(globalThis, 'fetch')
    .mockRejectedValueOnce(new TypeError('Reply lost.'))
    .mockResolvedValueOnce(new Response(text, { status, headers: { 'content-type': type, 'brewery-error': marker } }));
  const post = (request: typeof body, key: string) => api.queueJob(request, key);
  try {
    await submission.send('generate', body, post);
    await submission.send('generate', body, post);
    expect(submission.error).toContain('without reloading');
    const original = new Headers(fetch.mock.calls[0][1]?.headers).get('Idempotency-Key');
    expect(new Headers(fetch.mock.calls[1][1]?.headers).get('Idempotency-Key')).toBe(original);
    expect(await submission.send('generate', body, async (_, key) => key)).toBe(original);
  } finally { fetch.mockRestore(); }
});

it.each([70, 8])('allows new work after %d unresolved requests', async count => {
  const submission = new Submission();
  const prompt = count === 70 ? 'small' : 'a'.repeat(256 * 1024 - 128);
  let original = '';
  const lose = async (body: { offset: number }, key: string) => {
    if (!body.offset) original = key;
    throw new Error('Reply lost.');
  };
  for (let offset = 0; offset < count; offset++) await submission.send('generate', { prompt, offset }, lose);
  const post = vi.fn(async (_, key) => key);
  const newBody = { prompt: 'small', offset: count };
  expect(await submission.send('generate', newBody, post)).toMatch(/^conv-/);
  expect(await submission.send('generate', { prompt, offset: 0 }, post)).toBe(original);
});

it.each([
  { name: 'large ASCII', prompt: 'x'.repeat(1024 * 1024 - 128) },
  { name: 'UTF-8 byte limit', prompt: '\u00e9'.repeat(128 * 1024) },
])('rejects $name bodies before sending or affecting an earlier retry', async ({ prompt }) => {
  const submission = new Submission();
  const body = { prompt: 'small' };
  let original = '';
  await submission.send('generate', body, async (_, key) => {
    original = key;
    throw new Error('Reply lost.');
  });
  const post = vi.fn(async (_, key) => key);
  expect(await submission.send('generate', { prompt }, post)).toBeNull();
  expect(submission.error).toContain('256 KiB');
  expect(post).not.toHaveBeenCalled();
  expect(await submission.send('generate', body, post)).toBe(original);
});
