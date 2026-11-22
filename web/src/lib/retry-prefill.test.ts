import { describe, expect, it, vi } from 'vitest';
import { availableId, historicalPrompt, RetryKeys, RetryPrefill, omittedInputs } from './retry-prefill.svelte';
import { RouteActivity } from './route.svelte';

let slot = 0;
function setup() {
  const defaults = { host: '', limit: undefined as number | undefined, fields: { name: '', seed: 0 } };
  let draft = structuredClone(defaults);
  let resolve!: (name: string) => void;
  const load = vi.fn(() => new Promise<string>(done => { resolve = done; }));
  const focus = vi.fn();
  const clear = vi.fn();
  const restore = vi.fn((name: string) => ({ actions: ['queue'], patch: { host: 'worker', limit: 7, fields: { name, seed: 23 } }, focus: 'first' }));
  const form = {
    slot: `test-${++slot}`,
    defaults,
    read: () => draft,
    write: (value: typeof defaults) => { draft = value; },
    load,
    restore,
    focus,
    clear,
  };
  const retry = new RetryPrefill(form);
  return { retry, form, load, restore, focus, clear, draft: () => draft, resolve: (name: string) => resolve(name) };
}

describe('retry prefill', () => {
  it('never replaces an edited draft automatically; explicit restore focuses the first field once', async () => {
    const t = setup();
    t.retry.enter('entry', 'job');
    t.retry.setReady(true);
    t.retry.touch(); // Even when the edit was undone before the GET resolved.
    t.resolve('old input');
    await vi.waitFor(() => expect(t.retry.offered).toBe(true));
    expect(t.draft().fields.name).toBe('');
    expect(t.focus).not.toHaveBeenCalled();
    t.retry.usePrevious();
    t.retry.usePrevious();
    expect(t.draft().fields.name).toBe('old input');
    expect(t.retry.offered).toBe(false);
    await vi.waitFor(() => expect(t.focus).toHaveBeenCalledExactlyOnceWith('first'));
  });

  it('drops untouched fields, retains edits and revalidates when applying from history', async () => {
    const t = setup();
    t.retry.enter('entry', 'job');
    t.retry.setReady(true);
    t.resolve('old input');
    await vi.waitFor(() => expect(t.draft().limit).toBe(7));
    t.draft().fields.seed = 91;
    t.retry.enter('plain', null);
    expect(t.draft()).toEqual({ host: '', limit: undefined, fields: { name: '', seed: 91 } });
    expect(t.clear).toHaveBeenCalledOnce();
    t.retry.enter('entry', 'job');
    expect(t.load).toHaveBeenCalledTimes(1);
    expect(t.retry.offered).toBe(true);
    expect(t.draft().fields.seed).toBe(91);
    t.retry.usePrevious();
    expect(t.draft().fields).toEqual({ name: 'old input', seed: 23 });
    expect(t.restore).toHaveBeenCalledTimes(2);
  });

  it('caches the historical request, not the Library-dependent patch', async () => {
    const t = setup();
    t.retry.enter('entry', 'job');
    t.retry.setReady(true);
    t.resolve('known');
    await vi.waitFor(() => expect(t.draft().fields.name).toBe('known'));
    t.retry.dispose();
    const next = setup();
    next.restore.mockImplementation(name => ({ actions: ['queue'], patch: { host: '', limit: 7, fields: { name, seed: 23 } }, focus: 'first' }));
    const remount = new RetryPrefill({ ...next.form, slot: t.form.slot });
    remount.enter('entry', 'job');
    remount.setReady(true);
    expect(next.draft().fields.name).toBe('known');
    expect(next.draft().host).toBe('');
    expect(next.load).not.toHaveBeenCalled();
    expect(next.restore).toHaveBeenCalledOnce();
  });

  it('ignores a response from a departed route even when the transport ignores abort', async () => {
    const t = setup();
    t.retry.enter('old-entry', 'old-job');
    t.retry.setReady(true);
    t.retry.enter('plain', null);
    t.resolve('stale');
    await Promise.resolve();
    expect(t.draft().fields.name).toBe('');
    expect(t.retry.offered).toBe(false);
  });

  it('compares original inputs with the restored selection, not the current offer', () => {
    expect(omittedInputs(['A', 'B', 'B'], ['A'])).toEqual(['B']);
    expect(omittedInputs(['A', 'B'], ['A', 'B'])).toEqual([]);
  });

  it('restores only a currently available ID and falls back to the historical prompt text', () => {
    const id = 'library-id';
    const request = { prompt_id: id, prompt_text: 'Historical prompt.' };
    expect(availableId(id, [{ id, available: false }])).toBe('');
    expect(availableId(id, [{ id }])).toBe(id);
    expect(historicalPrompt(request, [{ id, text: 'New Library text.' }])).toEqual({ id: '', text: 'Historical prompt.', mode: 'text' });
    expect(historicalPrompt(request, [{ id, text: 'Historical prompt.' }])).toEqual({ id, text: 'Historical prompt.', mode: 'saved' });
  });

  it('blocks actions with a visible reason until both the request and inventory arrive', async () => {
    const t = setup();
    t.retry.enter('entry', 'job');
    t.resolve('old input');
    await Promise.resolve();
    expect(t.retry.loadingReason).toMatch(/Loading the previous/);
    expect(t.retry.action('queue').blocked).toBe(true);
    t.retry.setReady(true);
    expect(t.retry.loadingReason).toBe('');
    expect(t.retry.action('queue').blocked).toBe(false);
  });

  it('keeps the omission gate on an edited selection after returning to the plain route', async () => {
    let draft = { picked: [] as string[] };
    const retry = new RetryPrefill({
      slot: 'omission-test', defaults: { picked: [] as string[] },
      read: () => draft,
      write: value => { draft = value; },
      load: async () => ['original', 'deleted'],
      restore: prior => ({
        actions: ['transfer', 'preview'],
        patch: { picked: ['original'] },
        retains: value => value.picked.includes('original'),
        omissions: value => omittedInputs(prior, value.picked).map(id => ({ id, text: id })),
      }),
    });
    retry.enter('original', 'job');
    retry.setReady(true);
    await vi.waitFor(() => expect(draft.picked).toEqual(['original']));
    draft.picked.push('another');
    expect(retry.action('transfer').blocked).toBe(true);
    expect(retry.action('cleanup').blocked).toBe(false);
    expect(retry.action('cleanup').history).toBeUndefined();
    retry.enter('plain', null);
    expect(retry.action('preview').blocked).toBe(true);
    retry.acknowledge('transfer');
    expect(retry.action('preview').blocked).toBe(false);
    expect(retry.omissions('preview').map(item => item.id)).toEqual(['deleted']);
    draft.picked = ['original'];
    expect(retry.action('preview').blocked).toBe(false);
    draft.picked = ['another'];
    expect(retry.omissions('transfer')).toEqual([]);
  });
});

it('binds a retry key to its launcher, job kind and exact wire body', () => {
  let sequence = 0;
  const keys = new RetryKeys(() => String(++sequence));
  const body = { input_ids: ['A'], host: 'worker' };
  const retry = { slot: 'keys', id: 'failed-job', entry: 'one', action: 'editor' };
  expect(keys.take('plan', body, retry).key).toBe('1');
  expect(keys.take('plan', { host: 'worker', input_ids: ['A'] }, retry).key).toBe('1');
  expect(keys.take('plan', body, { ...retry, action: 'previous' }).key).toBe('2');
  expect(keys.take('inspect', body, retry).key).toBe('3');
  expect(keys.take('plan', { ...body, input_ids: ['B'] }, retry).key).toBe('4');
  const independent = keys.take('verify', body);
  independent.accept({ id: 'independent-job' });
  expect(keys.take('verify', body).key).not.toBe(independent.key);
});

it('replays an accepted retry on Back but retires its key for a fresh entry', () => {
  let sequence = 0;
  const launch = () => new RetryKeys<{ id: string }>(() => String(++sequence));
  const keys = launch();
  const body = { input_ids: ['A'] };
  const retry = (entry: string) => ({ slot: 'retire', id: 'failed-job', entry, action: 'run' });
  const initial = keys.take('plan', body, retry('one'));
  expect(launch().take('plan', body, retry('two')).key).toBe(initial.key);
  const withoutQuery = keys.take('plan', body);
  expect(withoutQuery.key).toBe(initial.key);
  withoutQuery.accept({ id: 'accepted-job' });
  expect(launch().take('plan', body, retry('one')).result?.id).toBe('accepted-job');
  expect(launch().take('plan', body, retry('two')).result?.id).toBe('accepted-job');
  const fresh = launch().take('plan', body, retry('three'));
  expect(fresh.key).not.toBe(initial.key);
  expect(fresh.result).toBeUndefined();
});

it('permits stored results but suppresses presentation after interaction or a round trip', () => {
  const activity = new RouteActivity();
  let sameLocation = true;
  const reply = activity.capture(() => sameLocation);
  expect(reply.canPresent()).toBe(true);
  activity.interact();
  expect(reply.current()).toBe(true);
  expect(reply.canPresent()).toBe(false);
  const second = activity.capture(() => sameLocation);
  sameLocation = false; // The browser moves before the router sees it.
  expect(second.canPresent()).toBe(false);
  activity.visit();
  activity.visit();
  sameLocation = true;
  expect(second.current()).toBe(false);
  expect(activity.capture().canPresent()).toBe(true);
});
