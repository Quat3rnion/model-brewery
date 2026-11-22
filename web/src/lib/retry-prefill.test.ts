import { describe, expect, it, vi } from 'vitest';
import { RetryKeys, RetryPrefill, omittedInputs } from './retry-prefill.svelte';

let slot = 0;
function setup() {
  const defaults = { host: '', limit: undefined as number | undefined, fields: { name: '', seed: 0 } };
  let draft = structuredClone(defaults);
  let resolve!: (name: string) => void;
  const load = vi.fn(() => new Promise<string>(done => { resolve = done; }));
  const focus = vi.fn();
  const clear = vi.fn();
  const restore = vi.fn((name: string) => ({ patch: { host: 'worker', limit: 7, fields: { name, seed: 23 } }, focus: 'first' }));
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

  it('drops retry fields but retains individually edited nested values; history reuses the same restore', async () => {
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
    expect(t.restore).toHaveBeenCalledOnce();
  });

  it('rehydrates once-restored history without a GET even after a view remount', async () => {
    const t = setup();
    t.retry.enter('entry', 'job');
    t.retry.setReady(true);
    t.resolve('known');
    await vi.waitFor(() => expect(t.draft().fields.name).toBe('known'));
    t.retry.dispose();
    const next = setup();
    const remount = new RetryPrefill({ ...next.form, slot: t.form.slot });
    remount.enter('entry', 'job');
    remount.setReady(true);
    expect(next.draft().fields.name).toBe('known');
    expect(next.load).not.toHaveBeenCalled();
    expect(next.restore).not.toHaveBeenCalled();
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
});

it('reuses unresolved keys across refetch and history; only body changes or a new accepted run rotate keys', () => {
  let sequence = 0;
  const keys = new RetryKeys(() => String(++sequence));
  const body = { input_ids: ['A'] };
  const first = keys.entry('convert', 'failed-job', 'one');
  expect(first.submission('run', body).key).toBe('1');
  expect(keys.entry('convert', 'failed-job', 'one').submission('run', body).key).toBe('1');
  const replay = keys.entry('convert', 'failed-job', 'two');
  const accepted = replay.submission('run', body);
  expect(accepted.key).toBe('1');
  expect(replay.submission('run', { input_ids: ['B'] }).key).toBe('2');
  expect(replay.submission('preview', body).key).toBe('3');
  accepted.accept();
  expect(keys.entry('convert', 'failed-job', 'one').submission('run', body).key).toBe('1');
  expect(keys.entry('convert', 'failed-job', 'three').submission('run', body).key).toBe('4');
});
