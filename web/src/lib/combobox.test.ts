import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { disambiguate, exactMatch, filterOptions, stepIndex, type ComboOption } from './combobox';
import { completionText, parseSuggestions, prefixFor, PathSuggester, type PathState } from './paths';
import { hostOptions, modelOptions, nameSuggestions, visibleChecklist } from './pickers';
import { ApiError } from './api';
import type { Host, Model } from './types';

const opts: ComboOption[] = [
  { value: 'h1', label: 'Frame', detail: 'frame.lan' },
  { value: 'h2', label: 'gpu2', detail: 'unavailable', disabled: true },
  { value: 'h3', label: 'Big Frame', keywords: ['strix'] },
  { value: 'h4', label: 'Café box' },
];

describe('filterOptions', () => {
  it('matches label, detail, value and keywords; every token must match', () => {
    expect(filterOptions(opts, 'frame').map((o) => o.value)).toEqual(['h1', 'h3']);
    expect(filterOptions(opts, 'strix').map((o) => o.value)).toEqual(['h3']);
    expect(filterOptions(opts, 'h2').map((o) => o.value)).toEqual(['h2']);
    expect(filterOptions(opts, 'big frame').map((o) => o.value)).toEqual(['h3']);
    expect(filterOptions(opts, 'cafe').map((o) => o.value)).toEqual(['h4']);
    expect(filterOptions(opts, '  ')).toBe(opts);
  });
  it('ranks label prefix before label contains before other fields', () => {
    const o: ComboOption[] = [{ value: 'a', label: 'x', detail: 'model' }, { value: 'b', label: 'my model' }, { value: 'c', label: 'model one' }];
    expect(filterOptions(o, 'model').map((x) => x.value)).toEqual(['c', 'b', 'a']);
  });
});

describe('labels and selection', () => {
  it('appends the ID to duplicate labels only', () => {
    const d = disambiguate([{ value: 'm1', label: 'Qwen' }, { value: 'm2', label: 'Qwen' }, { value: 'm3', label: 'Granite' }]);
    expect(d.map((o) => o.label)).toEqual(['Qwen (m1)', 'Qwen (m2)', 'Granite']);
  });
  it('exact match by label or value, never a disabled or ambiguous one', () => {
    expect(exactMatch(opts, 'frame')?.value).toBe('h1');
    expect(exactMatch(opts, 'H3')?.value).toBe('h3');
    expect(exactMatch(opts, 'gpu2')).toBeNull();
    expect(exactMatch([{ value: 'a', label: 'x' }, { value: 'b', label: 'X' }], 'x')).toBeNull();
  });
  it('steps over disabled options and stops at the ends', () => {
    expect(stepIndex(opts, 0, 1)).toBe(2);
    expect(stepIndex(opts, 2, -1)).toBe(0);
    expect(stepIndex(opts, 3, 1)).toBe(3);
    expect(stepIndex(opts, -1, 1)).toBe(0);
  });
});

describe('picker options', () => {
  const hosts: Host[] = [
    { id: 'frame', name: 'Frame', capabilities: { convert: true } },
    { id: 'gpu2', capabilities: { convert: false }, reasons: { convert: 'No converter' } },
  ];
  const models: Model[] = [
    { id: 'm1', name: 'Qwen', host: 'frame', path: '/m/qwen-a.gguf', kind: 'gguf' },
    { id: 'm2', name: 'Qwen', host: 'gpu2', path: '/m/qwen-b.gguf', kind: 'gguf' },
    { id: 'm3', host: 'frame', path: '/m/bundle/', kind: 'hgn' },
  ];
  it('hosts: capable first, reason in detail, ID searchable', () => {
    const o = hostOptions([...hosts].reverse(), 'convert');
    expect(o.map((x) => x.value)).toEqual(['frame', 'gpu2']);
    expect(o[1].detail).toMatch(/No converter/);
    expect(filterOptions(o, 'gpu2')[0].value).toBe('gpu2');
  });
  it('models: kind filter, exclusion, duplicate names get the ID, host and path searchable', () => {
    const o = modelOptions(models, hosts, new Set(['gguf']), ['m2']);
    expect(o.map((x) => x.value)).toEqual(['m1']);
    const both = modelOptions(models, hosts, new Set(['gguf']), []);
    expect(both.map((x) => x.label)).toEqual(['Qwen (m1)', 'Qwen (m2)']);
    expect(filterOptions(both, 'gpu2').map((x) => x.value)).toEqual(['m2']);
    expect(filterOptions(both, 'qwen-a').map((x) => x.value)).toEqual(['m1']);
    expect(modelOptions(models, hosts, new Set(['hgn']), [])[0].label).toBe('bundle');
  });
  it('registration names: basename first, then registered names marked as in use', () => {
    const n = nameSuggestions('/m/new-model.gguf', models);
    expect(n[0]).toMatchObject({ value: 'new-model.gguf', label: 'new-model.gguf' });
    expect(n.slice(1).map((x) => x.value)).toEqual(['Qwen']);
    expect(n[1].detail).toMatch(/2 entries/);
    expect(nameSuggestions('', models)[0].value).toBe('Qwen');
  });
});

describe('path suggestions', () => {
  it('parses items, drops relative or unknown kinds, keeps truncated', () => {
    expect(parseSuggestions({ items: [{ path: '/a', kind: 'directory' }, { path: 'rel', kind: 'file' }, { path: '/b', kind: 'link' }, { path: '/a', kind: 'directory' }], truncated: true })).toEqual({
      items: [{ path: '/a', kind: 'directory' }],
      truncated: true,
    });
    expect(parseSuggestions({ nope: 1 })).toBeNull();
  });
  it('completes directories with a slash, files as is; empty asks roots; relative is not asked', () => {
    expect(completionText({ path: '/models', kind: 'directory' })).toBe('/models/');
    expect(completionText({ path: '/models/', kind: 'directory' })).toBe('/models/');
    expect(completionText({ path: '/m/a.gguf', kind: 'file' })).toBe('/m/a.gguf');
    expect(prefixFor('')).toBe('');
    expect(prefixFor('models')).toBeNull();
    expect(prefixFor('/mo')).toBe('/mo');
  });

  describe('suggester', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());
    const setup = () => {
      const calls: { host: string; prefix: string; signal: AbortSignal; resolve: (v: unknown) => void; reject: (e: unknown) => void }[] = [];
      const states: PathState[] = [];
      const s = new PathSuggester(
        (host, prefix, signal) => new Promise((resolve, reject) => calls.push({ host, prefix, signal, resolve, reject })),
        (st) => states.push(st),
        100,
      );
      return { s, calls, states, last: () => states[states.length - 1] };
    };
    it('debounces typing into one request', async () => {
      const { s, calls } = setup();
      s.request('frame', '/m');
      s.request('frame', '/mo');
      s.request('frame', '/mod');
      await vi.advanceTimersByTimeAsync(100);
      expect(calls.map((c) => c.prefix)).toEqual(['/mod']);
    });
    it('aborts the running request and ignores its late answer after a host change', async () => {
      const { s, calls, last } = setup();
      s.request('frame', '/m');
      await vi.advanceTimersByTimeAsync(100);
      s.request('gpu2', '/m');
      expect(calls[0].signal.aborted).toBe(true);
      calls[0].resolve({ items: [{ path: '/m/frame-only', kind: 'file' }] });
      await vi.advanceTimersByTimeAsync(100);
      calls[1].resolve({ items: [{ path: '/m/gpu', kind: 'file' }], truncated: false });
      await vi.runAllTimersAsync();
      expect(last()).toMatchObject({ status: 'ready', host: 'gpu2', items: [{ path: '/m/gpu', kind: 'file' }] });
    });
    it('offline worker is an error with a message; missing endpoint is unavailable', async () => {
      const { s, calls, last } = setup();
      s.request('gpu2', '');
      await vi.advanceTimersByTimeAsync(100);
      expect(calls[0].prefix).toBe('');
      calls[0].reject(new ApiError('worker gpu2 is not reachable', 503));
      await vi.runAllTimersAsync();
      expect(last()).toMatchObject({ status: 'error', items: [] });
      expect(last().message).toMatch(/not reachable/);
      s.request('gpu2', '/');
      await vi.advanceTimersByTimeAsync(100);
      calls[1].reject(new ApiError('API endpoint not found', 404));
      await vi.runAllTimersAsync();
      expect(last().status).toBe('unavailable');
    });
    it('does not ask for a relative path or without a host', async () => {
      const { s, calls, last } = setup();
      s.request('frame', 'models');
      s.request('', '/m');
      await vi.runAllTimersAsync();
      expect(calls).toHaveLength(0);
      expect(last().status).toBe('idle');
    });
  });
});

describe('checklist filter', () => {
  const items = [
    { id: 'a', text: 'Qwen frame /m/qwen' },
    { id: 'b', text: 'Granite gpu2 /m/granite' },
    { id: 'c', text: 'Llama frame /m/llama' },
  ];
  it('keeps ticked items visible in list order even when they do not match', () => {
    expect(visibleChecklist(items, 'frame', ['b']).map((x) => x.id)).toEqual(['a', 'b', 'c']);
    expect(visibleChecklist(items, 'llama', ['a']).map((x) => x.id)).toEqual(['a', 'c']);
    expect(visibleChecklist(items, '', []).map((x) => x.id)).toEqual(['a', 'b', 'c']);
  });
});
