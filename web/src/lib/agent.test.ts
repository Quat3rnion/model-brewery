import { describe, expect, it } from 'vitest';
import {
  DEFAULT_PROVIDER,
  SESSION_STATUS,
  defaultBaseUrl,
  endpointChanged,
  entryText,
  goalProblem,
  parseProviderSettings,
  parseSession,
  parseSessions,
  providerDirty,
  providerProblems,
} from './agent';

const provider = { ...DEFAULT_PROVIDER, model: 'claude-test' };

describe('provider settings', () => {
  it('uses the controller defaults', () => {
    expect(DEFAULT_PROVIDER).toEqual({
      kind: 'anthropic', base_url: 'https://api.anthropic.com/v1', model: '',
      max_tokens: 1024, max_steps: 12, max_session_tokens: 20000, timeout_seconds: 180, request_timeout_seconds: 30,
    });
    expect(defaultBaseUrl('openai')).toBe('https://api.openai.com/v1');
  });

  it('parses the settings response and rejects a bad shape', () => {
    expect(parseProviderSettings({ provider: null, key_configured: false })).toEqual({ provider: null, keyConfigured: false });
    expect(parseProviderSettings({ provider, key_configured: true })).toEqual({ provider, keyConfigured: true });
    expect(() => parseProviderSettings({ provider: { ...provider, kind: 'other' }, key_configured: true })).toThrow();
    expect(() => parseProviderSettings({ provider, key_configured: 'yes' })).toThrow();
    expect(() => parseProviderSettings({ provider: { ...provider, max_steps: '3' }, key_configured: false })).toThrow();
  });

  it('reports each field out of range', () => {
    expect(providerProblems(provider)).toEqual([]);
    const fields = (p: typeof provider) => providerProblems(p).map((b) => b.field);
    expect(fields({ ...provider, model: '  ' })).toEqual(['model']);
    expect(fields({ ...provider, base_url: 'http://example.com' })).toEqual(['base_url']);
    expect(fields({ ...provider, base_url: 'http://127.0.0.1:8080/v1' })).toEqual([]);
    expect(fields({ ...provider, base_url: 'https://user:pw@example.com' })).toEqual(['base_url']);
    expect(fields({ ...provider, base_url: 'https://example.com/v1/chat/completions' })).toEqual(['base_url']);
    expect(fields({ ...provider, base_url: 'https://example.com/v1/messages' })).toEqual(['base_url']);
    expect(fields({ ...provider, base_url: 'https://example.com/?a=1' })).toEqual(['base_url']);
    expect(fields({ ...provider, max_steps: 33 })).toEqual(['max_steps']);
    expect(fields({ ...provider, max_steps: 1.5 })).toEqual(['max_steps']);
    expect(fields({ ...provider, timeout_seconds: 0 })).toEqual(['timeout_seconds']);
    expect(fields({ ...provider, request_timeout_seconds: 121 })).toEqual(['request_timeout_seconds']);
    expect(fields({ ...provider, max_tokens: 30000, max_session_tokens: 20000 })).toEqual(['max_tokens']);
    expect(fields({ ...provider, max_session_tokens: Number.NaN })).toEqual(['max_session_tokens']);
  });

  it('detects edits and endpoint changes that remove the key', () => {
    expect(providerDirty(provider, { ...provider })).toBe(false);
    expect(providerDirty(provider, { ...provider, max_steps: 3 })).toBe(true);
    expect(endpointChanged(null, provider)).toBe(false);
    expect(endpointChanged(provider, { ...provider, model: 'other' })).toBe(false);
    expect(endpointChanged(provider, { ...provider, base_url: 'https://api.anthropic.com/v1/' })).toBe(false);
    expect(endpointChanged(provider, { ...provider, kind: 'openai' })).toBe(true);
    expect(endpointChanged(provider, { ...provider, base_url: 'https://proxy.example' })).toBe(true);
  });
});

describe('agent sessions', () => {
  const summary = { id: 's1', goal: 'Lower refusals', model_id: 'm1', status: 'running', steps: 2, tokens: 300, scratch_ids: ['r1'], message: null };

  it('parses summaries and details', () => {
    expect(parseSessions([summary, { bad: true }])).toEqual([summary]);
    expect(parseSessions('nope')).toEqual([]);
    const detail = parseSession({ ...summary, entries: [
      { kind: 'assistant', text: 'Plan' },
      { kind: 'tool_call', name: 'trace', arguments: { layer: 3 } },
      { kind: 'tool_result', name: 'trace', result: { ok: true } },
      { kind: 'unknown' },
    ] });
    expect(detail?.entries.map((e) => e.kind)).toEqual(['assistant', 'tool_call', 'tool_result']);
    expect(parseSession({ ...summary, status: 'weird', entries: [] })).toBeNull();
  });

  it('labels every status with text', () => {
    expect(Object.keys(SESSION_STATUS).sort()).toEqual(['budget_exhausted', 'cancelled', 'completed', 'failed', 'running']);
    expect(SESSION_STATUS.budget_exhausted.label).toBe('Limit reached');
  });

  it('formats tool values and truncates long values', () => {
    expect(entryText({ a: 1 })).toBe('{\n  "a": 1\n}');
    expect(entryText('plain')).toBe('plain');
    const long = entryText('x'.repeat(5000));
    expect(long.length).toBeLessThan(4100);
    expect(long.endsWith('(Shortened for display.)')).toBe(true);
  });

  it('checks the goal size in UTF-8 bytes', () => {
    expect(goalProblem('  ')).toBe('Enter a goal.');
    expect(goalProblem('ok')).toBeNull();
    expect(goalProblem('é'.repeat(2049))).toBe('Write a goal of 4096 bytes or less.');
  });
});
