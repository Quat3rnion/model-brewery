// Built-in workbench agent (W15): hosted provider settings and agent sessions.
// Wire shapes follow crates/controller/src/agent.rs. The API key is write-only
// and never enters these types.
import type { Blocker } from './blockers';
import type { Tone } from './format';

export type ProviderKind = 'anthropic' | 'openai';
export interface Provider {
  kind: ProviderKind;
  base_url: string;
  model: string;
  max_tokens: number;
  max_steps: number;
  max_session_tokens: number;
  timeout_seconds: number;
  request_timeout_seconds: number;
}
export interface ProviderSettings { provider: Provider | null; keyConfigured: boolean }

export type SessionStatus = 'running' | 'completed' | 'cancelled' | 'failed' | 'budget_exhausted';
export interface SessionSummary {
  id: string;
  goal: string;
  model_id: string;
  status: SessionStatus;
  steps: number;
  tokens: number;
  scratch_ids: string[];
  message: string | null;
}
export type SessionEntry =
  | { kind: 'assistant'; text: string }
  | { kind: 'tool_call'; name: string; arguments: unknown }
  | { kind: 'tool_result'; name: string; result: unknown };
export interface SessionDetail extends SessionSummary { entries: SessionEntry[] }

export const PROVIDER_KINDS: { id: ProviderKind; label: string }[] = [
  { id: 'anthropic', label: 'Anthropic' },
  { id: 'openai', label: 'OpenAI-compatible' },
];

export function defaultBaseUrl(kind: ProviderKind): string {
  return kind === 'anthropic' ? 'https://api.anthropic.com/v1' : 'https://api.openai.com/v1';
}

export const DEFAULT_PROVIDER: Provider = {
  kind: 'anthropic', base_url: defaultBaseUrl('anthropic'), model: '',
  max_tokens: 1024, max_steps: 12, max_session_tokens: 20000, timeout_seconds: 180, request_timeout_seconds: 30,
};

type Limit = 'max_tokens' | 'max_steps' | 'max_session_tokens' | 'timeout_seconds' | 'request_timeout_seconds';
export const LIMITS: { field: Limit; label: string; max: number; hint: string }[] = [
  { field: 'max_steps', label: 'Maximum requests', max: 32, hint: 'Provider requests in one session, 1 to 32.' },
  { field: 'max_tokens', label: 'Maximum output tokens per request', max: 16384, hint: '1 to 16384. Not more than the session token limit.' },
  { field: 'max_session_tokens', label: 'Session token limit', max: 250000, hint: 'Input and output tokens in one session, 1 to 250000.' },
  { field: 'timeout_seconds', label: 'Session time limit (seconds)', max: 900, hint: '1 to 900 seconds.' },
  { field: 'request_timeout_seconds', label: 'Request time limit (seconds)', max: 120, hint: '1 to 120 seconds for each provider request.' },
];

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const isCount = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;

function parseProvider(raw: unknown): Provider {
  if (!isObject(raw) || (raw.kind !== 'anthropic' && raw.kind !== 'openai') ||
    typeof raw.base_url !== 'string' || typeof raw.model !== 'string' ||
    !LIMITS.every(({ field }) => isCount(raw[field]))) {
    throw new Error('The controller sent provider settings in an unknown format.');
  }
  return {
    kind: raw.kind, base_url: raw.base_url, model: raw.model,
    max_tokens: raw.max_tokens as number, max_steps: raw.max_steps as number,
    max_session_tokens: raw.max_session_tokens as number, timeout_seconds: raw.timeout_seconds as number,
    request_timeout_seconds: raw.request_timeout_seconds as number,
  };
}

export function parseProviderSettings(raw: unknown): ProviderSettings {
  if (!isObject(raw) || typeof raw.key_configured !== 'boolean') {
    throw new Error('The controller sent provider settings in an unknown format.');
  }
  return { provider: raw.provider === null ? null : parseProvider(raw.provider), keyConfigured: raw.key_configured };
}

function urlProblem(value: string): string | null {
  const bad = 'Use an HTTPS base URL without a user name, password, query, or fragment';
  let url: URL;
  try { url = new URL(value.trim()); } catch { return bad; }
  const loopback = /^127\.\d+\.\d+\.\d+$/.test(url.hostname) || url.hostname === '[::1]';
  if (!(url.protocol === 'https:' || (url.protocol === 'http:' && loopback))) return bad;
  if (url.username || url.password || url.search || url.hash || value.includes('?') || value.includes('#')) return bad;
  if (value.length > 512) return 'Use a base URL of 512 characters or less';
  const path = url.pathname.replace(/\/+$/, '');
  if (path.endsWith('/chat/completions') || path.endsWith('/messages')) {
    return 'Use the base URL only. The controller adds the request path';
  }
  return null;
}

export function providerProblems(p: Provider): Blocker[] {
  const out: Blocker[] = [];
  const url = urlProblem(p.base_url);
  if (url) out.push({ field: 'base_url', text: url });
  const model = p.model.trim();
  if (!model) out.push({ field: 'model', text: 'Enter a model ID' });
  else if (new TextEncoder().encode(model).length > 200) out.push({ field: 'model', text: 'Use a model ID of 200 bytes or less' });
  for (const { field, label, max } of LIMITS) {
    const v = p[field];
    if (!Number.isSafeInteger(v) || v < 1 || v > max) out.push({ field, text: `Set ${label.toLowerCase()} to a whole number from 1 to ${max}` });
  }
  if (!out.some((b) => b.field === 'max_tokens' || b.field === 'max_session_tokens') && p.max_tokens > p.max_session_tokens) {
    out.push({ field: 'max_tokens', text: 'Set maximum output tokens per request to the session token limit or less' });
  }
  return out;
}

const normalUrl = (u: string) => u.trim().replace(/\/+$/, '');

/** Wire body: trim the text fields as the controller does. */
export function providerBody(p: Provider): Provider {
  return { ...p, base_url: normalUrl(p.base_url), model: p.model.trim() };
}

export function providerDirty(a: Provider, b: Provider): boolean {
  return JSON.stringify(providerBody(a)) !== JSON.stringify(providerBody(b));
}

/** The controller removes the saved key when the kind or base URL changes. */
export function endpointChanged(saved: Provider | null, draft: Provider): boolean {
  return !!saved && (saved.kind !== draft.kind || normalUrl(saved.base_url) !== normalUrl(draft.base_url));
}

export const SESSION_STATUS: Record<SessionStatus, { label: string; tone: Tone }> = {
  running: { label: 'Running', tone: 'info' },
  completed: { label: 'Completed', tone: 'ok' },
  cancelled: { label: 'Cancelled', tone: 'muted' },
  failed: { label: 'Failed', tone: 'bad' },
  budget_exhausted: { label: 'Limit reached', tone: 'warn' },
};

function parseSummary(raw: unknown): SessionSummary | null {
  if (!isObject(raw) || typeof raw.id !== 'string' || !raw.id || typeof raw.goal !== 'string' ||
    typeof raw.model_id !== 'string' || typeof raw.status !== 'string' || !(raw.status in SESSION_STATUS) ||
    !isCount(raw.steps) || !isCount(raw.tokens) || !Array.isArray(raw.scratch_ids) ||
    !raw.scratch_ids.every((s) => typeof s === 'string') || !(raw.message === null || typeof raw.message === 'string')) {
    return null;
  }
  return {
    id: raw.id, goal: raw.goal, model_id: raw.model_id, status: raw.status as SessionStatus,
    steps: raw.steps, tokens: raw.tokens, scratch_ids: [...raw.scratch_ids as string[]], message: raw.message,
  };
}

export function parseSessions(raw: unknown): SessionSummary[] {
  return Array.isArray(raw) ? raw.map(parseSummary).filter((s): s is SessionSummary => !!s) : [];
}

function parseEntry(raw: unknown): SessionEntry | null {
  if (!isObject(raw)) return null;
  if (raw.kind === 'assistant' && typeof raw.text === 'string') return { kind: 'assistant', text: raw.text };
  if (raw.kind === 'tool_call' && typeof raw.name === 'string') return { kind: 'tool_call', name: raw.name, arguments: raw.arguments };
  if (raw.kind === 'tool_result' && typeof raw.name === 'string') return { kind: 'tool_result', name: raw.name, result: raw.result };
  return null;
}

export function parseSession(raw: unknown): SessionDetail | null {
  const summary = parseSummary(raw);
  if (!summary || !isObject(raw) || !Array.isArray(raw.entries)) return null;
  return { ...summary, entries: raw.entries.map(parseEntry).filter((e): e is SessionEntry => !!e) };
}

const MAX_SHOWN = 4000;
export function entryText(value: unknown): string {
  let text: string;
  if (typeof value === 'string') text = value;
  else {
    try { text = JSON.stringify(value, null, 2) ?? 'No value'; } catch { text = String(value); }
  }
  return text.length > MAX_SHOWN ? `${text.slice(0, MAX_SHOWN)}\n(Shortened for display.)` : text;
}

export function goalProblem(goal: string): string | null {
  if (!goal.trim()) return 'Enter a goal.';
  if (new TextEncoder().encode(goal).length > 4096) return 'Write a goal of 4096 bytes or less.';
  return null;
}
