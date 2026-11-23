import type { BrewState, EligibilityEntry, FixReason, JobDetail, JobSummary, Model, NewJob, NewModel, NewPrompt, Prompt } from './types';

export interface HostReadiness {
  id: string;
  ready: boolean;
  readiness: Record<'benchmark' | 'baseline' | 'evaluate' | 'generate', { ready: boolean; reason?: FixReason }>;
  evaluation: { status: 'managed' | 'profile_only'; reason?: string; reason_detail?: FixReason };
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly controllerError = false,
  ) {
    super(message);
  }
}

export type JobReceipt = JobSummary | { id: string; duplicate_of?: true };

const submitHeaders = (key: string, allowDuplicate: boolean) => ({
  'Idempotency-Key': key,
  ...(allowDuplicate ? { 'Brewery-Allow-Duplicate': '1' } : {}),
});

async function requestWithStatus<T>(method: string, path: string, body?: unknown, signal?: AbortSignal, extra?: Record<string, string>): Promise<{ data: T; status: number }> {
  let response: Response;
  const headers: Record<string, string> = { Accept: 'application/json' };
  // The controller requires JSON content type on POST as CSRF protection.
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (extra) Object.assign(headers, extra);
  try {
    response = await fetch(path, {
      method,
      signal,
      headers,
      credentials: 'same-origin',
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError('The controller is not reachable.', 0);
  }
  const text = await response.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      if (!response.ok) throw new ApiError(text.slice(0, 500), response.status);
      throw new ApiError('The controller sent a response that is not JSON.', response.status);
    }
  }
  if (!response.ok) {
    const error = data && typeof data === 'object' ? (data as { error?: unknown }).error : undefined;
    // Controller ApiError (including JSON extractor errors) is a JSON object
    // with exactly one field. Do not use a proxy's status code as provenance.
    const controllerError = typeof error === 'string' && Object.keys(data as object).length === 1 &&
      response.headers.get('content-type') === 'application/json' && response.headers.get('brewery-error') === '1';
    const message = typeof error === 'string' ? error : `Request failed with HTTP ${response.status}.`;
    throw new ApiError(message, response.status, controllerError);
  }
  return { data: data as T, status: response.status };
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal, extra?: Record<string, string>): Promise<T> {
  return (await requestWithStatus<T>(method, path, body, signal, extra)).data;
}

export const api = {
  state: (signal?: AbortSignal) => request<BrewState>('GET', '/api/state', undefined, signal),
  // Only job detail includes the submission-time prompt_text.
  job: (id: string, signal?: AbortSignal) => request<JobDetail>('GET', `/api/jobs/${encodeURIComponent(id)}`, undefined, signal),
  addModel: (model: NewModel) => request<Model>('POST', '/api/models', model),
  removeModel: (id: string) => request<void>('DELETE', `/api/models/${encodeURIComponent(id)}`),
  addPrompt: (prompt: NewPrompt) => request<Prompt>('POST', '/api/prompts', prompt),
  updatePrompt: (id: string, prompt: NewPrompt) => request<Prompt>('PUT', `/api/prompts/${encodeURIComponent(id)}`, prompt),
  deletePrompt: (id: string) => request<void>('DELETE', `/api/prompts/${encodeURIComponent(id)}`),
  queueJob: (job: NewJob, key: string, allowDuplicate = false) => request<JobReceipt>('POST', '/api/jobs', job, undefined, submitHeaders(key, allowDuplicate)),
  suggestPaths: (host: string, prefix: string, signal?: AbortSignal) => request<unknown>('POST', '/api/suggestions/paths', { host, prefix }, signal),
  // Durable conversion workflow (controller runs inspect → plan → convert → verify).
  conversionEligibility: (body: unknown, signal?: AbortSignal) => request<EligibilityEntry>('POST', '/api/conversions/eligibility', body, signal),
  startConversion: (body: unknown, key: string, allowDuplicate = false) => requestWithStatus<unknown>('POST', '/api/conversions', body, undefined, submitHeaders(key, allowDuplicate)),
  conversions: (signal?: AbortSignal) => request<unknown>('GET', '/api/conversions', undefined, signal),
  cancelConversion: (id: string) => request<unknown>('POST', `/api/conversions/${encodeURIComponent(id)}/cancel`, {}),
  settings: (signal?: AbortSignal) => request<{ hosts: HostReadiness[] }>('GET', '/api/settings', undefined, signal),
  saveSettings: (host: string, body: unknown) => request<HostReadiness>('PUT', `/api/settings/${encodeURIComponent(host)}`, body),
  adminState: () => request<unknown>('GET', '/api/admin/state'),
  planDirectory: (body: import('./model-directories').DirectoryChange) => request<unknown>('POST', '/api/admin/model-directories/plan', body),
  confirmDirectory: (body: {worker:string; revision:number; confirmation:string}) => request<unknown>('POST', '/api/admin/model-directories/confirm', body),
  adminAction: (action: unknown) => request<unknown>('POST', '/api/admin/actions', action),
  profileCatalog: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles/catalog', undefined, signal),
  ggufCatalog: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles/gguf-catalog', undefined, signal),
  profiles: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles', undefined, signal),
  saveProfile: (body: unknown) => request<unknown>('POST', '/api/profiles', body),
  cancelJob: (id: string) => request<JobSummary>('POST', `/api/jobs/${encodeURIComponent(id)}/cancel`, {}),
  // Weight-space programming: trace slices, scratch revisions, diffs.
  trace: (id: string, query: Record<string, string | number>, view: '' | 'tensor' | 'logit-lens' | 'model-logits' | 'features' = '', signal?: AbortSignal) =>
    request<unknown>('GET', `/api/jobs/${encodeURIComponent(id)}/trace${view ? `/${view}` : ''}?${new URLSearchParams(Object.entries(query).map(([k, v]) => [k, String(v)]))}`, undefined, signal),
  interventions: (signal?: AbortSignal) => request<unknown>('GET', '/api/interventions', undefined, signal),
  scratch: (signal?: AbortSignal) => request<unknown>('GET', '/api/scratch', undefined, signal),
  crosscoder: (id: string, signal?: AbortSignal) => request<unknown>('GET', `/api/crosscoders/${encodeURIComponent(id)}`, undefined, signal),
  diff: (kind: 'tensors' | 'traces' | 'behavior', query: string, signal?: AbortSignal) => request<unknown>('GET', `/api/diff/${kind}?${query}`, undefined, signal),
  // Built-in agent (W15). The provider API key has its own write-only calls in AgentProviderSettings.
  agentProvider: (signal?: AbortSignal) => request<unknown>('GET', '/api/settings/agent-provider', undefined, signal),
  saveAgentProvider: (body: unknown) => request<unknown>('PUT', '/api/settings/agent-provider', body),
  agentSessions: (signal?: AbortSignal) => request<unknown>('GET', '/api/agent/sessions', undefined, signal),
  agentSession: (id: string, signal?: AbortSignal) => request<unknown>('GET', `/api/agent/sessions/${encodeURIComponent(id)}`, undefined, signal),
  startAgentSession: (body: { goal: string; model_id: string }, key: string) =>
    request<unknown>('POST', '/api/agent/sessions', body, undefined, { 'Idempotency-Key': key }),
  cancelAgentSession: (id: string) => request<unknown>('POST', `/api/agent/sessions/${encodeURIComponent(id)}/cancel`, {}),
  // Operator only. The agent MCP interface has no route to this action.
  approveScratch: (id: string, sha256: string, key: string) =>
    request<unknown>('POST', `/api/admin/scratch/${encodeURIComponent(id)}/approve`, { sha256 }, undefined, { 'Idempotency-Key': key }),
};

export function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Only a controller rejection tells us it did not accept the POST. */
export function rejectedSubmission(error: unknown): boolean {
  return error instanceof ApiError && error.controllerError && error.status >= 400 && error.status < 500;
}

export function submissionError(error: unknown): string {
  const text = errorText(error);
  if (rejectedSubmission(error)) return text;
  return `${text} Check Jobs, or retry with the same inputs without reloading to reuse the key. The controller also checks for active and recent matches.`;
}
