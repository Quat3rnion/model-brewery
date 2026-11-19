import type { BrewState, JobDetail, JobSummary, Model, NewJob, NewModel, NewPrompt, Prompt } from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal, extra?: Record<string, string>): Promise<T> {
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
  if (response.status === 204) return undefined as T;
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
    const message =
      data && typeof data === 'object' && typeof (data as { error?: unknown }).error === 'string'
        ? (data as { error: string }).error
        : `Request failed with HTTP ${response.status}.`;
    throw new ApiError(message, response.status);
  }
  return data as T;
}

export const api = {
  state: (signal?: AbortSignal) => request<BrewState>('GET', '/api/state', undefined, signal),
  job: (id: string, signal?: AbortSignal) => request<JobDetail>('GET', `/api/jobs/${encodeURIComponent(id)}`, undefined, signal),
  addModel: (model: NewModel) => request<Model>('POST', '/api/models', model),
  removeModel: (id: string) => request<void>('DELETE', `/api/models/${encodeURIComponent(id)}`),
  addPrompt: (prompt: NewPrompt) => request<Prompt>('POST', '/api/prompts', prompt),
  submitJob: (job: NewJob) => request<JobSummary>('POST', '/api/jobs', job),
  suggestPaths: (host: string, prefix: string, signal?: AbortSignal) => request<unknown>('POST', '/api/suggestions/paths', { host, prefix }, signal),
  // Durable conversion workflow (controller runs inspect → plan → convert → verify).
  startConversion: (body: unknown, key: string) => request<unknown>('POST', '/api/conversions', body, undefined, { 'Idempotency-Key': key }),
  conversions: (signal?: AbortSignal) => request<unknown>('GET', '/api/conversions', undefined, signal),
  conversion: (id: string, signal?: AbortSignal) => request<unknown>('GET', `/api/conversions/${encodeURIComponent(id)}`, undefined, signal),
  cancelConversion: (id: string) => request<unknown>('POST', `/api/conversions/${encodeURIComponent(id)}/cancel`, {}),
  settings: (signal?: AbortSignal) => request<unknown>('GET', '/api/settings', undefined, signal),
  saveSettings: (host: string, body: unknown) => request<unknown>('PUT', `/api/settings/${encodeURIComponent(host)}`, body),
  adminState: () => request<unknown>('GET', '/api/admin/state'),
  planDirectory: (body: import('./model-directories').DirectoryChange) => request<unknown>('POST', '/api/admin/model-directories/plan', body),
  confirmDirectory: (body: {worker:string; revision:number; confirmation:string}) => request<unknown>('POST', '/api/admin/model-directories/confirm', body),
  adminAction: (action: unknown) => request<unknown>('POST', '/api/admin/actions', action),
  profileCatalog: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles/catalog', undefined, signal),
  ggufCatalog: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles/gguf-catalog', undefined, signal),
  profiles: (signal?: AbortSignal) => request<unknown>('GET', '/api/profiles', undefined, signal),
  saveProfile: (body: unknown) => request<unknown>('POST', '/api/profiles', body),
  cancelJob: (id: string) => request<JobSummary>('POST', `/api/jobs/${encodeURIComponent(id)}/cancel`, {}),
};

export function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
