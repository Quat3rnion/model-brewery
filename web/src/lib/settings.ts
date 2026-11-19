// Per-host operational settings (controller t_8a97b91c):
// GET /api/settings → {hosts:[HostSettingsWire]}
// PUT /api/settings/{host} {revision, benchmark_targets, enabled_suites, baseline_endpoints, evaluation_enabled}
//   → host object; 400 invalid, 404 unknown, 409 {error} locked/busy/conflict (refetch before retry).
// Writes use the ordinary authorized gateway session (no admin key).
// Saving never starts a job. Readiness comes from the controller, not from the form.

import { capability } from './format';
import { obj } from './reports';
import type { Host } from './types';

export interface Endpoint {
  id: string;
  label?: string;
  endpoint: string;
  model: string;
  /** Baseline endpoints only: exact BF16 source checkpoint and serving runtime. */
  source_sha256?: string;
  runtime_id?: string;
}

export interface HostSettings {
  id: string;
  name: string | null;
  managed: boolean;
  ready: boolean;
  reason: string | null;
  revision: number;
  /** Loopback ports the controller accepts for this worker's endpoints. */
  approvedPorts: number[];
  evaluationAllowed: boolean;
  /** false: the controller rejects new or changed baseline endpoints. null: not reported (older controller). */
  baselineAllowed: boolean | null;
  approvedSuites: { id: string; label?: string }[];
  benchmarkTargets: Endpoint[];
  enabledSuites: string[];
  baselineEndpoints: Endpoint[];
  evaluationEnabled: boolean;
}

const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);

function endpoints(v: unknown, baseline = false): Endpoint[] {
  if (!Array.isArray(v)) return [];
  return v
    .map(obj)
    .filter((e): e is Record<string, unknown> => !!e && typeof e.id === 'string')
    .map((e) => ({
      id: e.id as string,
      ...(typeof e.label === 'string' && e.label ? { label: e.label } : {}),
      endpoint: str(e.endpoint) ?? '',
      model: str(e.model) ?? '',
      ...(baseline ? { source_sha256: str(e.source_sha256) ?? '', runtime_id: str(e.runtime_id) ?? '' } : {}),
    }));
}

export function parseHost(v: unknown): HostSettings | null {
  const o = obj(v);
  if (!o || typeof o.id !== 'string' || !o.id) return null;
  return {
    id: o.id,
    name: str(o.name),
    managed: o.managed === true,
    ready: o.ready === true,
    reason: str(o.reason),
    revision: typeof o.revision === 'number' && Number.isInteger(o.revision) ? o.revision : 0,
    approvedPorts: Array.isArray(o.approved_ports) ? o.approved_ports.filter((x): x is number => Number.isInteger(x) && (x as number) > 0 && (x as number) < 65536) : [],
    evaluationAllowed: o.evaluation_allowed === true,
    baselineAllowed: typeof o.baseline_allowed === 'boolean' ? o.baseline_allowed : null,
    approvedSuites: (Array.isArray(o.approved_suites) ? o.approved_suites : [])
      .map(obj)
      .filter((x): x is Record<string, unknown> => !!x && typeof x.id === 'string')
      .map((x) => (typeof x.label === 'string' && x.label ? { id: x.id as string, label: x.label } : { id: x.id as string })),
    benchmarkTargets: endpoints(o.benchmark_targets),
    enabledSuites: Array.isArray(o.enabled_suites) ? o.enabled_suites.filter((x): x is string => typeof x === 'string') : [],
    baselineEndpoints: endpoints(o.baseline_endpoints, true),
    evaluationEnabled: o.evaluation_enabled === true,
  };
}

export function parseSettings(v: unknown): HostSettings[] {
  const hosts = obj(v)?.hosts;
  return Array.isArray(hosts) ? hosts.map(parseHost).filter((h): h is HostSettings => !!h) : [];
}

export const ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export function idProblem(id: string): string | null {
  if (!id) return 'Enter an ID.';
  return ID_RE.test(id) ? null : 'Use 1 to 64 lowercase letters, digits, _ or -, starting with a letter or digit.';
}

/** The controller accepts only numeric loopback HTTP on the worker: http://127.0.0.1:PORT/v1 or http://[::1]:PORT/v1. */
export function endpointProblem(url: string): string | null {
  if (!url) return 'Enter the endpoint URL.';
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return 'Not a valid URL.';
  }
  if (u.protocol !== 'http:') return 'Use plain http on the loopback address.';
  if (u.username || u.password) return 'Do not put credentials in the URL.';
  if (u.search || u.hash) return 'Remove the query or fragment.';
  if (u.hostname !== '127.0.0.1' && u.hostname !== '[::1]') return 'Use http://127.0.0.1:PORT/v1 or http://[::1]:PORT/v1 (loopback on the selected worker).';
  if (!u.port) return 'Add the port, for example :8000.';
  if (u.pathname.replace(/\/+$/, '') !== '/v1') return 'The path must be /v1.';
  return null;
}

function endpointProblems(kind: string, list: Endpoint[], ports: number[], baseline = false): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const e of list) {
    const id = idProblem(e.id);
    if (id) out.push(`${kind} "${e.id || '(no ID)'}": ${id}`);
    if (seen.has(e.id)) out.push(`Duplicate ${kind.toLowerCase()} ID ${e.id}.`);
    seen.add(e.id);
    const ep = endpointProblem(e.endpoint);
    if (ep) out.push(`${kind} ${e.id || '(no ID)'}: ${ep}`);
    else {
      const port = Number(new URL(e.endpoint).port);
      if (!ports.includes(port)) out.push(`${kind} ${e.id || '(no ID)'}: Port ${port} is not approved on this worker${ports.length ? ` (approved: ${ports.join(', ')})` : '; no port is approved'}.`);
    }
    if (baseline) {
      if (!/^[0-9a-f]{64}$/.test(e.source_sha256 ?? '')) out.push(`${kind} ${e.id || '(no ID)'}: enter the source SHA-256 of the exact BF16 checkpoint (64 lowercase hex).`);
      if (!(e.runtime_id ?? '').trim()) out.push(`${kind} ${e.id || '(no ID)'}: enter the runtime ID that serves it.`);
    }
    if (!e.model.trim()) out.push(`${kind} ${e.id || '(no ID)'}: enter the model name served by the endpoint.`);
  }
  return out;
}

const epKey = (e: Endpoint) => JSON.stringify([e.id, e.label ?? '', e.endpoint, e.model, e.source_sha256 ?? '', e.runtime_id ?? '']);

/**
 * Baseline endpoints added or changed relative to the saved host. Removing
 * one is not a change that needs permission.
 */
export function changedBaselines(draft: HostSettings, saved: HostSettings | undefined): Endpoint[] {
  const before = new Set((saved?.baselineEndpoints ?? []).map(epKey));
  return draft.baselineEndpoints.filter((e) => !before.has(epKey(e)));
}

export function validateHost(h: HostSettings, saved?: HostSettings): string[] {
  const approved = new Set(h.approvedSuites.map((s) => s.id));
  const blockedBaselines = h.baselineAllowed === false ? changedBaselines(h, saved) : [];
  return [
    ...blockedBaselines.map((e) => `Baseline ${e.id || '(no ID)'}: baseline endpoints are not allowed on this worker. Remove the new or changed entry.`),
    ...endpointProblems('Target', h.benchmarkTargets, h.approvedPorts),
    ...endpointProblems('Baseline', h.baselineEndpoints, h.approvedPorts, true),
    ...(h.evaluationEnabled && !h.evaluationAllowed ? ['GSM8K evaluation is not allowed on this worker (no approved Halogen runtime).'] : []),
    ...h.enabledSuites.filter((s) => !approved.has(s)).map((s) => `${s} is not an approved suite on this host.`),
  ];
}

export function buildPut(h: HostSettings) {
  const clean = (l: Endpoint[], baseline = false) =>
    l.map((e) => ({ id: e.id, ...(e.label ? { label: e.label } : {}), endpoint: e.endpoint, model: e.model, ...(baseline ? { source_sha256: e.source_sha256 ?? '', runtime_id: e.runtime_id ?? '' } : {}) }));
  return {
    revision: h.revision,
    benchmark_targets: clean(h.benchmarkTargets),
    enabled_suites: [...h.enabledSuites],
    baseline_endpoints: clean(h.baselineEndpoints, true),
    evaluation_enabled: h.evaluationEnabled,
  };
}

/** What the controller says this host can run now. Independent of the saved form. */
export function effectiveReadiness(h: HostSettings, host: Host | undefined): { what: string; allowed: boolean; reason: string | null }[] {
  const target = host ?? { id: h.id, name: h.name ?? undefined };
  return [
    ['Benchmarks', 'benchmark'],
    ['GSM8K evaluation', 'evaluate'],
    ['Source baseline', 'baseline'],
  ].map(([what, kind]) => {
    const c = capability(target, kind);
    return { what, allowed: c.allowed, reason: c.allowed ? null : c.reason };
  });
}

export function hostDirty(a: HostSettings, b: HostSettings): boolean {
  return JSON.stringify(buildPut(a)) !== JSON.stringify(buildPut(b));
}
