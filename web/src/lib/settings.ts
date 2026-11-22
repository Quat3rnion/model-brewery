// Per-host operational settings (controller t_8a97b91c):
// GET /api/settings → {hosts:[HostSettingsWire]}
// PUT /api/settings/{host} {revision, benchmark_targets, enabled_suites, baseline_endpoints, evaluation_enabled}
//   → host object; 400 invalid, 404 unknown, 409 {error} locked/busy/conflict (refetch before retry).
// Writes use the ordinary authorized gateway session (no admin key).
// Saving never starts a job. Readiness comes from the controller, not from the form.

import { capability } from './format';
import type { Blocker } from './blockers';
import { obj } from './reports';
import type { HostReadiness } from './api';
import type { Host } from './types';

type ReadinessKind = keyof HostReadiness['readiness'];
const readinessJobs = [
  ['Benchmarks', 'benchmark'],
  ['GSM8K evaluation', 'evaluate'],
  ['Source baseline', 'baseline'],
  ['Generation', 'generate'],
] as const satisfies readonly (readonly [string, ReadinessKind])[];

export interface Endpoint {
  id: string;
  label?: string;
  endpoint: string;
  model: string;
  /** Reference endpoints only: identity of the model actually served. */
  source_sha256?: string;
  runtime_id?: string;
  precision?: string;
}

export interface HostSettings {
  id: string;
  name: string | null;
  managed: boolean;
  managementAvailable: boolean | null;
  managementReason: string | null;
  evaluationStatus: 'managed'|'profile_only'|'not_enrolled'|'unavailable';
  savedProfileTargets: Endpoint[] | null;
  profileBinding: string | null;
  evaluationReason:string|null;
  ready: boolean;
  readiness: Partial<Record<ReadinessKind, { ready: boolean; reason: string | null }>>;
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
      ...(baseline ? { source_sha256: str(e.source_sha256) ?? '', runtime_id: str(e.runtime_id) ?? '', ...(typeof e.precision==='string' && e.precision ? {precision:e.precision} : {}) } : {}),
    }));
}

export function parseHost(v: unknown): HostSettings | null {
  const o = obj(v);
  if (!o || typeof o.id !== 'string' || !o.id) return null;
  const wireReadiness = obj(o.readiness);
  const readiness: HostSettings['readiness'] = {};
  for (const [, kind] of readinessJobs) {
    const row = obj(wireReadiness?.[kind]);
    if (typeof row?.ready === 'boolean') readiness[kind] = { ready: row.ready, reason: str(obj(row.reason)?.text) };
  }
  return {
    id: o.id,
    name: str(o.name),
    managed: o.managed === true,
    managementAvailable: typeof obj(o.management)?.available === 'boolean' ? obj(o.management)!.available as boolean : null,
    managementReason: str(obj(o.management)?.reason),
    evaluationStatus: obj(o.evaluation)?.status === 'profile_only' ? 'profile_only' : obj(o.evaluation)?.status === 'managed' ? 'managed' : obj(o.evaluation)?.status === 'not_enrolled' ? 'not_enrolled' : o.evaluation ? 'unavailable' : o.managed === true ? 'managed' : 'not_enrolled',
    savedProfileTargets: Array.isArray(obj(o.saved_profile)?.benchmark_targets) ? endpoints(obj(o.saved_profile)!.benchmark_targets) : null,
    profileBinding: str(o.profile_binding),
    evaluationReason: str(obj(o.evaluation)?.reason) ?? str(o.reason),
    ready: o.ready === true,
    readiness,
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
const loopback = /^http:\/\/(?:127\.0\.0\.1|\[::1\]):([1-9][0-9]{0,4})\/v1$/;
export function endpointProblem(url: string): string | null {
  if (!url) return 'Enter the endpoint URL.';
  const match = loopback.exec(url);
  return match && match[0] === url && Number(match[1]) <= 65535 ? null :
    'Use http://127.0.0.1:PORT/v1 or http://[::1]:PORT/v1 with a port from 1 to 65535 and no leading zero.';
}

function endpointProblems(kind: string, list: Endpoint[], ports: number[] | null, baseline = false): Blocker[] {
  const out: Blocker[] = [];
  const seen = new Set<string>();
  for (const [i, e] of list.entries()) {
    const prefix = `${baseline ? 'baselineEndpoints' : 'benchmarkTargets'}-${i}`;
    const add = (field: string, text: string) => out.push({ field: `${prefix}-${field}`, text });
    const id = idProblem(e.id);
    if (id) add('id', `${kind} "${e.id || '(no ID)'}": ${id}`);
    if (e.id && seen.has(e.id)) add('id', `Duplicate ${kind.toLowerCase()} ID ${e.id}.`);
    seen.add(e.id);
    const ep = endpointProblem(e.endpoint);
    if (ep) add('url', `${kind} ${e.id || '(no ID)'}: ${ep}`);
    else if (ports) {
      const port = Number(loopback.exec(e.endpoint)![1]);
      if (!ports.includes(port)) add('url', `${kind} ${e.id || '(no ID)'}: Port ${port} is not approved on this worker${ports.length ? ` (approved: ${ports.join(', ')})` : '; no port is approved'}.`);
    }
    if (baseline) {
      if (!/^[0-9a-f]{64}$/.test(e.source_sha256 ?? '')) add('sha', `${kind} ${e.id || '(no ID)'}: enter the reference model SHA-256 (64 lowercase hex).`);
      if (e.precision && !/^[A-Za-z0-9_.-]{1,32}$/.test(e.precision)) add('precision', `${kind} ${e.id || '(no ID)'}: precision must use at most 32 letters, digits, dots, hyphens or underscores.`);
      if (!(e.runtime_id ?? '').trim()) add('rt', `${kind} ${e.id || '(no ID)'}: enter the runtime ID that serves it.`);
    }
    if (!e.model.trim()) add('model', `${kind} ${e.id || '(no ID)'}: enter the model name served by the endpoint.`);
  }
  return out;
}

const epKey = (e: Endpoint) => JSON.stringify([e.id, e.label ?? '', e.endpoint, e.model, e.source_sha256 ?? '', e.runtime_id ?? '',e.precision ?? '']);

/**
 * Baseline endpoints added or changed relative to the saved host. Removing
 * one is not a change that needs permission.
 */
export function changedBaselines(draft: HostSettings, saved: HostSettings | undefined): Endpoint[] {
  const before = new Set((saved?.baselineEndpoints ?? []).map(epKey));
  return draft.baselineEndpoints.filter((e) => !before.has(epKey(e)));
}

export function validateHost(h: HostSettings, saved?: HostSettings): string[] {
  return evaluationBlockers(h, saved).map(({ text }) => text);
}

export function evaluationBlockers(h: HostSettings, saved?: HostSettings): Blocker[] {
  const approved = new Set(h.approvedSuites.map((s) => s.id));
  const blockedBaselines = h.baselineAllowed === false ? changedBaselines(h, saved) : [];
  return [
    ...blockedBaselines.map((e) => ({ field: `baselineEndpoints-${h.baselineEndpoints.indexOf(e)}-id`, text: `Reference endpoint ${e.id || '(no ID)'}: reference endpoints are not allowed on this worker. Remove the new or changed entry.` })),
    ...endpointProblems('Target', h.benchmarkTargets, h.evaluationStatus === 'profile_only' ? null : h.approvedPorts),
    ...endpointProblems('Reference endpoint', h.baselineEndpoints, h.approvedPorts, true),
    ...(h.evaluationEnabled && !h.evaluationAllowed ? [{ field: 'gsm8k', text: 'GSM8K evaluation is not allowed on this worker.' }] : []),
    ...h.enabledSuites.filter((s) => !approved.has(s)).map((s) => ({ field: 'suites', text: `${s} is not an approved suite on this host.` })),
  ];
}

export function buildPut(h: HostSettings) {
  const clean = (l: Endpoint[], baseline = false) =>
    l.map((e) => ({ id: e.id, ...(e.label ? { label: e.label } : {}), endpoint: e.endpoint, model: e.model, ...(baseline ? { source_sha256: e.source_sha256 ?? '', runtime_id: e.runtime_id ?? '', ...(e.precision ? {precision:e.precision} : {}) } : {}) }));
  return {
    revision: h.revision,
    ...(h.evaluationStatus === 'profile_only' && h.profileBinding ? {profile_binding:h.profileBinding} : {}),
    benchmark_targets: clean(h.benchmarkTargets),
    enabled_suites: [...h.enabledSuites],
    baseline_endpoints: clean(h.baselineEndpoints, true),
    evaluation_enabled: h.evaluationEnabled,
  };
}

/** What the controller says this host can run now. Independent of the saved form. */
export function effectiveReadiness(h: HostSettings, host: Host | undefined) {
  const target = host ?? { id: h.id, name: h.name ?? undefined };
  return readinessJobs.map(([what, kind]) => {
    const row = h.readiness[kind];
    const c = capability(target, kind);
    const allowed = row?.ready ?? c.allowed;
    const evaluationReason = kind === 'evaluate' && h.evaluationStatus === 'managed' ? h.evaluationReason : null;
    const reason = evaluationReason ?? row?.reason ?? host?.capability_reasons?.[kind]?.text ?? c.reason;
    return { kind, what, allowed, reason: allowed ? null : reason };
  });
}

export function hostDirty(a: HostSettings, b: HostSettings): boolean {
  return JSON.stringify(buildPut(a)) !== JSON.stringify(buildPut(b));
}
