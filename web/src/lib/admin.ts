// Infrastructure admin (task20): GET /api/admin/state, POST /api/admin/actions,
// header X-Brewery-Admin-Key: 64 lowercase hex. The key is kept in memory
// only (never stored, logged or put in a URL) and dropped on 401.

import { obj } from './reports';

export type AdminAction =
  | { action: 'enroll' | 'approve' | 'deactivate'; worker: string }
  | { action: 'enable_runtime' | 'disable_runtime' | 'install_runtime'; worker: string; runtime: string }
  | { action: 'set_resource_policy'; worker: string; allow_pool: boolean };

export interface AdminRuntime {
  id: string;
  label: string | null;
  enabled: boolean;
  available: boolean;
  installed: boolean;
  /** Install runs the operator-provisioned fixed helper on the worker. */
  can_install: boolean;
  reason: string | null;
}

export interface AdminWorker {
  id: string;
  name: string | null;
  status: string;
  can_enroll: boolean;
  can_approve: boolean;
  can_deactivate: boolean;
  allow_pool: boolean;
  /** True when this worker may be put in the conversion pool. */
  pool_ceiling: boolean;
  runtimes: AdminRuntime[];
}

export interface AdminState {
  workers: AdminWorker[];
  audit: { id: string; at: number | null; action: string; worker: string | null; runtime: string | null; allow_pool: boolean | null; outcome: string | null }[];
  /** Capabilities the backend does not implement, with its reason. Shown, never offered as controls. */
  unavailable: { what: string; reason: string }[];
}

export function isAdminKey(k: string): boolean {
  return /^[0-9a-f]{64}$/.test(k);
}

const b = (v: unknown) => v === true;
const s = (v: unknown) => (typeof v === 'string' ? v : null);

export function parseAdminState(v: unknown): AdminState | null {
  const o = obj(v);
  if (!o || !Array.isArray(o.workers)) return null;
  const UNAVAILABLE: Record<string, string> = { install_runtime: 'Install a runtime', host_provisioning: 'Provision a host', resource_controls: 'Other resource controls' };
  return {
    workers: o.workers
      .map(obj)
      .filter((w): w is Record<string, unknown> => !!w && typeof w.id === 'string')
      .map((w) => ({
        id: w.id as string,
        name: s(w.name),
        status: s(w.status) ?? 'unknown',
        can_enroll: b(w.can_enroll),
        can_approve: b(w.can_approve),
        can_deactivate: b(w.can_deactivate),
        allow_pool: b(w.allow_pool),
        pool_ceiling: b(w.pool_ceiling),
        runtimes: (Array.isArray(w.runtimes) ? w.runtimes : [])
          .map(obj)
          .filter((r): r is Record<string, unknown> => !!r && typeof r.id === 'string')
          .map((r) => ({ id: r.id as string, label: s(r.label), enabled: b(r.enabled), available: b(r.available), installed: b(r.installed), can_install: b(r.can_install), reason: s(r.reason) })),
      })),
    audit: (Array.isArray(o.audit) ? o.audit : [])
      .map(obj)
      .filter((a): a is Record<string, unknown> => !!a && typeof a.action === 'string')
      .map((a, i) => ({
        id: typeof a.id === 'number' ? String(a.id) : (s(a.id) ?? String(i)),
        at: typeof a.at_ms === 'number' ? a.at_ms : null,
        action: a.action as string,
        worker: s(a.worker),
        runtime: s(a.runtime),
        allow_pool: typeof a.allow_pool === 'boolean' ? a.allow_pool : null,
        outcome: s(a.outcome),
      })),
    unavailable: Object.entries(obj(o.unavailable) ?? {})
      .filter(([, r]) => typeof r === 'string')
      .map(([k, r]) => ({ what: UNAVAILABLE[k] ?? k.replace(/_/g, ' '), reason: r as string })),
  };
}

/** null when the backend says the action is possible; otherwise the reason it is not. */
export function adminActionAllowed(w: AdminWorker, a: AdminAction): string | null {
  switch (a.action) {
    case 'enroll':
      return w.can_enroll ? null : 'Enroll is not possible for this worker now.';
    case 'approve':
      return w.can_approve ? null : 'Approve is not possible for this worker now.';
    case 'deactivate':
      return w.can_deactivate ? null : 'Deactivate is not possible for this worker now.';
    case 'install_runtime': {
      const r = w.runtimes.find((x) => x.id === a.runtime);
      if (!r) return 'Unknown runtime.';
      if (r.installed) return 'Already installed.';
      return r.can_install ? null : (r.reason ?? 'This runtime cannot be installed on this worker.');
    }
    case 'enable_runtime':
    case 'disable_runtime': {
      const r = w.runtimes.find((x) => x.id === a.runtime);
      if (!r) return 'Unknown runtime.';
      if (a.action === 'enable_runtime' && !r.available) return r.reason ?? 'This runtime is not available on the worker.';
      return null;
    }
    case 'set_resource_policy':
      return a.allow_pool && !w.pool_ceiling ? 'This worker cannot join the conversion pool.' : null;
  }
}
