import { obj } from './reports';

export const limits = {
  concurrency: 'Concurrent jobs',
  cpu_threads: 'CPU threads per job',
  min_free_ram_bytes: 'Minimum free RAM (bytes)',
  disk_reserve_bytes: 'Disk reserve (bytes)',
};
export interface WeightEditorPolicy { available:boolean; enabled:boolean; capabilities:Record<string,boolean>; reason?:string }
function readWeightEditor(raw:unknown):WeightEditorPolicy|null {
  const w=obj(raw), caps=obj(w?.capabilities);
  if (!w || typeof w.available !== 'boolean' || typeof w.enabled !== 'boolean' || !caps || Object.values(caps).some(v=>typeof v !== 'boolean')) return null;
  return {available:w.available, enabled:w.enabled, capabilities:caps as Record<string,boolean>, ...(typeof w.reason === 'string' ? {reason:w.reason} : {})};
}
export interface HostPolicy {
  weightEditor:WeightEditorPolicy|null;
  id: string; name: string; revision: number; paused: boolean; draining: boolean;
  installed_jobs: string[]; allowed_jobs: string[]; disabled_jobs: string[];
  resource_bounds: Record<string, {min: number; max: number}>;
  resources: Record<string, number>; effective_resources: Record<string, number>;
  resourceEditable: boolean;
}
const integer = (n: unknown): n is number => typeof n === 'number' && Number.isSafeInteger(n) && n >= 0;
const strings = (x: unknown): x is string[] => Array.isArray(x) && x.every(v => typeof v === 'string');
function numbers(value: unknown) {
  const out: Record<string, number> = {};
  for (const [key,n] of Object.entries(obj(value) ?? {})) if (integer(n) && Object.hasOwn(limits,key)) out[key] = n;
  return out;
}
export function parseHostPolicies(value: unknown): HostPolicy[] | null {
  if (!Array.isArray(value)) return null;
  const out: HostPolicy[] = [];
  for (const raw of value) {
    const h = obj(raw);
    if (!h || typeof h.id !== 'string' || !integer(h.revision) || typeof h.paused !== 'boolean' || typeof h.draining !== 'boolean'
      || !strings(h.installed_jobs) || !strings(h.disabled_jobs) || !strings(h.allowed_jobs)) return null;
    const installed = h.installed_jobs;
    if (!h.disabled_jobs.every(k => installed.includes(k))) return null;
    const bounds: HostPolicy['resource_bounds'] = {};
    for (const [k,v] of Object.entries(obj(h.resource_bounds) ?? {})) {
      const range = obj(v);
      if (Object.hasOwn(limits,k) && integer(range?.min) && integer(range?.max) && range.min <= range.max) bounds[k] = {min: range.min,max: range.max};
    }
    const resources = numbers(h.resources);
    // Complete replacement must not drop a newer server's unknown overrides.
    const resourceEditable = !!obj(h.resources) && Object.keys(obj(h.resources)!).every(k => Object.hasOwn(bounds,k) && Object.hasOwn(resources,k));
    out.push({weightEditor:readWeightEditor(h.weight_editor), id:h.id, name:typeof h.name === 'string' ? h.name : h.id, revision:h.revision, paused:h.paused, draining:h.draining,
      installed_jobs:installed, allowed_jobs:h.allowed_jobs, disabled_jobs:h.disabled_jobs, resource_bounds:bounds,
      resources, effective_resources:numbers(h.effective_resources), resourceEditable});
  }
  return out;
}
export function resourceValues(h: HostPolicy, input: Record<string,string>): Record<string,number> | null {
  if (!h.resourceEditable) return null;
  const out: Record<string,number> = {};
  for (const [key,text] of Object.entries(input)) {
    if (!text.trim()) continue;
    const range = h.resource_bounds[key], n = Number(text);
    if (!range || !/^\d+$/.test(text) || !integer(n) || n < range.min || n > range.max) return null;
    out[key] = n;
  }
  return out;
}
export type PolicyAction =
  | {action:'set_weight_editor'; worker:string; revision:number; enabled:boolean}
  | {action:'set_host_policy'; worker:string; revision:number; paused:boolean; disabled_jobs:string[]}
  | {action:'set_host_resources'; worker:string; revision:number; resources:Record<string,number>};
