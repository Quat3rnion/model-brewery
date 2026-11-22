import { obj } from './reports';
import { readDirectories, type Directories } from './model-directories';
import type { Blocker } from './blockers';

export const limits = {
  concurrency: 'Concurrent jobs',
  cpu_threads: 'CPU threads per job',
  min_free_ram_bytes: 'Minimum free RAM',
  disk_reserve_bytes: 'Disk reserve',
};
const gib = 1024 ** 3;
export const resourceUnit = (key: string) => key.endsWith('_bytes') ? 'GiB' : '';
export function resourceInput(key: string, value: number) {
  const n = resourceUnit(key) ? value / gib : value;
  // Keep tiny byte limits in plain decimal so the displayed value is valid input.
  return n > 0 && n < 0.000001 ? n.toFixed(20).replace(/0+$/, '') : String(n);
}
export const resourceDisplay = (key: string, value: number) => `${resourceInput(key, value)}${resourceUnit(key) ? ' GiB' : ''}`;
interface Credential { id:string; label:string }
interface EditorSetup { status:string; model_directory:string|null; credentials:Credential[]; reason?:string; blocker?:string }
function isCredential(value:unknown):value is Credential {
  const c=obj(value);
  return typeof c?.id==='string' && c.id.length>0 && typeof c.label==='string';
}
export interface WeightEditorPolicy { available:boolean; enabled:boolean; capabilities:Record<string,boolean>; reason?:string; setup?:EditorSetup }
function readWeightEditor(raw:unknown):WeightEditorPolicy|null {
  const w=obj(raw), caps=obj(w?.capabilities);
  if (!w || typeof w.available !== 'boolean' || typeof w.enabled !== 'boolean' || !caps || Object.values(caps).some(v=>typeof v !== 'boolean')) return null;
  const setup=obj(w.setup);
  return {available:w.available, enabled:w.enabled, capabilities:caps as Record<string,boolean>, ...(typeof w.reason === 'string' ? {reason:w.reason} : {}),
    ...(setup && typeof setup.status==='string' && Array.isArray(setup.credentials) && setup.credentials.every(isCredential)
      ? {setup:{status:setup.status, model_directory:typeof setup.model_directory==='string'?setup.model_directory:null,
        credentials:setup.credentials, ...(typeof setup.reason==='string' ? {reason:setup.reason} : {}),
        ...(typeof setup.blocker==='string' ? {blocker:setup.blocker} : {})}} : {})};
}
export interface HostPolicy {
  directories:Directories|null;
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
    out.push({directories:readDirectories(h.model_directories),weightEditor:readWeightEditor(h.weight_editor), id:h.id, name:typeof h.name === 'string' ? h.name : h.id, revision:h.revision, paused:h.paused, draining:h.draining,
      installed_jobs:installed, allowed_jobs:h.allowed_jobs, disabled_jobs:h.disabled_jobs, resource_bounds:bounds,
      resources, effective_resources:numbers(h.effective_resources), resourceEditable});
  }
  return out;
}
export function validateResources(h: HostPolicy, input: Record<string,string>) {
  const values: Record<string,number> = {};
  const blockers: Blocker[] = [];
  for (const [key, raw] of Object.entries(input)) {
    const text = raw.trim();
    if (!text) continue;
    const range = h.resource_bounds[key];
    const usesGiB = !!resourceUnit(key);
    const number = usesGiB ? Math.round(Number(text) * gib) : Number(text);
    const pattern = usesGiB ? /^(?:\d+(?:\.\d*)?|\.\d+)$/ : /^\d+$/;
    if (!range || !pattern.test(text) || !integer(number) || number < range.min || number > range.max) {
      const name = limits[key as keyof typeof limits] ?? key;
      blockers.push({ field: key, text: range
        ? `${name}: enter ${usesGiB ? 'a value' : 'a whole number'} from ${resourceDisplay(key, range.min)} to ${resourceDisplay(key, range.max)}, or leave it empty to inherit.`
        : `${name}: the controller did not report a range.` });
    } else values[key] = number;
  }
  return { values: h.resourceEditable && !blockers.length ? values : null, blockers };
}
export function resourceValues(h: HostPolicy, input: Record<string,string>): Record<string,number> | null {
  return validateResources(h, input).values;
}
export type PolicyAction =
  | {action:'setup_weight_editor'; worker:string; revision:number; model_directory:string; credential:string}
  | {action:'set_weight_editor'; worker:string; revision:number; enabled:boolean}
  | {action:'set_host_policy'; worker:string; revision:number; paused:boolean; disabled_jobs:string[]}
  | {action:'set_host_resources'; worker:string; revision:number; resources:Record<string,number>};
