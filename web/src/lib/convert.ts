import type { Host, Model, NewJob } from './types';
import { BUNDLE_KINDS, CONVERT_SOURCE_KINDS, EXTRA_INPUT_KINDS } from './format';

export type ConvertStep = 'inspect' | 'plan' | 'convert' | 'verify';
type Extra = (typeof EXTRA_INPUT_KINDS)[number];

export interface ConvertForm {
  source: string;
  bundle: string;
  extras: Record<Extra, string>;
  host: string;
  /** Convert only: send host 'auto' so the controller picks an idle pool worker at launch. */
  anyHost: boolean;
  profile: string;
  ht_backend: string;
  i4r_backend: string;
  q4cp_backend: string;
  decode_backend: string;
  seed: number | null;
  threads: number | null;
  ht_beam: number | null;
}

/** Option values offered as selects. Update when the engine confirms its enums. */
export const CONVERT_ENUMS = {
  profile: ['v2', 'ht43'],
  backend: ['cpu', 'cuda', 'hip'],
} as const;

function intIn(v: number | null, min: number, max: number): boolean {
  return v === null || (Number.isInteger(v) && v >= min && v <= max);
}

/**
 * Build the POST /api/jobs body for a converter step.
 * Inputs: [source, ...extras] for inspect/plan/convert, where options.{assets,
 * ngram,vision,mtp} hold zero-based input indexes; [source, bundle] for verify.
 */
export function buildConvertJob(step: ConvertStep, f: ConvertForm, models: Model[], hosts: Host[] = []): { job: NewJob | null; problems: string[] } {
  const problems: string[] = [];
  const src = models.find((m) => m.id === f.source);
  if (!src) problems.push('Select a source checkpoint.');
  else if (!CONVERT_SOURCE_KINDS.has(src.kind)) problems.push('The source must be a checkpoint, GGUF or HGN entry.');

  const input_ids = f.source ? [f.source] : [];
  const options: Record<string, unknown> = {};

  if (step === 'verify') {
    const bundle = models.find((m) => m.id === f.bundle);
    if (!bundle) problems.push('Select the bundle to verify.');
    else if (!BUNDLE_KINDS.has(bundle.kind)) problems.push('The second input must be an HGN bundle.');
    else if (bundle.id === f.source) problems.push('Select a bundle that is not the source.');
    if (f.bundle) input_ids.push(f.bundle);
    if (f.decode_backend) options.decode_backend = f.decode_backend;
  }
  {
    for (const k of EXTRA_INPUT_KINDS) {
      const id = f.extras[k];
      if (!id) continue;
      if (input_ids.includes(id)) {
        problems.push(`The ${k} input repeats another input.`);
        continue;
      }
      options[k] = input_ids.length;
      input_ids.push(id);
    }
  }
  if (step !== 'verify') {
    if (f.profile) options.profile = f.profile;
    for (const key of ['ht_backend', 'i4r_backend', 'q4cp_backend'] as const) if (f[key]) options[key] = f[key];
    if (step !== 'inspect') {
      if (!intIn(f.seed, 0, Number.MAX_SAFE_INTEGER)) problems.push('Seed must be a whole number of 0 or more.');
      if (!intIn(f.threads, 1, 64)) problems.push('Threads must be 1 to 64.');
      if (!intIn(f.ht_beam, 1, 256)) problems.push('HT beam must be 1 to 256.');
      if (f.seed !== null) options.seed = f.seed;
      if (f.threads !== null) options.threads = f.threads;
      if (f.ht_beam !== null) options.ht_beam = f.ht_beam;
    }
  }

  const host = step === 'convert' && f.anyHost ? 'auto' : f.host || undefined;
  const allow = backendProblems(step, f, host === 'auto' ? undefined : (host ?? src?.host), hosts);
  problems.push(...allow);
  if (problems.length) return { job: null, problems };
  return { job: { kind: step, host, input_ids, options }, problems };
}

const BACKEND_KEYS = { ht_backend: 'ht', i4r_backend: 'i4r', q4cp_backend: 'q4cp' } as const;

/** Backends a host allows for one encoder; null when unrestricted. */
export function allowedBackends(host: Host | undefined, key: keyof typeof BACKEND_KEYS): string[] | null {
  const lists = host?.quantization_backends;
  if (!lists || typeof lists !== 'object') return null;
  const list = (lists as Record<string, unknown>)[BACKEND_KEYS[key]];
  return Array.isArray(list) ? list.filter((x): x is string => typeof x === 'string') : null;
}

function backendProblems(step: ConvertStep, f: ConvertForm, hostId: string | undefined, hosts: Host[]): string[] {
  if (step === 'verify' || !hostId) return [];
  const host = hosts.find((h) => h.id === hostId);
  const out: string[] = [];
  for (const key of Object.keys(BACKEND_KEYS) as (keyof typeof BACKEND_KEYS)[]) {
    const value = f[key];
    const allowed = allowedBackends(host, key);
    if (value && allowed && !allowed.includes(value))
      out.push(`${host?.name ?? hostId} does not allow ${value} for ${BACKEND_KEYS[key].toUpperCase()}; allowed: ${allowed.join(', ') || 'none'}.`);
  }
  return out;
}

/** Hosts in the conversion pool. Empty means auto falls back to the source host. */
export function poolHosts(hosts: Host[]): Host[] {
  return hosts.filter((h) => h.quantization_pool === true);
}
