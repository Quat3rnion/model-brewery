// Durable conversion workflow: one controller record runs
// inspect → plan → convert → verify. The browser only submits it once and
// then reads it; it never chains jobs itself.
//
// POST /api/conversions {host?, input_ids, options?, decode_backend?} (Idempotency-Key)
// GET  /api/conversions → Conversion[]
// GET  /api/conversions/{id} → Conversion
// POST /api/conversions/{id}/cancel → Conversion

import { buildConvertJob, type ConvertForm } from './convert';
import { obj } from './reports';
import type { Model } from './types';
import { rejectedSubmission, submissionError } from './api';
import { PlainAttempt } from './submission-key';

export const STAGES = ['inspect', 'plan', 'convert', 'verify'] as const;
export type Stage = (typeof STAGES)[number];
export const STAGE_LABEL: Record<Stage, string> = { inspect: 'Inspect', plan: 'Plan', convert: 'Convert', verify: 'Verify' };

export interface ConversionBody {
  host?: string;
  input_ids: string[];
  options?: Record<string, unknown>;
  decode_backend?: string;
  /** Saved custom profile revision (64 hex). Replaces options.profile. */
  profile_revision?: string;
  /** The controller defaults to HGN if omitted. */
  format?: 'gguf' | 'hgn';
}

/** Form value prefix for a saved profile revision in the profile select. */
export const REV_PREFIX = 'rev:';
export const DEFAULT_PROFILE = 'converter-default';

export function conversionRunHref(id: string, mode = 'hgn'): string {
  const params = new URLSearchParams(mode === 'hgn' ? { run: id } : { format: mode, run: id });
  return `#/convert?${params}`;
}

export interface Conversion {
  id: string;
  host: string | null;
  state: 'running' | 'succeeded' | 'failed' | 'cancelled' | (string & {});
  stage: Stage | null;
  jobIds: Partial<Record<Stage, string>>;
  cancelRequested: boolean;
  createdAt: number | null;
  updatedAt: number | null;
  error: string | null;
  artifactModelId: string | null;
  inputIds: string[];
  options: Record<string, unknown>;
  decodeBackend: string | null;
  requestedHost: string | null;
  profileRevision: string | null;
  profileName: string | null;
  format: string | null;
}

/** The form requires a profile choice, which may delegate to the converter. */
export function buildConversion(f: ConvertForm, models: Model[]): { body: ConversionBody | null; problems: string[] } {
  const rev = f.profile.startsWith(REV_PREFIX) ? f.profile.slice(REV_PREFIX.length) : null;
  const source = models.find((m) => m.id === f.source);
  // Reuse the converter step validation for inputs, extras and numeric options.
  const r = buildConvertJob('convert', rev !== null || f.profile === DEFAULT_PROFILE ? { ...f, profile: '' } : f, models);
  const problems = [...r.problems];
  if (!f.profile) problems.unshift('Choose the output profile.');
  if (rev !== null && !/^[0-9a-f]{64}$/.test(rev)) problems.unshift('The saved profile revision is not valid. Choose it again.');
  if (rev !== null && ((source && source.kind !== 'hf') ||
      [f.ht_backend, f.i4r_backend, f.q4cp_backend, f.decode_backend].some((backend) => backend && backend !== 'cpu'))) {
    problems.push('Select an HF source and CPU backends to use a custom profile.');
  }
  if (problems.length || !r.job) return { body: null, problems };
  const body: ConversionBody = { input_ids: r.job.input_ids, options: r.job.options };
  if (r.job.host) body.host = r.job.host;
  if (f.decode_backend) body.decode_backend = f.decode_backend;
  if (rev !== null) body.profile_revision = rev;
  return { body, problems };
}

const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function parseConversion(v: unknown): Conversion | null {
  const o = obj(v);
  if (!o || typeof o.id !== 'string' || typeof o.state !== 'string') return null;
  const ids = obj(o.job_ids) ?? {};
  const jobIds: Partial<Record<Stage, string>> = {};
  for (const s of STAGES) if (typeof ids[s] === 'string' && ids[s]) jobIds[s] = ids[s] as string;
  return {
    id: o.id,
    host: str(o.host),
    state: o.state,
    stage: (STAGES as readonly unknown[]).includes(o.stage) ? (o.stage as Stage) : null,
    jobIds,
    cancelRequested: o.cancel_requested === true,
    createdAt: num(o.created_at),
    updatedAt: num(o.updated_at),
    error: str(o.error),
    artifactModelId: str(o.artifact_model_id),
    inputIds: Array.isArray(o.input_ids) ? o.input_ids.filter((x): x is string => typeof x === 'string') : [],
    options: obj(o.options) ?? {},
    decodeBackend: str(o.decode_backend),
    requestedHost: str(o.requested_host),
    profileRevision: str(o.profile_revision),
    profileName: str(o.profile_name),
    format: str(o.format),
  };
}

export function parseConversions(v: unknown): Conversion[] {
  return Array.isArray(v) ? v.map(parseConversion).filter((x): x is Conversion => !!x) : [];
}

export function isActiveConversion(c: Conversion): boolean {
  return c.state === 'running';
}

export function newerConversion(current: Conversion, received: Conversion): Conversion {
  if (!isActiveConversion(current) && isActiveConversion(received)) return current;
  if (current.updatedAt !== null && (received.updatedAt === null || current.updatedAt > received.updatedAt)) return current;
  if (isActiveConversion(current) && isActiveConversion(received) && current.stage &&
      STAGES.indexOf(current.stage) > (received.stage ? STAGES.indexOf(received.stage) : -1)) return current;
  if (current.cancelRequested && !received.cancelRequested && isActiveConversion(received)) return current;
  return received;
}

export type StageStatus = 'pending' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'skipped' | 'unknown';

/**
 * One row per stage. Stages before the current one finished (the controller
 * only advances on success); the current stage shows its child job state;
 * later stages are pending, or skipped once the workflow ended early.
 */
export function stageRows(c: Conversion, jobStates: Record<string, string | undefined>): { stage: Stage; status: StageStatus; jobId: string | null }[] {
  const cur = c.stage ? STAGES.indexOf(c.stage) : -1;
  return STAGES.map((stage, i) => {
    const jobId = c.jobIds[stage] ?? null;
    let status: StageStatus;
    if (c.state === 'succeeded') status = 'succeeded';
    else if (i < cur) status = 'succeeded';
    else if (i > cur) status = c.state === 'running' ? 'pending' : 'skipped';
    else if (c.state === 'failed') status = 'failed';
    else if (c.state === 'cancelled') status = 'cancelled';
    else {
      const js = jobId ? jobStates[jobId] : undefined;
      status = js === 'queued' || js === 'running' ? js : js === 'succeeded' ? 'running' : jobId ? 'unknown' : 'queued';
    }
    return { stage, status, jobId };
  });
}

export type PostConversion = (body: ConversionBody, key: string) => Promise<{ data: unknown; status: number }>;

/**
 * One submission at a time. A retry of the same body reuses the key, so a
 * lost response cannot start a second workflow; a different body gets a new key.
 */
export class ConversionSubmitter {
  busy = false;
  error: string | null = null;
  replayed = false;
  #plain = new PlainAttempt();

  constructor(private launcher: string, private post: PostConversion) {}

  async submit(body: ConversionBody, current = () => true): Promise<Conversion | null> {
    if (this.busy) return null;
    this.error = null;
    this.busy = true;
    let plain: ReturnType<PlainAttempt['take']> | undefined;
    try {
      const snapshot: ConversionBody = JSON.parse(JSON.stringify(body));
      plain = this.#plain.take(this.launcher, snapshot, current);
      const { data, status } = await this.post(snapshot, plain.key);
      const c = parseConversion(data);
      if (!c?.id) throw new Error('The controller answered without a conversion record.');
      plain.complete();
      this.replayed = status === 200;
      return c;
    } catch (e) {
      if (rejectedSubmission(e)) plain?.complete();
      this.error = submissionError(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}
