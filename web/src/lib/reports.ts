// Derive the three separate report lines for a model bundle:
// file verification, generation, and benchmark quality.
// Each line uses only explicit evidence from controller job records.
// A finished job is never shown as a pass unless its result says so.

import type { JobDetail, JobSummary } from './types';
import { isActive } from './format';

export const NOT_RELEASE_QUALIFIED = 'Not release-qualified';

export type ReportStatus =
  | 'not-run'
  | 'queued'
  | 'running'
  | 'passed'
  | 'failed'
  | 'completed'
  | 'unvalidated'
  | 'not-qualifying'
  | 'cancelled'
  | 'interrupted';

export interface ReportLine {
  status: ReportStatus;
  label: string;
  detail: string;
  jobId: string | null;
}

export function obj(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Latest job of a kind whose inputs include the model id. Jobs are in queue order. */
export function latestJobFor(jobs: JobSummary[], kind: string, modelId: string): JobSummary | null {
  for (let i = jobs.length - 1; i >= 0; i -= 1) {
    const job = jobs[i];
    if (job.kind === kind && job.input_ids.includes(modelId)) return job;
  }
  return null;
}

function nonTerminal(job: JobSummary | null, what: string): ReportLine | null {
  if (!job) return { status: 'not-run', label: 'Not run', detail: `No ${what} job for this entry.`, jobId: null };
  if (job.state === 'queued') return { status: 'queued', label: 'Queued', detail: `${what} job waits in the queue.`, jobId: job.id };
  if (isActive(job.state)) return { status: 'running', label: 'Running', detail: `${what} job is running. No result yet.`, jobId: job.id };
  if (job.state === 'cancelled') return { status: 'cancelled', label: 'Cancelled', detail: `The last ${what} job was cancelled. No result.`, jobId: job.id };
  if (job.state === 'interrupted' || job.state === 'missing')
    return { status: 'interrupted', label: 'Interrupted', detail: `The last ${what} job was interrupted. Partial output is excluded.`, jobId: job.id };
  if (job.state === 'failed') return { status: 'failed', label: 'Failed', detail: job.error ?? `The ${what} job failed.`, jobId: job.id };
  return null;
}

/** Explicit verdict field in a report object, or null when the report has none. */
export function explicitVerdict(report: unknown): boolean | null {
  const r = obj(report);
  if (!r) return null;
  for (const key of ['passed', 'ok', 'verified']) {
    if (typeof r[key] === 'boolean') return r[key] as boolean;
  }
  for (const key of ['status', 'verdict']) {
    const v = r[key];
    if (v === 'passed' || v === 'pass' || v === 'verified') return true;
    if (v === 'failed' || v === 'fail' || v === 'mismatch') return false;
  }
  return null;
}

export function verificationLine(job: JobSummary | null, detail: JobDetail | undefined): ReportLine {
  const early = nonTerminal(job, 'verification');
  if (early) return early;
  const id = job!.id;
  const result = obj(detail?.result);
  if (!result) return { status: 'completed', label: 'Finished', detail: 'Result not loaded.', jobId: id };
  const v = fileVerificationStatus(result);
  if (v === 'passed') return { status: 'passed', label: 'Files verified', detail: 'File verification passed and the verify report states a pass.', jobId: id };
  if (v === 'failed') return { status: 'failed', label: 'Verification failed', detail: 'The verify result states a failure.', jobId: id };
  if (v === 'not_performed') return { status: 'completed', label: 'Not performed', detail: 'The job did not verify the files.', jobId: id };
  return { status: 'completed', label: 'No verdict', detail: 'The job finished, but the result has no explicit verdict. Read the report.', jobId: id };
}

/** Pass only when file_verification.status is passed AND report.passed is true. */
export function fileVerificationStatus(result: Record<string, unknown>): 'passed' | 'failed' | 'not_performed' | null {
  const fileStatus = obj(result.file_verification)?.status;
  const reportPassed = obj(result.report)?.passed;
  if (fileStatus === 'not_performed') return 'not_performed';
  if (fileStatus === 'passed' && reportPassed === true) return 'passed';
  if (explicitVerdict({ status: fileStatus }) === false || reportPassed === false) return 'failed';
  return null;
}

export function generationLine(job: JobSummary | null, detail: JobDetail | undefined): ReportLine {
  const early = nonTerminal(job, 'generation');
  if (early) return early;
  const result = obj(detail?.result);
  if (!result) return { status: 'completed', label: 'Finished', detail: 'Result not loaded.', jobId: job!.id };
  const proof = loadProofText(result.load_proof);
  if (typeof result.text !== 'string')
    return { status: 'failed', label: 'No text', detail: `The result has no generated text. ${proof}`, jobId: job!.id };
  return { status: 'completed', label: 'Text generated', detail: `Generation is not a quality result. ${proof}`, jobId: job!.id };
}

export function loadProofText(proof: unknown): string {
  if (proof === undefined || proof === null) return 'No load proof recorded.';
  const p = obj(proof);
  const verdict = explicitVerdict(p);
  if (verdict === true) return 'Load proof: opened files match the bundle.';
  if (verdict === false) return 'Load proof: opened files do not match the bundle.';
  return 'Load proof recorded without a verdict.';
}

/**
 * True only when the result states real scoring of the full pinned split:
 * result.dataset.full_split === true AND result.identity.synthetic === false.
 * False when either flag says otherwise; null when the flags are absent.
 */
export function isFullSplit(result: Record<string, unknown>): boolean | null {
  const full = obj(result.dataset)?.full_split;
  const synthetic = obj(result.identity)?.synthetic;
  if (full === true && synthetic === false) return true;
  if (full === false || synthetic === true) return false;
  return null;
}

export function isSynthetic(result: Record<string, unknown>): boolean {
  return obj(result.identity)?.synthetic === true;
}

export interface Score {
  correct: number;
  total: number;
  unparsed?: number;
  accuracy_basis_points?: number;
}

export function readScore(result: Record<string, unknown> | null): Score | null {
  const s = obj(result?.score);
  if (!s || typeof s.correct !== 'number' || typeof s.total !== 'number') return null;
  return s as unknown as Score;
}

export function accuracyText(score: Score): string {
  const bp = typeof score.accuracy_basis_points === 'number' ? score.accuracy_basis_points : score.total > 0 ? (score.correct * 10000) / score.total : NaN;
  const pct = Number.isFinite(bp) ? `${(bp / 100).toFixed(2)}%` : '—';
  return `${score.correct.toLocaleString()} / ${score.total.toLocaleString()} correct (${pct})`;
}

export function referenceInfo(r: Record<string, unknown> | null) {
  if (!r || !(r.comparison || r.baseline_identity || r.baseline_score || r.baseline_artifact)) return null;
  const identity = obj(r.baseline_identity);
  const provenance = obj(identity?.provenance);
  return {
    scope: typeof r.reference_scope === 'string' ? r.reference_scope : null,
    precision: typeof provenance?.precision === 'string' ? provenance.precision : r.reference_scope === 'attested_bf16_same_source' ? 'BF16' : null,
    identity: r.baseline_identity ?? null, score: r.baseline_score ?? null, artifact: r.baseline_artifact ?? null,
  };
}

export function benchmarkLine(job: JobSummary | null, detail: JobDetail | undefined): ReportLine {
  const disclaimer = `${NOT_RELEASE_QUALIFIED}.`;
  const early = nonTerminal(job, 'GSM8K evaluation');
  if (early) return { ...early, detail: `${early.detail} ${disclaimer}` };
  const id = job!.id;
  const result = obj(detail?.result);
  if (!result) return { status: 'completed', label: 'Finished', detail: `Result not loaded. ${disclaimer}`, jobId: id };
  const score = readScore(result);
  if (!score) return { status: 'completed', label: 'No score', detail: `The job finished without a score. ${disclaimer}`, jobId: id };
  const full = isFullSplit(result);
  if (full !== true)
    return {
      status: 'not-qualifying',
      label: isSynthetic(result) ? 'Synthetic fixture' : full === false ? 'Not the full split' : 'Split not confirmed',
      detail: `${accuracyText(score)}. ${disclaimer}`,
      jobId: id,
    };
  const comparison = obj(result.comparison);
  const delta = typeof comparison?.delta_correct === 'number' ? ` Candidate minus reference: ${comparison.delta_correct >= 0 ? '+' : ''}${comparison.delta_correct} correct.` : '';
  return {
    status: 'completed',
    label: comparison ? 'GSM8K score with reference' : 'Standalone GSM8K score',
    detail: `${accuracyText(score)}.${delta} ${disclaimer}`,
    jobId: id,
  };
}

export function lineTone(status: ReportStatus): 'ok' | 'bad' | 'warn' | 'info' | 'muted' {
  switch (status) {
    case 'passed':
      return 'ok';
    case 'failed':
      return 'bad';
    case 'unvalidated':
    case 'not-qualifying':
    case 'cancelled':
    case 'interrupted':
      return 'warn';
    case 'running':
    case 'queued':
      return 'info';
    default:
      return 'muted';
  }
}
