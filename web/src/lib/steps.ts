// Which check a job is in, who runs it, and why its counter restarted.
//
// The history is what this page observed for one job ID: the phases of its
// progress detail in producer-time order. It is never inferred from paths,
// inputs or other jobs; a different job ID always has its own history. After
// a reload the history starts again at the current phase, so a same-job note
// only appears for transitions this page saw.

import { kindTitle } from './format';
import { phaseLabel, type ProgressDetail } from './progress';
import type { JobSummary } from './types';

export type StepReason = 'first' | 'phase' | 'input' | 'restart';

export interface Step {
  phase: string;
  input_index: number | null;
  input_count: number | null;
  reason: StepReason;
  /** Producer time of the first sample of this step. */
  at: number;
}

interface JobSteps {
  steps: Step[];
  /** Steps dropped from the front once the list is full; keeps numbering true. */
  dropped: number;
  lastAt: number;
  lastCompleted: number;
}

const MAX_STEPS = 12;

export class StepTracker {
  #jobs = new Map<string, JobSteps>();

  /** Record one sample. Duplicates and older samples are ignored. Returns true when the history changed. */
  observe(jobId: string, d: ProgressDetail): boolean {
    const at = d.updated_at_ms;
    if (at === null) return false;
    const step = (reason: StepReason): Step => ({ phase: d.phase, input_index: d.input_index, input_count: d.input_count, reason, at });
    const j = this.#jobs.get(jobId);
    if (!j) {
      this.#jobs.set(jobId, { steps: [step('first')], dropped: 0, lastAt: at, lastCompleted: d.completed });
      return true;
    }
    if (at <= j.lastAt) return false;
    const cur = j.steps[j.steps.length - 1];
    const reason: StepReason | null =
      cur.phase !== d.phase ? 'phase' : cur.input_index !== d.input_index || cur.input_count !== d.input_count ? 'input' : d.completed < j.lastCompleted ? 'restart' : null;
    j.lastAt = at;
    j.lastCompleted = d.completed;
    if (!reason) return false;
    j.steps.push(step(reason));
    if (j.steps.length > MAX_STEPS) {
      j.steps.shift();
      j.dropped += 1;
    }
    return true;
  }

  history(jobId: string): readonly Step[] {
    return this.#jobs.get(jobId)?.steps ?? [];
  }

  /** Number of steps dropped before history(jobId)[0]. */
  dropped(jobId: string): number {
    return this.#jobs.get(jobId)?.dropped ?? 0;
  }

  /** Drop the history of jobs not in `keep` (terminal, cancelled, removed). */
  retain(keep: ReadonlySet<string>): boolean {
    let changed = false;
    for (const id of [...this.#jobs.keys()])
      if (!keep.has(id)) {
        this.#jobs.delete(id);
        changed = true;
      }
    return changed;
  }
}

const ENGINE_KINDS: ReadonlySet<string> = new Set(['compare', 'inspect', 'plan', 'convert', 'verify']);
const EVAL_KINDS: ReadonlySet<string> = new Set(['generate', 'evaluate', 'baseline']);
const TRANSFER_KINDS: ReadonlySet<string> = new Set(['transfer', 'transfer_preview']);

/** Who runs a phase and what it does. Only for phases whose producer is known from the workbench code. */
function actor(kind: string, phase: string, fingerprinted: boolean): { name: string; purpose: string | null } {
  switch (phase) {
    case 'hash_input':
      return TRANSFER_KINDS.has(kind)
        ? { name: 'Source worker fingerprint', purpose: 'The source host reads every file and records its SHA-256 before the copy.' }
        : { name: 'Worker fingerprint', purpose: 'The worker reads every input file and records its SHA-256 for this job.' };
    case 'pinned_manifest':
      return {
        name: 'Pinned manifest check',
        purpose: 'The worker uses the operator-pinned checksums. It checks file names and sizes only; it does not read file contents.',
      };
    case 'verify_pinned_input':
      // One full read against the pinned checksums. Not a reread, and not compared with a worker hash.
      return { name: 'Engine full read', purpose: 'The engine reads every input byte once and compares it with the operator-pinned checksums.' };
    case 'inventory_input':
      return { name: 'Worker inventory', purpose: 'The worker lists the input files before it fingerprints them.' };
    case 'verify_input':
      if (ENGINE_KINDS.has(kind))
        return {
          name: 'Engine reread',
          purpose: fingerprinted
            ? 'The engine reads the input again and compares it with the worker fingerprint from earlier in this job.'
            : 'The engine reads the input again and checks its SHA-256.',
        };
      if (EVAL_KINDS.has(kind)) return { name: 'Evaluator input check', purpose: 'The evaluator reads the input and checks it before the run.' };
      return { name: 'Input check', purpose: 'The input is read again and its SHA-256 is checked.' };
    case 'copy_input':
      return { name: 'Worker copy', purpose: 'The worker copies the bundle to this host.' };
    case 'verify_cache':
      return { name: 'Worker cache check', purpose: 'The worker rereads the cached bundle and compares it with the bundle manifest.' };
    case 'verify_staged':
      return { name: 'Worker staged-copy check', purpose: 'The worker rereads the copied files and compares them with the bundle manifest.' };
    case 'verify_output':
      return { name: 'Worker output check', purpose: 'The worker reads the engine output and records its SHA-256.' };
    default:
      return { name: phaseLabel(phase), purpose: null };
  }
}

const PROFILES: Record<string, string> = { v2: 'HGN v2', ht43: 'HT43' };

function profileOf(job: Pick<JobSummary, 'options'>): string | null {
  const p = job.options?.profile;
  return typeof p === 'string' && p ? `${PROFILES[p] ?? p} profile` : null;
}

/** "Converter plan, HT43 profile". The profile only when the job options carry one. */
export function jobIdentity(job: Pick<JobSummary, 'kind' | 'options'>): string {
  const profile = profileOf(job);
  return `${kindTitle(job.kind)}${profile ? `, ${profile}` : ''}`;
}

export interface StepView {
  /** "Worker fingerprint · Converter plan, HT43 profile · job 282d971f" */
  role: string;
  /** For the compact banner, which already shows the job kind and ID: "Worker fingerprint, HT43 profile". */
  compactRole: string;
  /** One sentence on what the step does; null for an unknown phase. */
  purpose: string | null;
  /** Why the counter restarted in this job, when this page saw it happen. */
  transition: string | null;
  compactTransition: string | null;
  /** The checks this page saw for this job, when there is more than one. */
  sequence: string | null;
}

export function describeStep(job: JobSummary, d: ProgressDetail, history: readonly Step[], dropped = 0): StepView {
  // Steps of this phase and input that the page saw; the last one is current.
  let cur = -1;
  for (let i = history.length - 1; i >= 0; i--)
    if (history[i].phase === d.phase && history[i].input_index === d.input_index && history[i].input_count === d.input_count) {
      cur = i;
      break;
    }
  const earlier = cur >= 0 ? history.slice(0, cur) : [];
  const fingerprinted = earlier.some((s) => s.phase === 'hash_input' && s.input_index === d.input_index && s.input_count === d.input_count);
  const a = actor(job.kind, d.phase, fingerprinted);
  const role = `${a.name} · ${jobIdentity(job)} · job ${job.id.slice(0, 8)}`;
  let transition: string | null = null;
  let compactTransition: string | null = null;
  const step = cur >= 0 ? history[cur] : null;
  if (step && cur > 0) {
    const prev = history[cur - 1];
    if (step.reason === 'phase') {
      transition = `Same job, new check: the counter restarted because this job moved from ${phaseLabel(prev.phase)} to ${phaseLabel(step.phase)}.`;
      compactTransition = 'Same job, new check';
    } else if (step.reason === 'input') {
      const n = step.input_index !== null ? `input ${step.input_index}${step.input_count !== null ? ` of ${step.input_count}` : ''}` : 'the next input';
      transition = `Same job, next input: the counter restarted for ${n}.`;
      compactTransition = 'Same job, next input';
    } else if (step.reason === 'restart') {
      transition = 'Same job: the counter restarted because the job reported a lower value.';
      compactTransition = 'Same job, counter restarted';
    }
  }
  const sequence =
    cur > 0
      ? `Checks seen on this page for this job: ${history
          .slice(0, cur + 1)
          .map((s, i) => `${dropped + i + 1}. ${actor(job.kind, s.phase, false).name}${i === cur ? ' (now)' : ''}`)
          .join(' → ')}`
      : null;
  const profile = profileOf(job);
  const compactRole = `${a.name}${profile ? `, ${profile}` : ''}`;
  return { role, compactRole, purpose: a.purpose, transition, compactTransition, sequence };
}
