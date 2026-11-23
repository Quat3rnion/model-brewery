import { api, rejectedSubmission, submissionError } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';
import { bodyKey, PlainAttempt } from './submission-key';
import { captureSubmissionVisit } from './route.svelte';
import { captureResultFocus } from './focus';

type AcceptedJob = JobSummary | Pick<JobSummary, 'id'>;
const MULTI_CLICK_MS = 500;

/** One launcher request scope. Use a separate scope for independent reads. */
export class Submission {
  busy = $state(false);
  error = $state<string | null>(null);
  #plain = new PlainAttempt();
  #sent?: { request: string; at: number; current: () => boolean };

  async send<Body extends object, Result>(
    launcher: string,
    body: Body,
    post: (body: Body, key: string) => Promise<Result>,
    event?: Pick<MouseEvent, 'detail' | 'timeStamp'>,
  ): Promise<Result | null> {
    if (this.busy) return null;
    const current = captureSubmissionVisit();
    const request = bodyKey([launcher, body])!;
    const prior = this.#sent;
    if (event && event.detail > 1 && prior?.request === request && prior.current() &&
        event.timeStamp >= prior.at && event.timeStamp - prior.at <= MULTI_CLICK_MS) return null;

    const plain = this.#plain.take(launcher, body, current);
    this.busy = true;
    this.error = null;
    try {
      const pending = post(body, plain.key);
      this.#sent = event && event.detail > 0 ? { request, at: event.timeStamp, current } : undefined;
      const result = await pending;
      plain.complete();
      return result;
    } catch (e) {
      if (rejectedSubmission(e)) plain.complete();
      this.error = submissionError(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

/** Tracks the result of queuing a job. */
export class Submitter extends Submission {
  job = $state<AcceptedJob | null>(null);
  replayed = $state(false);
  statusElement?: HTMLElement;

  constructor(private launcher: string | ((job: NewJob) => string)) { super(); }

  async submit(job: NewJob, event: MouseEvent, focusStatus = true): Promise<AcceptedJob | null> {
    const focus = captureResultFocus(event);
    const trigger = event.currentTarget;
    // Use the same wire body for the key comparison and the POST.
    const body: NewJob = JSON.parse(JSON.stringify(stripEmpty(job)));
    const launcher = typeof this.launcher === 'string' ? this.launcher : this.launcher(body);
    const sent = this.send(launcher, body, async (snapshot, key) => {
      const created = await api.queueJob(snapshot, key);
      if (typeof created?.id !== 'string' || !created.id) throw new Error('The controller did not return a job ID.');
      return created;
    }, event);
    const created = await sent;
    if (created) {
      this.job = created;
      this.replayed = !('state' in created);
      if (focusStatus) void focus?.(() => this.statusElement?.querySelector('a'));
      void brewery.refresh();
    } else if (this.error && trigger instanceof HTMLButtonElement) {
      void focus?.(() => trigger.isConnected && !trigger.disabled ? trigger
        : this.statusElement?.querySelector<HTMLElement>('[role="alert"]'));
    }
    return created;
  }
}

/** Remove empty optional values so the controller applies its own defaults. */
export function stripEmpty(job: NewJob): NewJob {
  const checklist = job.kind === 'transfer' || job.kind === 'transfer_preview';
  const out: NewJob = { kind: job.kind, input_ids: checklist ? [...new Set(job.input_ids)].sort() : job.input_ids };
  if (job.host) out.host = job.host;
  if (job.prompt_id) out.prompt_id = job.prompt_id;
  else if (job.prompt) out.prompt = job.prompt;
  if (job.options) {
    const options: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(job.options)) {
      if (v === undefined || v === null || v === '' || (typeof v === 'number' && Number.isNaN(v))) continue;
      options[k] = v;
    }
    const coords = options.indices;
    if (job.kind === 'weight_tune' && Array.isArray(coords) && coords.every(pair =>
      Array.isArray(pair) && pair.length === 2 && pair.every(Number.isSafeInteger))) {
      // Coordinate order is not a control in the editor. Preserve each tuple's axes.
      options.indices = [...coords].sort(([ar, ac], [br, bc]) => ar - br || ac - bc);
    }
    if (Object.keys(options).length) out.options = options;
  }
  return out;
}

/** Reserve a matched read's slot until a detail or state snapshot supplies its state. */
export function acceptedJobSummary(accepted: AcceptedJob, request: NewJob): JobSummary {
  return 'state' in accepted ? accepted : {
    id: accepted.id, state: 'queued', kind: request.kind, host: request.host,
    input_ids: request.input_ids, options: request.options,
  };
}
