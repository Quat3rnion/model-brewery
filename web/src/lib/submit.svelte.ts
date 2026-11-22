import { api, rejectedSubmission, submissionError } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';
import { PlainAttempt, retryKey, type RetrySubmission } from './retry-keys';
import { captureSubmissionVisit } from './route.svelte';

type AcceptedJob = JobSummary | Pick<JobSummary, 'id'>;

/** Tracks one form's job submission: busy flag, error text and the queued job. */
export class Submitter {
  busy = $state(false);
  error = $state<string | null>(null);
  job = $state<AcceptedJob | null>(null);
  replayed = $state(false);
  #plain = new PlainAttempt();

  constructor(private launcher: string | ((job: NewJob) => string)) {}

  async submit(job: NewJob, retry?: RetrySubmission): Promise<AcceptedJob | null> {
    if (this.busy || retry?.blocked) return null;
    this.busy = true;
    this.error = null;
    const failedId = retry?.failedId;
    let plain: ReturnType<PlainAttempt['take']> | undefined;
    try {
      // Freeze the actual wire body before yielding to Web Crypto.
      const body: NewJob = JSON.parse(JSON.stringify(stripEmpty(job)));
      const launcher = typeof this.launcher === 'string' ? this.launcher : this.launcher(body);
      if (!failedId) plain = this.#plain.take(launcher, body, captureSubmissionVisit());
      const key = plain?.key ?? await retryKey(failedId!, launcher, body);
      const created = await api.queueJob(body, key);
      if (typeof created?.id !== 'string' || !created.id) throw new Error('The controller did not return a job ID.');
      plain?.complete();
      this.job = created;
      this.replayed = !('state' in created);
      void brewery.refresh();
      return created;
    } catch (e) {
      if (rejectedSubmission(e)) plain?.complete();
      this.error = submissionError(e, !!failedId);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

/** Remove empty optional values so the controller applies its own defaults. */
export function stripEmpty(job: NewJob): NewJob {
  const out: NewJob = { kind: job.kind, input_ids: job.input_ids };
  if (job.host) out.host = job.host;
  if (job.prompt_id) out.prompt_id = job.prompt_id;
  else if (job.prompt) out.prompt = job.prompt;
  if (job.options) {
    const options: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(job.options)) {
      if (v === undefined || v === null || v === '' || (typeof v === 'number' && Number.isNaN(v))) continue;
      options[k] = v;
    }
    if (Object.keys(options).length) out.options = options;
  }
  return out;
}
