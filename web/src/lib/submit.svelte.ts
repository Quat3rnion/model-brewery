import { api, submissionError } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';
import { RetryKeys, type RetrySubmission } from './retry-keys';

/** Tracks one form's job submission: busy flag, error text and the queued job. */
export class Submitter {
  busy = $state(false);
  error = $state<string | null>(null);
  job = $state<JobSummary | null>(null);
  replayed = $state(false);
  #keys = new RetryKeys<JobSummary>();

  async submit(job: NewJob, retry?: RetrySubmission): Promise<JobSummary | null> {
    if (this.busy || retry?.blocked) return null;
    this.busy = true;
    this.error = null;
    try {
      const body = stripEmpty(job);
      const attempt = this.#keys.take(body.kind, body, retry?.history);
      const created = attempt.result ?? await api.submitJob(body, attempt.key);
      if (!created?.id) throw new Error('The controller did not return a job ID.');
      attempt.accept(created);
      this.job = created;
      this.replayed = !!attempt.result;
      if (!attempt.result) void brewery.refresh();
      return created;
    } catch (e) {
      this.error = submissionError(e);
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
