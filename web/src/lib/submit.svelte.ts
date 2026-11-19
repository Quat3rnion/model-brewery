import { api, errorText } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';

/** Tracks one form's job submission: busy flag, error text and the queued job. */
export class Submitter {
  busy = $state(false);
  error = $state<string | null>(null);
  job = $state<JobSummary | null>(null);

  async submit(job: NewJob): Promise<JobSummary | null> {
    this.busy = true;
    this.error = null;
    try {
      const created = await api.submitJob(stripEmpty(job));
      this.job = created;
      void brewery.refresh();
      return created;
    } catch (e) {
      this.error = errorText(e);
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
