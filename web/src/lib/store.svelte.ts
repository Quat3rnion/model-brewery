import { api, errorText } from './api';
import type { BrewState, JobDetail } from './types';
import { isTerminal } from './format';

export type Connection = 'connecting' | 'live' | 'retrying';

const FAST_MS = 1500;
const MAX_BACKOFF_MS = 15000;

/**
 * One shared view of the controller. The controller owns all state; the
 * browser polls GET /api/state, so a reload or a lost connection only
 * needs the next successful poll to show the current queue again.
 */
class Brewery {
  state = $state<BrewState | null>(null);
  connection = $state<Connection>('connecting');
  lastError = $state<string | null>(null);
  lastUpdated = $state<Date | null>(null);
  details = $state<Record<string, JobDetail>>({});

  #timer: ReturnType<typeof setTimeout> | null = null;
  #failures = 0;
  #inflight: AbortController | null = null;
  #started = false;

  start() {
    if (this.#started) return;
    this.#started = true;
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') this.refresh();
    });
    window.addEventListener('online', () => this.refresh());
    this.refresh();
  }

  async refresh() {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    this.#inflight?.abort();
    const controller = new AbortController();
    this.#inflight = controller;
    try {
      const next = await api.state(controller.signal);
      if (controller.signal.aborted) return;
      this.state = normalize(next);
      this.connection = 'live';
      this.lastError = null;
      this.lastUpdated = new Date();
      this.#failures = 0;
      this.#dropStaleDetails();
    } catch (error) {
      if (controller.signal.aborted) return;
      this.#failures += 1;
      this.connection = this.state ? 'retrying' : 'connecting';
      this.lastError = errorText(error);
    } finally {
      if (this.#inflight === controller) this.#inflight = null;
    }
    const delay = this.#failures === 0 ? FAST_MS : Math.min(MAX_BACKOFF_MS, FAST_MS * 2 ** this.#failures);
    const hidden = document.visibilityState === 'hidden';
    this.#timer = setTimeout(() => this.refresh(), hidden ? Math.max(delay, 10000) : delay);
  }

  /** Full job record with result and event log. Terminal results are immutable, so they are cached. */
  async loadJob(id: string, force = false): Promise<JobDetail> {
    const cached = this.details[id];
    if (cached && !force && isTerminal(cached.state)) return cached;
    const detail = await api.job(id);
    this.details[id] = detail;
    return detail;
  }

  #dropStaleDetails() {
    // A cached detail of a job that was not terminal must be refreshed by its view.
    const jobs = this.state?.jobs ?? [];
    for (const job of jobs) {
      const cached = this.details[job.id];
      if (cached && cached.state !== job.state) delete this.details[job.id];
    }
  }
}

function normalize(s: BrewState): BrewState {
  return {
    ...s,
    hosts: Array.isArray(s?.hosts) ? s.hosts : [],
    models: Array.isArray(s?.models) ? s.models : [],
    prompts: Array.isArray(s?.prompts) ? s.prompts : [],
    jobs: Array.isArray(s?.jobs) ? s.jobs.map((j) => ({ ...j, input_ids: Array.isArray(j.input_ids) ? j.input_ids : [] })) : [],
  };
}

export const brewery = new Brewery();
