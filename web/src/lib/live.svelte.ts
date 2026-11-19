// Poll the optional legacy live observation file. The controller dist may
// serve /live-progress/current.json; when it does not (404), nothing changes.
// Polling runs only while some job is running without controller
// progress_detail, and stops otherwise. A failed poll keeps the last sample;
// lib/progress.ts hides it once it is stale.
//
// The same loop feeds the shared rate tracker (lib/rate.ts) from the job list
// and the live sample. It makes no extra requests.

import { untrack } from 'svelte';
import { brewery } from './store.svelte';
import { feedRates, LIVE_POLL_IDLE_MS, LIVE_POLL_MS, LIVE_URL, parseLive, pollPlan, type LiveSample, type ProgressDetail } from './progress';
import { RateTracker, type RateSource, type RateView } from './rate';
import { StepTracker, type Step } from './steps';

/** Age labels update at this rate while progress is shown, without network requests. */
const TICK_MS = 1_000;

const MAX_BYTES = 64 * 1024;

class Live {
  sample = $state<LiveSample | null>(null);
  /** Clock for freshness checks, updated on every poll tick. */
  now = $state(Date.now());
  /** True after the file answered 404 at least once: the deployment has no live file. */
  absent = $state(false);
  /** Bumped when the rate tracker changes; read by rate() so views update. */
  #rateVersion = $state(0);
  #rates = new RateTracker();
  #steps = new StepTracker();
  #timer: ReturnType<typeof setTimeout> | null = null;
  #clock: ReturnType<typeof setInterval> | null = null;
  #started = false;

  start() {
    if (this.#started) return;
    this.#started = true;
    $effect.root(() => {
      $effect(() => {
        // Re-evaluate when the job list changes.
        const plan = this.#plan();
        // One shared tracker, fed once per job list: every view of a job shows the same rate.
        this.#feed();
        if (plan.tick && !this.#clock) this.#clock = setInterval(() => this.#onClock(), TICK_MS);
        if (plan.fetch && !this.#timer) this.#schedule(0);
      });
    });
  }

  #plan() {
    return pollPlan(brewery.state?.jobs ?? []);
  }

  #feed() {
    const jobs = brewery.state?.jobs ?? [];
    const sample = untrack(() => this.sample);
    if (feedRates(this.#rates, jobs, sample, Date.now(), this.#steps)) untrack(() => (this.#rateVersion += 1));
  }

  /** Phases this page saw for one job, oldest first, and how many were dropped before them. */
  steps(jobId: string): { history: readonly Step[]; dropped: number } {
    void this.#rateVersion;
    return { history: this.#steps.history(jobId), dropped: this.#steps.dropped(jobId) };
  }

  /** Measured rate of the counter `detail` of this job shows, from the shared tracker. */
  rate(jobId: string, source: RateSource, detail: ProgressDetail): RateView | null {
    void this.#rateVersion;
    return this.#rates.view(jobId, source, detail, this.now);
  }

  #onClock() {
    this.now = Date.now();
    if (!this.#plan().tick && this.#clock) {
      clearInterval(this.#clock);
      this.#clock = null;
    }
  }

  #schedule(ms: number) {
    this.#timer = setTimeout(() => void this.#tick(), ms);
  }

  async #tick() {
    this.#timer = null;
    this.now = Date.now();
    if (!this.#plan().fetch) return; // the effect restarts polling when needed again
    try {
      const res = await fetch(LIVE_URL, { cache: 'no-store', credentials: 'same-origin', headers: { Accept: 'application/json' } });
      if (res.status === 404) this.absent = true;
      else if (res.ok) {
        this.absent = false;
        const len = Number(res.headers.get('content-length') ?? '0');
        if (len <= MAX_BYTES) {
          const text = await res.text();
          if (text.length <= MAX_BYTES) {
            const parsed = parseLive(JSON.parse(text));
            if (parsed) {
              this.sample = parsed;
              this.#feed();
            }
          }
        }
      }
    } catch {
      /* network or JSON error: keep the last sample */
    }
    this.now = Date.now();
    const hidden = typeof document !== 'undefined' && document.visibilityState === 'hidden';
    this.#schedule(this.absent || hidden ? LIVE_POLL_IDLE_MS : LIVE_POLL_MS);
  }
}

export const live = new Live();
