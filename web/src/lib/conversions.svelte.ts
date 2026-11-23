// Reads the durable conversion workflows. Polls GET only, and only while the
// Convert page shows them; it never POSTs on its own. The list survives
// reloads because it lives in the controller.

import { api, errorText } from './api';
import { ConversionSubmitter, isActiveConversion, newerConversion, parseConversions, type Conversion, type ConversionBody } from './conversion';
import type { RetrySubmission } from './retry-prefill.svelte';
import { captureSubmissionVisit } from './route.svelte';
import { bodyKey, openOnce, type OpenPlain } from './retry-keys';

const ACTIVE_MS = 2_000;
const IDLE_MS = 10_000;

class Conversions {
  list = $state<Conversion[]>([]);
  loaded = $state(false);
  /** 'unavailable' when the controller has no conversion API (404/405). */
  status = $state<'ok' | 'error' | 'unavailable'>('ok');
  error = $state<string | null>(null);
  #timer: ReturnType<typeof setTimeout> | null = null;
  #users = 0;
  #inflight: Promise<void> | null = null;

  /** Start polling while a view is mounted; returns the stop function. */
  watch(): () => void {
    this.#users += 1;
    if (this.#users === 1) void this.refresh();
    return () => {
      this.#users -= 1;
      if (this.#users === 0 && this.#timer) {
        clearTimeout(this.#timer);
        this.#timer = null;
      }
    };
  }

  refresh(): Promise<void> {
    this.#inflight ??= this.#load().finally(() => { this.#inflight = null; });
    return this.#inflight;
  }

  async #load() {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
    try {
      this.list = parseConversions(await api.conversions()).sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
      this.status = 'ok';
      this.error = null;
    } catch (e) {
      const st = (e as { status?: number }).status;
      this.status = st === 404 || st === 405 ? 'unavailable' : 'error';
      this.error = errorText(e);
    }
    this.loaded = true;
    if (this.#users > 0) {
      const delay = this.status === 'unavailable' ? 60_000 : this.list.some(isActiveConversion) ? ACTIVE_MS : IDLE_MS;
      this.#timer = setTimeout(() => void this.refresh(), delay);
    }
  }

  /** Merge a record returned by POST or cancel without waiting for the next poll. */
  upsert(c: Conversion) {
    const i = this.list.findIndex((x) => x.id === c.id);
    this.list = i >= 0 ? this.list.map((x, j) => (j === i ? newerConversion(x, c) : x)) : [c, ...this.list];
  }
}

export const conversions = new Conversions();

/** Svelte-reactive wrapper of the single-submission logic. */
export class ConversionStarter {
  busy = $state(false);
  error = $state<string | null>(null);
  last = $state<Conversion | null>(null);
  /** True when the last click replayed an earlier identical request instead of starting a new workflow. */
  replayed = $state(false);
  acceptedBodyKey = $state<string>();
  newRun = $state.raw<OpenPlain>();
  #inner: ConversionSubmitter;

  constructor(launcher: string) { this.#inner = new ConversionSubmitter(launcher, api.startConversion); }

  async start(body: ConversionBody, retry?: RetrySubmission): Promise<Conversion | null> {
    if (this.busy || retry?.blocked) return null;
    this.busy = true;
    this.error = null;
    const submitted = bodyKey(body);
    const prepare = retry?.preparePlain?.(body);
    const c = await this.#inner.submit(body, retry, captureSubmissionVisit());
    this.busy = false;
    this.error = this.#inner.error;
    if (c) {
      this.replayed = this.#inner.replayed;
      this.last = c;
      this.acceptedBodyKey = submitted;
      this.newRun = this.replayed && prepare ? openOnce(prepare, () => { this.newRun = undefined; }) : undefined;
      conversions.upsert(c);
      void conversions.refresh();
    }
    return c;
  }
}
