// Reads the durable conversion workflows. Polls GET only, and only while the
// Convert page shows them; it never POSTs on its own. The list survives
// reloads because it lives in the controller.

import { api, errorText } from './api';
import { isActiveConversion, newerConversion, parseConversion, parseConversions, type Conversion, type ConversionBody } from './conversion';
import { Submission } from './submit.svelte';
import { captureResultFocus } from './focus';

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

type PostConversion = (body: ConversionBody, key: string) => Promise<{ data: unknown; status: number }>;

export class ConversionSubmitter extends Submission {
  replayed = $state(false);

  constructor(private launcher: string, private post: PostConversion) { super(); }

  async submit(body: ConversionBody, event?: MouseEvent): Promise<Conversion | null> {
    const snapshot: ConversionBody = JSON.parse(JSON.stringify(body));
    const result = await this.send(this.launcher, snapshot, async (request, key) => {
      const { data, status } = await this.post(request, key);
      const c = parseConversion(data);
      if (!c?.id) throw new Error('The controller answered without a conversion record.');
      return { c, status };
    }, event);
    if (!result) return null;
    this.replayed = result.status === 200;
    return result.c;
  }
}

/** Present the accepted workflow and update the shared run list. */
export class ConversionStarter extends ConversionSubmitter {
  last = $state<Conversion | null>(null);

  constructor(launcher: string) { super(launcher, api.startConversion); }

  async start(body: ConversionBody, event: MouseEvent, result: () => HTMLElement | undefined): Promise<Conversion | null> {
    const focus = captureResultFocus(event);
    const trigger = event.currentTarget;
    const c = await this.submit(body, event);
    if (c) {
      this.last = c;
      void focus?.(result);
      conversions.upsert(c);
      void conversions.refresh();
    } else if (this.error && trigger instanceof HTMLElement) {
      void focus?.(() => trigger);
    }
    return c;
  }
}
