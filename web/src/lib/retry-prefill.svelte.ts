import { onDestroy, tick, untrack } from 'svelte';
import { newKey } from './conversion';
import { hashParam, route } from './route.svelte';
import { readRetry } from './rerun';

export function omittedInputs(original: readonly string[], restored: readonly string[]): string[] {
  return [...new Set(original)].filter(id => !restored.includes(id));
}

// Compare submitted values, not object insertion order or form-only state.
function bodyKey(body: unknown): string | undefined {
  return JSON.stringify(body, (_, value: unknown) => value && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value);
}

type Attempt = { key: string; accepted: boolean };
export interface RetrySubmission {
  submission(action: string, body: unknown): { key: string; accept: () => void };
}

/** One tab's retry attempts. A history entry pins its keys; a new entry joins
 * an unresolved attempt for the same request, but not a confirmed submission. */
export class RetryKeys {
  #entries = new Map<string, Map<string, Attempt>>();
  #unconfirmed = new Map<string, Attempt>();

  constructor(private mint = newKey) {}

  entry(slot: string, id: string, entry: string): RetrySubmission {
    const entryKey = JSON.stringify([slot, id, entry]);
    const attempts = this.#entries.get(entryKey) ?? new Map<string, Attempt>();
    this.#entries.set(entryKey, attempts);
    return {
      submission: (action, body) => {
        const request = JSON.stringify([slot, id, action, bodyKey(body)]);
        const attempt = attempts.get(request) ?? this.#unconfirmed.get(request) ?? { key: this.mint(), accepted: false };
        attempts.set(request, attempt);
        if (!attempt.accepted) this.#unconfirmed.set(request, attempt);
        return { key: attempt.key, accept: () => {
          attempt.accepted = true;
          if (this.#unconfirmed.get(request) === attempt) this.#unconfirmed.delete(request);
        } };
      },
    };
  }
}

const keys = new RetryKeys();

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

// Drafts are plain data, possibly Svelte proxies; preserve undefined values.
function copy<D>(value: D): D {
  if (Array.isArray(value)) return value.map(copy) as D;
  if (isRecord(value)) return Object.fromEntries(Object.entries(value).map(([key, field]) => [key, copy(field)])) as D;
  return value;
}

function resetRestored<D>(value: D, restored: unknown, defaults: unknown): D {
  if (isRecord(value) && isRecord(restored) && isRecord(defaults)) {
    return Object.fromEntries(Object.entries(value).flatMap(([k, v]) => {
      if (!Object.hasOwn(restored, k)) return [[k, v]];
      const reset = resetRestored(v, restored[k], defaults[k]);
      return reset === undefined && !Object.hasOwn(defaults, k) ? [] : [[k, reset]];
    })) as D;
  }
  return bodyKey(value) === bodyKey(restored) ? copy(defaults) as D : value;
}

interface Restoration<D, C> {
  patch: Partial<D>;
  context?: C;
  focus?: string;
  autoFocus?: string;
}

interface Prefill<D, R, C> {
  slot: string;
  defaults: D;
  read: () => D;
  write: (draft: D) => void;
  load: (id: string, signal: AbortSignal) => Promise<R>;
  restore: (request: R) => Restoration<D, C>;
  applied?: (patch: Partial<D>) => void;
  focus?: (id: string) => void;
  clear?: (patch: Partial<D>) => void;
}

interface HistoryRecord<D, R, C> {
  request: R;
  restoration?: Restoration<D, C>;
}

// A browser history entry may outlive the component that first filled it.
const historyDrafts = new Map<string, HistoryRecord<object, unknown, unknown>>();

/** GET -> offer/apply -> exit. This class never queues jobs. Views supply
 * only plain form data and a conversion from the prior request to a patch. */
export class RetryPrefill<D extends object, R, C = never> {
  loading = $state(false);
  failed = $state(false);
  context = $state.raw<C | undefined>(undefined);
  pending = $state.raw<R | undefined>(undefined);
  get offered() { return this.pending !== undefined; }
  submission = $state.raw<RetrySubmission | undefined>(undefined);
  #route = '';
  #abort: AbortController | undefined;
  #ready = false;
  #edited = false;
  #isRetry = false;
  #baseline: string | undefined;
  #restored: Partial<D> | undefined;
  #record: HistoryRecord<D, R, C> | undefined;

  constructor(private form: Prefill<D, R, C>) {}

  enter(entry: string, id: string | null) {
    const routeKey = JSON.stringify([this.form.slot, entry, id]);
    if (routeKey === this.#route) return;
    const edited = this.#edited;
    const wasRetry = this.#isRetry;
    this.dispose();
    this.#route = routeKey;
    this.loading = false;
    this.failed = false;
    this.pending = undefined;
    this.#record = undefined;
    if (this.#restored) {
      const plain = resetRestored(this.form.read(), this.#restored, this.form.defaults);
      this.form.write(plain);
      this.form.clear?.(this.#restored);
    }
    this.#restored = undefined;
    this.context = undefined;
    this.submission = id ? keys.entry(this.form.slot, id, entry) : undefined;
    this.#baseline = bodyKey(this.form.read());
    this.#edited = (edited || wasRetry) && this.#baseline !== bodyKey(this.form.defaults);
    this.#isRetry = !!id;
    if (!id) return;
    this.loading = true;
    const abort = new AbortController();
    this.#abort = abort;
    // Each slot provides one consistent draft and request type.
    const saved = historyDrafts.get(routeKey) as HistoryRecord<D, R, C> | undefined;
    if (saved) {
      this.#receive(saved);
    } else {
      void this.form.load(id, abort.signal).then(request => {
        if (abort.signal.aborted) return;
        const record = { request };
        historyDrafts.set(routeKey, record);
        this.#receive(record);
      }).catch(() => { if (!abort.signal.aborted) this.#fail(); });
    }
  }

  setReady(ready: boolean) {
    this.#ready = ready;
    this.#settle();
  }

  touch() { this.#edited = true; }

  /** A confirmed plain submission is no longer an unsent draft. */
  markClean() { this.#edited = false; }

  /** An inventory update may prune a checklist; do not mistake that for a user edit on exit. */
  prune<K extends keyof D>(field: K, filter: (value: D[K]) => D[K]) {
    const value = this.#restored?.[field];
    if (value !== undefined && this.#restored) this.#restored[field] = filter(value);
  }

  // Capture real field changes, including the combobox option buttons.
  touchField = (event: Event) => {
    if (event.type === 'click' && !(event.target instanceof Element && event.target.closest('[role="option"], [role="tab"]'))) return;
    if (event instanceof KeyboardEvent && !(event.key === 'Enter' && event.target instanceof Element && event.target.closest('[role="combobox"]'))) return;
    this.touch();
  };

  usePrevious = () => {
    if (this.pending !== undefined) this.#apply(true);
  };

  dispose() { this.#abort?.abort(); }

  #receive(record: HistoryRecord<D, R, C>) {
    this.#record = record;
    this.#settle();
  }

  #settle() {
    if (!this.#ready || !this.#record || !this.loading) return;
    const { request } = this.#record;
    this.loading = false;
    if (this.#edited || bodyKey(this.form.read()) !== this.#baseline) this.pending = request;
    else this.#apply(false);
  }

  #apply(explicit: boolean) {
    const record = this.#record;
    if (!record) return;
    let restoration: Restoration<D, C>;
    try {
      // History rehydrates the same patch, not a new restore against a
      // refreshed offer that could silently add a previously omitted input.
      restoration = record.restoration ??= this.form.restore(record.request);
    } catch {
      this.#fail();
      return;
    }
    this.#restored = copy(restoration.patch);
    this.form.write({ ...this.form.read(), ...copy(restoration.patch) });
    this.context = restoration.context;
    this.pending = undefined;
    this.form.applied?.(restoration.patch);
    const target = explicit ? restoration.focus : restoration.autoFocus;
    if (target) {
      const abort = this.#abort;
      const focus = this.form.focus ?? ((id: string) => {
        document.getElementById(id)?.focus();
        if (explicit && document.activeElement === document.body) document.querySelector<HTMLElement>('main h1')?.focus();
      });
      void tick().then(() => { if (!abort?.signal.aborted) focus(target); });
    }
  }

  #fail() {
    this.loading = false;
    this.pending = undefined;
    this.failed = true;
  }
}

export const loadRetryJob = (kinds: readonly string[]) => (id: string, signal: AbortSignal) => readRetry(id, kinds, signal);

/** Bind a prefill to the hash router; inventory-dependent restores wait until
 * the first snapshot, not until every subsequent poll. */
export function watchPrefill<D extends object, R, C>(
  retry: RetryPrefill<D, R, C>,
  ready: () => boolean = () => true,
  id: (hash: string) => string | null = hash => hashParam(hash, 'retry'),
) {
  $effect.pre(() => {
    const entry = route.entry;
    const requestId = id(route.hash);
    untrack(() => retry.enter(entry, requestId));
  });
  $effect(() => {
    const value = ready();
    untrack(() => retry.setReady(value));
  });
  onDestroy(() => retry.dispose());
}
