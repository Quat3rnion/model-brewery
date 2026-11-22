import { onDestroy, tick, untrack } from 'svelte';
import { bodyKey, type RetryIdentity, type RetrySubmission } from './retry-keys';
import { hashParam, route } from './route.svelte';
import { readRetry } from './rerun';

export { RetryKeys, type RetrySubmission } from './retry-keys';

export function omittedInputs(original: readonly string[], restored: readonly string[]): string[] {
  return [...new Set(original)].filter(id => !restored.includes(id));
}

export function availableId(id: string | null | undefined, choices: readonly { id: string; available?: boolean | null }[]): string {
  return id && choices.some(choice => choice.id === id && choice.available !== false) ? id : '';
}

export function historicalPrompt(
  request: { prompt_id?: string | null; prompt_text?: string | null },
  prompts: readonly { id: string; text: string }[],
) {
  const saved = prompts.find(item => item.id === request.prompt_id);
  const id = saved && (request.prompt_text == null || saved.text === request.prompt_text) ? saved.id : '';
  return { id, text: request.prompt_text ?? '', mode: id ? 'saved' as const : 'text' as const };
}

export interface RetryOmission {
  id: string;
  text: string;
  /** Queue needs a replacement, not just confirmation. */
  required?: boolean;
}

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
  actions: readonly string[];
  patch: Partial<D>;
  context?: C;
  focus?: string;
  autoFocus?: string;
  omissions?: (draft: D) => readonly RetryOmission[];
  retains?: (draft: D) => boolean;
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

interface HistoryRecord<R> {
  request: R;
}

// A browser history entry may outlive the component that first filled it.
const historyRequests = new Map<string, HistoryRecord<unknown>>();

type Origin = Omit<RetryIdentity, 'action'>;
interface Applied<D> {
  origin: Origin;
  actions: readonly string[];
  omissions?: (draft: D) => readonly RetryOmission[];
  retains?: (draft: D) => boolean;
}

/**
 * Retry invariants:
 * - Load only with GET. Prefill never submits. A delayed response never
 *   overwrites an edited draft; explicit restore focuses the restored field.
 * - Cache the historical request, not the restored patch. On every apply,
 *   including Back/Forward, validate IDs against the current state. Use the
 *   historical prompt text if the saved prompt no longer matches.
 * - Remove unavailable inputs. Show what was omitted and require a valid
 *   replacement or explicit acknowledgment before an affected action queues.
 *   Keep the warning on edits and plain routes while the draft still depends
 *   on restored inputs. One acknowledgment covers only the current omissions.
 * - During retry loading, block Queue and show the reason. Discard work from
 *   departed route entries; clean up untouched retry fields on exit.
 * - Bind each key to one launcher and one request body. Record the actual
 *   key used, including supplied retry keys, so an identical resubmit after
 *   an ambiguous response or retry query removal replays. Changing the body
 *   gets a different key. An unrelated launcher never inherits the retry.
 *   A fresh retry after confirmed acceptance is intentional new work.
 *   Keys do not survive reload; say so when a reply is ambiguous.
 */
export class RetryPrefill<D extends object, R, C = never> {
  loading = $state(false);
  get loadingReason() { return this.loading ? 'Loading the previous job options.' : ''; }
  failed = $state(false);
  context = $state.raw<C | undefined>(undefined);
  pending = $state.raw<R | undefined>(undefined);
  get offered() { return this.pending !== undefined; }
  #origin = $state.raw<Origin | undefined>(undefined);
  #applied = $state.raw<Applied<D>[]>([]);
  #acknowledged = $state.raw(new Map<Applied<D>, string>());
  #route = '';
  #abort: AbortController | undefined;
  #ready = false;
  #edited = false;
  #isRetry = false;
  #baseline: string | undefined;
  #restored: Partial<D> | undefined;
  #record: HistoryRecord<R> | undefined;
  #deferredFocus?: string;

  constructor(private form: Prefill<D, R, C>) {}

  #policy(action: string) {
    return this.#applied.find(p => p.actions.includes(action) && (p.origin === this.#origin || p.retains?.(this.form.read())));
  }

  omissions(action: string): readonly RetryOmission[] {
    return this.#policy(action)?.omissions?.(this.form.read()) ?? [];
  }

  #omissionKey(items: readonly RetryOmission[]) {
    return JSON.stringify(items.map(item => item.id).sort());
  }

  acknowledged(action: string): boolean {
    const policy = this.#policy(action);
    return !!policy && this.#acknowledged.get(policy) === this.#omissionKey(this.omissions(action));
  }

  acknowledge(action: string, accepted = true) {
    const policy = this.#policy(action);
    if (!policy) return;
    const next = new Map(this.#acknowledged);
    if (accepted) next.set(policy, this.#omissionKey(this.omissions(action)));
    else next.delete(policy);
    this.#acknowledged = next;
  }

  action(action: string): RetrySubmission {
    const policy = this.#policy(action);
    const missing = this.omissions(action);
    return {
      blocked: this.loading || (missing.length > 0 && (missing.some(item => item.required) || !this.acknowledged(action))),
      history: policy && { ...policy.origin, action },
    };
  }

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
    this.#origin = id ? { slot: this.form.slot, id, entry } : undefined;
    this.#applied = this.#applied.filter(policy => policy.retains?.(this.form.read()));
    this.#acknowledged = new Map([...this.#acknowledged].filter(([policy]) => this.#applied.includes(policy)));
    this.#baseline = bodyKey(this.form.read());
    this.#edited = (edited || wasRetry) && this.#baseline !== bodyKey(this.form.defaults);
    this.#isRetry = !!id;
    if (!id) return;
    this.loading = true;
    const abort = new AbortController();
    this.#abort = abort;
    // Each slot provides one consistent draft and request type.
    const saved = historyRequests.get(routeKey) as HistoryRecord<R> | undefined;
    if (saved) {
      this.#receive(saved);
    } else {
      void this.form.load(id, abort.signal).then(request => {
        if (abort.signal.aborted) return;
        const record = { request };
        historyRequests.set(routeKey, record);
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

  dispose() {
    this.#abort?.abort();
    this.#deferredFocus = undefined;
  }

  /** Retry an explicit focus move after a dependent control mounts. */
  resumeFocus() {
    const id = this.#deferredFocus;
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    this.#deferredFocus = undefined;
    if (document.activeElement === document.body || document.activeElement?.matches('main h1')) target.focus();
  }

  #receive(record: HistoryRecord<R>) {
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
    const origin = this.#origin;
    if (!record || !origin) return;
    let restoration: Restoration<D, C>;
    try {
      restoration = this.form.restore(record.request);
    } catch {
      this.#fail();
      return;
    }
    this.#restored = copy(restoration.patch);
    this.form.write({ ...this.form.read(), ...copy(restoration.patch) });
    const { actions, omissions, retains } = restoration;
    this.#applied = [{ origin, actions, omissions, retains },
      ...this.#applied.filter(policy => !policy.actions.some(action => actions.includes(action)))];
    this.context = restoration.context;
    this.pending = undefined;
    this.form.applied?.(restoration.patch);
    const target = explicit ? restoration.focus : restoration.autoFocus;
    if (target) {
      const abort = this.#abort;
      const focus = this.form.focus ?? ((id: string) => {
        const control = document.getElementById(id);
        control?.focus();
        if (explicit && !control) {
          this.#deferredFocus = id;
          if (document.activeElement === document.body) document.querySelector<HTMLElement>('main h1')?.focus();
        }
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
