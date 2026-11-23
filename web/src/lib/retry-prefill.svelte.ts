import { onDestroy, tick, untrack } from 'svelte';
import { bodyKey, type RetrySubmission } from './retry-keys';
import { hashParam, route } from './route.svelte';
import { readRetry } from './rerun';

export type { RetrySubmission } from './retry-keys';

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
  /** Other actions can share omission checks without using the retry key. */
  retryActions?: readonly string[];
  /** Use the original run or job ID if the loader needed a prefixed selector. */
  failedId?: string;
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

interface Applied<D> {
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
 * - On a retry route, the matching action gets a deterministic key from
 *   the failed ID, launcher and wire body, even if the operator enters the
 *   inputs by hand. Bind the action before enabling Queue. The controller
 *   matches it across visits and reloads and returns the accepted record.
 *   On a plain route, a hand-rebuilt identical body is a new run by design.
 *   Plain keys can only be reused for an unconfirmed submission in its visit.
 * - Async replies may update visible results. Only the visit that started
 *   the request may navigate or move focus, and only before the user moves
 *   on. Returning to the same history entry starts a different visit.
 */
export class RetryPrefill<D extends object, R, C = never> {
  loading = $state(false);
  get loadingReason() { return this.loading ? 'Loading the previous job options.' : ''; }
  failed = $state(false);
  context = $state.raw<C | undefined>(undefined);
  pending = $state.raw<R | undefined>(undefined);
  get offered() { return this.pending !== undefined; }
  #id: string | null = null;
  #binding = $state.raw<{ failedId: string; actions: readonly string[] } | undefined>(undefined);
  #current = $state.raw<Applied<D> | undefined>(undefined);
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
  #visit?: ReturnType<typeof route.capture>;
  #deferredFocus?: { id: string; canPresent: () => boolean };

  constructor(private form: Prefill<D, R, C>) {}

  #policy(action: string) {
    return this.#applied.find(p => p.actions.includes(action) && (p === this.#current || p.retains?.(this.form.read())));
  }

  restoredAction(action: string) { return this.#current?.actions.includes(action) ?? false; }

  omissions(action: string): readonly RetryOmission[] {
    return this.#policy(action)?.omissions?.(this.form.read()) ?? [];
  }

  #omissionKey(items: readonly RetryOmission[]) {
    return JSON.stringify(items.map(item => [item.id, !!item.required] as const).sort());
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

  reconcileAcknowledgments() {
    const next = new Map(this.#acknowledged);
    const draft = this.form.read();
    for (const [policy, key] of this.#acknowledged) {
      if (key !== this.#omissionKey(policy.omissions?.(draft) ?? [])) next.delete(policy);
    }
    if (next.size !== this.#acknowledged.size) this.#acknowledged = next;
  }

  action(action: string): RetrySubmission {
    const missing = this.omissions(action);
    return {
      blocked: this.loading || (missing.length > 0 && (missing.some(item => item.required) || !this.acknowledged(action))),
      failedId: this.#binding?.actions.includes(action) ? this.#binding.failedId : undefined,
    };
  }

  enter(entry: string, id: string | null, visit?: ReturnType<typeof route.capture>) {
    const routeKey = JSON.stringify([this.form.slot, entry, id]);
    if (routeKey === this.#route) return;
    const edited = this.#edited;
    const wasRetry = this.#isRetry;
    this.dispose();
    this.#visit = visit;
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
    this.#id = id;
    this.#binding = undefined;
    this.#current = undefined;
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
        if (abort.signal.aborted || visit?.current() === false) return;
        const record = { request };
        historyRequests.set(routeKey, record);
        this.#receive(record);
      }).catch(() => { if (!abort.signal.aborted && visit?.current() !== false) this.#fail(); });
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
    const deferred = this.#deferredFocus;
    if (!deferred) return;
    const { id, canPresent } = deferred;
    if (!canPresent()) {
      this.#deferredFocus = undefined;
      return;
    }
    const target = document.getElementById(id);
    if (!target) return;
    this.#deferredFocus = undefined;
    if (document.activeElement === document.body || document.activeElement?.matches('main h1')) {
      target.focus();
      if (document.activeElement !== target) document.querySelector<HTMLElement>('main h1')?.focus();
    }
  }

  #receive(record: HistoryRecord<R>) {
    this.#record = record;
    this.#settle();
  }

  #settle() {
    if (!this.#ready || !this.#record || !this.loading || this.#visit?.current() === false) return;
    const { request } = this.#record;
    const restoration = this.#restore(request);
    if (!restoration) return;
    this.#binding = { failedId: restoration.failedId ?? this.#id!, actions: restoration.retryActions ?? restoration.actions };
    if (this.#edited || bodyKey(this.form.read()) !== this.#baseline) this.pending = request;
    else this.#apply(false, restoration);
    this.loading = false;
  }

  #restore(request: R): Restoration<D, C> | undefined {
    try {
      return this.form.restore(request);
    } catch {
      this.#fail();
    }
  }

  #apply(explicit: boolean, initial?: Restoration<D, C>) {
    const record = this.#record;
    if (!record || !this.#id || this.#visit?.current() === false) return;
    const restoration = initial ?? this.#restore(record.request);
    if (!restoration) return;
    this.#restored = copy(restoration.patch);
    this.form.write({ ...this.form.read(), ...copy(restoration.patch) });
    const { actions, omissions, retains } = restoration;
    const applied = { actions, omissions, retains };
    this.#current = applied;
    this.#applied = [applied,
      ...this.#applied.filter(policy => !policy.actions.some(action => actions.includes(action)))];
    this.#acknowledged = new Map([...this.#acknowledged].filter(([policy]) => this.#applied.includes(policy)));
    this.context = restoration.context;
    this.pending = undefined;
    this.form.applied?.(restoration.patch);
    const target = explicit ? restoration.focus : restoration.autoFocus;
    if (target) {
      const abort = this.#abort;
      const visit = explicit && this.#visit ? route.capture() : this.#visit;
      const canPresent = () => !abort?.signal.aborted && visit?.canPresent() !== false;
      const focus = this.form.focus ?? ((id: string) => {
        const control = document.getElementById(id);
        control?.focus();
        if (explicit && (!control || document.activeElement !== control)) {
          if (!control) this.#deferredFocus = { id, canPresent };
          document.querySelector<HTMLElement>('main h1')?.focus();
        }
      });
      void tick().then(() => { if (canPresent()) focus(target); });
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
    untrack(() => retry.enter(entry, requestId, route.capture()));
  });
  $effect(() => {
    const value = ready();
    untrack(() => retry.setReady(value));
  });
  $effect(() => retry.reconcileAcknowledgments());
  onDestroy(() => retry.dispose());
}
