import { SvelteSet } from 'svelte/reactivity';
import { api, rejectedSubmission, submissionError, type JobReceipt } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';
import { PlainAttempt, requestKey } from './submission-key';
import { route } from './route.svelte';
import { captureResultFocus } from './focus';
import { isTerminal } from './format';

type Click = Pick<MouseEvent, 'detail'> & Partial<Pick<PointerEvent, 'pointerType' | 'button' | 'clientX' | 'clientY'>>;
type ClickStep = { detail: number; previousLauncher?: string; launcher?: string };
type Tap = { x: number; y: number };
const pointerClicks = new Map<string, ClickStep>();
const clickSteps = new WeakMap<Click, ClickStep>();
const disabledPresses = new Map<string, { ended: boolean }>();
const clickStream = (event: Click) => `${event.pointerType || 'mouse'}:${event.button ?? 0}`;

function clickStep(event: Click): ClickStep | undefined {
  if (event.detail <= 0) return;
  const seen = clickSteps.get(event);
  if (seen) return seen;
  const stream = clickStream(event);
  disabledPresses.delete(stream);
  const prior = pointerClicks.get(stream);
  const step = {
    detail: event.detail,
    previousLauncher: event.detail > 1 && prior?.detail === event.detail - 1 ? prior.launcher : undefined,
  };
  pointerClicks.set(stream, step);
  clickSteps.set(event, step);
  return step;
}

if (typeof window !== 'undefined') {
  // If a disabled press ends without a click, reset before the next press.
  // Blink can still click a button that was disabled on down but enabled on up.
  window.addEventListener('click', clickStep, { capture: true });
  window.addEventListener('pointerdown', event => {
    const stream = clickStream(event);
    if (disabledPresses.get(stream)?.ended) pointerClicks.delete(stream);
    disabledPresses.delete(stream);
    if (event.target instanceof Element && event.target.closest(':disabled')) disabledPresses.set(stream, { ended: false });
  }, { capture: true });
  window.addEventListener('pointerup', event => {
    const press = disabledPresses.get(clickStream(event));
    if (press) press.ended = true;
  }, { capture: true });
  window.addEventListener('pointercancel', event => {
    for (const stream of disabledPresses.keys()) {
      if (!stream.startsWith(`${event.pointerType || 'mouse'}:`)) continue;
      disabledPresses.delete(stream);
      pointerClicks.delete(stream);
    }
  }, { capture: true });
}

/** Do not follow a read result link that appears under a repeat submit click. */
export function guardReadLink(event: MouseEvent, launcher: string): void {
  const step = clickStep(event);
  if (step?.previousLauncher === launcher || isRepeatTap(launcher, tapPoint(event), performance.now())) {
    if (step) step.launcher = launcher;
    event.preventDefault();
  }
}

const DOUBLE_CLICK_MS = 800;
const TAP_SLOP_PX = 24;
let lastTap: (Tap & { launcher: string; time: number; current: () => boolean }) | undefined;

function tapPoint(event?: Click): Tap | undefined {
  if (event?.pointerType === 'touch' && event.detail > 0 && event.clientX !== undefined && event.clientY !== undefined) {
    return { x: event.clientX, y: event.clientY };
  }
}

function isRepeatTap(launcher: string, tap: Tap | undefined, now: number): boolean {
  return !!(tap && lastTap?.launcher === launcher && lastTap.current()
    && now - lastTap.time < DOUBLE_CLICK_MS
    && Math.hypot(tap.x - lastTap.x, tap.y - lastTap.y) <= TAP_SLOP_PX);
}

// DefaultBodyLimit for both queue endpoints in crates/controller/src/web.rs.
const MAX_BODY_BYTES = 256 * 1024;
const encoder = new TextEncoder();
const recent = new Map<string, { sent: number; accepted?: number }>();
let recentVisit: (() => boolean) | undefined;

/**
 * Submission rules:
 * 1. Suppress a native repeat click (detail > 1) if the previous click reached
 *    an enabled submit of the same launcher, even if the body changed. Touch
 *    contacts in one sequence can have different pointer IDs. A disabled press
 *    only clears the sequence if it ends without dispatching a click.
 * 2. Touch browsers can reset the click count before a replacement control is
 *    ready. Also guard a nearby tap on the same launcher within 800 ms of send.
 * 3. Also suppress the same launcher and wire body within 800 ms of send or
 *    acceptance, a conservative bound for the platform double-click window.
 *    A keyboard activation (detail = 0) only uses the acceptance window.
 * K (keys). Each explicit submission gets a random key. Resending the same
 *    request after an unknown outcome reuses its key in this visit. Only a
 *    4xx with Brewery-Error: 1, application/json, and exactly {error: string}
 *    establishes a controller rejection. The controller matches active and
 *    recent work across keys. Only explicit repeat controls can opt out.
 * R (reads). A failed read or a suppressed press leaves existing results
 *    intact and does not block a new read. Only an admitted send may change
 *    the results. Show a terminal failure as an error with the read action
 *    still available, and do not carry that error across source switches.
 */
export class Submission {
  busy = $state(false);
  error = $state<string | null>(null);
  ambiguous = $state(false);

  constructor(private attempts = new PlainAttempt()) {}

  async send<Body extends object, Result>(
    launcher: string,
    body: Body,
    post: (body: Body, key: string) => Promise<Result>,
    event?: Click,
    onAttempt?: (admitted: boolean) => void,
    allowDuplicate = false,
  ): Promise<Result | null> {
    const step = event && clickStep(event);
    if (step?.previousLauncher === launcher) {
      step.launcher = launcher;
      return null;
    }
    if (this.busy) return null;

    const current = route.capture().current;
    if (!recentVisit?.()) {
      recent.clear();
      recentVisit = current;
    }
    const json = JSON.stringify(body);
    if (json.length > MAX_BODY_BYTES || encoder.encode(json).length > MAX_BODY_BYTES) {
      this.ambiguous = false;
      this.error = 'The request is too large. The controller limit is 256 KiB.';
      onAttempt?.(false);
      return null;
    }
    const now = performance.now();
    const tap = tapPoint(event);
    if (isRepeatTap(launcher, tap, now)) {
      if (step) step.launcher = launcher;
      return null;
    }
    for (const [key, value] of recent) {
      if (now - Math.max(value.sent, value.accepted ?? -Infinity) >= DOUBLE_CLICK_MS) recent.delete(key);
    }
    const request = requestKey(allowDuplicate ? `${launcher}:repeat` : launcher, body);
    const last = recent.get(request);
    const since = event?.detail === 0 ? last?.accepted : Math.max(last?.sent ?? -Infinity, last?.accepted ?? -Infinity);
    if (step) step.launcher = launcher;
    if (event && since !== undefined && now - since < DOUBLE_CLICK_MS) return null;

    const plain = this.attempts.takeRequest(request, current);
    recent.set(request, { sent: now, accepted: last?.accepted });
    if (tap) lastTap = { ...tap, launcher, time: now, current };
    this.busy = true;
    this.error = null;
    this.ambiguous = false;
    try {
      onAttempt?.(true);
      const result = await post(body, plain.key);
      plain.complete();
      if (current()) {
        // A different submitter can send the same body before this reply arrives.
        recent.set(request, { sent: recent.get(request)?.sent ?? now, accepted: performance.now() });
      }
      return result;
    } catch (e) {
      this.ambiguous = !rejectedSubmission(e);
      if (!this.ambiguous) plain.complete();
      this.error = submissionError(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

/** Tracks the result of queuing a job. */
export class Submitter extends Submission {
  job = $state<JobReceipt | null>(null);
  replayed = $state(false);
  statusElement?: HTMLElement;
  repeat = $state<((event: MouseEvent) => void) | null>(null);

  constructor(private launcher: string | ((job: NewJob) => string), attempts?: PlainAttempt) { super(attempts); }

  #launcherId(body: NewJob) {
    return typeof this.launcher === 'string' ? this.launcher : this.launcher(body);
  }

  async submit(job: NewJob, event: MouseEvent, focusStatus = true,
    onAttempt?: (admitted: boolean) => void,
    onAccepted?: (job: JobReceipt, focus: ReturnType<typeof captureResultFocus>) => void,
    allowDuplicate = false): Promise<JobReceipt | null> {
    const focus = captureResultFocus(event);
    const trigger = event.currentTarget;
    // Use the same wire body for the key comparison and the POST.
    const body: NewJob = JSON.parse(JSON.stringify(stripEmpty(job)));
    const launcher = this.#launcherId(body);
    let attempted = false;
    const sent = this.send(launcher, body, async (snapshot, key) => {
      const created = await api.queueJob(snapshot, key, allowDuplicate);
      if (typeof created?.id !== 'string' || !created.id) throw new Error('The controller did not return a job ID.');
      return created;
    }, event, admitted => {
      if (!allowDuplicate) this.repeat = null;
      attempted = true;
      onAttempt?.(admitted);
    }, allowDuplicate);
    const created = await sent;
    if (created) {
      this.job = created;
      this.replayed = !('state' in created);
      this.repeat = 'duplicate_of' in created && created.duplicate_of === true
        ? event => { void this.submit(body, event, focusStatus, onAttempt, onAccepted, true); }
        : null;
      onAccepted?.(created, focus);
      if (focusStatus) void focus?.(() => this.statusElement?.querySelector('a'));
      void brewery.refresh();
    } else if (attempted && this.error && trigger instanceof HTMLButtonElement) {
      void focus?.(() => {
        const alert = this.statusElement?.querySelector<HTMLElement>('[role="alert"]');
        if (event.detail === 0) alert?.scrollIntoView({ block: 'center' });
        if (trigger.isConnected && !trigger.disabled) {
          // Keep the retry control focused without scrolling the alert away.
          trigger.focus({ preventScroll: true });
          return;
        }
        return alert;
      });
    }
    return created;
  }
}

/** Compare job reads using the same omitted fields as the POST. */
export function jobRequestKey(launcher: string, job: NewJob): string {
  return requestKey(launcher, stripEmpty(job));
}

/** Remove empty optional values so the controller applies its own defaults. */
export function stripEmpty(job: NewJob): NewJob {
  const checklist = job.kind === 'transfer' || job.kind === 'transfer_preview';
  const out: NewJob = { kind: job.kind, input_ids: checklist ? [...new Set(job.input_ids)].sort() : job.input_ids };
  if (job.host) out.host = job.host;
  if (job.prompt_id) out.prompt_id = job.prompt_id;
  else if (job.prompt) out.prompt = job.prompt;
  if (job.options) {
    const options: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(job.options)) {
      if (v === undefined || v === null || v === '' || (typeof v === 'number' && Number.isNaN(v))) continue;
      options[k] = v;
    }
    const coords = options.indices;
    if (job.kind === 'weight_tune' && Array.isArray(coords) && coords.every(pair =>
      Array.isArray(pair) && pair.length === 2 && pair.every(Number.isSafeInteger))) {
      // Coordinate order is not a control in the editor. Preserve each tuple's axes.
      options.indices = [...coords].sort(([ar, ac], [br, bc]) => ar - br || ac - bc);
    }
    if (Object.keys(options).length) out.options = options;
  }
  return out;
}

/** Reserve a matched read's slot until a detail or state snapshot supplies its state. */
export function acceptedJobSummary(accepted: JobReceipt, request: NewJob): JobSummary {
  return 'state' in accepted ? accepted : {
    id: accepted.id, state: 'queued', kind: request.kind, host: request.host,
    input_ids: request.input_ids, options: request.options,
  };
}

/** Reuse active and successful reads across sources, but keep errors local. */
export class ReadVisit {
  #started = new SvelteSet<string>();

  remember(id: string) { this.#started.add(id); }

  observe(job?: Pick<JobSummary, 'id' | 'state'> | null) {
    if (job && !isTerminal(job.state)) this.remember(job.id);
  }

  shows(job: Pick<JobSummary, 'id' | 'state'>): boolean {
    return job.state === 'succeeded' || !isTerminal(job.state) || this.#started.has(job.id);
  }
}
