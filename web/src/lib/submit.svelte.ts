import { api, submissionError } from './api';
import { brewery } from './store.svelte';
import type { JobSummary, NewJob } from './types';
import { PlainAttempt, requestKey } from './submission-key';
import { route } from './route.svelte';
import { captureResultFocus } from './focus';

type AcceptedJob = JobSummary | Pick<JobSummary, 'id'>;
type Click = Pick<MouseEvent, 'detail'> & Partial<Pick<PointerEvent, 'pointerId' | 'pointerType' | 'button'>>;
type ClickStep = { detail: number; pointer: number; repeat: boolean; consumed: boolean };
const pointerClicks = new Map<string, ClickStep>();
const clickSteps = new WeakMap<Click, ClickStep>();
const clickStream = (event: Click) => `${event.pointerType ?? 'mouse'}:${event.button ?? 0}`;

function clickStep(event: Click): ClickStep | undefined {
  if (event.detail <= 0) return;
  const seen = clickSteps.get(event);
  if (seen) return seen;
  const stream = clickStream(event);
  const pointer = event.pointerId ?? 0;
  const prior = pointerClicks.get(stream);
  const repeat = event.detail > 1 && prior?.pointer === pointer && prior.detail === event.detail - 1 && prior.consumed;
  const step = { detail: event.detail, pointer, repeat, consumed: false };
  pointerClicks.set(stream, step);
  clickSteps.set(event, step);
  return step;
}

if (typeof window !== 'undefined') {
  // Observe non-submit clicks and presses on disabled controls, which can
  // suppress the click entirely. Neither has consumed a submit activation.
  window.addEventListener('click', clickStep, { capture: true });
  window.addEventListener('pointerdown', event => {
    if (event.target instanceof Element && event.target.closest(':disabled')) {
      pointerClicks.delete(clickStream(event));
    }
  }, { capture: true });
}

/** Do not follow a read result link that appears under a repeat submit click. */
export function guardReadLink(event: MouseEvent): void {
  const step = clickStep(event);
  if (step?.repeat) {
    step.consumed = true;
    event.preventDefault();
  }
}

/**
 * Submission rules:
 * 1. Use the native click count, not elapsed time. A repeat click (detail > 1)
 *    cannot queue again after a submit in that pointer sequence, even if the
 *    control or request changes. If the preceding click was disabled or blocked
 *    and did not reach an enabled submit, the repeat may queue.
 * 2. Identify a request by launcher ID and its wire body, not UI metadata.
 *    Keep each key until acceptance is confirmed in this visit. A retry uses
 *    its original key even after other sends. A different body gets its own key.
 */
export class Submission {
  busy = $state(false);
  error = $state<string | null>(null);

  constructor(private attempts = new PlainAttempt()) {}

  async send<Body extends object, Result>(
    launcher: string,
    body: Body,
    post: (body: Body, key: string) => Promise<Result>,
    event?: Click,
  ): Promise<Result | null> {
    const step = event && clickStep(event);
    if (step?.repeat) {
      step.consumed = true;
      return null;
    }
    if (this.busy) return null;

    const plain = this.attempts.take(launcher, body, route.capture().current);
    this.busy = true;
    this.error = null;
    if (step) step.consumed = true;
    try {
      const result = await post(body, plain.key);
      plain.complete();
      return result;
    } catch (e) {
      this.error = submissionError(e);
      return null;
    } finally {
      this.busy = false;
    }
  }
}

/** Tracks the result of queuing a job. */
export class Submitter extends Submission {
  job = $state<AcceptedJob | null>(null);
  replayed = $state(false);
  statusElement?: HTMLElement;

  constructor(private launcher: string | ((job: NewJob) => string), attempts?: PlainAttempt) { super(attempts); }

  #launcherId(body: NewJob) {
    return typeof this.launcher === 'string' ? this.launcher : this.launcher(body);
  }

  async submit(job: NewJob, event: MouseEvent, focusStatus = true): Promise<AcceptedJob | null> {
    const focus = captureResultFocus(event);
    const trigger = event.currentTarget;
    // Use the same wire body for the key comparison and the POST.
    const body: NewJob = JSON.parse(JSON.stringify(stripEmpty(job)));
    const launcher = this.#launcherId(body);
    const sent = this.send(launcher, body, async (snapshot, key) => {
      const created = await api.queueJob(snapshot, key);
      if (typeof created?.id !== 'string' || !created.id) throw new Error('The controller did not return a job ID.');
      return created;
    }, event);
    const created = await sent;
    if (created) {
      this.job = created;
      this.replayed = !('state' in created);
      if (focusStatus) void focus?.(() => this.statusElement?.querySelector('a'));
      void brewery.refresh();
    } else if (this.error && trigger instanceof HTMLButtonElement) {
      void focus?.(() => trigger.isConnected && !trigger.disabled ? trigger
        : this.statusElement?.querySelector<HTMLElement>('[role="alert"]'));
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
export function acceptedJobSummary(accepted: AcceptedJob, request: NewJob): JobSummary {
  return 'state' in accepted ? accepted : {
    id: accepted.id, state: 'queued', kind: request.kind, host: request.host,
    input_ids: request.input_ids, options: request.options,
  };
}
