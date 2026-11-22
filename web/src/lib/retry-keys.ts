// Compare wire values, not property insertion order or form-only state.
export function bodyKey(body: unknown): string | undefined {
  return JSON.stringify(body, (_, value: unknown) => value && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value);
}

/** Fresh ASCII key within the controller's size and character limits. */
export function newKey(): string {
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  return `conv-${[...b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
}

export interface RetryIdentity {
  slot: string;
  id: string;
  entry: string;
  action: string;
}

export interface RetrySubmission {
  blocked: boolean;
  history?: RetryIdentity;
}

type Attempt = { key: string; accepted: boolean };
// Retry entries outlive a mounted launcher, but not the browser document.
const entries = new Map<string, Attempt>();
const unconfirmed = new Map<string, Attempt>();

/** Each launcher owns its last body. History sharing is limited to that
 * logical action and the exact retry body, including after a remount. */
export class RetryKeys {
  #last?: { body: string | undefined; attempt: Attempt; retry: boolean };

  constructor(private mint = newKey, private replayAccepted = false) {}

  take(body: unknown, history?: RetryIdentity, supplied?: string) {
    const json = bodyKey(body);
    const last = this.#last;
    const make = (key = this.mint()): Attempt => ({ key, accepted: false });
    const request = history && JSON.stringify([history.slot, history.action, history.id, json]);
    const entry = history && JSON.stringify([request, history.entry]);
    let attempt: Attempt;
    if (request && entry) {
      attempt = entries.get(entry) ?? unconfirmed.get(request) ?? make();
      entries.set(entry, attempt);
      if (!attempt.accepted) unconfirmed.set(request, attempt);
    } else if (supplied) {
      attempt = make(supplied);
    } else if (last && last.body === json && (!last.attempt.accepted || last.retry || this.replayAccepted)) {
      attempt = last.attempt;
    } else {
      attempt = make();
    }
    this.#last = { body: json, attempt, retry: !!history || !!supplied || (last?.attempt === attempt && last.retry) };
    return { key: attempt.key, accept: () => {
      attempt.accepted = true;
      if (request && unconfirmed.get(request) === attempt) unconfirmed.delete(request);
    } };
  }
}
