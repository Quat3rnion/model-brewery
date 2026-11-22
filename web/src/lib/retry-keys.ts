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

type Attempt<T> = { key: string; request?: string; result?: T };
const attempts = new Map<string, Attempt<unknown>>();
// Retry entries outlive a mounted launcher, but not the browser document.
const unconfirmed = new Map<string, Attempt<unknown>>();

/** One instance per launcher. History sharing requires the same action, job
 * kind and wire body, including after a remount. */
export class RetryKeys<T extends object> {
  #last?: { request: string; attempt: Attempt<T>; retry: boolean };

  constructor(private mint = newKey, private replayAccepted = false) {}

  take(kind: string, body: unknown, history?: RetryIdentity) {
    const request = JSON.stringify([kind, bodyKey(body)]);
    const last = this.#last;
    const retryRequest = history && JSON.stringify([history.slot, history.action, history.id, request]);
    const entry = history && JSON.stringify([retryRequest, history.entry]);
    const make = (): Attempt<T> => ({ key: this.mint(), request: retryRequest });
    let attempt: Attempt<T>;
    if (retryRequest && entry) {
      // The same action and kind in the same slot use one response type.
      const prior = (attempts.get(entry) ?? unconfirmed.get(retryRequest)) as Attempt<T> | undefined;
      attempt = prior ?? make();
      attempts.set(entry, attempt);
      if (!attempt.result) unconfirmed.set(retryRequest, attempt);
    } else if (last && last.request === request && (!last.attempt.result || last.retry || this.replayAccepted)) {
      attempt = last.attempt;
    } else {
      attempt = make();
    }
    this.#last = { request, attempt, retry: !!history || (last?.attempt === attempt && last.retry) };
    return {
      key: attempt.key,
      result: attempt.result,
      accept: (result: T) => {
        attempt.result ??= result;
        const request = attempt.request;
        if (request && unconfirmed.get(request) === attempt) unconfirmed.delete(request);
      },
    };
  }
}
