// Compare wire values, not property insertion order or form-only state.
export function bodyKey(body: unknown): string | undefined {
  const json = JSON.stringify(body);
  return json === undefined ? undefined : sortedJson(JSON.parse(json));
}

function sortedJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(sortedJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    const fields = Object.entries(value).sort(([a], [b]) => {
      // Rust sorts strings by UTF-8 bytes, which have Unicode code point order.
      const left = [...a], right = [...b];
      for (let i = 0; i < Math.min(left.length, right.length); i++) {
        const diff = left[i].codePointAt(0)! - right[i].codePointAt(0)!;
        if (diff) return diff;
      }
      return left.length - right.length;
    });
    // Emit keys directly: JSON.stringify on an object reorders integer-like keys.
    return `{${fields.map(([key, field]) => `${JSON.stringify(key)}:${sortedJson(field)}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

async function sha256(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return hex(new Uint8Array(bytes));
}

const hex = (bytes: Uint8Array) => [...bytes].map(x => x.toString(16).padStart(2, '0')).join('');
const keyChars = /^[A-Za-z0-9._-]+$/;
const keyLimit = 128;

/** Stable over document reloads. The launcher is a fixed, short ASCII action ID. */
export async function retryKey(failedId: string, launcher: string, body: object): Promise<string> {
  if (!keyChars.test(launcher) || launcher.length > 60) throw new Error('Invalid retry launcher ID.');
  const id = keyChars.test(failedId) && failedId.length + launcher.length + 4 < keyLimit
    ? failedId : `sha256.${(await sha256(failedId)).slice(0, 32)}`;
  const prefix = `r-${id}-${launcher}-`;
  return prefix + (await sha256(bodyKey(body)!)).slice(0, keyLimit - prefix.length);
}

export interface RetrySubmission {
  blocked: boolean;
  failedId?: string;
  /** Canonical launcher when several controls submit the same failed operation. */
  launcher?: string;
  /** Capture the draft now; open the plain form only on a later explicit press. */
  preparePlain?: () => () => boolean;
}

export function resultOfPreviousSubmission(accepted: string | undefined, current: unknown): boolean {
  return accepted !== undefined && accepted !== bodyKey(current);
}

/** Keep only a plain launcher’s unconfirmed request in the current visit.
 * Retry keys and accepted results never enter this cache. */
export class PlainAttempt {
  #pending?: { request: string; key: string; current: () => boolean };

  take(launcher: string, body: object, current: () => boolean) {
    const request = bodyKey([launcher, body])!;
    const prior = this.#pending;
    const pending = prior?.current() && prior.request === request ? prior : {
      request, key: `conv-${hex(crypto.getRandomValues(new Uint8Array(16)))}`, current,
    };
    this.#pending = pending;
    return {
      key: pending.key,
      complete: () => { if (this.#pending === pending) this.#pending = undefined; },
    };
  }
}
