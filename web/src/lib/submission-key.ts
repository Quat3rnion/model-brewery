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

const hex = (bytes: Uint8Array) => [...bytes].map(x => x.toString(16).padStart(2, '0')).join('');

/** Keep only an unconfirmed request in the current visit. */
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
