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

export const requestKey = (launcher: string, body: object): string => bodyKey([launcher, body])!;

/** Keep unconfirmed requests in the current visit. */
export class PlainAttempt {
  #pending = new Map<string, string>();
  #current?: () => boolean;

  take(launcher: string, body: object, current: () => boolean) {
    if (!this.#current?.()) {
      this.#pending.clear();
      this.#current = current;
    }
    const request = requestKey(launcher, body);
    const key = this.#pending.get(request) ?? `conv-${hex(crypto.getRandomValues(new Uint8Array(16)))}`;
    this.#pending.set(request, key);
    return {
      key,
      complete: () => { if (this.#pending.get(request) === key) this.#pending.delete(request); },
    };
  }
}
