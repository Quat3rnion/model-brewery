import { SvelteMap } from 'svelte/reactivity';

/** Keep the name of each selection until the operator clears it. */
export class Checklist {
  #picked = new SvelteMap<string, string>();

  get ids(): string[] { return [...this.#picked.keys()]; }

  has(id: string): boolean { return this.#picked.has(id); }

  set(id: string, label: string, checked: boolean) {
    if (checked) this.#picked.set(id, label);
    else this.#picked.delete(id);
  }

  missing(available: readonly { id: string }[]): { id: string; label: string }[] {
    const ids = new Set(available.map(item => item.id));
    return [...this.#picked].flatMap(([id, label]) => ids.has(id) ? [] : [{ id, label }]);
  }

  remove(items: readonly { id: string }[]) {
    for (const item of items) this.#picked.delete(item.id);
  }
}
