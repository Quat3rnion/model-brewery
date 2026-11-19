// Pure helpers for the searchable pickers (components/Combobox.svelte).
// The picker keeps the real value (host ID, model ID, prompt ID); the label
// is only for display and search.

export interface ComboOption {
  value: string;
  label: string;
  /** Secondary line: kind, host, path. Searched. */
  detail?: string;
  /** Extra search terms that are not shown, for example the ID. */
  keywords?: string[];
  disabled?: boolean;
}

export function normalize(s: string): string {
  return s.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
}

/**
 * Options whose label, detail, value or keywords contain every whitespace
 * separated token of the query. Label prefix matches come first, then
 * label matches, then the rest, each in source order.
 */
export function filterOptions(options: ComboOption[], query: string): ComboOption[] {
  const q = normalize(query.trim());
  if (!q) return options;
  const tokens = q.split(/\s+/);
  const ranked: { o: ComboOption; rank: number; i: number }[] = [];
  options.forEach((o, i) => {
    const label = normalize(o.label);
    const hay = `${label}\n${normalize(o.detail ?? '')}\n${normalize(o.value)}\n${(o.keywords ?? []).map(normalize).join('\n')}`;
    if (!tokens.every((t) => hay.includes(t))) return;
    ranked.push({ o, i, rank: label.startsWith(q) ? 0 : label.includes(q) ? 1 : 2 });
  });
  return ranked.sort((a, b) => a.rank - b.rank || a.i - b.i).map((x) => x.o);
}

/** Labels that occur more than once get their value appended, so the input text names one option. */
export function disambiguate(options: ComboOption[]): ComboOption[] {
  const counts = new Map<string, number>();
  for (const o of options) counts.set(o.label, (counts.get(o.label) ?? 0) + 1);
  return options.map((o) => ((counts.get(o.label) ?? 0) > 1 ? { ...o, label: `${o.label} (${o.value})` } : o));
}

/** The single enabled option whose label or value equals the text (case-insensitive), or null. */
export function exactMatch(options: ComboOption[], text: string): ComboOption | null {
  const t = normalize(text.trim());
  if (!t) return null;
  const hits = options.filter((o) => !o.disabled && (normalize(o.label) === t || normalize(o.value) === t));
  return hits.length === 1 ? hits[0] : null;
}

/** Next enabled index from `from` in direction `dir`; stays put at the ends. -1 when none. */
export function stepIndex(options: ComboOption[], from: number, dir: 1 | -1): number {
  let i = from;
  for (let n = 0; n < options.length; n += 1) {
    i += dir;
    if (i < 0 || i >= options.length) return from >= 0 && from < options.length && !options[from].disabled ? from : -1;
    if (!options[i].disabled) return i;
  }
  return -1;
}
