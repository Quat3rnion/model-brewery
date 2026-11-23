// Option lists for the searchable host, model and name pickers.

import { disambiguate, normalize, type ComboOption } from './combobox';
import { basename, capability, hostName, kindLabel, modelLabel } from './format';
import type { EligibilityEntry, Host, Model } from './types';

/** Show available hosts first and disable hosts rejected by workflow eligibility. */
export function hostOptions(hosts: Host[], kind?: string, eligibility?: (host: Host) => EligibilityEntry | undefined): ComboOption[] {
  const rows = hosts.map((h, i) => {
    const entry = eligibility?.(h);
    const c = entry
      ? { allowed: entry.eligible, reason: entry.reasons.map((r) => r.text).join(' ') || 'This host cannot run this workflow.' }
      : kind ? capability(h, kind) : { allowed: true, reason: null };
    const name = h.name ?? h.id;
    const detail = [name !== h.id ? h.id : '', c.allowed ? '' : `Unavailable: ${c.reason ?? ''}`].filter(Boolean).join(' · ');
    const o: ComboOption = { value: h.id, label: name, detail: detail || undefined, keywords: [h.id] };
    if (entry?.eligible === false) o.disabled = true;
    return { o, ok: c.allowed, i };
  });
  rows.sort((a, b) => Number(b.ok) - Number(a.ok) || a.i - b.i);
  return disambiguate(rows.map((r) => r.o));
}

/** Library entries of the given kinds; the value is the model ID. */
export function modelOptions(models: Model[], hosts: Host[], kinds: ReadonlySet<string>, exclude: string[], unavailable?: (model: Model) => string | null): ComboOption[] {
  return disambiguate(
    models
      .filter((m) => kinds.has(m.kind) && !exclude.includes(m.id))
      .map((m) => {
        const reason = unavailable?.(m);
        const status = reason ? ` · Unavailable: ${reason}` : m.available === false ? ' · unavailable' : '';
        return {
          value: m.id,
          label: modelLabel(m),
          detail: `${kindLabel(m.kind)} · ${hostName(hosts, m.host)} · ${m.path}${status}`,
          keywords: [m.id, m.host, m.kind, m.path],
          ...(reason ? { disabled: true } : {}),
        };
      }),
  );
}

/** Choices given as {id, label?} (benchmark targets, baseline sources). */
export function idOptions(list: { id: string; label?: string }[]): ComboOption[] {
  return disambiguate(list.map((x) => ({ value: x.id, label: x.label ?? x.id, detail: x.label && x.label !== x.id ? x.id : undefined, keywords: [x.id] })));
}

/** Free-form name suggestions: the basename of the path, then names already in the library. */
export function nameSuggestions(path: string, models: Model[]): ComboOption[] {
  const out: ComboOption[] = [];
  const base = path.trim() ? basename(path.trim()) : '';
  if (base && base !== '/') out.push({ value: base, label: base, detail: 'File or directory name' });
  const used = new Map<string, number>();
  for (const m of models) if (m.name) used.set(m.name, (used.get(m.name) ?? 0) + 1);
  for (const [name, n] of used) {
    if (name === base) continue;
    out.push({ value: name, label: name, detail: `In use by ${n} ${n === 1 ? 'entry' : 'entries'}` });
  }
  return out;
}

/** Items that match every token, plus the ticked ones, in list order. */
export function visibleChecklist<T extends { id: string; text: string }>(items: T[], query: string, ticked: string[]): T[] {
  const tokens = normalize(query.trim()).split(/\s+/).filter(Boolean);
  if (!tokens.length) return items;
  return items.filter((it) => ticked.includes(it.id) || tokens.every((t) => normalize(it.text).includes(t)));
}

/** Saved prompts by name; the ID is searchable and the value. Prompt text is not searched or shown. */
export function promptOptions(prompts: { id: string; name: string }[]): ComboOption[] {
  return disambiguate(prompts.map((p) => ({ value: p.id, label: p.name, keywords: [p.id] })));
}
