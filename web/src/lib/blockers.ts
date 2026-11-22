export interface Blocker {
  text: string;
  field?: string;
  host?: string;
  fixAnchor?: string;
}

export function blockerText({ text, host }: Blocker): string {
  const message = text.trim();
  return `${host ? `${host}: ` : ''}${message}${/[.!?]$/.test(message) ? '' : '.'}`;
}

export function visibleBlockers(blockers: Blocker[], touched: Readonly<Record<string, boolean>>, attempted = false): Blocker[] {
  return attempted ? blockers : blockers.filter(({ field }) => field !== undefined && touched[field] === true);
}

export function fieldError(blockers: Blocker[], field: string): string | undefined {
  const first = blockers.find((blocker) => blocker.field === field);
  return first && blockerText(first);
}
