import { obj } from './reports';

export interface TokenRow {
  key: string;
  pos: number | string;
  id: number | string | null;
  text: string;
  phase: string;
  /** Candidates after the evaluated step that ends at this token, if any. */
  candidates: string;
}

function tokenText(t: unknown): string {
  const o = obj(t);
  if (!o) return typeof t === 'string' ? t : String(t);
  for (const k of ['piece', 'text', 'token']) if (typeof o[k] === 'string') return o[k] as string;
  return '';
}

function tokenId(t: unknown): number | string | null {
  if (typeof t === 'number') return t;
  const o = obj(t);
  if (o && (typeof o.id === 'number' || typeof o.id === 'string')) return o.id as number | string;
  return null;
}

function candidateText(list: unknown): string {
  if (!Array.isArray(list)) return '';
  return list
    .map((c) => {
      const o = obj(c);
      if (!o) return String(c);
      const p = typeof o.p === 'number' ? ` p=${o.p.toPrecision(3)}` : '';
      return `${JSON.stringify(tokenText(o))}#${tokenId(o) ?? '?'}${p}`;
    })
    .join(', ');
}

/**
 * Rows for the per-token table, in the Lanternfish shape:
 * tokens [{i,id,piece,phase}], steps [{first,n,phase}], and
 * next_candidates[k] is the candidate list after step k. It belongs to the
 * last token of that step (first + n - 1), not to token index k.
 */
export function tokenRows(trace: Record<string, unknown>): TokenRow[] {
  const tokens = Array.isArray(trace.tokens) ? trace.tokens : [];
  const steps = Array.isArray(trace.steps) ? trace.steps : [];
  const next = Array.isArray(trace.next_candidates) ? trace.next_candidates : [];
  const byEnd = new Map<number, unknown>();
  steps.forEach((step, k) => {
    const s = obj(step);
    if (s && typeof s.first === 'number' && typeof s.n === 'number' && next[k] !== undefined) byEnd.set(s.first + s.n - 1, next[k]);
  });
  return tokens.map((t, index) => {
    const o = obj(t);
    const pos = o && typeof o.i === 'number' ? o.i : index;
    return {
      key: `${index}`,
      pos,
      id: tokenId(t),
      text: tokenText(t),
      phase: o && typeof o.phase === 'string' ? o.phase : '',
      candidates: candidateText(byEnd.get(typeof pos === 'number' ? pos : index)),
    };
  });
}

/**
 * Trace values are compared directly only when the comparison record says so.
 * The engine reports alignment.status: aligned, prefix_only (only the first
 * matched_prefix_tokens positions pair up) or not_aligned (no pairing).
 */
export function comparability(alignment: unknown): { comparable: boolean; status: string; prefix: number | null; text: string } {
  const a = obj(alignment);
  const reason = a && typeof a.reason === 'string' && a.reason ? ` ${a.reason}` : '';
  if (!a || typeof a.status !== 'string')
    return { comparable: false, status: 'unknown', prefix: null, text: 'The result has no alignment status. Outputs are shown side by side; trace values are not compared.' };
  const prefix = typeof a.matched_prefix_tokens === 'number' ? a.matched_prefix_tokens : null;
  switch (a.status) {
    case 'aligned':
      return { comparable: true, status: a.status, prefix, text: `Inputs, positions and signal meanings match. Trace values can be compared directly.${reason}` };
    case 'prefix_only':
      // Matching token IDs do not assert matching layer or expert meanings.
      return {
        comparable: false,
        status: a.status,
        prefix,
        text: `The first ${prefix ?? '?'} token positions have the same input tokens. The engine does not assert that layer or expert signals mean the same thing, so trace values are not compared directly.${reason}`,
      };
    case 'not_aligned':
      return { comparable: false, status: a.status, prefix, text: `No positions pair up. Outputs are shown side by side only.${reason}` };
    default:
      return { comparable: false, status: a.status, prefix, text: `Unknown alignment status "${a.status}". Values are not compared.${reason}` };
  }
}
