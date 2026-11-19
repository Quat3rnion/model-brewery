import { obj } from './reports';

/** Most layer rows the UI renders for one selected position. Stated in the view. */
export const MAX_LAYER_ROWS = 256;
/** Most experts listed per cell. Stated in the view. */
export const MAX_EXPERTS_SHOWN = 16;

export interface SignalCell {
  layer: number;
  kind: string;
  attn: number | null;
  ffn: number | null;
  experts: number[] | null;
  weights: number[] | null;
}

export interface Coverage {
  total_layers: number | null;
  total_tokens: number | null;
  captured_cells: number | null;
  included_cells: number | null;
  layer_span: [number, number] | null;
  token_span: [number, number] | null;
  truncated: boolean | null;
}

export interface Signals {
  schema: number;
  /** Positions that have at least one included cell, ascending. */
  positions: number[];
  coverage: Coverage;
  omissions: Record<string, unknown> | null;
  /** Layer indices that appear anywhere in the record, ascending. */
  layers: number[];
  /** Cells for one position, ascending by layer. */
  at(position: number): SignalCell[];
  /** One row per recorded layer at a position; null where the engine has no cell. */
  rowsAt(position: number): { layer: number; cell: SignalCell | null }[];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const nums = (v: unknown): number[] | null => (Array.isArray(v) && v.every((x) => typeof x === 'number') ? (v as number[]) : null);
const span = (v: unknown): [number, number] | null => {
  const a = nums(v);
  return a && a.length === 2 ? [a[0], a[1]] : null;
};

/**
 * Read traces[].signals (schema 1). Returns null for receipts without
 * signals or with an unknown schema, so older results still render.
 * A missing cell field stays null: not captured, never zero.
 */
export function readSignals(raw: unknown): Signals | null {
  const s = obj(raw);
  if (!s || s.schema !== 1 || !Array.isArray(s.layers)) return null;
  const byPos = new Map<number, SignalCell[]>();
  const layerSet = new Set<number>();
  for (const l of s.layers) {
    const layer = obj(l);
    const index = num(layer?.layer);
    if (!layer || index === null || !Array.isArray(layer.cells)) continue;
    layerSet.add(index);
    const kind = typeof layer.kind === 'string' ? layer.kind : '';
    for (const c of layer.cells) {
      const cell = obj(c);
      const token = num(cell?.token);
      if (!cell || token === null) continue;
      const list = byPos.get(token) ?? [];
      list.push({ layer: index, kind, attn: num(cell.attn), ffn: num(cell.ffn), experts: nums(cell.experts), weights: nums(cell.weights) });
      byPos.set(token, list);
    }
  }
  for (const list of byPos.values()) list.sort((a, b) => a.layer - b.layer);
  const layers = [...layerSet].sort((a, b) => a - b);
  const cov = obj(s.coverage);
  return {
    schema: 1,
    positions: [...byPos.keys()].sort((a, b) => a - b),
    coverage: {
      total_layers: num(cov?.total_layers),
      total_tokens: num(cov?.total_tokens),
      captured_cells: num(cov?.captured_cells),
      included_cells: num(cov?.included_cells),
      layer_span: span(cov?.layer_span),
      token_span: span(cov?.token_span),
      truncated: typeof cov?.truncated === 'boolean' ? cov.truncated : null,
    },
    omissions: obj(s.omissions),
    layers,
    at: (p) => byPos.get(p) ?? [],
    rowsAt: (p) => {
      const cells = new Map((byPos.get(p) ?? []).map((c) => [c.layer, c]));
      return layers.map((layer) => ({ layer, cell: cells.get(layer) ?? null }));
    },
  };
}

/** Shows the recorded value (up to 10 significant digits, no padding zeros). */
export function fmtSignal(v: number | null): string {
  if (v === null) return '';
  return String(Number(v.toPrecision(10)));
}

/** "e12 (0.412)" pairs; weights shown only when lengths match. */
export function fmtExperts(experts: number[] | null, weights: number[] | null): string {
  if (!experts) return '';
  const shown = experts.slice(0, MAX_EXPERTS_SHOWN);
  const w = weights && weights.length === experts.length ? weights : null;
  const text = shown.map((e, i) => (w ? `${e} (${fmtSignal(w[i])})` : `${e}`)).join(', ');
  return experts.length > shown.length ? `${text}, +${experts.length - shown.length} more` : text;
}
