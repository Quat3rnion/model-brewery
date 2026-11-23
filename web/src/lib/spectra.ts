// Spectral weight health (X2): parse a weight_spectra job result and build
// per-layer series and chart geometry. The values are not release-qualified.

export type MetricKey = 'effective_rank' | 'stable_rank' | 'alpha';
export const METRICS: { key: MetricKey; label: string; hint: string }[] = [
  { key: 'effective_rank', label: 'Effective rank', hint: 'Entropy of the normalized singular values, as a rank.' },
  { key: 'stable_rank', label: 'Stable rank', hint: 'Squared Frobenius norm divided by the squared largest singular value.' },
  { key: 'alpha', label: 'Alpha', hint: 'Power-law exponent of the eigenvalue tail (WeightWatcher style). Lower is more heavy-tailed.' },
];

export interface SpectraSide {
  dtype: string;
  shape: number[];
  method: string;
  singular_values: number[];
  effective_rank: number | null;
  stable_rank: number | null;
  alpha: number | null;
}
export interface SpectraTensor {
  name: string;
  layer: number | null;
  source?: SpectraSide;
  reference?: SpectraSide;
  delta?: Partial<Record<MetricKey | 'sigma_max', number | null>> | null;
  status?: string;
}
export interface SpectraModel { id: string; sha256: string; format: string }
export interface SpectraResult {
  qualification: string;
  source: SpectraModel;
  reference?: SpectraModel;
  tensors: SpectraTensor[];
}

const rec = (v: unknown): Record<string, unknown> | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');

function model(v: unknown): SpectraModel | undefined {
  const r = rec(v);
  return r && typeof r.id === 'string' ? { id: r.id, sha256: str(r.sha256), format: str(r.format) } : undefined;
}
function sideOf(v: unknown): SpectraSide | undefined {
  const r = rec(v);
  if (!r) return undefined;
  return {
    dtype: str(r.dtype),
    shape: Array.isArray(r.shape) ? r.shape.filter((x): x is number => typeof x === 'number') : [],
    method: str(r.method),
    singular_values: Array.isArray(r.singular_values) ? r.singular_values.filter((x): x is number => typeof x === 'number' && Number.isFinite(x)) : [],
    effective_rank: num(r.effective_rank),
    stable_rank: num(r.stable_rank),
    alpha: num(r.alpha),
  };
}

export function parseSpectra(raw: unknown): SpectraResult | null {
  const r = rec(raw);
  if (!r || r.kind !== 'weight_spectra' || !Array.isArray(r.tensors)) return null;
  const source = model(r.source);
  if (!source) return null;
  const tensors: SpectraTensor[] = [];
  for (const t of r.tensors) {
    const o = rec(t);
    if (!o || typeof o.name !== 'string') continue;
    const d = rec(o.delta);
    tensors.push({
      name: o.name,
      layer: typeof o.layer === 'number' && Number.isInteger(o.layer) ? o.layer : null,
      source: sideOf(o.source),
      reference: sideOf(o.reference),
      delta: d ? { effective_rank: num(d.effective_rank), stable_rank: num(d.stable_rank), alpha: num(d.alpha), sigma_max: num(d.sigma_max) } : o.delta === null ? null : undefined,
      status: typeof o.status === 'string' ? o.status : undefined,
    });
  }
  return { qualification: str(r.qualification) || 'not release-qualified', source, reference: model(r.reference), tensors };
}

/** Tensor name without the layer prefix, so the same role groups across layers. */
export function tensorGroup(t: Pick<SpectraTensor, 'name' | 'layer'>): string {
  if (t.layer === null) return t.name;
  const parts = t.name.split('.');
  const i = parts.indexOf(String(t.layer));
  return i >= 0 && i < parts.length - 1 ? parts.slice(i + 1).join('.') : t.name;
}

export function tensorGroups(r: SpectraResult): string[] {
  return [...new Set(r.tensors.filter((t) => t.layer !== null).map(tensorGroup))].sort();
}

export interface LayerPoint {
  layer: number;
  source: number | null;
  reference: number | null;
  delta: number | null;
  tensors: number;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** Mean of each metric per layer, over all tensors or one tensor group. */
export function layerSeries(r: SpectraResult, metric: MetricKey, group = ''): LayerPoint[] {
  const by = new Map<number, SpectraTensor[]>();
  for (const t of r.tensors) {
    if (t.layer === null || (group && tensorGroup(t) !== group)) continue;
    by.set(t.layer, [...(by.get(t.layer) ?? []), t]);
  }
  return [...by.entries()].sort(([a], [b]) => a - b).map(([layer, ts]) => {
    const s: number[] = [], ref: number[] = [], d: number[] = [];
    for (const t of ts) {
      const a = t.source?.[metric] ?? null, b = t.reference?.[metric] ?? null;
      if (a !== null) s.push(a);
      if (b !== null) ref.push(b);
      const given = t.delta?.[metric] ?? null;
      const dv = t.delta === undefined && a !== null && b !== null ? a - b : given;
      if (dv !== null) d.push(dv);
    }
    return { layer, source: mean(s), reference: mean(ref), delta: mean(d), tensors: ts.length };
  });
}

/** A relative change of 1% or more of the reference value. */
export const CHANGE_REL = 0.01;
export function changed(delta: number | null, reference: number | null): boolean {
  if (delta === null) return false;
  const base = Math.abs(reference ?? 0);
  return base > 1e-12 ? Math.abs(delta) / base >= CHANGE_REL : Math.abs(delta) > 1e-6;
}

/** The layer with the largest relative change, if any layer changed. */
export function largestChange(points: LayerPoint[]): LayerPoint | null {
  let best: LayerPoint | null = null, score = 0;
  for (const p of points) {
    if (!changed(p.delta, p.reference)) continue;
    const s = Math.abs(p.delta!) / Math.max(Math.abs(p.reference ?? 0), 1e-12);
    if (s > score) [best, score] = [p, s];
  }
  return best;
}

/** SVG path that starts a new segment after each missing value. */
export function linePath(points: [number, number | null][], x: (v: number) => number, y: (v: number) => number): string {
  const out: string[] = [];
  let pen = false;
  for (const [a, b] of points) {
    if (b === null) {
      pen = false;
      continue;
    }
    out.push(`${pen ? 'L' : 'M'}${round(x(a))},${round(y(b))}`);
    pen = true;
  }
  return out.join(' ');
}
const round = (v: number) => Math.round(v * 10) / 10;

/** Round tick values from min to max. */
export function ticks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** [index (1-based), value] of positive singular values, at most about `max` points. */
export function spectrumPoints(values: number[], max = 256): [number, number][] {
  const pos = values.map((v, i) => [i + 1, v] as [number, number]).filter(([, v]) => v > 0);
  if (pos.length <= max) return pos;
  const step = (pos.length - 1) / (max - 1);
  const out: [number, number][] = [];
  for (let k = 0; k < max; k++) out.push(pos[Math.round(k * step)]);
  return out;
}

export interface SpectraForm {
  compare: string;
  representation: 'safetensors' | 'gguf';
  compareRepresentation: 'safetensors' | 'gguf';
  layer: string;
  tensor: string;
}
export interface SpectraRequest {
  id: string;
  compare_id?: string;
  representation?: 'safetensors' | 'gguf';
  compare_representation?: 'safetensors' | 'gguf';
  layer?: number;
  tensor?: string;
}

export function spectraBlocker(f: SpectraForm): string {
  const l = f.layer.trim();
  return l && !/^\d{1,4}$/.test(l) ? 'The layer must be a whole number.' : '';
}

/** POST /api/weight-spectra body. Defaults (SafeTensors, the parent) are omitted. */
export function spectraRequest(id: string, f: SpectraForm): SpectraRequest {
  const out: SpectraRequest = { id };
  if (f.compare) out.compare_id = f.compare;
  if (f.representation !== 'safetensors') out.representation = f.representation;
  if (f.compareRepresentation !== 'safetensors') out.compare_representation = f.compareRepresentation;
  if (f.layer.trim()) out.layer = Number(f.layer.trim());
  if (f.tensor.trim()) out.tensor = f.tensor.trim();
  return out;
}
