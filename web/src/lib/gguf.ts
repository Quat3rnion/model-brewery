// GGUF output profiles (llama.cpp tensor types). Separate from the HGN
// v2/HT43 catalog: different names, types and rules.
//
// GET  /api/profiles/gguf-catalog → {presets:[{id,requires_imatrix}],
//        tensor_types:[{id,block_size,block_bytes,requires_imatrix}]}   (static, not per source)
// POST /api/profiles {format:'gguf', name, preset, overrides:{[exactName]: type}} → revision
// POST /api/conversions {format:'gguf', input_ids:[source, imatrix?], profile_revision? | options:{preset?, threads?}}
// Tensor names come from inspect/plan result.report.gguf_tensors [{name, shape, current_type}];
// the worker validates every override. Sizes are only computed from block metrics and a known shape.

import { obj } from './reports';

/** Not a real llama.cpp type. Never offered, even if a catalog lists it. */
export const NEVER_OFFER = new Set(['IQ1_XXS']);
export const IMATRIX_KIND = 'imatrix';

export interface GgufPreset {
  id: string;
  label: string | null;
  requiresImatrix: boolean;
}
export interface GgufType {
  id: string;
  blockSize: number | null;
  blockBytes: number | null;
  requiresImatrix: boolean;
}
export interface GgufCatalog {
  presets: GgufPreset[];
  types: GgufType[];
  typeById: Map<string, GgufType>;
}

const pos = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null);

export function parseGgufCatalog(v: unknown): GgufCatalog | null {
  const o = obj(v);
  if (!o || !Array.isArray(o.presets) || !Array.isArray(o.tensor_types)) return null;
  const presets = o.presets
    .map(obj)
    .filter((p): p is Record<string, unknown> => !!p && typeof p.id === 'string' && !!p.id && !NEVER_OFFER.has(p.id as string))
    .map((p) => ({ id: p.id as string, label: typeof p.label === 'string' ? p.label : null, requiresImatrix: p.requires_imatrix === true }));
  const types = o.tensor_types
    .map(obj)
    .filter((t): t is Record<string, unknown> => !!t && typeof t.id === 'string' && !!t.id && !NEVER_OFFER.has(t.id as string))
    .map((t) => ({ id: t.id as string, blockSize: pos(t.block_size), blockBytes: pos(t.block_bytes), requiresImatrix: t.requires_imatrix === true }));
  return { presets, types, typeById: new Map(types.map((t) => [t.id, t])) };
}

export interface GgufTensor {
  name: string;
  layer: number | null;
  /** Name after blk.N. (or the whole name for non-layer tensors). */
  role: string;
  shape: number[] | null;
  currentType: string | null;
}

const BLK = /^blk\.(\d+)\.(.+)$/;

export function layerOf(name: string): number | null {
  const m = BLK.exec(name);
  return m ? Number(m[1]) : null;
}

/** result.report.gguf_tensors from an inspect or plan job; null when the result has none. */
export function ggufTensorsFromResult(result: unknown): GgufTensor[] | null {
  const list = obj(obj(result)?.report)?.gguf_tensors;
  if (!Array.isArray(list)) return null;
  const seen = new Set<string>();
  const out: GgufTensor[] = [];
  for (const x of list) {
    const t = obj(x);
    if (!t || typeof t.name !== 'string' || !t.name || seen.has(t.name)) continue;
    seen.add(t.name);
    const m = BLK.exec(t.name);
    out.push({
      name: t.name,
      layer: m ? Number(m[1]) : null,
      role: m ? m[2] : t.name,
      shape: Array.isArray(t.shape) && t.shape.length && t.shape.every((d) => Number.isInteger(d) && (d as number) > 0) ? (t.shape as number[]) : null,
      currentType: typeof t.current_type === 'string' ? t.current_type : null,
    });
  }
  return out.sort((a, b) => (a.layer ?? 1e9) - (b.layer ?? 1e9) || a.name.localeCompare(b.name));
}

/** Payload bytes from the catalog's block metrics; null unless the element count divides into blocks. */
export function tensorBytes(c: GgufCatalog, type: string, shape: number[] | null): number | null {
  const t = c.typeById.get(type);
  if (!t || !t.blockSize || !t.blockBytes || !shape) return null;
  const n = shape.reduce((a, b) => a * b, 1);
  return n % t.blockSize === 0 ? (n / t.blockSize) * t.blockBytes : null;
}

/** Set (type) or clear (null) every tensor of the role in layers from..to. Expands to exact names. */
export function rangeOverrides(tensors: GgufTensor[], o: Record<string, string>, r: { from: number; to: number; role: string | null; type: string | null }): Record<string, string> {
  const [a, b] = r.from <= r.to ? [r.from, r.to] : [r.to, r.from];
  const next = { ...o };
  for (const t of tensors) {
    if (t.layer === null || t.layer < a || t.layer > b) continue;
    if (r.role !== null && t.role !== r.role) continue;
    if (r.type === null) delete next[t.name];
    else next[t.name] = r.type;
  }
  return Object.fromEntries(Object.keys(next).sort().map((k) => [k, next[k]]));
}

export function imatrixRequired(c: GgufCatalog, preset: string, overrides: Record<string, string>): boolean {
  if (c.presets.find((p) => p.id === preset)?.requiresImatrix) return true;
  return Object.values(overrides).some((t) => c.typeById.get(t)?.requiresImatrix);
}

export function buildGgufProfileBody(c: GgufCatalog, name: string, preset: string, overrides: Record<string, string>) {
  const problems: string[] = [];
  const n = name.trim();
  if (!n) problems.push('Enter a profile name.');
  if (!c.presets.some((p) => p.id === preset)) problems.push('Choose a base preset.');
  const clean: Record<string, string> = {};
  for (const k of Object.keys(overrides).sort()) {
    const ty = overrides[k];
    if (!k.trim()) continue;
    if (!c.typeById.has(ty)) problems.push(`${k}: ${ty} is not a supported tensor type.`);
    else clean[k] = ty;
  }
  if (!problems.length && !Object.keys(clean).length) problems.push('Add at least one tensor override, or convert with the preset alone.');
  return { body: problems.length ? null : { format: 'gguf' as const, name: n, preset, overrides: clean }, problems };
}

export interface GgufConvertForm {
  source: string;
  sourceKind?: string;
  assets?: string;
  preset: string;
  revision: { revision: string; needsImatrix: boolean } | null;
  imatrix: string;
  threads: number | null;
  host: string;
}

export function buildGgufConversion(c: GgufCatalog, f: GgufConvertForm) {
  const problems: string[] = [];
  if (!f.source) problems.push('Select a source checkpoint.');
  if (f.sourceKind==='safetensors' && !f.assets) problems.push('Select matching config and tokenizer assets.');
  if (!f.revision && !c.presets.some((p) => p.id === f.preset)) problems.push('Choose a GGUF preset or a saved GGUF profile.');
  const needs = f.revision ? f.revision.needsImatrix : !!c.presets.find((p) => p.id === f.preset)?.requiresImatrix;
  if (needs && !f.imatrix) problems.push('This choice needs an imatrix file. Select one from the Library.');
  if (f.threads !== null && (!Number.isInteger(f.threads) || f.threads < 1 || f.threads > 64)) problems.push('Threads must be 1 to 64.');
  if (problems.length) return { body: null, problems };
  const body: Record<string, unknown> = { format: 'gguf' };
  if (f.host) body.host = f.host;
  body.input_ids = [f.source, ...(f.sourceKind==='safetensors' ? [f.assets!] : []), ...(f.imatrix ? [f.imatrix] : [])];
  if (f.revision) {
    body.profile_revision = f.revision.revision;
    if (f.threads !== null) body.options = { threads: f.threads };
  } else body.options = { preset: f.preset, ...(f.threads !== null ? { threads: f.threads } : {}) };
  return { body, problems };
}
