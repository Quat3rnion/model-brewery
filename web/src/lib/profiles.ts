// Custom conversion profiles: per-tensor choice between the two built-in
// encodings (v2 and HT43). The controller owns the catalog and the saved
// revisions; the browser only edits a list of overrides and previews sizes.
//
// GET  /api/profiles/catalog → {schema:1, architecture, source_sha256, catalog_sha256,
//        base_bytes:{v2,ht43}, tensors:[{name,file,group,layer,role,v2,ht43,delta_bytes,editable,reason?}]}
// GET  /api/profiles → [{revision, name, base, catalog_sha256, overrides, size, created_at}]
// POST /api/profiles {name, base, catalog_sha256, overrides} → revision (201 new, 200 identical)
// A revision is an immutable 64-hex content address. Editing saves a new revision.

import { obj } from './reports';

export type Base = 'v2' | 'ht43';
export type Choice = Base;
export type Overrides = Record<string, Choice>;
export const GROUPS = ['trunk', 'vision', 'mtp', 'global', 'engram'] as const;
export const GROUP_LABEL: Record<string, string> = { trunk: 'Trunk layers', vision: 'Vision tower', mtp: 'MTP head', global: 'Global (embeddings, output)', engram: 'Engram lookup' };
export const BASE_LABEL: Record<Base, string> = { v2: 'HGN v2', ht43: 'HT43' };
export const BUILT_IN_NAMES = new Set(['v2', 'ht43', 'hgn v2', 'hgn-v2']);

export interface Tensor {
  name: string;
  file: string;
  group: string;
  layer: number | null;
  role: string;
  v2: string;
  ht43: string;
  /** Signed bytes, HT43 minus v2, including companions. null when the catalog did not state it. */
  delta: number | null;
  editable: boolean;
  reason: string | null;
}

export interface Catalog {
  architecture: string;
  sourceSha256: string;
  catalogSha256: string;
  baseBytes: Record<Base, number | null>;
  tensors: Tensor[];
  byName: Map<string, Tensor>;
  trunkLayers: number[];
  fixedGroups: { group: string; count: number; editable: number }[];
  roles: string[];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);

export function parseCatalog(v: unknown): Catalog | null {
  const o = obj(v);
  if (!o || o.schema !== 1 || !Array.isArray(o.tensors)) return null;
  const tensors: Tensor[] = [];
  const seen = new Set<string>();
  for (const x of o.tensors) {
    const e = obj(x);
    if (!e || typeof e.name !== 'string' || !e.name || seen.has(e.name) || typeof e.v2 !== 'string' || typeof e.ht43 !== 'string') continue;
    seen.add(e.name);
    tensors.push({
      name: e.name,
      file: str(e.file) ?? '',
      group: str(e.group) ?? 'global',
      layer: Number.isInteger(e.layer) ? (e.layer as number) : null,
      role: str(e.role) ?? '',
      v2: e.v2,
      ht43: e.ht43,
      delta: num(e.delta_bytes),
      editable: e.editable === true,
      reason: str(e.reason),
    });
  }
  const gi = (g: string) => {
    const i = (GROUPS as readonly string[]).indexOf(g);
    return i < 0 ? GROUPS.length : i;
  };
  tensors.sort((a, b) => gi(a.group) - gi(b.group) || (a.layer ?? -1) - (b.layer ?? -1) || a.name.localeCompare(b.name));
  const bb = obj(o.base_bytes) ?? {};
  const trunk = [...new Set(tensors.filter((t) => t.group === 'trunk' && t.layer !== null).map((t) => t.layer as number))].sort((a, b) => a - b);
  const fixedGroups = [...new Set(tensors.filter((t) => t.group !== 'trunk').map((t) => t.group))]
    .sort((a, b) => gi(a) - gi(b))
    .map((group) => {
      const g = tensors.filter((t) => t.group === group);
      return { group, count: g.length, editable: g.filter((t) => t.editable).length };
    });
  return {
    architecture: str(o.architecture) ?? '?',
    sourceSha256: str(o.source_sha256) ?? '',
    catalogSha256: str(o.catalog_sha256) ?? '',
    baseBytes: { v2: num(bb.v2), ht43: num(bb.ht43) },
    tensors,
    byName: new Map(tensors.map((t) => [t.name, t])),
    trunkLayers: trunk,
    fixedGroups,
    roles: [...new Set(tensors.map((t) => t.role).filter(Boolean))].sort(),
  };
}

/** Why a tensor cannot be changed; null when it can. Uses the catalog reason when present. */
export function fixedReason(t: Tensor): string | null {
  if (t.editable && t.v2 !== t.ht43) return null;
  if (t.reason) return t.reason;
  if (t.v2 === t.ht43) return 'Same encoding in v2 and HT43';
  return 'Not editable in the converter catalog';
}

export function isLegal(t: Tensor): boolean {
  return fixedReason(t) === null;
}

export function effective(base: Base, o: Overrides, t: Tensor): Choice {
  return o[t.name] ?? base;
}

export function encodingOf(t: Tensor, c: Choice): string {
  return c === 'v2' ? t.v2 : t.ht43;
}

/** Only legal tensors whose choice differs from the base, in sorted name order. */
export function normalizeOverrides(c: Catalog, base: Base, o: Record<string, string>): Overrides {
  const out: Overrides = {};
  for (const name of Object.keys(o).sort()) {
    const t = c.byName.get(name);
    const v = o[name];
    if (!t || !isLegal(t) || (v !== 'v2' && v !== 'ht43') || v === base) continue;
    out[name] = v;
  }
  return out;
}

export function predictSize(c: Catalog, base: Base, o: Overrides): { base: number | null; predicted: number | null; delta: number | null; changed: number } {
  let delta = 0;
  let unknown = false;
  let changed = 0;
  for (const [name, v] of Object.entries(o)) {
    const t = c.byName.get(name);
    if (!t || v === base) continue;
    changed += 1;
    if (t.delta === null) unknown = true;
    else delta += v === 'ht43' ? t.delta : -t.delta;
  }
  const b = c.baseBytes[base];
  return { base: b, predicted: b === null || unknown ? null : b + delta, delta: unknown ? null : delta, changed };
}

export interface LayerStat {
  layer: number;
  total: number;
  editable: number;
  ht43: number;
  changed: number;
}

/** Per trunk layer: tensor count, legal count, how many use HT43 now, how many differ from the base. */
export function layerSummary(c: Catalog, base: Base, o: Overrides): Map<number, LayerStat> {
  const m = new Map<number, LayerStat>();
  for (const l of c.trunkLayers) m.set(l, { layer: l, total: 0, editable: 0, ht43: 0, changed: 0 });
  for (const t of c.tensors) {
    if (t.group !== 'trunk' || t.layer === null) continue;
    const s = m.get(t.layer)!;
    s.total += 1;
    if (!isLegal(t)) continue;
    s.editable += 1;
    const ch = effective(base, o, t);
    if (ch === 'ht43') s.ht43 += 1;
    if (ch !== base) s.changed += 1;
  }
  return m;
}

export function rangeSelect(layers: number[], from: number, to: number): number[] {
  const [a, b] = from <= to ? [from, to] : [to, from];
  return layers.filter((l) => l >= a && l <= b);
}

/** Set every legal tensor of the role (or all roles when null) in the given layers. 'base' removes overrides. */
export function applyPolicy(c: Catalog, base: Base, o: Overrides, p: { layers: number[]; role: string | null; choice: Choice | 'base' }): Overrides {
  const next: Record<string, string> = { ...o };
  const layers = new Set(p.layers);
  for (const t of c.tensors) {
    if (t.group !== 'trunk' || t.layer === null || !layers.has(t.layer) || !isLegal(t)) continue;
    if (p.role !== null && t.role !== p.role) continue;
    if (p.choice === 'base') delete next[t.name];
    else next[t.name] = p.choice;
  }
  return normalizeOverrides(c, base, next);
}

/** Set every legal tensor of a non-trunk group. */
export function applyGroup(c: Catalog, base: Base, o: Overrides, group: string, choice: Choice | 'base'): Overrides {
  const next: Record<string, string> = { ...o };
  for (const t of c.tensors) {
    if (t.group !== group || !isLegal(t)) continue;
    if (choice === 'base') delete next[t.name];
    else next[t.name] = choice;
  }
  return normalizeOverrides(c, base, next);
}

export interface SearchFilter {
  q?: string;
  group?: string;
  layers?: number[];
  changedOnly?: boolean;
  editableOnly?: boolean;
}

export function searchTensors(c: Catalog, base: Base, o: Overrides, f: SearchFilter): Tensor[] {
  const tokens = (f.q ?? '').toLowerCase().split(/\s+/).filter(Boolean);
  const layers = f.layers?.length ? new Set(f.layers) : null;
  return c.tensors.filter((t) => {
    if (f.group && t.group !== f.group) return false;
    if (layers && (t.group !== 'trunk' || t.layer === null || !layers.has(t.layer))) return false;
    if (f.editableOnly && !isLegal(t)) return false;
    if (f.changedOnly && effective(base, o, t) === base) return false;
    if (tokens.length) {
      const hay = `${t.name} ${t.role} ${t.group} ${t.v2} ${t.ht43} ${t.file}`.toLowerCase();
      return tokens.every((k) => hay.includes(k));
    }
    return true;
  });
}

export function buildProfileBody(c: Catalog, base: Base, name: string, o: Overrides) {
  const problems: string[] = [];
  const n = name.trim();
  if (!n) problems.push('Enter a profile name.');
  else if (BUILT_IN_NAMES.has(n.toLowerCase())) problems.push('That name belongs to a built-in profile.');
  else if (n.length > 80) problems.push('Use at most 80 characters for the name.');
  const overrides = normalizeOverrides(c, base, o);
  if (!Object.keys(overrides).length) problems.push(`There is no change from ${BASE_LABEL[base]}; use the built-in profile instead.`);
  return { body: problems.length ? null : { name: n, base, catalog_sha256: c.catalogSha256, overrides }, problems };
}

export interface SavedProfile {
  revision: string;
  name: string;
  base: Base;
  catalogSha256: string;
  overrides: Overrides;
  size: { base: number | null; predicted: number | null; delta: number | null } | null;
  createdAt: number | null;
  /** Made for the current catalog. Others are shown but cannot be edited or used. */
  compatible: boolean;
}

export function parseProfile(v: unknown, c: Catalog | null): SavedProfile | null {
  const o = obj(v);
  if (!o || typeof o.revision !== 'string' || !/^[0-9a-f]{64}$/.test(o.revision) || typeof o.name !== 'string') return null;
  // GGUF recipes have their own editor and are never shown as HGN profiles.
  if (o.format !== undefined && o.format !== 'hgn') return null;
  const base: Base = o.base === 'ht43' ? 'ht43' : 'v2';
  const ov = obj(o.overrides) ?? {};
  const overrides: Overrides = {};
  for (const k of Object.keys(ov).sort()) if (ov[k] === 'v2' || ov[k] === 'ht43') overrides[k] = ov[k] as Choice;
  const sz = obj(o.size);
  const cat = str(o.catalog_sha256) ?? '';
  return {
    revision: o.revision,
    name: o.name,
    base,
    catalogSha256: cat,
    overrides,
    size: sz ? { base: num(sz.base_bytes), predicted: num(sz.predicted_bytes), delta: num(sz.delta_bytes) } : null,
    createdAt: num(o.created_at),
    compatible: !!c && cat === c.catalogSha256,
  };
}

export function parseProfiles(v: unknown, c: Catalog | null): SavedProfile[] {
  return (Array.isArray(v) ? v.map((x) => parseProfile(x, c)).filter((x): x is SavedProfile => !!x) : []).sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
}

export const shortRev = (r: string) => r.slice(0, 10);

export interface SavedGgufProfile {
  encoderFamily?:string;
  encoderRevision?:string;
  revision: string;
  name: string;
  preset: string;
  overrides: Record<string, string>;
  createdAt: number | null;
}

export function parseGgufProfiles(v: unknown): SavedGgufProfile[] {
  if (!Array.isArray(v)) return [];
  const out: SavedGgufProfile[] = [];
  for (const x of v) {
    const o = obj(x);
    if (!o || o.format !== 'gguf' || typeof o.revision !== 'string' || !/^[0-9a-f]{64}$/.test(o.revision) || typeof o.name !== 'string' || typeof o.preset !== 'string') continue;
    const ov = obj(o.overrides) ?? {};
    const overrides: Record<string, string> = {};
    for (const k of Object.keys(ov).sort()) if (typeof ov[k] === 'string') overrides[k] = ov[k] as string;
    out.push({ revision: o.revision, name: o.name, preset: o.preset, overrides, createdAt: num(o.created_at),
      ...(typeof o.encoder_family==='string' ? {encoderFamily:o.encoder_family} : {}),
      ...(typeof o.encoder_revision==='string' ? {encoderRevision:o.encoder_revision} : {}) });
  }
  return out.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
}
