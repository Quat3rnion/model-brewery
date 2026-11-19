import { describe, expect, it } from 'vitest';
import {
  applyPolicy,
  buildProfileBody,
  effective,
  encodingOf,
  fixedReason,
  layerSummary,
  normalizeOverrides,
  parseCatalog,
  parseProfiles,
  predictSize,
  rangeSelect,
  searchTensors,
} from './profiles';

const t = (name: string, over: Record<string, unknown> = {}) => ({ name, file: 'model.hgn', group: 'trunk', layer: 0, role: 'attn_q', v2: 'Q6G64', ht43: 'Q8G64', delta_bytes: 100, editable: true, ...over });
const raw = {
  schema: 1,
  architecture: 'qwen3_next',
  source_sha256: 'a'.repeat(64),
  catalog_sha256: 'c'.repeat(64),
  base_bytes: { v2: 10_000, ht43: 10_500 },
  tensors: [
    t('l0.q'),
    t('l0.down', { role: 'ffn_down', v2: 'I4r', ht43: 'Ht3', delta_bytes: -300 }),
    t('l1.q', { layer: 1 }),
    t('l1.norm', { layer: 1, role: 'norm', v2: 'F16', ht43: 'F16', delta_bytes: 0, editable: false }),
    t('l2.q', { layer: 2, delta_bytes: 50 }),
    t('vis.patch', { group: 'vision', layer: null, role: 'vision', editable: false }),
    t('mtp.head', { group: 'mtp', layer: null, role: 'mtp', editable: false, reason: 'MTP head follows the base profile' }),
    t('embed', { group: 'global', layer: null, role: 'embed' }),
    t('engram.0', { group: 'engram', layer: null, role: 'engram', editable: false }),
    { name: 'bad' },
  ],
};

describe('catalog', () => {
  it('parses tensors, skips malformed ones and sorts by group, layer and name', () => {
    const c = parseCatalog(raw)!;
    expect(c.tensors).toHaveLength(9);
    expect(c.tensors[0].name).toBe('l0.down');
    expect(c.trunkLayers).toEqual([0, 1, 2]);
    expect(c.fixedGroups.map((g) => [g.group, g.count])).toEqual([['vision', 1], ['mtp', 1], ['global', 1], ['engram', 1]]);
    expect(parseCatalog({ schema: 2, tensors: [] })).toBeNull();
  });
  it('fixed reasons come from the catalog when given, otherwise from facts', () => {
    const c = parseCatalog(raw)!;
    const by = (n: string) => c.tensors.find((x) => x.name === n)!;
    expect(fixedReason(by('l0.q'))).toBeNull();
    expect(fixedReason(by('l1.norm'))).toBe('Same encoding in v2 and HT43');
    expect(fixedReason(by('mtp.head'))).toBe('MTP head follows the base profile');
    expect(fixedReason(by('vis.patch'))).toBe('Not editable in the converter catalog');
  });
});

describe('overrides and sizes', () => {
  const c = parseCatalog(raw)!;
  it('normalizes: only editable tensors that differ from the base, sorted', () => {
    const o = normalizeOverrides(c, 'v2', { 'l2.q': 'ht43', 'l0.q': 'v2', 'l1.norm': 'ht43', 'l0.down': 'ht43', ghost: 'ht43' });
    expect(Object.keys(o)).toEqual(['l0.down', 'l2.q']);
  });
  it('effective choice and encoding', () => {
    const tq = c.tensors.find((x) => x.name === 'l0.q')!;
    expect(effective('v2', {}, tq)).toBe('v2');
    expect(effective('v2', { 'l0.q': 'ht43' }, tq)).toBe('ht43');
    expect(encodingOf(tq, 'ht43')).toBe('Q8G64');
  });
  it('predicted size uses signed delta (HT43 minus v2) from either base', () => {
    expect(predictSize(c, 'v2', {})).toEqual({ base: 10_000, predicted: 10_000, delta: 0, changed: 0 });
    expect(predictSize(c, 'v2', { 'l0.q': 'ht43', 'l0.down': 'ht43' })).toEqual({ base: 10_000, predicted: 9_800, delta: -200, changed: 2 });
    expect(predictSize(c, 'ht43', { 'l0.q': 'v2' })).toEqual({ base: 10_500, predicted: 10_400, delta: -100, changed: 1 });
  });
});

describe('layer map and bulk policy', () => {
  const c = parseCatalog(raw)!;
  it('summarizes each trunk layer: editable, HT43 share, changed', () => {
    const s = layerSummary(c, 'v2', { 'l0.q': 'ht43' });
    expect(s.get(0)).toEqual({ layer: 0, total: 2, editable: 2, ht43: 1, changed: 1 });
    expect(s.get(1)).toEqual({ layer: 1, total: 2, editable: 1, ht43: 0, changed: 0 });
  });
  it('range select is inclusive in either direction', () => {
    expect(rangeSelect([0, 1, 2, 3], 3, 1)).toEqual([1, 2, 3]);
  });
  it('applies a role policy to selected layers only, editable tensors only', () => {
    const o = applyPolicy(c, 'v2', {}, { layers: [0, 1], role: 'attn_q', choice: 'ht43' });
    expect(Object.keys(o)).toEqual(['l0.q', 'l1.q']);
    const all = applyPolicy(c, 'v2', o, { layers: [0], role: null, choice: 'ht43' });
    expect(Object.keys(all)).toEqual(['l0.down', 'l0.q', 'l1.q']);
    const back = applyPolicy(c, 'v2', all, { layers: [0, 1], role: null, choice: 'base' });
    expect(back).toEqual({});
  });
});

describe('search', () => {
  const c = parseCatalog(raw)!;
  it('matches name, role, group and encodings; filters by layers and changed', () => {
    expect(searchTensors(c, 'v2', {}, { q: 'ht3' }).map((x) => x.name)).toEqual(['l0.down']);
    expect(searchTensors(c, 'v2', {}, { q: 'vision' }).map((x) => x.name)).toEqual(['vis.patch']);
    expect(searchTensors(c, 'v2', {}, { layers: [1] }).map((x) => x.name)).toEqual(['l1.norm', 'l1.q']);
    expect(searchTensors(c, 'v2', { 'l2.q': 'ht43' }, { changedOnly: true }).map((x) => x.name)).toEqual(['l2.q']);
    expect(searchTensors(c, 'v2', {}, { editableOnly: true, group: 'global' }).map((x) => x.name)).toEqual(['embed']);
  });
});

describe('save body and saved revisions', () => {
  const c = parseCatalog(raw)!;
  it('POST body: name, base, catalog digest, sorted normalized overrides; built-in names refused', () => {
    expect(buildProfileBody(c, 'v2', ' Fast attn ', { 'l2.q': 'ht43', 'l0.q': 'ht43', 'l1.norm': 'ht43' })).toEqual({
      body: { name: 'Fast attn', base: 'v2', catalog_sha256: 'c'.repeat(64), overrides: { 'l0.q': 'ht43', 'l2.q': 'ht43' } },
      problems: [],
    });
    expect(buildProfileBody(c, 'v2', 'v2', { 'l0.q': 'ht43' }).problems.join()).toMatch(/built-in/);
    expect(buildProfileBody(c, 'v2', '', {}).problems.join()).toMatch(/name/);
    expect(buildProfileBody(c, 'v2', 'x', {}).problems.join()).toMatch(/no change/);
  });
  it('parses saved revisions and marks other catalogs', () => {
    const p = parseProfiles(
      [
        { revision: 'd'.repeat(64), name: 'Fast attn', base: 'v2', catalog_sha256: 'c'.repeat(64), overrides: { 'l0.q': 'ht43' }, size: { base_bytes: 10_000, predicted_bytes: 10_100, delta_bytes: 100 }, created_at: 5 },
        { revision: 'e'.repeat(64), name: 'Old', base: 'ht43', catalog_sha256: 'f'.repeat(64), overrides: {}, created_at: 1 },
        { name: 'no rev' },
      ],
      c,
    );
    expect(p.map((x) => [x.name, x.compatible])).toEqual([['Fast attn', true], ['Old', false]]);
    expect(p[0].size?.delta).toBe(100);
  });
});

describe('format split', () => {
  it('HGN parsing ignores GGUF revisions; GGUF revisions are parsed separately', async () => {
    const { parseGgufProfiles } = await import('./profiles');
    const c = parseCatalog(raw)!;
    const list = [
      { revision: 'd'.repeat(64), name: 'HGN one', base: 'v2', catalog_sha256: 'c'.repeat(64), overrides: { 'l0.q': 'ht43' }, created_at: 2 },
      { revision: 'e'.repeat(64), format: 'gguf', name: 'GGUF one', preset: 'Q4_K_M', overrides: { 'blk.1.ffn_down.weight': 'IQ1_S' }, created_at: 3 },
    ];
    expect(parseProfiles(list, c).map((p) => p.name)).toEqual(['HGN one']);
    expect(parseGgufProfiles(list)).toEqual([{ revision: 'e'.repeat(64), name: 'GGUF one', preset: 'Q4_K_M', overrides: { 'blk.1.ffn_down.weight': 'IQ1_S' }, createdAt: 3 }]);
  });
});
