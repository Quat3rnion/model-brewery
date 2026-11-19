import { describe, expect, it } from 'vitest';
import { COMPONENT_ROLES, componentBlockers, defaultText, readComponents, roleRows } from './components';
import type { Host, Model } from './types';

const hf: Model = { id: 'src', host: 'frame', path: '/m/ckpt', kind: 'hf' };
const gguf: Model = { id: 'g', host: 'frame', path: '/m/a.gguf', kind: 'gguf' };
const hgn: Model = { id: 'h', host: 'frame', path: '/m/b', kind: 'hgn' };

describe('roles', () => {
  it('has plain labels for the four existing kinds', () => {
    expect(COMPONENT_ROLES.map((r) => [r.kind, r.label])).toEqual([
      ['assets', 'Tokenizer and config files'],
      ['vision', 'Vision tower'],
      ['ngram', 'Engram (n-gram lookup)'],
      ['mtp', 'MTP draft head'],
    ]);
  });
  it('default text never claims a component exists', () => {
    expect(defaultText(hf, 'vision')).toBe('From the checkpoint, if it has one');
    expect(defaultText(gguf, 'assets')).toMatch(/Required for GGUF/);
    expect(defaultText(gguf, 'vision')).toBe('None unless you add one');
    expect(defaultText(hgn, 'mtp')).toBe('From the bundle, if it has one');
    expect(defaultText(undefined, 'mtp')).toBe('Select a source first');
    for (const m of [hf, gguf, hgn]) for (const r of COMPONENT_ROLES) expect(defaultText(m, r.kind)).not.toMatch(/detected|found|included/i);
  });
});

describe('host gate', () => {
  const on: Host = { id: 'frame', capabilities: { convert_components: true } };
  const off: Host = { id: 'frame', capabilities: { convert: true }, reasons: { convert_components: 'Pinned qualification worker' } };
  it('blocks explicit replacements of an HF source on a host without convert_components', () => {
    expect(componentBlockers(hf, { assets: '', ngram: '', vision: 'v', mtp: '' }, off)).toEqual(['Pinned qualification worker']);
    expect(componentBlockers(hf, { assets: '', ngram: '', vision: 'v', mtp: '' }, on)).toEqual([]);
    expect(componentBlockers(hf, { assets: '', ngram: '', vision: '', mtp: '' }, off)).toEqual([]);
    expect(componentBlockers(gguf, { assets: 'a', ngram: '', vision: '', mtp: '' }, off)).toEqual([]);
  });
  it('without a reason it names the host and does not assume support', () => {
    expect(componentBlockers(hf, { assets: '', ngram: 'n', vision: '', mtp: '' }, { id: 'gpu2' })[0]).toMatch(/gpu2.*replacement/);
  });
});

describe('result evidence', () => {
  it('reads result.components and keeps only stated sources', () => {
    const c = readComponents({ components: { vision: { source: 'input', input_id: 'v1', format: 'safetensors', sha256: 'a'.repeat(64) }, mtp: { source: 'embedded' }, ngram: { source: 'none' }, assets: { source: 'guess' } } })!;
    expect(c.vision).toMatchObject({ source: 'input', inputId: 'v1', format: 'safetensors' });
    expect(c.mtp?.source).toBe('embedded');
    expect(c.ngram?.source).toBe('none');
    expect(c.assets).toBeUndefined();
    expect(readComponents({ report: {} })).toBeNull();
  });
  it('rows: requested vs reported; legacy results say not reported', () => {
    const rows = roleRows({ assets: '', ngram: '', vision: 'v1', mtp: '' }, readComponents({ components: { vision: { source: 'input', input_id: 'v1' }, mtp: { source: 'embedded' } } }));
    expect(rows.find((r) => r.kind === 'vision')).toMatchObject({ requested: 'v1', reported: 'input', match: true });
    expect(rows.find((r) => r.kind === 'mtp')).toMatchObject({ requested: null, reported: 'embedded' });
    expect(rows.find((r) => r.kind === 'assets')).toMatchObject({ reported: null });
    const legacy = roleRows({ assets: '', ngram: '', vision: 'v1', mtp: '' }, null);
    expect(legacy.every((r) => r.reported === null)).toBe(true);
    const mismatch = roleRows({ assets: '', ngram: '', vision: 'v1', mtp: '' }, readComponents({ components: { vision: { source: 'input', input_id: 'other' } } }));
    expect(mismatch.find((r) => r.kind === 'vision')?.match).toBe(false);
  });
});

describe('carry selection through the workflow', () => {
  it('recovers replacements from input_ids and 0-based option indexes', async () => {
    const { selectionFrom } = await import('./components');
    expect(selectionFrom(['src', 'a', 'v'], { assets: 1, vision: 2, profile: 'ht43' })).toEqual({ assets: 'a', vision: 'v', ngram: '', mtp: '' });
    expect(selectionFrom(['src'], { vision: 4 })).toEqual({ assets: '', vision: '', ngram: '', mtp: '' });
  });
});
