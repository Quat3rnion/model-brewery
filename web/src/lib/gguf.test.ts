import { describe, expect, it } from 'vitest';
import {
  buildGgufConversion,
  buildGgufProfileBody,
  ggufTensorsFromResult,
  imatrixRequired,
  layerOf,
  parseGgufCatalog,
  rangeOverrides,
  tensorBytes,
} from './gguf';

const raw = {
  presets: [{ id: 'Q4_K_M', requires_imatrix: false }, { id: 'IQ2_XXS', requires_imatrix: true }, { id: 'IQ1_XXS', requires_imatrix: true }, { bad: 1 }],
  tensor_types: [
    { id: 'BF16', block_size: 1, block_bytes: 2, requires_imatrix: false },
    { id: 'Q4_K', block_size: 256, block_bytes: 144, requires_imatrix: false },
    { id: 'IQ1_S', block_size: 256, block_bytes: 50, requires_imatrix: true },
    { id: 'IQ1_M', block_size: 256, block_bytes: 56, requires_imatrix: true },
    { id: 'IQ2_XXS', block_size: 256, block_bytes: 66, requires_imatrix: true },
    { id: 'IQ1_XXS', block_size: 256, block_bytes: 40, requires_imatrix: true },
  ],
};

describe('catalog', () => {
  it('keeps presets and real tensor types; IQ1_XXS is never offered', () => {
    const c = parseGgufCatalog(raw)!;
    expect(c.presets.map((p) => p.id)).toEqual(['Q4_K_M', 'IQ2_XXS']);
    expect(c.types.map((t) => t.id)).toEqual(['BF16', 'Q4_K', 'IQ1_S', 'IQ1_M', 'IQ2_XXS']);
    expect(parseGgufCatalog({})).toBeNull();
  });
});

describe('source tensors', () => {
  const result = { report: { gguf_tensors: [{ name: 'blk.0.ffn_down.weight', shape: [4096, 1536], current_type: 'BF16' }, { name: 'token_embd.weight', shape: [4096, 151936], current_type: 'BF16' }, { name: 'blk.12.attn_q.weight', shape: [4096, 4096] }, { bad: true }] } };
  it('reads gguf_tensors from an inspect or plan result', () => {
    const t = ggufTensorsFromResult(result)!;
    expect(t.map((x) => x.name)).toEqual(['blk.0.ffn_down.weight', 'blk.12.attn_q.weight', 'token_embd.weight']);
    expect(t[0]).toMatchObject({ layer: 0, role: 'ffn_down.weight', currentType: 'BF16' });
    expect(t[2]).toMatchObject({ layer: null, role: 'token_embd.weight' });
    expect(ggufTensorsFromResult({ report: {} })).toBeNull();
  });
  it('layer comes only from llama.cpp blk.N names', () => {
    expect(layerOf('blk.7.attn_k.weight')).toBe(7);
    expect(layerOf('model.layers.7.self_attn.k_proj.weight')).toBeNull();
  });
  it('bytes from catalog block metrics only when the shape divides; otherwise unknown', () => {
    const c = parseGgufCatalog(raw)!;
    expect(tensorBytes(c, 'Q4_K', [4096, 1536])).toBe((4096 * 1536 / 256) * 144);
    expect(tensorBytes(c, 'Q4_K', [100])).toBeNull();
    expect(tensorBytes(c, 'Q4_K', null)).toBeNull();
    expect(tensorBytes(c, 'UNKNOWN', [256])).toBeNull();
  });
  it('range overrides expand to exact names in the selected layers and role', () => {
    const t = ggufTensorsFromResult({ report: { gguf_tensors: [0, 1, 2, 3].flatMap((l) => [{ name: `blk.${l}.ffn_down.weight`, shape: [256] }, { name: `blk.${l}.attn_q.weight`, shape: [256] }]) } })!;
    expect(Object.keys(rangeOverrides(t, {}, { from: 1, to: 2, role: 'ffn_down.weight', type: 'IQ1_S' }))).toEqual(['blk.1.ffn_down.weight', 'blk.2.ffn_down.weight']);
    expect(Object.keys(rangeOverrides(t, { 'blk.0.attn_q.weight': 'Q4_K' }, { from: 0, to: 0, role: null, type: null }))).toEqual([]);
  });
});

describe('imatrix and bodies', () => {
  const c = parseGgufCatalog(raw)!;
  it('imatrix is required by the preset or by any override type', () => {
    expect(imatrixRequired(c, 'Q4_K_M', {})).toBe(false);
    expect(imatrixRequired(c, 'IQ2_XXS', {})).toBe(true);
    expect(imatrixRequired(c, 'Q4_K_M', { 'blk.0.ffn_down.weight': 'IQ1_M' })).toBe(true);
  });
  it('profile body: format gguf, preset, exact-name overrides sorted; no imatrix id', () => {
    const r = buildGgufProfileBody(c, 'Tight FFN', 'Q4_K_M', { 'blk.2.ffn_down.weight': 'IQ1_S', 'blk.1.ffn_down.weight': 'BF16', bad: 'IQ1_XXS' });
    expect(r.problems).toEqual(['bad: IQ1_XXS is not a supported tensor type.']);
    const ok = buildGgufProfileBody(c, 'Tight FFN', 'Q4_K_M', { 'blk.2.ffn_down.weight': 'IQ1_S', 'blk.1.ffn_down.weight': 'BF16' });
    expect(ok.body).toEqual({ format: 'gguf', name: 'Tight FFN', preset: 'Q4_K_M', overrides: { 'blk.1.ffn_down.weight': 'BF16', 'blk.2.ffn_down.weight': 'IQ1_S' } });
    expect(buildGgufProfileBody(c, 'x', 'NOPE', { a: 'BF16' }).problems.join()).toMatch(/preset/);
  });
  it('conversion body: preset or revision, imatrix as second input when required', () => {
    expect(buildGgufConversion(c, { source: 'src', preset: 'Q4_K_M', revision: null, imatrix: '', threads: 8, host: '' }).body).toEqual({ format: 'gguf', input_ids: ['src'], options: { preset: 'Q4_K_M', threads: 8 } });
    expect(buildGgufConversion(c, { source: 'src', preset: 'IQ2_XXS', revision: null, imatrix: '', threads: null, host: '' }).problems.join()).toMatch(/imatrix/);
    const rev = 'a'.repeat(64);
    expect(buildGgufConversion(c, { source: 'src', preset: '', revision: { revision: rev, needsImatrix: true }, imatrix: 'im1', threads: null, host: 'frame' }).body).toEqual({ format: 'gguf', host: 'frame', input_ids: ['src', 'im1'], profile_revision: rev });
  });
  it('requires assets for standalone SafeTensors before an optional importance matrix', () => {
    const form={source:'weights', sourceKind:'safetensors', assets:'', preset:'Q4_K_M',revision:null,imatrix:'',threads:null,host:''};
    expect(buildGgufConversion(c,form).body).toBeNull();
    expect(buildGgufConversion(c,{...form,assets:'cfg'}).body?.input_ids).toEqual(['weights','cfg']);
    expect(buildGgufConversion(c,{...form,assets:'cfg',preset:'IQ2_XXS',imatrix:'importance'}).body?.input_ids).toEqual(['weights','cfg','importance']);
    expect(buildGgufConversion(c,{...form,sourceKind:'hf',assets:'cfg'}).body?.input_ids).toEqual(['weights']);
  });
});

describe('route', () => {
  it('#/profiles?format=gguf opens the profiles view', async () => {
    const { parse } = await import('./route.svelte');
    expect(parse('#/profiles?format=gguf').view).toBe('profiles');
    expect(parse('#/profiles').view).toBe('profiles');
  });
});
