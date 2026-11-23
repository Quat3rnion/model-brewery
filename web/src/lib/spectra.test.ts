import { describe, expect, it } from 'vitest';
import {
  METRICS,
  changed,
  layerSeries,
  linePath,
  parseSpectra,
  spectraBlocker,
  spectraRequest,
  spectrumPoints,
  tensorGroup,
  tensorGroups,
  ticks,
} from './spectra';

const side = (er: number | null, sr: number | null, alpha: number | null, sv: number[] | null = [4, 2, 1]) => ({
  dtype: 'BF16', shape: [4, 3], method: 'exact', singular_values: sv, effective_rank: er, stable_rank: sr, alpha,
});

const raw = {
  kind: 'weight_spectra',
  qualification: 'not release-qualified',
  source: { id: 'rev-1', sha256: 'a'.repeat(64), format: 'safetensors' },
  reference: { id: 'parent-1', sha256: 'b'.repeat(64), format: 'safetensors' },
  tensors: [
    { name: 'model.layers.1.mlp.down_proj.weight', layer: 1, source: side(10, 4, 3), reference: side(10, 4, 3), delta: { effective_rank: 0, stable_rank: 0, alpha: 0 } },
    { name: 'model.layers.0.mlp.down_proj.weight', layer: 0, source: side(8, 2, 4), reference: side(10, 3, 3), delta: { effective_rank: -2, stable_rank: -1, alpha: 1, sigma_max: 0.5 } },
    { name: 'model.layers.0.self_attn.q_proj.weight', layer: 0, source: side(12, 6, 2), reference: side(12, 6, 2), delta: { effective_rank: 0, stable_rank: 0, alpha: 0 } },
    { name: 'model.layers.0.input_layernorm.weight', layer: 0, source: { ...side(null, null, null, null), shape: [4] }, status: 'unsupported_1d' },
    { name: 'model.embed_tokens.weight', layer: null, source: side(30, 9, 2.5) },
    { name: 42 },
  ],
};

describe('parseSpectra', () => {
  it('keeps valid tensors and rejects other job results', () => {
    const r = parseSpectra(raw)!;
    expect(r.tensors).toHaveLength(5);
    expect(r.reference?.id).toBe('parent-1');
    expect(r.tensors[3].source?.effective_rank).toBeNull();
    expect(r.tensors[3].source?.singular_values).toEqual([]);
    expect(parseSpectra({ kind: 'trace' })).toBeNull();
    expect(parseSpectra(null)).toBeNull();
  });
});

describe('tensor groups', () => {
  it('drops the layer prefix', () => {
    expect(tensorGroup({ name: 'model.layers.12.mlp.up_proj.weight', layer: 12 })).toBe('mlp.up_proj.weight');
    expect(tensorGroup({ name: 'blk.3.ffn_down.weight', layer: 3 })).toBe('ffn_down.weight');
    expect(tensorGroup({ name: 'lm_head.weight', layer: null })).toBe('lm_head.weight');
    expect(tensorGroups(parseSpectra(raw)!)).toEqual(['input_layernorm.weight', 'mlp.down_proj.weight', 'self_attn.q_proj.weight']);
  });
});

describe('layerSeries', () => {
  it('averages finite values per layer in layer order and skips layerless tensors', () => {
    const s = layerSeries(parseSpectra(raw)!, 'effective_rank');
    expect(s.map((p) => p.layer)).toEqual([0, 1]);
    expect(s[0]).toMatchObject({ source: 10, reference: 11, delta: -1, tensors: 3 });
    expect(s[1]).toMatchObject({ source: 10, reference: 10, delta: 0 });
  });
  it('filters by tensor group', () => {
    const s = layerSeries(parseSpectra(raw)!, 'alpha', 'mlp.down_proj.weight');
    expect(s.map((p) => [p.layer, p.source, p.reference, p.delta])).toEqual([[0, 4, 3, 1], [1, 3, 3, 0]]);
  });
  it('computes the delta when the result omits it and keeps nulls for missing data', () => {
    const r = parseSpectra({ ...raw, reference: undefined, tensors: [
      { name: 'blk.0.a.weight', layer: 0, source: side(5, 2, 2), reference: side(4, 2, 2) },
      { name: 'blk.1.a.weight', layer: 1, source: side(5, 2, 2) },
    ] })!;
    const s = layerSeries(r, 'effective_rank');
    expect(s[0].delta).toBe(1);
    expect(s[1]).toMatchObject({ reference: null, delta: null });
  });
  it('does not invent a delta when the controller reports a shape mismatch', () => {
    const r = parseSpectra({ ...raw, tensors: [{
      name: 'blk.0.a.weight', layer: 0, source: side(5, 2, 2),
      reference: { ...side(4, 2, 2), shape: [3, 4] }, status: 'shape_mismatch', delta: null,
    }] })!;
    expect(layerSeries(r, 'effective_rank')[0]).toMatchObject({ source: 5, reference: 4, delta: null });
  });
  it('lists the three metrics with labels', () => {
    expect(METRICS.map((m) => m.key)).toEqual(['effective_rank', 'stable_rank', 'alpha']);
  });
});

describe('changed', () => {
  it('flags a relative change of 1% or more', () => {
    expect(changed(0.2, 10)).toBe(true);
    expect(changed(0.05, 10)).toBe(false);
    expect(changed(null, 10)).toBe(false);
    expect(changed(1e-9, 0)).toBe(false);
    expect(changed(0.5, 0)).toBe(true);
  });
});

describe('chart geometry', () => {
  it('breaks the line at missing values', () => {
    expect(linePath([[0, 1], [1, null], [2, 3], [3, 4]], (x) => x * 10, (y) => y)).toBe('M0,1 M20,3 L30,4');
  });
  it('makes round ticks that cover the range', () => {
    expect(ticks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(ticks(3, 3, 4)).toEqual([3]);
  });
  it('decimates a long spectrum and keeps the first and last values', () => {
    const sv = Array.from({ length: 1000 }, (_, i) => 1000 - i);
    const p = spectrumPoints(sv, 100);
    expect(p.length).toBeLessThanOrEqual(101);
    expect(p[0]).toEqual([1, 1000]);
    expect(p.at(-1)).toEqual([1000, 1]);
    expect(spectrumPoints([2, 0, -1, 1])).toEqual([[1, 2], [4, 1]]);
  });
});

describe('spectraRequest', () => {
  it('omits empty optional fields', () => {
    expect(spectraRequest('rev-1', { compare: '', representation: 'safetensors', compareRepresentation: 'safetensors', layer: '', tensor: ' ' }))
      .toEqual({ id: 'rev-1' });
    expect(spectraRequest('rev-1', { compare: 'm-2', representation: 'gguf', compareRepresentation: 'gguf', layer: ' 7 ', tensor: 'blk.7.ffn_down.weight' }))
      .toEqual({ id: 'rev-1', compare_id: 'm-2', representation: 'gguf', compare_representation: 'gguf', layer: 7, tensor: 'blk.7.ffn_down.weight' });
  });
  it('blocks a layer that is not a whole number', () => {
    expect(spectraBlocker({ compare: '', representation: 'safetensors', compareRepresentation: 'safetensors', layer: '1.5', tensor: '' })).toBe('The layer must be a whole number.');
    expect(spectraBlocker({ compare: '', representation: 'safetensors', compareRepresentation: 'safetensors', layer: '', tensor: '' })).toBe('');
  });
});
