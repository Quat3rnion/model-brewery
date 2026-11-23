import { describe, expect, it } from 'vitest';
import {
  pairSplit,
  classCounts,
  crosscoderHref,
  filterFeatures,
  isCrosscoderId,
  parseCrosscoder,
  scatterPoints,
  weightsUrl,
  type CrosscoderFeature,
  type CrosscoderPair,
} from './crosscoder';
import { hashParam, parse } from './route.svelte';

const ID = 'c'.repeat(64);
const side = { mse: 0.01, explained_variance: 0.9, l0: 12.5, dead_features: 3 };
const feat = (feature: number, classification: CrosscoderFeature['classification'], parent_norm: number, child_norm: number, decoder_cosine: number | null): CrosscoderFeature => ({
  feature,
  classification,
  parent_norm,
  child_norm,
  decoder_cosine,
  child_norm_ratio: parent_norm + child_norm === 0 ? null : child_norm / (parent_norm + child_norm),
});
const features: CrosscoderFeature[] = [
  feat(0, 'parent_unique', 1, 0.05, 0.2),
  feat(1, 'child_unique', 0.04, 1.2, 0.1),
  feat(2, 'changed', 1, 1.1, 0.3),
  feat(3, 'shared', 1, 1, 0.99),
  feat(4, 'dead', 0, 0, null),
  feat(12, 'changed', 1, 0.9, -0.2),
  feat(21, 'parent_unique', 2, 0, null),
];
const report = (over: Record<string, unknown> = {}) => ({
  id: ID,
  schema_version: 1,
  qualification: 'not_release_qualified',
  parent: { id: 'model-parent', sha256: 'a'.repeat(64) },
  child: { id: 'rev-child', sha256: 'b'.repeat(64) },
  layer: 6,
  role: 'l_out',
  metrics: { parent: side, child: { ...side, mse: 0.02 } },
  features,
  ...over,
});

describe('crosscoder ids and links', () => {
  it('accepts only 64 lowercase hex characters', () => {
    expect(isCrosscoderId(ID)).toBe(true);
    expect(isCrosscoderId(` ${ID}`)).toBe(false);
    expect(isCrosscoderId('C'.repeat(64))).toBe(false);
    expect(isCrosscoderId('c'.repeat(63))).toBe(false);
    expect(isCrosscoderId('../x')).toBe(false);
  });
  it('builds the report route under Edits and the weights URL', () => {
    expect(crosscoderHref(ID, 'rev-child')).toBe(`#/edits/crosscoder?id=${ID}&rev=rev-child`);
    expect(crosscoderHref(ID)).toBe(`#/edits/crosscoder?id=${ID}`);
    expect(weightsUrl(ID)).toBe(`/api/crosscoders/${ID}/safetensors`);
  });
  it('parses the nested route', () => {
    const hash = crosscoderHref(ID, 'rev-child');
    expect(parse(hash)).toEqual({ view: 'crosscoder', jobId: null });
    expect(hashParam(hash, 'id')).toBe(ID);
    expect(parse('#/edits')).toEqual({ view: 'edits', jobId: null });
  });
});

describe('parseCrosscoder', () => {
  it('accepts a schema version 1 report', () => {
    const r = parseCrosscoder(report());
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.features).toHaveLength(features.length);
      expect(r.value.metrics.child.mse).toBe(0.02);
    }
  });
  it('refuses another schema version', () => {
    const r = parseCrosscoder(report({ schema_version: 2 }));
    expect(r).toEqual({ ok: false, error: 'This view reads crosscoder schema version 1. The report has version 2.' });
  });
  it('refuses missing metrics or a bad feature class', () => {
    expect(parseCrosscoder(report({ metrics: { parent: side } })).ok).toBe(false);
    expect(parseCrosscoder(report({ features: [{ ...features[0], classification: 'other' }] })).ok).toBe(false);
    expect(parseCrosscoder(null).ok).toBe(false);
  });
  it('keeps null ratio and cosine', () => {
    const r = parseCrosscoder(report());
    expect(r.ok && r.value.features[4].child_norm_ratio).toBe(null);
    expect(r.ok && r.value.features[4].decoder_cosine).toBe(null);
  });
});

describe('classCounts', () => {
  it('counts every class, also the empty ones', () => {
    expect(classCounts(features)).toEqual({ parent_unique: 2, child_unique: 1, changed: 2, shared: 1, dead: 1 });
    expect(classCounts([])).toEqual({ parent_unique: 0, child_unique: 0, changed: 0, shared: 0, dead: 0 });
  });
});

describe('filterFeatures', () => {
  it('keeps one class and sorts by strength by default', () => {
    expect(filterFeatures(features, 'parent_unique', '', 'strength').map((f) => f.feature)).toEqual([21, 0]);
  });
  it('sorts changed features by lowest cosine', () => {
    expect(filterFeatures(features, 'changed', '', 'cosine').map((f) => f.feature)).toEqual([12, 2]);
  });
  it('sorts by distance of the child share from one half; null is last', () => {
    expect(filterFeatures(features, 'parent_unique', '', 'ratio').map((f) => f.feature)).toEqual([21, 0]);
    expect(filterFeatures([feat(5, 'dead', 0, 0, null), feat(6, 'dead', 1, 2, null), feat(7, 'dead', 1, 1, null)], 'dead', '', 'ratio').map((f) => f.feature)).toEqual([6, 7, 5]);
  });
  it('keeps every class for all', () => {
    expect(filterFeatures(features, 'all', '', 'feature').map((f) => f.feature)).toEqual([0, 1, 2, 3, 4, 12, 21]);
  });
  it('finds a feature number by prefix', () => {
    expect(filterFeatures(features, 'changed', '1', 'feature').map((f) => f.feature)).toEqual([12]);
    expect(filterFeatures(features, 'changed', ' 2 ', 'feature').map((f) => f.feature)).toEqual([2]);
    expect(filterFeatures(features, 'changed', 'x', 'feature')).toEqual([]);
  });
});

describe('pairSplit', () => {
  it('counts train and held-out pairs', () => {
    const pair = (split: CrosscoderPair['split']) => ({ parent_job_id: 'p', child_job_id: 'c', split });
    expect(pairSplit([pair('train'), pair('train'), pair('heldout')])).toEqual({ total: 3, train: 2, heldout: 1 });
    expect(pairSplit(undefined)).toBe(null);
  });
});

describe('parseCrosscoder optional fields', () => {
  it('reads width as activation size and pairs as an array', () => {
    const r = parseCrosscoder(report({ width: 896, pairs: [{ parent_job_id: 'p', child_job_id: 'c', split: 'heldout' }] }));
    expect(r.ok && r.value.width).toBe(896);
    expect(r.ok && r.value.pairs).toEqual([{ parent_job_id: 'p', child_job_id: 'c', split: 'heldout' }]);
    const bad = parseCrosscoder(report({ pairs: 4096 }));
    expect(bad.ok && bad.value.pairs).toBe(undefined);
  });
});

describe('scatterPoints', () => {
  it('plots the stored child share against cosine and omits null values', () => {
    const s = scatterPoints(features);
    expect(s.omitted).toBe(2);
    expect(s.points.map((p) => p.feature)).toEqual([0, 1, 2, 3, 12]);
    const p3 = s.points.find((p) => p.feature === 3)!;
    expect(p3.x).toBe(0.5);
    expect(p3.y).toBe(0.99);
  });
});
