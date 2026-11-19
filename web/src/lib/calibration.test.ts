import { describe, expect, it } from 'vitest';
import { codecTransitions, MAX_FILE_BYTES, parseReportText, readFileBounded, signedBytes, topExperts, usageCoverage } from './calibration';

const usage = () => ({
  schema: 'brewery-expert-usage-v1',
  model: { name: 'qwen3-30b-a3b', sha256: 'a'.repeat(64), nested: { x: 1 } },
  runtime: { name: 'halogen' },
  tokenizer: { sha256: 'b'.repeat(64) },
  traces: [{ sha256: 'c'.repeat(64), prefill_tokens: 12, decode_tokens: 20 }],
  totals: { trace_count: 1, prefill_tokens: 12, decode_tokens: 20, observed_positions: 60, missing_positions: 4, selections: 480 },
  layers: [
    {
      layer: 1,
      prefill: { observed_positions: 12, missing_positions: 0, selections: 96, experts: [{ expert: 7, selections: 10, observed_positions: 12, selection_fraction: 0.8333, weight_samples: 10, weight_sum: 2.5, weight_mean: 0.25 }] },
      decode: { observed_positions: 16, missing_positions: 4, selections: 128, experts: [] },
    },
    { layer: 0, prefill: { observed_positions: 12, missing_positions: 0, selections: 96, experts: [] }, decode: null },
  ],
  caveats: ['Positions without routing capture are missing, not zero.'],
});

const preview = () => ({
  schema: 'brewery.profile_preview.v1',
  executable: false,
  release_eligible: false,
  source_compatibility: { status: 'metadata_only', weight_identity: 'unverified', matching_metadata: { source_format: 'hf', source_dtypes: { Bf16: 3 } } },
  limits: ['Changes only; unchanged tensors are omitted.'],
  resolved_recipe_sha256: 'd'.repeat(64),
  resolved_recipe: {
    schema_version: 1,
    base: { profile: 'v2', identity: 'x', plan_sha256: 'e'.repeat(64) },
    alternate: { profile: 'ht43', identity: 'y', plan_sha256: 'f'.repeat(64) },
    changes: [
      { filename: 'w-0001.hgn', tensor: 'blk.3.ffn_down', layer: 3, role: 'ffn_down', kind: 'override', before: { encoding: 'q4cp', shape: [4096, 1536], payload_bytes: 3_000_000 }, after: { encoding: 'ht', shape: [4096, 1536], payload_bytes: 2_000_000 } },
      { filename: 'w-0001.hgn', tensor: 'blk.3.ffn_down.scale', layer: 3, role: 'scale', kind: 'companion', before: { encoding: 'f16', shape: [2048], payload_bytes: 4096 }, after: { encoding: 'f16', shape: [4096], payload_bytes: 8192 } },
    ],
  },
  size: { base_hgn_bytes: 10_000_000, predicted_hgn_bytes: 9_004_096, delta_hgn_bytes: -995_904, files: [{ filename: 'w-0001.hgn', base_hgn_bytes: 10_000_000, predicted_hgn_bytes: 9_004_096, delta_hgn_bytes: -995_904, base_overhead_bytes: 512, predicted_overhead_bytes: 512 }] },
});

describe('report detection', () => {
  it('reads a usage report with layers sorted and missing phases kept as absent', () => {
    const r = parseReportText(JSON.stringify(usage()));
    expect(r.kind).toBe('usage');
    if (r.kind !== 'usage') return;
    expect(r.report.layers.map((l) => l.layer)).toEqual([0, 1]);
    expect(r.report.layers[0].decode).toBeNull();
    expect(r.report.layers[1].prefill?.experts[0]).toMatchObject({ expert: 7, selection_fraction: 0.8333, weight_mean: 0.25 });
    expect(r.report.identity.model).toEqual([['name', 'qwen3-30b-a3b'], ['sha256', 'a'.repeat(64)]]);
  });
  it('reads a profile preview and never reports it as runnable', () => {
    const r = parseReportText(JSON.stringify({ ...preview(), executable: true, release_eligible: true }));
    expect(r.kind).toBe('preview');
    if (r.kind !== 'preview') return;
    expect(r.report.executable).toBe(false);
    expect(r.report.release_eligible).toBe(false);
    expect(r.report.claimedRunnable).toBe(true);
    expect(r.report.changes[0].delta).toBe(-1_000_000);
    expect(r.report.changes[1]).toMatchObject({ kind: 'companion', delta: 4096, encodingChanged: false, shapeChanged: true });
    expect(r.report.size).toMatchObject({ base: 10_000_000, predicted: 9_004_096, delta: -995_904 });
    expect(r.report.compatibility).toEqual([['status', 'metadata_only'], ['weight_identity', 'unverified'], ['matching_metadata.source_format', 'hf']]);
    expect(r.report.files[0]).toMatchObject({ filename: 'w-0001.hgn', delta: -995_904, baseOverhead: 512 });
  });
  it('a change with an absent side has an unknown delta, not zero', () => {
    const p = preview();
    (p.resolved_recipe.changes[1] as Record<string, unknown>).before = null;
    const r = parseReportText(JSON.stringify(p));
    if (r.kind !== 'preview') throw new Error('expected preview');
    expect(r.report.changes[1].before).toBeNull();
    expect(r.report.changes[1].delta).toBeNull();
  });
  it('gives short errors for invalid JSON, unknown schemas and wrong shapes, without dumping content', () => {
    const bad = parseReportText('{not json');
    expect(bad).toMatchObject({ kind: 'error' });
    const other = parseReportText(JSON.stringify({ schema: 'halogen-trace-v3', tokens: Array(5).fill({ secret: 'raw' }) }));
    expect(other.kind).toBe('error');
    if (other.kind === 'error') {
      expect(other.message).toMatch(/not supported/i);
      expect(other.message).not.toMatch(/secret|raw/);
      expect(other.message.length).toBeLessThan(160);
    }
    expect(parseReportText(JSON.stringify({ ...usage(), layers: 'x' })).kind).toBe('error');
    expect(parseReportText(JSON.stringify({ ...preview(), resolved_recipe: null })).kind).toBe('error');
    expect(parseReportText('[]').kind).toBe('error');
  });
  it('keeps non-finite or absent numbers as null, not zero', () => {
    const u = usage();
    (u.layers[0].prefill!.experts[0] as Record<string, unknown>).weight_mean = 'NaN';
    const r = parseReportText(JSON.stringify(u));
    if (r.kind !== 'usage') throw new Error('expected usage');
    expect(r.report.layers[1].prefill?.experts[0].weight_mean).toBeNull();
  });
});

describe('usage derived values', () => {
  it('computes coverage from observed and missing positions only', () => {
    expect(usageCoverage({ observed_positions: 16, missing_positions: 4 })).toBe(0.8);
    expect(usageCoverage({ observed_positions: 0, missing_positions: 0 })).toBeNull();
    expect(usageCoverage({ observed_positions: null, missing_positions: 4 })).toBeNull();
  });
  it('lists top experts by selections with a stated remainder', () => {
    const experts = Array.from({ length: 12 }, (_, i) => ({ expert: i, selections: i, observed_positions: 12, selection_fraction: i / 12, weight_samples: i, weight_sum: null, weight_mean: null }));
    const t = topExperts(experts, 5);
    expect(t.shown.map((e) => e.expert)).toEqual([11, 10, 9, 8, 7]);
    expect(t.hidden).toBe(7);
  });
});

describe('bounded reading', () => {
  it('rejects a file above the cap before reading it', async () => {
    let read = false;
    const big = { name: 'x.json', size: MAX_FILE_BYTES + 1, text: async () => ((read = true), '{}') };
    const r = await readFileBounded(big);
    expect(r.kind).toBe('error');
    expect(read).toBe(false);
  });
  it('reads a small file', async () => {
    const r = await readFileBounded({ name: 'u.json', size: 10, text: async () => JSON.stringify(usage()) });
    expect(r.kind).toBe('usage');
  });
});

describe('speculation report', () => {
  const spec = () => ({
    schema: 'brewery-halogen-speculation-v1',
    producer_version: '0.18.22',
    model: 'granite-4.0-h-small',
    requests: 40,
    total_with_speculation: 32,
    missing_speculation: 8,
    methods: {
      head: { rounds: 100, drafted: 300, accepted: 210, acceptance_fraction: 0.7, accepted_per_round: 2.1 },
      lookup: { rounds: 0, drafted: 0, accepted: 0, acceptance_fraction: null, accepted_per_round: null },
      grammar: null,
    },
  });
  it('reads methods in a fixed order and keeps null methods and denominators as unknown', () => {
    const r = parseReportText(JSON.stringify(spec()));
    expect(r.kind).toBe('speculation');
    if (r.kind !== 'speculation') return;
    expect(r.report.methods.map((m) => m.name)).toEqual(['head', 'lookup', 'grammar']);
    expect(r.report.methods[0].stats).toMatchObject({ drafted: 300, accepted: 210, acceptance_fraction: 0.7 });
    expect(r.report.methods[1].stats?.acceptance_fraction).toBeNull();
    expect(r.report.methods[2].stats).toBeNull();
    expect(r.report).toMatchObject({ producerVersion: '0.18.22', model: 'granite-4.0-h-small', requests: 40, withSpeculation: 32, missingSpeculation: 8, hasByDepth: false, checkpointSha256: null });
  });
  it('does not compute an acceptance fraction the report did not state', () => {
    const p = spec();
    (p.methods.head as Record<string, unknown>).acceptance_fraction = null;
    const r = parseReportText(JSON.stringify(p));
    if (r.kind !== 'speculation') throw new Error('expected speculation');
    expect(r.report.methods[0].stats?.acceptance_fraction).toBeNull();
  });
  it('rejects a report without a methods object', () => {
    expect(parseReportText(JSON.stringify({ ...spec(), methods: [] })).kind).toBe('error');
  });
});

describe('preview summaries', () => {
  it('groups codec transitions by kind and keeps unknown deltas apart', () => {
    const r = parseReportText(JSON.stringify(preview()));
    if (r.kind !== 'preview') throw new Error('expected preview');
    expect(codecTransitions(r.report.changes)).toEqual([
      { from: 'q4cp', to: 'ht', kind: 'override', count: 1, delta: -1_000_000, unknown: 0 },
      { from: 'f16', to: 'f16', kind: 'companion', count: 1, delta: 4096, unknown: 0 },
    ]);
    expect(signedBytes(-1024)).toBe('−1.00 KiB');
    expect(signedBytes(null)).toBe('—');
  });
});
