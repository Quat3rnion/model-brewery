import { describe, expect, it } from 'vitest';
import {
  approvalState,
  attentionMatrix,
  buildTraceJob,
  checkRows,
  decodeF32,
  estimateTraceBytes,
  lensGrid,
  lineage,
  needsFullDump,
  normMatrix,
  parseIndexList,
  traceDiffQuery,
  type ScratchRevision,
  type TraceForm,
} from './weightspace';
import { hashParam, parse } from './route.svelte';

const SHA = 'a'.repeat(64);
const SHA_B = 'b'.repeat(64);

function f32(values: number[]): string {
  const bytes = new Uint8Array(new Float32Array(values).buffer);
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

const form = (over: Partial<TraceForm> = {}): TraceForm => ({
  model: 'm1',
  host: '',
  prompts: ['The capital of France is'],
  layers: '0-1',
  roles: ['l_out'],
  positions: '0-3',
  topK: 5,
  byteCapMiB: null,
  maxByteCap: 16 << 20,
  fullDump: false,
  artifact: null,
  patches: [],
  ...over,
});

describe('parseIndexList', () => {
  it('expands ranges, removes duplicates and sorts', () => {
    expect(parseIndexList('3, 0-2, 2', 4096)).toEqual({ ok: true, value: [0, 1, 2, 3] });
  });
  it('accepts an empty list only when allowed', () => {
    expect(parseIndexList('  ', 4096, true)).toEqual({ ok: true, value: [] });
    expect(parseIndexList('', 4096).ok).toBe(false);
  });
  it('rejects bad tokens, reversed ranges and values at or above the limit', () => {
    expect(parseIndexList('a', 10).ok).toBe(false);
    expect(parseIndexList('5-2', 10).ok).toBe(false);
    expect(parseIndexList('10', 10).ok).toBe(false);
    expect(parseIndexList('-1', 10).ok).toBe(false);
  });
});

describe('trace capture size', () => {
  it('needs a full dump above the controller limits', () => {
    expect(needsFullDump(64, 32, 1)).toBe(false); // 2048 selections
    expect(needsFullDump(65, 1, 1)).toBe(true);
    expect(needsFullDump(1, 129, 1)).toBe(true);
    expect(needsFullDump(32, 64, 2)).toBe(true); // 4096 selections > 2048
  });

  it('estimates F32 rows, attention rows and logits per prompt', () => {
    const e = estimateTraceBytes({ prompts: 2, layers: 2, positions: 4, roles: ['l_out', 'ffn_out'], topK: 0, hidden: 8, heads: null, keys: 4 });
    // 2 prompts * 2 layers * 4 positions * 2 roles * 8 * 4 bytes = 1024 plus manifest overhead.
    expect(e.payload).toBe(1024);
    expect(e.bytes).toBeGreaterThan(1024);
    expect(e.complete).toBe(true);
  });

  it('marks the estimate incomplete when the hidden size or head count is unknown', () => {
    expect(estimateTraceBytes({ prompts: 1, layers: 1, positions: 1, roles: ['l_out'], topK: 0, hidden: null, heads: null, keys: 1 }).complete).toBe(false);
    expect(estimateTraceBytes({ prompts: 1, layers: 1, positions: 1, roles: ['kq_soft_max'], topK: 0, hidden: 8, heads: null, keys: 1 }).complete).toBe(false);
    const attn = estimateTraceBytes({ prompts: 1, layers: 1, positions: 2, roles: ['kq_soft_max'], topK: 0, hidden: null, heads: 2, keys: 3 });
    expect(attn.payload).toBe(2 * 2 * 3 * 4);
    expect(attn.complete).toBe(true);
  });

  it('counts logit lens rows per layer and model logits per position', () => {
    const e = estimateTraceBytes({ prompts: 1, layers: 2, positions: 3, roles: [], topK: 4, hidden: null, heads: null, keys: 3 });
    expect(e.payload).toBe((2 * 3 + 3) * 4 * 16);
    expect(e.complete).toBe(true);
  });
});

describe('buildTraceJob', () => {
  it('builds a trace job with one prompt as the prompt field', () => {
    const r = buildTraceJob(form({ byteCapMiB: 4 }));
    expect(r.blockers).toEqual([]);
    expect(r.job).toEqual({
      kind: 'trace',
      input_ids: ['m1'],
      prompt: 'The capital of France is',
      options: { layers: [0, 1], roles: ['l_out'], positions: [0, 1, 2, 3], top_k: 5, byte_cap: 4 << 20 },
    });
  });

  it('sends two or more prompts in options.prompts and keeps the host', () => {
    const r = buildTraceJob(form({ prompts: ['a', 'b'], host: 'frame' }));
    expect(r.job?.prompt).toBeUndefined();
    expect(r.job?.host).toBe('frame');
    expect((r.job?.options as Record<string, unknown>).prompts).toEqual(['a', 'b']);
  });

  it('adds a saved intervention version and activation patches', () => {
    const r = buildTraceJob(
      form({
        artifact: { id: 'iv1', version: 2 },
        patches: [{ source: 'job-1', prompt: 0, layer: '3', role: 'l_out', positions: '1-2' }],
      }),
    );
    expect(r.blockers).toEqual([]);
    const o = r.job?.options as Record<string, unknown>;
    expect(o.intervention_artifact).toEqual({ id: 'iv1', version: 2 });
    expect(o.activation_patches).toEqual([{ source_job_id: 'job-1', source_prompt: 0, layer: 3, role: 'l_out', positions: [1, 2] }]);
  });

  it('lists the reasons that block the request', () => {
    const r = buildTraceJob(form({ model: '', prompts: [' '], layers: 'x', roles: [], topK: 0, byteCapMiB: 32 }));
    expect(r.job).toBeNull();
    expect(r.blockers).toContain('Select a GGUF model.');
    expect(r.blockers).toContain('Enter each prompt.');
    expect(r.blockers.some((b) => b.startsWith('Layers:'))).toBe(true);
    expect(r.blockers).toContain('The byte cap must be at most 16 MiB.');
  });

  it('requires roles or logits, and layers for roles', () => {
    expect(buildTraceJob(form({ roles: [], topK: 0 })).blockers).toContain('Select at least one role, or set top-k above 0.');
    expect(buildTraceJob(form({ layers: '' })).blockers).toContain('Select layers for the roles.');
    expect(buildTraceJob(form({ layers: '', roles: [], topK: 3 })).blockers).toEqual([]);
  });

  it('requires full dump for a large selection', () => {
    expect(buildTraceJob(form({ layers: '0-64' })).blockers).toContain('This selection is large. Select "Full dump" to send it.');
    const ok = buildTraceJob(form({ layers: '0-64', fullDump: true }));
    expect(ok.blockers).toEqual([]);
    expect((ok.job?.options as Record<string, unknown>).full_dump).toBe(true);
  });

  it('rejects patches that are not complete or that target attention', () => {
    const r = buildTraceJob(form({ patches: [{ source: '', prompt: 0, layer: '1', role: 'kq_soft_max', positions: '0' }] }));
    expect(r.blockers).toContain('Patch 1: select a source trace.');
    expect(r.blockers).toContain('Patch 1: attention probabilities are read-only.');
  });
});

describe('decodeF32 and matrices', () => {
  it('decodes little-endian F32 rows', () => {
    expect(Array.from(decodeF32(f32([1, -2, 0.5])))).toEqual([1, -2, 0.5]);
  });

  it('builds a layer by position L2 norm matrix with missing cells', () => {
    const m = normMatrix(
      [
        { layer: 0, rows: [{ position: 0, data_base64: f32([3, 4]) }], missing_positions: [1] },
        { layer: 1, rows: [{ position: 1, data_base64: f32([0, 2]) }], missing_positions: [] },
      ],
      [0, 1],
    );
    expect(m.rows).toEqual([0, 1]);
    expect(m.cols).toEqual([0, 1]);
    expect(m.values).toEqual([
      [5, null],
      [null, 2],
    ]);
    expect(m.max).toBe(5);
  });

  it('reads one head of a kq_soft_max capture as query by key', () => {
    const t = {
      rows: [
        { position: 0, shape: [2, 3], data_base64: f32([1, 0, 0, 0.5, 0.5, 0]) },
        { position: 1, shape: [2, 3], data_base64: f32([0.2, 0.8, 0, 0.1, 0.1, 0.8]) },
      ],
    };
    const h1 = attentionMatrix(t, 1);
    expect(h1?.heads).toBe(2);
    expect(h1?.rows).toEqual([0, 1]);
    expect(h1?.cols).toEqual([0, 1, 2]);
    expect(h1?.values[1].map((v) => Number(v?.toFixed(2)))).toEqual([0.1, 0.1, 0.8]);
    expect(attentionMatrix({ rows: [{ position: 0, shape: [6], data_base64: f32([1, 2, 3, 4, 5, 6]) }] }, 0)).toBeNull();
  });

  it('indexes logit lens rows by layer and position', () => {
    const g = lensGrid([
      { layer: 1, position: 2, top: [{ token_id: 9, logit: 3 }] },
      { layer: 0, position: 2, top: [{ token_id: 4, logit: 1 }] },
    ]);
    expect(g.layers).toEqual([0, 1]);
    expect(g.positions).toEqual([2]);
    expect(g.at(1, 2)?.[0].token_id).toBe(9);
    expect(g.at(1, 3)).toBeUndefined();
  });
});

const rev = (id: string, parent: string, over: Partial<ScratchRevision> = {}): ScratchRevision => ({
  id,
  kind: 'scratch',
  state: 'scratch',
  sha256: SHA,
  qualification: 'not release-qualified',
  library_model_id: null,
  provenance: {
    parent_id: parent,
    parent_sha256: SHA_B,
    revision_id: id,
    revision_sha256: SHA,
    worker: 'frame',
    method: 'direct',
    operations: [],
    tensors: [],
    tool: 'brewery-weight-edit',
    tool_version: '1',
    runtime_version: '1',
    created_at_ms: 0,
  },
  delta: { changed_tensors: 1, changed_bytes: 4, delta_bytes: 8, delta_sha256: SHA },
  ...over,
});

describe('lineage', () => {
  it('groups scratch revisions under their Library source, oldest first', () => {
    const a = rev('a', 'lib', { provenance: { ...rev('a', 'lib').provenance, method: 'copy', created_at_ms: 1 } });
    const c = rev('c', 'a', { provenance: { ...rev('c', 'a').provenance, created_at_ms: 3 } });
    const b = rev('b', 'a', { provenance: { ...rev('b', 'a').provenance, created_at_ms: 2 } });
    const d = rev('d', 'b');
    const roots = lineage([c, d, a, b]);
    expect(roots.map((r) => r.source)).toEqual(['lib']);
    expect(roots[0].nodes.map((n) => n.rev.id)).toEqual(['a']);
    expect(roots[0].nodes[0].children.map((n) => n.rev.id)).toEqual(['b', 'c']);
    expect(roots[0].nodes[0].children[0].children.map((n) => n.rev.id)).toEqual(['d']);
    expect(roots[0].nodes[0].children[0].children[0].depth).toBe(3);
  });
});

describe('checks and approval', () => {
  const bundle = {
    id: 'bundle-1',
    revision_sha256: SHA,
    parent_sha256: SHA_B,
    created_at_ms: 5,
    qualification: 'not release-qualified',
    integrity: { status: 'passed' },
    smoke: { status: 'failed', reason: 'no output' },
    behavior: { status: 'not_run', reason: 'no report' },
    job_refs: [],
    acknowledged_at_ms: 9,
  };

  it('lists each check with a tone and reason', () => {
    expect(checkRows(bundle)).toEqual([
      { name: 'Integrity', status: 'passed', label: 'Passed', tone: 'ok', reason: '' },
      { name: 'Load and generate', status: 'failed', label: 'Failed', tone: 'bad', reason: 'no output' },
      { name: 'Behavior', status: 'not_run', label: 'Not run', tone: 'muted', reason: 'no report' },
    ]);
    expect(checkRows(undefined)).toEqual([]);
  });

  it('allows approval only for a publish request with a bundle for the same hash', () => {
    expect(approvalState(rev('x', 'p', { state: 'publish_requested', check_bundle: bundle })).can).toBe(true);
    expect(approvalState(rev('x', 'p', { state: 'scratch', check_bundle: bundle })).can).toBe(false);
    expect(approvalState(rev('x', 'p', { state: 'publish_requested' })).reason).toBe('The revision has no check bundle.');
    const other = approvalState(rev('x', 'p', { state: 'publish_requested', check_bundle: { ...bundle, revision_sha256: SHA_B } }));
    expect(other.can).toBe(false);
    expect(other.reason).toBe('The check bundle is for a different hash.');
  });

  it('shows program correctness for a compiled root', () => {
    const compiled = { ...bundle, program_correctness: { status: 'passed' as const } };
    expect(checkRows(compiled).map(({ name, label }) => [name, label])).toEqual([
      ['Integrity', 'Passed'],
      ['Load and run', 'Failed'],
      ['Behavior', 'Not run'],
      ['Program correctness', 'Passed'],
    ]);
  });
});

describe('trace diff query', () => {
  it('encodes the pair, prompt and optional layer', () => {
    expect(traceDiffQuery('a', 'b', 1, '')).toBe('a=a&b=b&prompt=1');
    expect(traceDiffQuery('a', 'b', 0, '4')).toBe('a=a&b=b&prompt=0&layer=4');
  });
});

describe('routes', () => {
  it('opens the trace and edit session views', () => {
    expect(parse('#/trace').view).toBe('trace');
    expect(parse('#/trace?job=abc').view).toBe('trace');
    expect(hashParam('#/trace?job=abc', 'job')).toBe('abc');
    expect(parse('#/edits?rev=r1').view).toBe('edits');
  });
});
