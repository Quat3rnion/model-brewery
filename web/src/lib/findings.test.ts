import { describe, expect, it } from 'vitest';
import {
  agentSessionHref,
  apiHref,
  compareRows,
  findingHref,
  findingIdFromHash,
  parseFinding,
  parseFindingSummaries,
  viewHref,
  type Finding,
} from './findings';
import { parse, VIEWS } from './route.svelte';

const SHA = 'a'.repeat(64);
const summary = {
  id: 'fnd-2',
  version: 2,
  hypothesis: 'Layer 6 MLP stores the capital fact.',
  parent_id: 'fnd-1',
  created_at_ms: 1_790_000_000_000,
  state: 'ready',
  model: { id: 'model-a', sha256: SHA },
};
const detail = {
  ...summary,
  experiments: [
    { spec: { kind: 'trace', layers: [6] }, artifact: { id: 'job-1', job_id: 'job-1', sha256: SHA, api_url: '/api/jobs/job-1', view_url: '#/jobs/job-1' }, metrics: [{ path: 'logit_diff', value: 2.5 }] },
    { spec: { kind: 'patch' }, artifact: { id: 'art-2', sha256: SHA, api_url: '/api/x/art-2', view_url: 'javascript:alert(1)' }, metrics: [{ path: 'recovered', value: 0.8 }, { path: 'kl', value: 0.01 }] },
  ],
  verdict: { status: 'supported', reasoning: 'Patching layer 6 recovers 80% of the logit difference.' },
  provenance: { agent: { kind: 'builtin', id: 'session-9' }, tool_version: '0.9.0' },
  comparison: [
    { experiment: 1, path: 'logit_diff', original: 2.5, replay: 2.5 },
    { experiment: 2, path: 'recovered', original: 0.8, replay: 0.6 },
  ],
};

describe('findings routes', () => {
  it('adds Findings to the sections and parses list and detail hashes', () => {
    expect(VIEWS.map((v) => v.id)).toContain('findings');
    expect(VIEWS.find((v) => v.id === 'findings')?.label).toBe('Findings');
    expect(parse('#/findings')).toEqual({ view: 'findings', jobId: null });
    expect(parse(findingHref('a/b c'))).toEqual({ view: 'findings', jobId: null });
    expect(findingHref('a/b c')).toBe('#/findings/a%2Fb%20c');
    expect(findingIdFromHash('#/findings/a%2Fb%20c')).toBe('a/b c');
    expect(findingIdFromHash('#/findings')).toBeNull();
    expect(findingIdFromHash('#/findings/%E0%A4%A')).toBeNull();
  });
});

describe('parseFindingSummaries', () => {
  it('accepts the list and rejects bad rows', () => {
    const r = parseFindingSummaries([summary, { ...summary, id: 'fnd-1', version: 1, parent_id: null }]);
    expect(r.ok && r.value.map((f) => f.id)).toEqual(['fnd-2', 'fnd-1']);
    expect(parseFindingSummaries([{ ...summary, state: 'other' }]).ok).toBe(false);
    expect(parseFindingSummaries([{ ...summary, model: null }]).ok).toBe(false);
    expect(parseFindingSummaries({}).ok).toBe(false);
  });
});

describe('parseFinding', () => {
  it('reads a ready finding with evidence, verdict, provenance and comparison', () => {
    const r = parseFinding(detail);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.value.experiments[0].spec.kind).toBe('trace');
    expect(r.value.verdict?.status).toBe('supported');
    expect(r.value.provenance?.agent).toEqual({ kind: 'builtin', id: 'session-9' });
    expect(r.value.comparison).toHaveLength(2);
  });
  it('accepts a running finding without verdict or comparison, and a failed one with an error', () => {
    const running = parseFinding({ ...summary, state: 'running', experiments: [], verdict: null, provenance: detail.provenance });
    expect(running.ok && running.value.verdict).toBeNull();
    expect(running.ok && running.value.comparison).toEqual([]);
    const failed = parseFinding({ ...summary, state: 'failed', experiments: [], error: 'trace job failed', provenance: detail.provenance });
    expect(failed.ok && failed.value.error).toBe('trace job failed');
  });
  it('rejects bad verdicts and metrics', () => {
    expect(parseFinding({ ...detail, verdict: { status: 'maybe', reasoning: '' } }).ok).toBe(false);
    expect(parseFinding({ ...detail, experiments: [{ ...detail.experiments[0], metrics: [{ path: 'x', value: 'big' }] }] }).ok).toBe(false);
  });
});

describe('links', () => {
  it('links only hash views and controller API paths', () => {
    expect(viewHref('#/jobs/job-1')).toBe('#/jobs/job-1');
    expect(viewHref('javascript:alert(1)')).toBeNull();
    expect(viewHref('https://example.com')).toBeNull();
    expect(apiHref('/api/jobs/job-1')).toBe('/api/jobs/job-1');
    expect(apiHref('//evil/api')).toBeNull();
  });
  it('links built-in agent sessions to the Agent view only', () => {
    expect(agentSessionHref({ kind: 'builtin', id: 's 1' })).toBe('#/agent?session=s%201');
    expect(agentSessionHref({ kind: 'external_mcp', id: 'client' })).toBeNull();
  });
});

describe('compareRows', () => {
  it('labels each row with its experiment and marks changed values', () => {
    const f = (parseFinding(detail) as { ok: true; value: Finding }).value;
    const rows = compareRows(f.comparison, f.experiments);
    expect(rows.map((r) => [r.experiment, r.path, r.changed])).toEqual([
      ['1. trace', 'logit_diff', false],
      ['2. patch', 'recovered', true],
    ]);
    expect(rows[1].delta).toBeCloseTo(-0.2);
    expect(rows[0].delta).toBe(0);
  });
  it('keeps an unknown experiment index visible', () => {
    expect(compareRows([{ experiment: 9, path: 'p', original: 1, replay: 1 }], [])[0].experiment).toBe('9');
  });
});
