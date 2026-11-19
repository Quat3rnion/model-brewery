import { describe, expect, it } from 'vitest';
import {
  MAX_COORDINATES,
  MAX_EDITS,
  appliesOf,
  applyJob,
  checkEdits,
  exact,
  jobsFor,
  parseCoordinates,
  parseEditLines,
  parseIndex,
  parseMaxUpdate,
  parseTensor,
  parseValue,
  percent,
  proposeJob,
  readApplied,
  readInspect,
  readProposal,
  readTuneReport,
  relChange,
  rounded,
  sentence,
  tuneJob,
  PROPOSAL_KINDS,
  type EditRow,
} from './weights';
import type { JobSummary } from './types';
import { hashParam, parse } from './route.svelte';

const SHA = 'a'.repeat(64);
const row = (over: Partial<EditRow> = {}): EditRow => ({ key: '1', tensor: 'blk.0.attn_q.weight', index: '0, 1', op: 'set', value: '0.5', ...over });

describe('input parsing', () => {
  it('accepts tensor names the controller accepts', () => {
    expect(parseTensor(' blk.0.ffn_up.weight ')).toEqual({ ok: true, value: 'blk.0.ffn_up.weight' });
    expect(parseTensor('').ok).toBe(false);
    expect(parseTensor('a b').ok).toBe(false);
    expect(parseTensor('x'.repeat(128)).ok).toBe(false);
  });

  it('parses comma, space and bracket coordinates', () => {
    expect(parseIndex('3, 17')).toEqual({ ok: true, value: [3, 17] });
    expect(parseIndex('[0,1,2]')).toEqual({ ok: true, value: [0, 1, 2] });
    expect(parseIndex('4 5')).toEqual({ ok: true, value: [4, 5] });
    expect(parseIndex('-1, 2').ok).toBe(false);
    expect(parseIndex('1.5').ok).toBe(false);
    expect(parseIndex('1,,2').ok).toBe(false);
    expect(parseIndex('0,0,0,0,0').ok).toBe(false);
    expect(parseIndex('4294967296').ok).toBe(false);
    expect(parseIndex('1', 2).ok).toBe(false);
  });

  it('keeps exact decimal digits and refuses non-finite values', () => {
    expect(parseValue('-0.125')).toEqual({ ok: true, value: -0.125 });
    expect(parseValue('3e-4')).toEqual({ ok: true, value: 3e-4 });
    expect(parseValue('.5')).toEqual({ ok: true, value: 0.5 });
    for (const t of ['', 'NaN', 'Infinity', '1e999', '0x10', '1,5']) expect(parseValue(t).ok).toBe(false);
    expect(parseMaxUpdate('0').ok).toBe(false);
    expect(parseMaxUpdate('-1').ok).toBe(false);
    expect(parseMaxUpdate('0.01')).toEqual({ ok: true, value: 0.01 });
  });
});

describe('manual edits', () => {
  it('builds edits for valid rows', () => {
    const r = checkEdits([row(), row({ key: '2', index: '0,2', op: 'delta', value: '-1e-3' })]);
    expect(r.blocker).toBeNull();
    expect(r.edits).toEqual([
      { tensor: 'blk.0.attn_q.weight', index: [0, 1], op: 'set', value: 0.5 },
      { tensor: 'blk.0.attn_q.weight', index: [0, 2], op: 'delta', value: -0.001 },
    ]);
  });

  it('marks duplicates, zero deltas and bad fields per row', () => {
    const r = checkEdits([row(), row({ key: '2', index: '[0,1]' }), row({ key: '3', index: '1,1', op: 'delta', value: '0' }), row({ key: '4', tensor: '' })]);
    expect(r.edits).toBeNull();
    expect(r.errors[1].index).toMatch(/already/);
    expect(r.errors[2].value).toMatch(/delta of 0/);
    expect(r.errors[3].tensor).toBeDefined();
  });

  it('enforces the edit count limit', () => {
    expect(checkEdits([]).blocker).toMatch(/at least one/);
    const many = Array.from({ length: MAX_EDITS + 1 }, (_, i) => row({ key: `${i}`, index: `0, ${i}` }));
    expect(checkEdits(many).blocker).toMatch(/at most 256/);
  });
});

describe('bulk edit lines', () => {
  let n = 0;
  const key = () => `k${n++}`;
  it('parses lines, skips comments and lower-cases the operation', () => {
    const r = parseEditLines('# header\nblk.0.w 3,17 delta -0.002\n\nblk.0.w [1, 2] SET 1e-3\n', key);
    expect(r.ok && r.value.map(({ tensor, index, op, value }) => [tensor, index, op, value])).toEqual([
      ['blk.0.w', '3,17', 'delta', '-0.002'],
      ['blk.0.w', '[1,2]', 'set', '1e-3'],
    ]);
  });
  it('names the first malformed line', () => {
    expect(parseEditLines('a 1 set 2\nb 1 add 2', key)).toEqual({ ok: false, error: 'Line 2: write <tensor> <coordinate> <set|delta> <value>.' });
    expect(parseEditLines('\n#x\n', key).ok).toBe(false);
  });
});

describe('weights route', () => {
  it('opens the hidden weights view with a source parameter', () => {
    expect(parse('#/weights?source=abc').view).toBe('weights');
    expect(hashParam('#/weights?source=a%20b', 'source')).toBe('a b');
    expect(hashParam('#/weights', 'source')).toBeNull();
  });
});

describe('tuning selection', () => {
  it('parses one pair per line and refuses duplicates and other ranks', () => {
    expect(parseCoordinates('0,1\n\n 2, 3 \n')).toEqual({ ok: true, value: [[0, 1], [2, 3]] });
    expect(parseCoordinates('0,1\n0, 1')).toEqual({ ok: false, error: 'Line 2: 0,1 is selected twice.' });
    expect(parseCoordinates('0,1,2').ok).toBe(false);
    expect(parseCoordinates('').ok).toBe(false);
    const many = Array.from({ length: MAX_COORDINATES + 1 }, (_, i) => `0,${i}`).join('\n');
    expect(parseCoordinates(many).ok).toBe(false);
  });
});

describe('job envelopes match the controller contract', () => {
  it('sends closed option objects', () => {
    expect(proposeJob('m', 'h', [{ tensor: 't', index: [1], op: 'set', value: 2 }])).toEqual({
      kind: 'weight_propose', host: 'h', input_ids: ['m'], options: { edits: [{ tensor: 't', index: [1], op: 'set', value: 2 }] },
    });
    expect(tuneJob('m', 'h', 'c', 'v', 't', [[0, 1]], 0.01)).toEqual({
      kind: 'weight_tune', host: 'h', input_ids: ['m', 'c', 'v'], options: { tensor: 't', indices: [[0, 1]], max_update: 0.01 },
    });
    expect(applyJob('m', 'h', 'p')).toEqual({ kind: 'weight_apply', host: 'h', input_ids: ['m'], options: { proposal_job_id: 'p', reviewed: true } });
  });
});

const proposal = {
  schema: 'brewery.weight-edit-proposal.v1',
  source_sha256: SHA,
  edits: [{ tensor: 't', index: [0, 1], op: 'delta', value: 0.1, dtype: 'F16', before: 0.5, after: 0.60009765625 }],
};

describe('result readers', () => {
  it('reads a proposal from report.proposal or top-level proposal', () => {
    expect(readProposal({ report: { proposal } })?.edits[0].after).toBe(0.60009765625);
    expect(readProposal({ proposal })?.source_sha256).toBe(SHA);
    expect(readProposal({ report: { proposal: { ...proposal, schema: 'other' } } })).toBeNull();
    expect(readProposal({ report: { proposal: { ...proposal, edits: [{ ...proposal.edits[0], after: 'x' }] } } })).toBeNull();
    expect(readProposal({ report: { proposal: { ...proposal, edits: [] } } })).toBeNull();
  });

  it('reads tune metrics and derives held-out regression from the numbers', () => {
    const tune = {
      schema: 'brewery.weight-tune-report.v1', tensor: 't', method: 'rounded-coordinate-descent-v1', selected: 2, proposed: 1, max_update: 0.1,
      calibration: { before_mse: 2, after_mse: 1 }, held_out: { before_mse: 1, after_mse: 1.5, regression: false },
    };
    const r = readTuneReport({ report: { tune } });
    expect(r?.calibration).toEqual({ before: 2, after: 1 });
    expect(r?.held_out.regression).toBe(true);
    expect(readTuneReport({ report: { tune: { ...tune, held_out: {} } } })).toBeNull();
  });

  it('reads inspect and apply receipts', () => {
    expect(readInspect({ report: { tensor: 't', index: [0], dtype: 'BF16', value: 1, shape: [4], source_sha256: SHA } })?.dtype).toBe('BF16');
    expect(readInspect({ report: { tensor: 't' } })).toBeNull();
    const a = readApplied({ library_model_id: 'new', report: { schema: 'brewery.weight-edit-result.v1', source_sha256: SHA, output_sha256: SHA, edit_count: 1, verified: true } });
    expect(a).toEqual({ source_sha256: SHA, output_sha256: SHA, edit_count: 1, verified: true, library_model_id: 'new' });
    expect(readApplied({ report: { schema: 'x' } })).toBeNull();
  });
});

describe('job lists and display', () => {
  const job = (id: string, kind: string, input: string, options?: Record<string, unknown>): JobSummary => ({ id, kind, state: 'succeeded', input_ids: [input], options });
  it('filters proposals by source and applies by proposal', () => {
    const jobs = [job('1', 'weight_propose', 'a'), job('2', 'weight_tune', 'b'), job('3', 'weight_apply', 'a', { proposal_job_id: '1' }), job('4', 'convert', 'a')];
    expect(jobsFor(jobs, 'a', PROPOSAL_KINDS).map((j) => j.id)).toEqual(['1']);
    expect(appliesOf(jobs, '1').map((j) => j.id)).toEqual(['3']);
  });

  it('ends controller reasons as sentences', () => {
    expect(sentence('Weight editing is not configured on this worker')).toBe('Weight editing is not configured on this worker.');
    expect(sentence('Done.')).toBe('Done.');
  });

  it('formats exact values and relative change', () => {
    expect(exact(0.1)).toBe('0.1');
    expect(exact(-0)).toBe('-0');
    expect(relChange(0, 1)).toBeNull();
    expect(percent(relChange(2, 1))).toBe('-50.0 %');
    expect(percent(relChange(1, 1.005))).toBe('+0.500 %');
  });

  it('flags edits changed by dtype rounding', () => {
    expect(rounded(proposal.edits[0] as never)).toBe(true);
    expect(rounded({ ...proposal.edits[0], op: 'set', value: 0.75, after: 0.75 } as never)).toBe(false);
  });
});
