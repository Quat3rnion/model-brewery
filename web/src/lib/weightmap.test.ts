import { describe, expect, it } from 'vitest';
import {
  blockedReason,
  cellIndex,
  clusters,
  divergent,
  findRead,
  inventoryJob,
  matches,
  niceRanges,
  plan,
  readInventory,
  readTile,
  roleOf,
  roundTo,
  step,
  tileJob,
  tileRequest,
} from './weightmap';
import type { JobSummary } from './types';

const SHA = 'b'.repeat(64);

// Reference values from Python struct with the BF16 algorithm of tools/weight_edit.py.
const REF: [number, number | null, number | null][] = [
  [0.1, 0.0999755859375, 0.10009765625],
  [-0.1, -0.0999755859375, -0.10009765625],
  [1 / 3, 0.333251953125, 0.333984375],
  [0.0399921875, 0.03997802734375, 0.0400390625],
  [0.3, 0.300048828125, 0.30078125],
  [65504, 65504, 65536],
  [65519, 65504, 65536],
  [65520, null, 65536],
  [1e-8, 0, 1.0011717677116394e-8],
  [6e-8, 5.960464477539063e-8, 6.007030606269836e-8],
  [-2.5e-5, -2.4974346160888672e-5, -2.5033950805664062e-5],
  [3.14159, 3.140625, 3.140625],
  [1.00390625, 1.00390625, 1.0],
  [1.01171875, 1.01171875, 1.015625],
  [1.005859375, 1.005859375, 1.0078125],
  [3.3895313892515355e38, null, 3.3895313892515355e38],
  [3.4e38, null, null],
  [1e-40, 0, 9.183549615799121e-41],
  [123456.789, null, 123392],
];

describe('stored-type rounding matches the worker', () => {
  it.each(REF)('%s', (v, half, bf) => {
    expect(roundTo('F16', v)).toBe(half);
    expect(roundTo('BF16', v)).toBe(bf);
    expect(roundTo('F32', v)).toBe(Math.fround(v));
  });
});

describe('step', () => {
  it('moves by one representable value', () => {
    expect(step('F16', 1, 1)).toBe(1 + 2 ** -10);
    expect(step('F16', 1, -1)).toBe(1 - 2 ** -11);
    expect(step('BF16', 1, 1)).toBe(1 + 2 ** -7);
    expect(step('F32', 1, 1)).toBe(1 + 2 ** -23);
  });
  it('crosses zero and stays finite', () => {
    expect(step('F16', 0, -1)).toBe(-(2 ** -24));
    expect(step('F16', 2 ** -24, -2)).toBe(-(2 ** -24));
    expect(step('F16', 65504, 5)).toBe(65504);
    expect(step('BF16', -0, 1)).toBeGreaterThan(0);
  });
  it('is reversible', () => {
    for (const d of ['F16', 'BF16', 'F32'] as const) {
      const v = roundTo(d, 0.0419921875)!;
      expect(step(d, step(d, v, 10), -10)).toBe(v);
    }
  });
});

const inv = (tensors: unknown[], extra: Record<string, unknown> = {}) => ({
  report: { mode: 'inventory', source_sha256: SHA, offset: 0, total: tensors.length, next_offset: null, tensors, ...extra },
});

describe('inventory', () => {
  it('keeps the Flash-Next vision patch kernel in the HF inventory without opening it for edits', () => {
    const row = {name:'model.visual.patch_embed.proj.weight', shape:[1152,3,2,16,16], elements:1769472, dtype:'BF16', editable:false, reason:'SafeTensors shape is read-only'};
    const page = (patch:Record<string,unknown> = {}) => readInventory(inv([{...row,...patch}], {source_format:'hf'}), 'hf');
    expect(page()?.tensors[0]).toMatchObject({shape:row.shape,reason:row.reason,elements:row.elements});
    for (const patch of [{editable:true}, {editable:undefined}, {shape:[2.5]}, {shape:[-1]}, {shape:Array(9).fill(1)}, {elements:99}])
      expect(page(patch)).toBeNull();
    for (const [shape,elements] of [[[],1], [[2,0],0]] as const)
      expect(page({shape,elements})?.tensors[0].reason).toBe(row.reason);
  });
  it('reads tensors with role and layer from the name', () => {
    const p = readInventory(inv([
      { name: 'token_embd.weight', shape: [8, 4], dtype: 'F16', elements: 32 },
      { name: 'blk.0.attn_q.weight', shape: [4, 4], dtype: 'F32', elements: 16 },
      { name: 'blk.0.ffn_down_exps.weight', shape: [2, 4, 8], dtype: 'BF16', elements: 64 },
    ]));
    expect(p?.tensors.map((t) => [t.role, t.layer, t.order])).toEqual([['embed', null, 0], ['attn', 0, 1], ['expert', 0, 2]]);
    expect(clusters(p!.tensors).map((c) => c.layer)).toEqual([null, 0]);
  });
  it('refuses wrong elements, dtypes and fingerprints', () => {
    expect(readInventory(inv([{ name: 'a', shape: [2], dtype: 'F16', elements: 3 }]))).toBeNull();
    expect(readInventory(inv([{ name: 'a', shape: [2], dtype: 'Q4_K' }]))).toBeNull();
    expect(readInventory(inv([], { source_sha256: 'x' }))).toBeNull();
  });
  it('filters by text, L<n> and role', () => {
    const t = readInventory(inv([{ name: 'blk.12.attn_k.weight', shape: [2, 2], dtype: 'F16' }]))!.tensors[0];
    expect(matches(t, 'l12 attn', new Set())).toBe(true);
    expect(matches(t, 'l1', new Set())).toBe(false);
    expect(matches(t, '', new Set(['ffn']))).toBe(false);
    expect(roleOf('blk.3.ffn_gate_inp.weight')).toBe('router');
  });
  it('blocks names the controller refuses', () => {
    const t = { name: 'x'.repeat(64) } as never;
    expect(blockedReason(t)).toMatch(/does not accept/);
  });
});

describe('tile planning', () => {
  it('clips the last tile and keeps full rank', () => {
    const p = plan([3, 70, 40]);
    expect([p.gridRows, p.gridCols]).toEqual([3, 2]);
    expect(tileRequest(p, { lead: [2], tr: 2, tc: 1 })).toEqual({ origin: [2, 64, 32], shape: [1, 6, 8] });
    expect(tileRequest(plan([5000]), { lead: [], tr: 0, tc: 4 })).toEqual({ origin: [4096], shape: [904] });
  });
  it('maps cells to weight_edit coordinates', () => {
    expect(cellIndex({ origin: [2, 64, 32], shape: [1, 6, 8] }, 1, 3)).toEqual([2, 65, 35]);
    expect(cellIndex({ origin: [1024], shape: [10] }, 0, 3)).toEqual([1027]);
  });
  it('sends the contract options', () => {
    expect(tileJob('m', 'h', SHA, 't', { origin: [0, 0], shape: [2, 2] }).options).toEqual({ mode: 'tile', tensor: 't', origin: [0, 0], shape: [2, 2], source_sha256: SHA });
    expect(inventoryJob('m', 'h', 0).options).toEqual({ mode: 'inventory', offset: 0, limit: 128 });
    expect(inventoryJob('m', 'h', 128, SHA).options).toEqual({ mode: 'inventory', offset: 128, limit: 128, source_sha256: SHA });
  });
});

describe('tile reports', () => {
  const want = { tensor: 't', origin: [0, 0], shape: [2, 3], sha: SHA, dtype: 'F16' as const, full: [4, 3] };
  const rep = (over: Record<string, unknown> = {}) => ({
    report: { mode: 'tile', source_sha256: SHA, tensor: 't', dtype: 'F16', shape: [4, 3], origin: [0, 0], tile_shape: [2, 3], values: [[1, -2, 0], [0.5, 0, 0]], ...over },
  });
  it('accepts a matching tile and computes the scale', () => {
    const t = readTile(rep(), want);
    expect(typeof t === 'object' && t.absMax).toBe(2);
  });
  it('names a changed source and refuses mismatches', () => {
    expect(readTile(rep({ source_sha256: 'c'.repeat(64) }), want)).toMatch(/changed/);
    expect(readTile(rep({ origin: [1, 0] }), want)).toMatch(/does not match/);
    expect(readTile(rep({ values: [[1, 2, 3]] }), want)).toMatch(/shape/);
  });
});

describe('job reuse', () => {
  const j = (id: string, state: string, options: Record<string, unknown>, created_at = 0): JobSummary => ({ id, kind: 'weight_inspect', state, input_ids: ['m'], options, created_at });
  it('prefers succeeded reads with equal options', () => {
    const o = { mode: 'tile', tensor: 't', origin: [0, 0], shape: [2, 2], source_sha256: SHA };
    const jobs = [j('a', 'failed', o, 3), j('b', 'succeeded', o, 1), j('c', 'running', o, 2), j('d', 'succeeded', { ...o, origin: [0, 32] }, 4)];
    expect(findRead(jobs, 'm', o)?.id).toBe('b');
    expect(findRead(jobs, 'other', o)).toBeNull();
  });
});

describe('display', () => {
  it('builds colours and ranges', () => {
    expect(divergent(0, 1, 'dark')).toBe('rgb(35,28,51)');
    expect(divergent(5, 1, 'dark')).toBe(divergent(1, 1, 'dark'));
    expect(niceRanges(0.37)).toEqual([0.005, 0.05, 0.5]);
    expect(niceRanges(0)).toEqual([0.00001, 0.0001, 0.001]);
  });
});

it('keeps read-only native HGN stores beside editable weights and exposes the worker reason', () => {
  const page = readInventory(inv([{name:'float.weight',shape:[2],dtype:'F16',editable:true}, {name:'packed',shape:[2],dtype:'HT43',editable:false,reason:'Packed store has no write adapter.'}, {name:'companion',shape:[2],dtype:'F16',editable:false,reason:'Companion tensor.'}], {source_format:'hgn'}));
  expect(page?.tensors.map(t => blockedReason(t))).toEqual([null,'Packed store has no write adapter.','Companion tensor.']);
});

it('enables a packed tile only with strictly ordered worker-reported decoded neighbors', () => {
  const want = { tensor:'codes', origin:[0,0], shape:[1,2], full:[1,2], sha:SHA, dtype:'Q4_0' as const };
  const report = { mode:'tile', tensor:'codes', origin:[0,0], tile_shape:[1,2], shape:[1,2], dtype:'Q4_0', source_sha256:SHA, values:[[0,0.25]] };
  const steps = [[{lower:null,upper:0.25},{lower:0,upper:null}]];
  const t = readTile({report:{...report,packed_steps:steps}},want);
  expect(typeof t === 'object' && t.steps).toEqual(steps);
  for (const invalid of [undefined,[],[[{lower:0,upper:1}]], [[{lower:0,upper:1},{lower:0,upper:1}]]]) expect(typeof readTile({report:{...report,packed_steps:invalid}},want)).toBe('string');
});

it('rejects an HF inventory whose reported source format differs from the registered kind', () => {
  const receipt = inv([{ name:'layer.weight',shape:[2,2],dtype:'BF16',editable:true },{ name:'tokens',shape:[2],dtype:'I64',editable:false,reason:'I64 is not editable.' }], {source_format:'hf'});
  expect(readInventory(receipt,'safetensors')).toBeNull();
  expect(readInventory(receipt,'hf')?.tensors.map(blockedReason)).toEqual([null,'I64 is not editable.']);
});

it('opens inventory-listed native keys without widening GGUF or HGN names', () => {
  const long = 'model.language_model.hyper_connection_mixer.input_mix_weight_down.weight';
  const keys = [long, '-with / spaces', 'x'.repeat(95), 'x'.repeat(96), '__metadata__', 'bad\n', 'é'];
  const rows = keys.map(name => ({ name, shape: [2, 2], dtype: 'BF16', editable: true }));
  for (const kind of ['hf', 'safetensors']) {
    const reasons = readInventory(inv(rows, {source_format: kind}), kind)?.tensors.map(blockedReason);
    expect(reasons?.slice(0, 3)).toEqual([null, null, null]);
    expect(reasons?.slice(3).every(r => r?.includes('does not accept'))).toBe(true);
  }
  for (const kind of ['gguf', 'hgn']) {
    const page = readInventory(inv(rows, {source_format: kind}));
    expect(page?.tensors.map(t => blockedReason(t) === null)).toEqual([false, false, false, false, true, false, false]);
  }
});
