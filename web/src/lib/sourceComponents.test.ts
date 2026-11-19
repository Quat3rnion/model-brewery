import { expect, it } from 'vitest';
import { resolveSourceComponents, sourceComponentJobs } from './sourceComponents';
import type { JobDetail, Model } from './types';

const sha = 'a'.repeat(64);
const source: Model = { id: 'left', kind: 'hf', host: 'local-sm120', path: '/model' };
const alias: Model = { ...source, id: 'right', host: 'local-components' };
const samePath: Model = { ...source, id: 'unverified' };
const origin = (name: string) => ({ origin: { Tensor: { name } } });
function detail(model: Model, kind: 'inspect' | 'plan' | 'weight_inspect', report: object, rest = {}): JobDetail {
  return {
    id: `${model.id}-${kind}`, kind, state: 'succeeded', input_ids: [model.id],
    options: kind === 'weight_inspect' ? { mode: 'inventory' } : {},
    result: { inputs: [{ id: model.id, kind: 'hf', sha256: sha, bundle_sha256: sha }], report, ...rest },
  };
}

it('joins independently attested physical HF bundles and detects integrated MTP', () => {
  const inspection = detail(source, 'inspect', {
    assets: ['tokenizer.json', 'config.json'],
    outputs: [{ filename: 'vision.hgn', identity: 'qwen3.8-flash-next-vision' },
              { filename: 'lookup.hgn', identity: 'qwen3.8-flash-next-ngram' }],
  });
  const plan = detail(source, 'plan', { plan: { files: [{
    filename: 'model.hgn', identity: 'qwen3.8-flash-next-v2',
    tensors: [origin('mtp.layers.0.self_attn.q_proj.weight')],
  }] } });
  const page = detail(alias, 'weight_inspect', { mode: 'inventory', source_format: 'hf', source_sha256: sha, tensors: [{ name: 'embed_tokens.weight' }] });
  const jobs = [inspection, plan, page];
  const models = [source, alias, samePath];
  for (const model of [source, alias]) {
    const candidates = sourceComponentJobs(model, jobs, models);
    const evidence = resolveSourceComponents(model, candidates, models);
    expect(Object.values(evidence).map(role => role.status)).toEqual(Array(4).fill('detected'));
    expect(evidence.mtp.summary).toContain('Integrated MTP');
  }
  expect(sourceComponentJobs(samePath, jobs, models)).toEqual([]);
  expect(Object.values(resolveSourceComponents(samePath, jobs, models)).map(role => role.status)).toEqual(Array(4).fill('unknown'));
});

it('ignores replacement evidence and does not equate silence with absence', () => {
  const overlay: JobDetail = {
    ...detail(source, 'inspect', {
      outputs: [{ filename: 'vision.hgn', identity: 'flash-next-vision' }],
    }, { components: { vision: { source: 'input' }, ngram: { source: 'none' } } }),
    input_ids: [source.id, 'replacement'], options: { vision: 1 },
  };
  const evidence = resolveSourceComponents(source, [overlay]);
  expect(evidence.vision.status).toBe('unknown');
  expect(evidence.ngram.status).toBe('none');
  expect(evidence.mtp.status).toBe('unknown');
});

it('refuses aliases with mismatched receipts and ignores lookalike tensor names', () => {
  const direct = detail(source, 'inspect', { outputs: [{ filename: 'vision.hgn', identity: 'main' }] });
  const other = { ...alias, sha256: 'b'.repeat(64) };
  const page = detail(other, 'weight_inspect', { mode: 'inventory', source_format: 'hf', source_sha256: sha,
    tensors: [{ name: 'model.visual.blocks.0.attn.qkv.weight' }] });
  const lookalikes = detail(source, 'weight_inspect', { mode: 'inventory', source_format: 'hf', source_sha256: sha,
    tensors: [{ name: 'model.layers.0.mtp.weight' }, { name: 'model.vision_projection.weight' }] });
  const failed = { ...direct, state: 'failed' as const, result: {
    inputs: [{ id: source.id, kind: 'hf', bundle_sha256: sha }], components: { ngram: { source: 'embedded' } },
  } };
  expect(Object.values(resolveSourceComponents(source, [direct, page, lookalikes, failed], [other]))
    .map(e => e.status)).toEqual(Array(4).fill('unknown'));
});

it('detects a positive saved MTP config without treating a zero as absence', () => {
  for (const depth of [0, 1]) {
    const saved = detail(source, 'inspect', { config: { text_config: { mtp_num_hidden_layers: depth } } });
    expect(resolveSourceComponents(source, [saved]).mtp.status).toBe(depth ? 'detected' : 'unknown');
  }
});
