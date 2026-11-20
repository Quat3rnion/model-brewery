import { readFileSync } from 'node:fs';
import { expect, it, vi } from 'vitest';
import { plannedMainTensors, publishedGgufPlan } from './publishedGgufPlan';

it('selects the Flash-Next main plan by verified digest, not by Library name or path', async () => {
  const sha = '0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1';
  const url = `/tensor-plans/v2/${sha}.json`;
  const file = readFileSync(new URL(`../../public${url}`, import.meta.url), 'utf8');
  const fetchPlan = vi.fn(async () => Response.json(JSON.parse(file)));
  vi.stubGlobal('fetch', fetchPlan);
  try {
    const model = {host: 'local', id: 'registered', kind: 'hf', path: '/source'};
    expect(await publishedGgufPlan(model)).toBeNull();
    expect(fetchPlan).not.toHaveBeenCalled();
    const rows = await publishedGgufPlan({...model, bundle_sha256: sha});
    expect(fetchPlan).toHaveBeenCalledWith(url);
    expect(rows).toHaveLength(1256);
    expect(new Set(rows?.map(t => t.currentType))).toEqual(new Set(['BF16']));
    expect(rows?.find(t => t.name === 'blk.0.attn_gate.weight')?.shape).toEqual([2560, 6144]);
    expect(await publishedGgufPlan({...model, sha256: 'a'.repeat(64)})).toBeNull();
  } finally {
    vi.unstubAllGlobals();
  }
});

it('recognizes planned main tensor names without turning projector or metadata keys into overrides', () => {
  const result = {report: {hf_export: {
    llama_cpp_commit: '184f6536024668e8908ab8c86033d56708e62b40',
    source_coverage_plan: {schema_version: 1, scope: 'safetensors_header_coverage', sources: {q: {dtype:'BF16'}, k: {dtype:'F32'}}, artifacts: {
      model: {tensors: {'blk.0.query.weight': {gguf_dimensions: [16, 32], sources:['q','k']}}, metadata: {offsets: {}}},
      mmproj: {tensors: {'v.proj.weight': {gguf_dimensions: [1]}}},
    }},
  }}};
  expect(plannedMainTensors(result)?.map(t => [t.name, t.shape, t.currentType])).toEqual([
    ['blk.0.query.weight', [16, 32], 'BF16 + F32'],
  ]);
  result.report.hf_export.llama_cpp_commit = 'stale';
  expect(plannedMainTensors(result)).toBeNull();
});
