import { ggufTensorsFromResult, type GgufTensor } from './gguf';
import { obj } from './reports';
import type { Model } from './types';

const mappingRevision = '184f6536024668e8908ab8c86033d56708e62b40';

function readDimensions(dims: Record<string, unknown> | null): GgufTensor[] | null {
  if (!dims) return null;
  const rows = Object.entries(dims).map(([name, shape]) => ({name, shape}));
  if (!rows.length || rows.some(({name, shape}) => !/^[a-zA-Z0-9_.-]{1,200}$/.test(name) ||
      !Array.isArray(shape) || !shape.length || shape.length > 4 ||
      !shape.every(n => Number.isSafeInteger(n) && n > 0))) return null;
  return ggufTensorsFromResult({report: {gguf_tensors: rows}});
}

/** The exporter may know every name even when it cannot publish the model yet. */
export function plannedMainTensors(result: unknown): GgufTensor[] | null {
  const hf = obj(obj(obj(result)?.report)?.hf_export);
  if (hf?.llama_cpp_commit !== mappingRevision) return null;
  const coverage = obj(hf.source_coverage_plan);
  if (!coverage) return readDimensions(obj(hf.expected_gguf_tensors));
  if (coverage.schema_version !== 1 || coverage.scope !== 'safetensors_header_coverage') return null;
  const tensors = obj(obj(obj(coverage.artifacts)?.model)?.tensors);
  return tensors && readDimensions(Object.fromEntries(
    Object.entries(tensors).map(([name, t]) => [name, obj(t)?.gguf_dimensions]),
  ));
}

/** Optional header plan, addressed by physical HF bundle digest, never by path or model name. */
export async function publishedGgufPlan(model: Model): Promise<GgufTensor[] | null> {
  const sha = model.bundle_sha256 ?? model.sha256;
  if (model.kind !== 'hf' || typeof sha !== 'string' || !/^[0-9a-f]{64}$/.test(sha) ||
      (model.sha256 !== undefined && model.sha256 !== sha)) return null;
  const response = await fetch(`/tensor-plans/${sha}.json`);
  if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) return null;
  const plan = obj(await response.json());
  if (plan?.schema !== 1 || plan.bundle_sha256 !== sha || plan.artifact !== 'model' ||
      plan.llama_cpp_commit !== mappingRevision) return null;
  return readDimensions(obj(plan.tensors));
}
