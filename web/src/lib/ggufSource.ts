import { capability, isSafeSource } from './format';
import { ggufTensorsFromResult, type GgufTensor } from './gguf';
import { plannedMainTensors, publishedGgufPlan } from './publishedGgufPlan';
import { obj } from './reports';
import type { Host, JobDetail, JobSummary, Model } from './types';

export const validSha = (v: unknown) => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v) ? v : undefined;

export function inventoryBlocker(source: Model, host: Host | undefined): string | null {
  if (source.available === false) return source.reason ?? 'This source is not available.';
  if (host?.scheduling_paused) return 'Resume job scheduling for the input worker.';
  for (const key of ['weight_inspect', 'weight_visual', ...(isSafeSource(source.kind) ? ['weight_safetensors'] : source.kind === 'hgn' ? ['weight_hgn'] : [])]) {
    const cap = capability(host, key);
    if (!cap.allowed) return host?.capability_reasons?.[key]?.text ?? cap.reason ?? 'Native tensor inventory is not enabled.';
  }
  return null;
}

const inventory = (j: JobSummary) => j.kind === 'weight_inspect' && j.options?.mode === 'inventory' && (j.options.offset ?? 0) === 0;
const eligible = (source: Model, j: JobSummary) => j.state === 'succeeded' && j.has_result !== false && j.input_ids[0] === source.id &&
  (j.kind === 'inspect' || j.kind === 'plan' || (source.kind === 'hf' && inventory(j)));

/** Fetch only selected-model evidence. Prefer inspect and the first inventory page to larger plans. */
export function ggufEvidenceIds(source: Model, jobs: readonly JobSummary[]): string[] {
  const rank = (j: JobSummary) => j.kind === 'inspect' ? 0 : inventory(j) ? 1 : 2;
  const time = (j: JobSummary) => j.finished_at ?? j.updated_at ?? j.created_at ?? 0;
  return jobs.filter(j => eligible(source, j)).sort((a, b) => rank(a) - rank(b) || time(b) - time(a)).map(j => j.id);
}

export type ExactInventory = { tensors: GgufTensor[]; planned: boolean };

export async function resolveGgufSource(
  source: Model,
  ids: string[],
  load: (id: string) => Promise<JobDetail>,
  current: () => boolean,
): Promise<ExactInventory | null> {
  const file = validSha(source.sha256);
  const bundle = validSha(source.bundle_sha256);
  const tried = new Set<string>();
  async function published(tree: string | undefined): Promise<ExactInventory | null> {
    if (!tree || source.kind !== 'hf' || tried.has(tree) || (file && file !== tree) || (bundle && bundle !== tree)) return null;
    tried.add(tree);
    const tensors = await publishedGgufPlan({ ...source, bundle_sha256: tree }).catch(() => null);
    return tensors?.length ? { tensors, planned: true } : null;
  }
  const pinned = await published(bundle ?? file);
  if (pinned) return pinned;
  for (const id of ids) {
    if (!current()) return null;
    const job = await load(id).catch(() => null);
    if (!job || job.id !== id || !eligible(source, job)) continue;
    const result = obj(job.result);
    const report = obj(result?.report);
    const input = obj(Array.isArray(result?.inputs) ? result.inputs[0] : undefined);
    if (!input || input.id !== source.id || input.kind !== source.kind ||
        (input.source_host !== undefined && input.source_host !== source.host) ||
        (report?.source_format !== undefined && report.source_format !== source.kind)) continue;
    const tree = validSha(input.bundle_sha256);
    const sha = validSha(input.sha256);
    if ((file && sha !== file) || (bundle && tree !== bundle)) continue;
    if (job.kind === 'inspect' || job.kind === 'plan') {
      const measured = source.kind !== 'gguf' ? plannedMainTensors(result) : null;
      const formats = new Map(measured?.map(t => [t.name, t.currentType]));
      const tensors = ggufTensorsFromResult(result)?.map(t => ({...t,
        currentType: source.kind === 'gguf' ? t.currentType : formats.get(t.name) ?? null}));
      if (tensors?.length && tensors.some(t => !t.currentType) && tree && tree === sha) {
        const fromPin = await published(tree);
        if (fromPin) {
          const pinnedFormats = new Map(fromPin.tensors.map(t => [t.name, t.currentType]));
          for (const t of tensors) t.currentType ??= pinnedFormats.get(t.name) ?? null;
        }
      }
      if (tensors?.length) return { tensors, planned: source.kind !== 'gguf' };
      if (measured?.length) return { tensors: measured, planned: true };
    }
    if (!current()) return null;
    if ((!inventory(job) || (report?.source_format === 'hf' && report.mode === 'inventory' && report.source_sha256 === tree)) &&
        (input.sha256 == null || sha === tree)) {
      const fromPin = await published(tree);
      if (fromPin) return fromPin;
    }
  }
  return null;
}
