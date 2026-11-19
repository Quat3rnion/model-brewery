// Saved job evidence only. This module does not inspect or schedule a model.
import { COMPONENT_ROLES, readComponents, type RoleKind } from './components';
import { obj } from './reports';
import type { JobDetail, JobSummary, Model } from './types';

export interface SourceComponentEvidence {
  status: 'detected' | 'none' | 'unknown';
  summary: string;
  jobId?: string;
}

const digest = (value: unknown) => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value) ? value : undefined;
const list = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const order = (a: JobSummary, b: JobSummary) =>
  (b.finished_at ?? b.updated_at ?? b.created_at ?? 0) - (a.finished_at ?? a.updated_at ?? a.created_at ?? 0);

/** Candidate details to GET. Other HF rows are only candidates; IDs, names and paths do not prove equivalence. */
export function sourceComponentJobs(source: Model, jobs: readonly JobSummary[], models: readonly Model[] = []): JobSummary[] {
  if (source.kind !== 'hf') return [];
  const pin = modelDigest(source);
  const ids = new Set([source.id, ...models.filter(m => m.kind === 'hf' &&
    (!pin || !modelDigest(m) || modelDigest(m) === pin)).map(m => m.id)]);
  if (!pin && !jobs.some(j => eligible(j) && j.input_ids[0] === source.id)) return [];
  return jobs.filter(j => eligible(j) && ids.has(j.input_ids[0])).sort((a, b) =>
    Number(b.input_ids[0] === source.id) - Number(a.input_ids[0] === source.id) || order(a, b));
}

function eligible(job: JobSummary): boolean {
  return job.state === 'succeeded' && job.has_result !== false &&
    (job.kind === 'inspect' || job.kind === 'plan' ||
      (job.kind === 'weight_inspect' && job.options?.mode === 'inventory'));
}

function modelDigest(model: Model): string | undefined {
  return digest(model.bundle_sha256) ?? digest(model.sha256);
}

function receipt(job: JobDetail) {
  const result = obj(job.result);
  const first = obj(list(result?.inputs)[0]);
  if (!result || !first || first.id !== job.input_ids[0] || first.kind !== 'hf') return null;
  // Use the first input's physical HF tree digest, not the conversion's
  // result.source_sha256 (which can include role prefixes or other inputs).
  const tree = digest(first.bundle_sha256);
  const sha = digest(first.sha256);
  if (!tree || (sha && sha !== tree)) return null;
  const report = obj(result.report);
  if (job.kind === 'weight_inspect' && (report?.mode !== 'inventory' ||
      report.source_format !== 'hf' || digest(report.source_sha256) !== tree)) return null;
  return { job, result, report, tree };
}

// Native role namespaces. These are anchored rather than matching words such
// as "vision" or "mtp" anywhere in an unrelated language tensor.
const tensorRole = (name: string): RoleKind | undefined => {
  if (/^model\.visual\./.test(name)) return 'vision';
  if (/^model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.(?:shard_\d+\.)?weight$/.test(name)) return 'ngram';
  if (/^mtp\.(?:layers\.\d+\.|(?:fc_embedding|fc_hidden|pre_fc_norm_embedding|pre_fc_norm_hidden)\.weight$)/.test(name)) return 'mtp';
  return undefined;
};

const tokenizer = new Set(['tokenizer.json', 'tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model', 'vocab.json', 'vocab.txt']);

export function resolveSourceComponents(source: Model, jobs: readonly JobDetail[], models: readonly Model[] = []):
Record<RoleKind, SourceComponentEvidence> {
  const empty: SourceComponentEvidence = { status: 'unknown', summary: 'No saved source evidence for this role.' };
  const evidence: Record<RoleKind, SourceComponentEvidence> = { assets: empty, vision: empty, ngram: empty, mtp: empty };
  if (source.kind !== 'hf') return evidence;

  const rows = new Map([...models, source].filter(m => m.kind === 'hf').map(m => [m.id, m]));
  const saved = jobs.filter(j => eligible(j) && rows.has(j.input_ids[0])).sort(order)
    .flatMap(j => {
      const bound = receipt(j);
      const row = rows.get(j.input_ids[0]);
      const pin = row && modelDigest(row);
      return bound && (!pin || pin === bound.tree) ? [bound] : [];
    });
  const exact = saved.filter(e => e.job.input_ids[0] === source.id);
  const trees = new Set(exact.map(e => e.tree));
  const tree = modelDigest(source) ?? (trees.size === 1 ? exact[0]?.tree : undefined);
  if (!tree) return evidence;

  const scores = { assets: 0, vision: 0, ngram: 0, mtp: 0 };
  for (const { job, result, report } of saved.filter(e => e.tree === tree)) {
    const explicit = readComponents(result);
    const plan = obj(report?.plan);
    const native = obj(obj(obj(report?.summary)?.source_selection ?? report?.source_selection)?.roles);
    const put = (kind: RoleKind, status: 'detected' | 'none', summary: string, score: number) => {
      if (job.options?.[kind] !== undefined || explicit?.[kind]?.source === 'input' || native?.[kind] === 'selected') return;
      if (score <= scores[kind]) return;
      scores[kind] = score;
      evidence[kind] = { status, summary, jobId: job.id };
    };
    for (const { kind } of COMPONENT_ROLES) {
      if (explicit?.[kind]?.source === 'embedded') put(kind, 'detected', 'Saved job confirms this component comes from the checkpoint.', 4);
      if (explicit?.[kind]?.source === 'none') put(kind, 'none', 'Saved job explicitly reports no checkpoint component.', 4);
    }
    const assets = list(report?.assets ?? plan?.assets);
    if (assets.includes('config.json') && assets.some(a => typeof a === 'string' && tokenizer.has(a))) {
      put('assets', 'detected', 'Saved source report lists checkpoint tokenizer and config files.', 2);
    }
    for (const output of [...list(report?.outputs), ...list(plan?.files)]) {
      const file = obj(output);
      if (!file || typeof file.filename !== 'string' || !file.filename.endsWith('.hgn')) continue;
      // Native inspect/plan names sidecars by output identity, not by the
      // name of the Library entry or the original HF directory.
      for (const role of ['vision', 'ngram', 'mtp'] as const) {
        if (typeof file.identity === 'string' && new RegExp(`(?:^|[-.])${role}$`).test(file.identity))
          put(role, 'detected', `Saved native report identifies ${file.filename} as a ${role === 'ngram' ? 'lookup' : role} output.`, 2);
      }
      for (const raw of list(file.tensors)) {
        const name = obj(obj(obj(raw)?.origin)?.Tensor)?.name;
        const role = typeof name === 'string' ? tensorRole(name) : undefined;
        if (role) put(role, 'detected', role === 'mtp' && file.filename === 'model.hgn'
          ? `Integrated MTP: saved plan maps ${name} into the main model.`
          : `Saved plan maps source tensor ${name} to ${file.filename}.`, 3);
      }
    }
    if (job.kind === 'weight_inspect') {
      for (const raw of list(report?.tensors)) {
        const name = obj(raw)?.name;
        const role = typeof name === 'string' ? tensorRole(name) : undefined;
        if (role) put(role, 'detected', `${role === 'mtp' ? 'Integrated MTP: saved' : 'Saved'} inventory lists ${name}.`, 2);
      }
    }
    // A saved config can advertise MTP even without an output sidecar. Zero
    // or missing layers do not prove absence from the actual tensor inventory.
    const config = obj(report?.config);
    for (const section of [config, obj(config?.text_config)]) {
      if (['mtp_num_hidden_layers', 'num_nextn_predict_layers'].some(key =>
        typeof section?.[key] === 'number' && Number.isInteger(section[key]) && section[key] > 0))
        put('mtp', 'detected', 'Saved checkpoint config reports MTP layers.', 1);
    }
  }
  return evidence;
}
