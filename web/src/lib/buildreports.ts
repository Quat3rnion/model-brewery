// Offline build analysis reports for the Compare view:
//   brewery-build-diff-v1      (task18): artifact and identity differences of two builds
//   brewery-paired-quality-v1  (task6):  paired GSM8K outcomes of two evaluation jobs
// Files are read in the browser only. A missing value stays null and is
// shown as unknown, never as 0. The two schemas carry no digest that binds
// one to the other, so the viewer never merges or matches them.

import { obj } from './reports';
import type { FileLike } from './calibration';

export const BUILD_DIFF_SCHEMA = 'brewery-build-diff-v1';
export const PAIRED_QUALITY_SCHEMA = 'brewery-paired-quality-v1';
export const MAX_BUILD_FILE_BYTES = 16 * 1024 * 1024;
/** Rendering caps; the report is never truncated. */
export const IDS_SHOWN = 24;
export const FILES_PAGE = 100;

const n = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const s = (v: unknown): string | null => (typeof v === 'string' && v !== '' ? v : null);
const strings = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);
/** Number, or a finite numeric string (exact values can be sent as strings). */
const num = (v: unknown): number | null => {
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return n(v);
};
const ints = (v: unknown): number[] => (Array.isArray(v) ? v.filter((x): x is number => Number.isInteger(x) && (x as number) >= 1) : []);

// ---- Build diff ----

export type IdState = 'same' | 'changed' | 'unknown';
export const COMPARISON_KEYS = ['checkpoint', 'source', 'profile', 'options', 'tool', 'encoder', 'runtime'] as const;
export const ROLES = ['checkpoint', 'sidecar', 'assets', 'manifest'] as const;

export interface BuildSide {
  jobId: string | null;
  source: { kind: string | null; sha256: string | null; alias: string | null; model: string | null; revision: string | null; branch: string | null };
  profile: string | null;
  policy: string | null;
  tool: { name: string | null; sha256: string | null; encoderSha256: string | null };
  bundleSha256: string | null;
  bundleBytes: number | null;
  hasRuntime: boolean;
}

export interface FileRow {
  path: string;
  role: string;
  status: string;
  left: { bytes: number | null; sha256: string | null } | null;
  right: { bytes: number | null; sha256: string | null } | null;
  /** As reported (right minus left); null when the report has none. */
  deltaBytes: number | null;
}

export interface BuildDiff {
  left: BuildSide;
  right: BuildSide;
  comparison: { key: (typeof COMPARISON_KEYS)[number]; state: IdState }[];
  /** True only for a literal true in the report. */
  quantizationOnly: boolean;
  files: FileRow[];
  roles: { role: string; leftBytes: number | null; rightBytes: number | null; deltaBytes: number | null }[];
  limits: string[];
}

function buildSide(v: unknown): BuildSide | null {
  const o = obj(v);
  if (!o) return null;
  const src = obj(o.source) ?? {};
  const tool = obj(o.tool) ?? {};
  const art = obj(o.artifact) ?? {};
  return {
    jobId: s(o.job_id),
    source: { kind: s(src.kind), sha256: s(src.sha256), alias: s(src.alias), model: s(src.model), revision: s(src.revision), branch: s(src.branch) },
    profile: s(o.profile),
    policy: s(o.policy),
    tool: { name: s(tool.name), sha256: s(tool.sha256), encoderSha256: s(tool.encoder_sha256) },
    bundleSha256: s(art.bundle_sha256),
    bundleBytes: n(art.bytes),
    hasRuntime: obj(o.runtime) !== null,
  };
}

function fileSide(v: unknown): FileRow['left'] {
  const o = obj(v);
  return o ? { bytes: n(o.bytes), sha256: s(o.sha256) } : null;
}

function readDiff(r: Record<string, unknown>): BuildDiff | string {
  const left = buildSide(r.left);
  const right = buildSide(r.right);
  if (!left || !right) return 'The build diff has no left and right build.';
  if (!Array.isArray(r.files)) return 'The build diff has no file list.';
  const cmp = obj(r.comparison) ?? {};
  const byRole = obj(r.bytes_by_role) ?? {};
  const roleKeys = [...ROLES.filter((k) => k in byRole), ...Object.keys(byRole).filter((k) => !(ROLES as readonly string[]).includes(k))];
  return {
    left,
    right,
    comparison: COMPARISON_KEYS.map((key) => ({ key, state: cmp[key] === 'same' || cmp[key] === 'changed' ? (cmp[key] as IdState) : 'unknown' })),
    quantizationOnly: cmp.quantization_only === true,
    files: r.files
      .map(obj)
      .filter((f): f is Record<string, unknown> => !!f)
      .map((f) => ({ path: s(f.path) ?? '?', role: s(f.role) ?? '?', status: s(f.status) ?? 'unknown', left: fileSide(f.left), right: fileSide(f.right), deltaBytes: n(f.delta_bytes) })),
    roles: roleKeys.map((role) => {
      const x = obj(byRole[role]) ?? {};
      return { role, leftBytes: n(x.left_bytes), rightBytes: n(x.right_bytes), deltaBytes: n(x.delta_bytes) };
    }),
    limits: strings(r.limits),
  };
}

// ---- Paired quality ----

export interface QualitySide {
  jobId: string | null;
  kind: string | null;
  host: string | null;
  sourceSha256: string | null;
  predictionsSha256: string | null;
  score: { correct: number | null; total: number | null; unparsed: number | null; accuracyBasisPoints: number | null } | null;
  provenanceMethod: string | null;
  manifestSha256: string | null;
}

export interface PairedQuality {
  dataset: { name: string | null; revision: string | null; split: string | null; sha256: string | null; rows: number | null };
  datasetFull: boolean | null;
  protocolId: string | null;
  /** null when the report does not state it. */
  synthetic: boolean | null;
  left: QualitySide;
  right: QualitySide;
  transitions: { bothCorrect: number | null; bothIncorrect: number | null; improvements: number | null; regressions: number | null };
  improvementIds: number[];
  regressionIds: number[];
  parsedChangeIds: number[];
  /** Right minus left, as reported. */
  deltaCorrect: number | null;
  deltaPp: number | null;
  uncertainty: { method: string | null; discordantPairs: number | null; pValue: number | null; pExact: string | null; assumption: string | null };
  warnings: string[];
}

function qualitySide(v: unknown): QualitySide | null {
  const o = obj(v);
  if (!o) return null;
  const sc = obj(o.score);
  const pv = obj(o.provenance) ?? {};
  return {
    jobId: s(o.job_id),
    kind: s(o.kind),
    host: s(o.host),
    sourceSha256: s(o.source_sha256),
    predictionsSha256: s(obj(o.predictions)?.sha256),
    score: sc ? { correct: n(sc.correct), total: n(sc.total), unparsed: n(sc.unparsed), accuracyBasisPoints: n(sc.accuracy_basis_points) } : null,
    provenanceMethod: s(pv.method),
    manifestSha256: s(pv.manifest_sha256),
  };
}

function readQuality(r: Record<string, unknown>): PairedQuality | string {
  const left = qualitySide(r.left);
  const right = qualitySide(r.right);
  if (!left || !right) return 'The paired quality report has no left and right evaluation.';
  const ds = obj(r.dataset) ?? {};
  const t = obj(r.transitions) ?? {};
  const u = obj(r.uncertainty) ?? {};
  const ex = obj(u.p_value_exact);
  const exN = ex ? (typeof ex.numerator === 'string' || typeof ex.numerator === 'number' ? String(ex.numerator) : null) : null;
  const exD = ex ? (typeof ex.denominator === 'string' || typeof ex.denominator === 'number' ? String(ex.denominator) : null) : null;
  return {
    dataset: { name: s(ds.name), revision: s(ds.revision), split: s(ds.split), sha256: s(ds.sha256), rows: n(ds.rows) },
    datasetFull: typeof ds.full_split === 'boolean' ? ds.full_split : null,
    protocolId: s(r.protocol_id),
    synthetic: typeof r.synthetic === 'boolean' ? r.synthetic : null,
    left,
    right,
    transitions: { bothCorrect: n(t.both_correct), bothIncorrect: n(t.both_incorrect), improvements: n(t.improvements), regressions: n(t.regressions) },
    improvementIds: ints(r.improvement_ids),
    regressionIds: ints(r.regression_ids),
    parsedChangeIds: ints(r.parsed_answer_change_ids),
    deltaCorrect: num(r.delta_correct),
    deltaPp: num(r.delta_percentage_points),
    uncertainty: {
      method: s(u.method),
      discordantPairs: n(u.discordant_pairs),
      pValue: num(u.p_value),
      pExact: exN && exD ? `${exN}/${exD}` : null,
      assumption: s(u.assumption),
    },
    warnings: strings(r.warnings),
  };
}

// ---- Detection ----

export type BuildParsed =
  | { kind: 'build-diff'; report: BuildDiff }
  | { kind: 'paired-quality'; report: PairedQuality }
  | { kind: 'error'; message: string };

export function parseBuildReport(text: string): BuildParsed {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { kind: 'error', message: 'The file is not valid JSON.' };
  }
  const r = obj(data);
  if (!r) return { kind: 'error', message: 'The file is not a JSON object.' };
  if (r.schema === BUILD_DIFF_SCHEMA) {
    const d = readDiff(r);
    return typeof d === 'string' ? { kind: 'error', message: d } : { kind: 'build-diff', report: d };
  }
  if (r.schema === PAIRED_QUALITY_SCHEMA) {
    const q = readQuality(r);
    return typeof q === 'string' ? { kind: 'error', message: q } : { kind: 'paired-quality', report: q };
  }
  const label = typeof r.schema === 'string' && /^[A-Za-z0-9._-]{1,64}$/.test(r.schema) ? `schema ${r.schema}` : 'no known schema';
  return { kind: 'error', message: `Not supported (${label}). Open a ${BUILD_DIFF_SCHEMA} or ${PAIRED_QUALITY_SCHEMA} report.` };
}

export async function readBuildFileBounded(file: FileLike): Promise<BuildParsed> {
  if (file.size > MAX_BUILD_FILE_BYTES) return { kind: 'error', message: `The file is larger than ${MAX_BUILD_FILE_BYTES / 1024 / 1024} MiB.` };
  try {
    return parseBuildReport(await file.text());
  } catch {
    return { kind: 'error', message: 'The file could not be read.' };
  }
}

export function pValueText(u: PairedQuality['uncertainty']): string {
  if (u.pValue !== null) return u.pValue < 0.0001 ? u.pValue.toExponential(2) : u.pValue.toFixed(4);
  if (u.pExact) return `${u.pExact} (exact; too small for a decimal)`;
  return 'Not reported';
}
