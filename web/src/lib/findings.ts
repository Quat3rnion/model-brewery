// Findings (X10): versioned hypotheses with experiment evidence, a verdict and
// provenance. The workbench only reads findings (GET). Each evidence or verdict
// update makes a new version whose parent_id is the prior version. A replay can
// name its original in replay_of and has a comparison when it is finished.
import type { Tone } from './format';
import type { Parsed } from './weightspace';

export type FindingState = 'ready' | 'running' | 'failed';
export type VerdictStatus = 'supported' | 'refuted' | 'inconclusive';
export interface FindingSummary {
  id: string;
  version: number;
  hypothesis: string;
  parent_id: string | null;
  replay_of?: string | null;
  created_at_ms: number;
  state: FindingState;
  model: { id: string; sha256: string };
  verdict?: Verdict | null;
}
export interface Metric { path: string; value: number }
export interface Experiment {
  spec: { kind: string; [key: string]: unknown };
  artifact: { id: string; job_id?: string; sha256: string; api_url: string; view_url: string; pins?: unknown };
  metrics: Metric[];
}
export interface Verdict { status: VerdictStatus; reasoning: string }
export interface Agent { kind: string; id: string }
export interface Comparison { experiment: number; path: string; original: number; replay: number }
export interface Finding extends FindingSummary {
  experiments: Experiment[];
  verdict: Verdict | null;
  provenance: { agent: Agent; tool_version: string } | null;
  comparison: Comparison[];
  error?: string;
}

export const STATE_INFO: Record<FindingState, { label: string; tone: Tone }> = {
  ready: { label: 'Ready', tone: 'ok' },
  running: { label: 'Running', tone: 'info' },
  failed: { label: 'Failed', tone: 'bad' },
};
export const VERDICT_INFO: Record<VerdictStatus, { label: string; tone: Tone }> = {
  supported: { label: 'Supported', tone: 'ok' },
  refuted: { label: 'Refuted', tone: 'bad' },
  inconclusive: { label: 'Inconclusive', tone: 'warn' },
};
const AGENT_LABELS: Record<string, string> = {
  builtin: 'Built-in agent',
  external_mcp: 'External agent (MCP)',
  external_rest: 'External client (REST)',
};
export const agentLabel = (kind: string) => AGENT_LABELS[kind] ?? kind;

export const findingHref = (id: string) => `#/findings/${encodeURIComponent(id)}`;
export function findingIdFromHash(hash: string): string | null {
  const m = /^#?\/?findings\/([^/?]+)/.exec(hash);
  if (!m) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return null;
  }
}
/** Only workbench hash routes. Other schemes and origins are not links. */
export const viewHref = (url: string) => (/^#\/[^\s]*$/.test(url) ? url : null);
/** Only same-origin controller API paths. */
export const apiHref = (url: string) => (/^\/api\/[^\s]*$/.test(url) ? url : null);
export const agentSessionHref = (agent: Agent) => (agent.kind === 'builtin' ? `#/agent?session=${encodeURIComponent(agent.id)}` : null);

const rec = (v: unknown): Record<string, unknown> | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const str = (v: unknown): v is string => typeof v === 'string';
const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const optStr = (v: unknown) => v === undefined || v === null || str(v);

function verdict(v: unknown): Verdict | null | undefined {
  if (v === undefined || v === null) return null;
  const r = rec(v);
  if (!r || !str(r.reasoning) || !(str(r.status) && r.status in VERDICT_INFO)) return undefined;
  return { status: r.status as VerdictStatus, reasoning: r.reasoning };
}

function summary(v: unknown): FindingSummary | null {
  const r = rec(v);
  const model = rec(r?.model);
  if (!r || !str(r.id) || !num(r.version) || !str(r.hypothesis) || !num(r.created_at_ms)) return null;
  if (!(r.parent_id === null || r.parent_id === undefined || str(r.parent_id)) || !optStr(r.replay_of)) return null;
  if (!(str(r.state) && r.state in STATE_INFO) || !model || !str(model.id) || !str(model.sha256)) return null;
  const v2 = verdict(r.verdict);
  if (v2 === undefined) return null;
  return {
    id: r.id, version: r.version, hypothesis: r.hypothesis, parent_id: (r.parent_id as string | null | undefined) ?? null,
    replay_of: (r.replay_of as string | null | undefined) ?? null, created_at_ms: r.created_at_ms,
    state: r.state as FindingState, model: { id: model.id, sha256: model.sha256 }, verdict: v2,
  };
}

export function parseFindingSummaries(v: unknown): Parsed<FindingSummary[]> {
  if (!Array.isArray(v)) return { ok: false, error: 'The controller sent findings in an unknown format.' };
  const rows = v.map(summary);
  if (rows.some((r) => !r)) return { ok: false, error: 'The controller sent a finding in an unknown format.' };
  return { ok: true, value: rows as FindingSummary[] };
}

function experiment(v: unknown): Experiment | null {
  const r = rec(v);
  const spec = rec(r?.spec);
  const a = rec(r?.artifact);
  if (!r || !spec || !str(spec.kind) || !a || !str(a.id) || !str(a.sha256) || !str(a.api_url) || !str(a.view_url) || !optStr(a.job_id)) return null;
  if (!Array.isArray(r.metrics)) return null;
  const metrics: Metric[] = [];
  for (const m of r.metrics) {
    const mr = rec(m);
    if (!mr || !str(mr.path) || !num(mr.value)) return null;
    metrics.push({ path: mr.path, value: mr.value });
  }
  return {
    spec: spec as Experiment['spec'],
    artifact: { id: a.id, sha256: a.sha256, api_url: a.api_url, view_url: a.view_url, ...(str(a.job_id) ? { job_id: a.job_id } : {}), ...(a.pins !== undefined ? { pins: a.pins } : {}) },
    metrics,
  };
}

export function parseFinding(v: unknown): Parsed<Finding> {
  const bad = { ok: false as const, error: 'The controller sent a finding in an unknown format.' };
  const base = summary(v);
  const r = rec(v);
  if (!base || !r) return bad;
  const experiments = (Array.isArray(r.experiments) ? r.experiments : []).map(experiment);
  if (r.experiments !== undefined && !Array.isArray(r.experiments)) return bad;
  if (experiments.some((e) => !e)) return bad;
  let provenance: Finding['provenance'] = null;
  if (r.provenance !== undefined && r.provenance !== null) {
    const p = rec(r.provenance);
    const agent = rec(p?.agent);
    if (!p || !agent || !str(agent.kind) || !str(agent.id) || !str(p.tool_version)) return bad;
    provenance = { agent: { kind: agent.kind, id: agent.id }, tool_version: p.tool_version };
  }
  const comparison: Comparison[] = [];
  if (r.comparison !== undefined && r.comparison !== null) {
    if (!Array.isArray(r.comparison)) return bad;
    for (const c of r.comparison) {
      const cr = rec(c);
      if (!cr || !num(cr.experiment) || !str(cr.path) || !num(cr.original) || !num(cr.replay)) return bad;
      comparison.push({ experiment: cr.experiment, path: cr.path, original: cr.original, replay: cr.replay });
    }
  }
  if (!optStr(r.error)) return bad;
  return {
    ok: true,
    value: {
      ...base, verdict: base.verdict ?? null, experiments: experiments as Experiment[], provenance, comparison,
      ...(str(r.error) ? { error: r.error } : {}),
    },
  };
}

export interface CompareRow { key: string; experiment: string; path: string; original: number; replay: number; delta: number; changed: boolean }
/** Original against replay values. experiment is a 1-based index into experiments. */
export function compareRows(rows: Comparison[], experiments: Experiment[]): CompareRow[] {
  return rows.map((c, i) => {
    const kind = experiments[c.experiment - 1]?.spec.kind;
    const delta = c.replay - c.original;
    return {
      key: `${c.experiment}:${c.path}:${i}`, experiment: kind ? `${c.experiment}. ${kind}` : String(c.experiment),
      path: c.path, original: c.original, replay: c.replay, delta: Object.is(delta, -0) ? 0 : delta, changed: c.replay !== c.original,
    };
  });
}
