import type { Host, JobState, JobSummary, Model, Progress } from './types';

export const TERMINAL: ReadonlySet<string> = new Set(['succeeded', 'failed', 'cancelled', 'interrupted', 'missing']);
export const ACTIVE: ReadonlySet<string> = new Set(['queued', 'running', 'cancelling']);

export function isTerminal(state: JobState | undefined): boolean {
  return state !== undefined && TERMINAL.has(state);
}

export function isActive(state: JobState | undefined): boolean {
  return state !== undefined && ACTIVE.has(state);
}

export type Tone = 'ok' | 'bad' | 'warn' | 'info' | 'muted';

export function stateTone(state: JobState): Tone {
  switch (state) {
    case 'succeeded':
      return 'ok';
    case 'failed':
    case 'missing':
      return 'bad';
    case 'cancelled':
    case 'interrupted':
      return 'warn';
    case 'running':
    case 'cancelling':
      return 'info';
    default:
      return 'muted';
  }
}

export function stateLabel(state: JobState): string {
  switch (state) {
    case 'succeeded':
      return 'Finished';
    case 'missing':
      return 'Lost on worker';
    case 'interrupted':
      return 'Interrupted';
    default:
      return state ? state[0].toUpperCase() + state.slice(1) : 'Unknown';
  }
}

const UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];

export function formatBytes(n: unknown): string {
  if (typeof n !== 'number' || !Number.isFinite(n) || n < 0) return '—';
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value} B` : `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${UNITS[unit]}`;
}

/** Fraction 0..1 when the controller reported one; null when progress is unknown. */
export function progressFraction(p: Progress): number | null {
  if (typeof p === 'string') {
    const pct = /(\d+(?:\.\d+)?)\s*%/.exec(p);
    if (pct) return clamp01(Number(pct[1]) / 100);
    const ratio = /(\d+)\s*\/\s*(\d+)/.exec(p);
    if (ratio && Number(ratio[2]) > 0) return clamp01(Number(ratio[1]) / Number(ratio[2]));
    return null;
  }
  if (typeof p === 'number') return Number.isFinite(p) ? clamp01(p > 1 && p <= 100 ? p / 100 : p) : null;
  if (!p || typeof p !== 'object') return null;
  if (typeof p.fraction === 'number' && Number.isFinite(p.fraction)) return clamp01(p.fraction);
  if (typeof p.done === 'number' && typeof p.total === 'number' && p.total > 0) return clamp01(p.done / p.total);
  return null;
}

export function progressText(p: Progress): string {
  if (typeof p === 'string') return p;
  const fraction = progressFraction(p);
  const parts: string[] = [];
  if (p && typeof p === 'object') {
    if (typeof p.phase === 'string') parts.push(p.phase);
    if (typeof p.done === 'number' && typeof p.total === 'number') {
      parts.push(
        p.unit === 'bytes' || p.unit === 'B'
          ? `${formatBytes(p.done)} of ${formatBytes(p.total)}`
          : `${p.done.toLocaleString()} of ${p.total.toLocaleString()}${p.unit ? ` ${p.unit}` : ''}`,
      );
    }
    if (typeof p.message === 'string') parts.push(p.message);
  }
  if (fraction !== null) parts.push(`${Math.round(fraction * 100)}%`);
  return parts.join(' · ');
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

export function hostName(hosts: Host[], id: string | undefined | null): string {
  if (!id) return 'no host';
  if (id === 'auto') return 'any pool worker (not yet assigned)';
  return hosts.find((h) => h.id === id)?.name ?? id;
}

/**
 * Whether a host may run a job kind. The controller decides; the UI only
 * reads `capabilities` and shows `reasons`. A missing entry is unknown and
 * treated as not available, with an explanation.
 */
export function capability(host: Host | undefined, kind: string): { allowed: boolean; reason: string | null } {
  if (!host) return { allowed: false, reason: 'Select a host.' };
  const value = host.capabilities?.[kind];
  const reason = host.reasons?.[kind] ?? null;
  if (value === true) return { allowed: true, reason };
  if (value === false) return { allowed: false, reason: reason ?? `${host.name ?? host.id} cannot run ${kind}.` };
  return { allowed: false, reason: reason ?? `The controller did not report ${kind} support for ${host.name ?? host.id}.` };
}

export function modelLabel(m: Model | undefined): string {
  if (!m) return 'Removed entry';
  return m.name || basename(m.path);
}

export function basename(path: string): string {
  const trimmed = path.replace(/\/+$/, '');
  const i = trimmed.lastIndexOf('/');
  return i >= 0 ? trimmed.slice(i + 1) || trimmed : trimmed;
}

export function shortHash(h: unknown): string {
  return typeof h === 'string' && h.length > 16 ? `${h.slice(0, 12)}…` : typeof h === 'string' ? h : '—';
}

/** Controller timestamps are Unix milliseconds; ISO strings are also accepted. */
export function toMillis(at: unknown): number {
  if (typeof at === 'number') return at;
  if (typeof at === 'string') return Date.parse(at);
  return NaN;
}

export function formatTime(at: number | string | undefined): string {
  if (at === undefined || at === null || at === '') return '';
  const d = new Date(toMillis(at));
  if (Number.isNaN(d.getTime())) return String(at);
  return d.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' });
}

/** Jobs newest first. Prefers controller timestamps, falls back to queue order. */
export function newestFirst(jobs: JobSummary[]): JobSummary[] {
  return jobs
    .map((job, index) => ({ job, index }))
    .sort((a, b) => {
      const ta = toMillis(a.job.created_at);
      const tb = toMillis(b.job.created_at);
      if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb - ta;
      return b.index - a.index;
    })
    .map((x) => x.job);
}

export const MODEL_KINDS: { value: string; label: string; hint: string }[] = [
  { value: 'gguf', label: 'GGUF model', hint: 'A single .gguf file. Compare and trace.' },
  { value: 'hf', label: 'Hugging Face checkpoint', hint: 'Checkpoint directory. For weight editing, register the directory that contains the SafeTensors index and shards.' },
  { value: 'safetensors', label: 'SafeTensors file', hint: 'A standalone .safetensors file. For an indexed checkpoint, register its directory as a Hugging Face checkpoint instead.' },
  { value: 'hgn', label: 'HGN model', hint: 'For weight edits, choose a native HGN file. For bundle operations, choose a directory with manifest.json.' },
  { value: 'brewery-bundle', label: 'Brewery bundle', hint: 'Converted bundle directory with manifest.json.' },
  { value: 'hgn-v2', label: 'HGN v2 bundle', hint: 'Bundle directory, HGN v2 profile.' },
  { value: 'ht43', label: 'HT43 bundle', hint: 'Bundle directory, HT43 profile.' },
  { value: 'assets', label: 'Tokenizer and config files', hint: 'Directory with tokenizer and config files. Replaces or supplies them in a conversion.' },
  { value: 'vision', label: 'Vision tower', hint: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
  { value: 'ngram', label: 'Engram (n-gram lookup)', hint: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
  { value: 'mtp', label: 'MTP draft head', hint: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
  { value: 'imatrix', label: 'GGUF importance matrix (imatrix)', hint: 'llama.cpp imatrix file computed for this source model. Required by IQ1/IQ2 types.' },
  { value: 'weight-calibration', label: 'Weight calibration split', hint: 'JSON file with activations and targets for one layer, at most 8 MiB and 1,024 rows. Register calibration and held-out data as two entries.' },
  { value: 'gsm8k-baseline', label: 'GSM8K reference results', hint: 'Scored reference results file.' },
  { value: 'gsm8k-predictions', label: 'GSM8K source predictions', hint: 'Per-question predictions with provenance.' },
];

export const BUNDLE_KINDS: ReadonlySet<string> = new Set(['hgn', 'brewery-bundle', 'hgn-v2', 'ht43']);
export const CONVERT_SOURCE_KINDS: ReadonlySet<string> = new Set(['hf', 'gguf', 'hgn']);
export const EXTRA_INPUT_KINDS = ['assets', 'ngram', 'vision', 'mtp'] as const;

export function kindLabel(kind: string): string {
  if (kind === 'hgn-archive') return 'Structural HGN archive';
  return MODEL_KINDS.find((k) => k.value === kind)?.label ?? kind;
}

const KIND_TITLES: Record<string, string> = {
  transfer_preview: 'Transfer preview',
  transfer: 'Transfer',
  cache_cleanup: 'Cache cleanup',
  compare: 'Prompt comparison',
  inspect: 'Converter inspect',
  plan: 'Converter plan',
  convert: 'Convert step',
  verify: 'Bundle verify',
  generate: 'Generation',
  evaluate: 'GSM8K evaluation',
  baseline: 'GSM8K reference',
  benchmark: 'Benchmark run',
  weight_inspect: 'Weight inspect',
  hf_download: 'Download from Hugging Face',
  weight_propose: 'Weight edit proposal',
  weight_tune: 'Weight tune proposal',
  weight_apply: 'Weight edit apply',
};

export function kindTitle(kind: string): string {
  return KIND_TITLES[kind] ?? kind;
}

export const WEIGHT_SOURCES = ['gguf', 'hgn', 'safetensors', 'hf'];
export const isSafeSource = (kind?: string) => kind === 'safetensors' || kind === 'hf';
