// Model components a conversion can take from the checkpoint or from a
// separate Library entry: tokenizer/config assets, vision tower, engram
// (n-gram lookup) and MTP draft head. The library kinds and the
// options.<kind> input index are unchanged; this module only adds labels,
// the host gate and the reading of result.components evidence.

import { capability } from './format';
import { obj } from './reports';
import type { Host, Model } from './types';

export type RoleKind = 'assets' | 'vision' | 'ngram' | 'mtp';
export type RoleSelection = Record<RoleKind, string>;

export const COMPONENT_ROLES: { kind: RoleKind; label: string; formats: string }[] = [
  { kind: 'assets', label: 'Tokenizer and config files', formats: 'Directory with tokenizer and config files.' },
  { kind: 'vision', label: 'Vision tower', formats: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
  { kind: 'ngram', label: 'Engram (n-gram lookup)', formats: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
  { kind: 'mtp', label: 'MTP draft head', formats: 'Role-only safetensors file, indexed safetensors directory, or HGN sidecar.' },
];

export function roleLabel(kind: string): string {
  return COMPONENT_ROLES.find((r) => r.kind === kind)?.label ?? kind;
}

/**
 * What the conversion uses when no Library entry is chosen. Wording is
 * conditional: the page has no evidence that the checkpoint contains the
 * component until a result says so.
 */
export function defaultText(source: Model | undefined, kind: RoleKind): string {
  if (!source) return 'Select a source first';
  if (source.kind === 'hf') return 'From the checkpoint, if it has one';
  if (source.kind === 'hgn' || source.kind === 'hgn-v2' || source.kind === 'ht43' || source.kind === 'brewery-bundle') return 'From the bundle, if it has one';
  if (source.kind === 'gguf') {
    if (kind === 'assets') return 'Required for GGUF: add matching tokenizer and config files';
    if (kind === 'ngram') return 'From the GGUF, if it has one';
    return 'None unless you add one';
  }
  return 'Converter default';
}

/** Replacements on an HF source need a host that accepts component inputs. */
export function componentBlockers(source: Model | undefined, sel: RoleSelection, host: Host | undefined): string[] {
  if (!source || source.kind !== 'hf') return [];
  if (!COMPONENT_ROLES.some((r) => sel[r.kind])) return [];
  const c = capability(host, 'convert_components');
  if (c.allowed) return [];
  if (host?.reasons?.convert_components) return [host.reasons.convert_components];
  return [`${host?.name ?? host?.id ?? 'The host'} does not accept component replacements for a Hugging Face checkpoint. Remove them or choose another host.`];
}

export interface RoleEvidence {
  source: 'input' | 'embedded' | 'none' | 'unknown';
  inputId: string | null;
  format: string | null;
  sha256: string | null;
}

const SOURCES = new Set(['input', 'embedded', 'none', 'unknown']);

/** result.components, keyed by role. Absent for legacy results (null). Unknown source values are dropped. */
export function readComponents(result: unknown): Partial<Record<RoleKind, RoleEvidence>> | null {
  const c = obj(obj(result)?.components);
  if (!c) return null;
  const out: Partial<Record<RoleKind, RoleEvidence>> = {};
  for (const { kind } of COMPONENT_ROLES) {
    const e = obj(c[kind]);
    if (!e || typeof e.source !== 'string' || !SOURCES.has(e.source)) continue;
    out[kind] = {
      source: e.source as RoleEvidence['source'],
      inputId: typeof e.input_id === 'string' ? e.input_id : null,
      format: typeof e.format === 'string' ? e.format : null,
      sha256: typeof e.sha256 === 'string' ? e.sha256 : null,
    };
  }
  return out;
}

export interface RoleRow {
  kind: RoleKind;
  label: string;
  requested: string | null;
  reported: RoleEvidence['source'] | null;
  evidence: RoleEvidence | null;
  /** For a requested input: does the reported input_id equal it? null when not comparable. */
  match: boolean | null;
}

export function roleRows(requested: RoleSelection, evidence: Partial<Record<RoleKind, RoleEvidence>> | null): RoleRow[] {
  return COMPONENT_ROLES.map(({ kind, label }) => {
    const ev = evidence?.[kind] ?? null;
    const req = requested[kind] || null;
    return {
      kind,
      label,
      requested: req,
      reported: ev?.source ?? null,
      evidence: ev,
      match: req && ev?.source === 'input' && ev.inputId ? ev.inputId === req : req && ev && ev.source !== 'input' ? false : null,
    };
  });
}

export function evidenceText(ev: RoleEvidence | null): string {
  if (!ev) return 'Not reported';
  switch (ev.source) {
    case 'input':
      return `Library entry${ev.format ? ` (${ev.format})` : ''}`;
    case 'embedded':
      return 'From the checkpoint (confirmed)';
    case 'none':
      return 'None';
    default:
      return 'Unknown';
  }
}

/** Requested replacements from an input list and its options.<kind> 0-based indexes. */
export function selectionFrom(inputIds: string[], options: Record<string, unknown> | undefined): RoleSelection {
  const sel: RoleSelection = { assets: '', vision: '', ngram: '', mtp: '' };
  for (const { kind } of COMPONENT_ROLES) {
    const i = options?.[kind];
    if (Number.isInteger(i) && (i as number) >= 0 && (i as number) < inputIds.length) sel[kind] = inputIds[i as number];
  }
  return sel;
}
