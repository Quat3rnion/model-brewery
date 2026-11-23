import { capability } from './format';
import type { BrewState, EligibilityEntry, GgufSourceKind, Workflow, WorkflowEligibility } from './types';

export type GgufEligibility = { kind: GgufSourceKind; rocmfpx?: boolean };

/** Use the more specific admission decision for GGUF requests with a known source kind. */
export function hostWorkflowEligibility(
  snapshot: WorkflowEligibility | undefined,
  hostId: string,
  workflow: Workflow,
  gguf?: GgufEligibility,
): EligibilityEntry | undefined {
  const host = snapshot?.hosts[hostId];
  if (workflow === 'gguf_output' && gguf) {
    return (gguf.rocmfpx ? host?.gguf_rocmfpx_by_kind : host?.gguf_output_by_kind)?.[gguf.kind];
  }
  return host?.[workflow];
}

/** Compare and Transfers also work with controllers that only publish host capabilities. */
export function globalJobEligibility(state: BrewState | null, kind: 'compare' | 'transfer' | 'transfer_preview'): EligibilityEntry | undefined {
  if (!state) return undefined;
  const entry = state.eligibility?.global[kind];
  if (entry) return entry;
  const reasons: EligibilityEntry['reasons'] = [];
  for (const host of state.hosts) {
    const hostEntry = hostWorkflowEligibility(state.eligibility, host.id, kind);
    const support = capability(host, kind);
    if (hostEntry?.eligible ?? support.allowed) return { eligible: true, reasons: [] };
    const hint = host.capability_reasons?.[kind];
    reasons.push(...(hostEntry?.reasons.length ? hostEntry.reasons : [{
      text: `${host.name ?? host.id}: ${hint?.text ?? support.reason ?? 'This job is not available.'}`,
      fix_anchor: hint?.fix_anchor ?? `#/settings/infrastructure?host=${encodeURIComponent(host.id)}&section=policy`,
    }]));
  }
  return { eligible: false, reasons: reasons.length ? reasons : [{
    text: 'Configure a worker in Settings.',
    fix_anchor: '#/settings/infrastructure',
  }] };
}
