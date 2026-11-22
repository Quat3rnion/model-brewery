import type { EligibilityEntry, GgufSourceKind, Workflow, WorkflowEligibility } from './types';

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
