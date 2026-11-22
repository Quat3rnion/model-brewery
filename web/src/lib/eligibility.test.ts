import { describe, expect, it } from 'vitest';
import { hostWorkflowEligibility } from './eligibility';
import { hostOptions } from './pickers';
import type { Host, WorkflowEligibility } from './types';

const paused = { text: 'Resume job scheduling.', fix_anchor: '#/settings/infrastructure?host=worker-b&section=policy' };
const enabled = { eligible: true, reasons: [] };
const disabled = { eligible: false, reasons: [paused] };
const snapshot = {
  global: { compare: enabled },
  hosts: { 'worker-a': { compare: enabled }, 'worker-b': { compare: disabled, gguf_output: enabled, gguf_output_by_kind: { hf: disabled } } },
} as unknown as WorkflowEligibility;
const a: Host = { id: 'worker-a', name: 'Worker A', capabilities: { compare: true } };
const b: Host = { id: 'worker-b', name: 'Worker B', capabilities: { compare: true } };

describe('workflow eligibility', () => {
  it('marks an advertised host unavailable when workflow admission rejects it', () => {
    expect(hostOptions([b, a], 'compare', (host) => hostWorkflowEligibility(snapshot, host.id, 'compare')))
      .toEqual([
        { value: 'worker-a', label: 'Worker A', detail: 'worker-a', keywords: ['worker-a'] },
        { value: 'worker-b', label: 'Worker B', detail: 'worker-b · Unavailable: Resume job scheduling.', keywords: ['worker-b'], disabled: true },
      ]);
  });

  it('uses the source-specific GGUF decision instead of the generic one', () => {
    expect(hostWorkflowEligibility(snapshot, b.id, 'gguf_output')).toBe(enabled);
    expect(hostWorkflowEligibility(snapshot, b.id, 'gguf_output', { kind: 'hf' })).toBe(disabled);
    expect(hostWorkflowEligibility(undefined, a.id, 'compare')).toBeUndefined();
  });
});
