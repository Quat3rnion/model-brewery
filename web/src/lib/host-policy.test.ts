import { describe, expect, it } from 'vitest';
import { parseHostPolicies, resourceValues } from './host-policy';

const row = { id: 'cpu', name: 'CPU', paused: false, draining: false, revision: 2,
  installed_jobs: ['convert'], allowed_jobs: ['convert'], disabled_jobs: [], resources: {},
  resource_bounds: { cpu_threads: { min: 1, max: 8 } }, effective_resources: { cpu_threads: 6 } };
describe('all-host policy', () => {
  it('accepts a non-evaluator, but never edits a missing revision or an unreported job kind', () => {
    expect(parseHostPolicies([row])?.[0].id).toBe('cpu');
    expect(parseHostPolicies([{ ...row, revision: null }])).toBeNull();
    expect(parseHostPolicies([{ ...row, disabled_jobs: ['evaluate'] }])).toBeNull();
  });
  it('keeps inherited resource limits absent, and rejects values outside controller bounds', () => {
    const host = parseHostPolicies([row])![0];
    expect(resourceValues(host, {})).toEqual({});
    expect(resourceValues(host, { cpu_threads: '4' })).toEqual({ cpu_threads: 4 });
    expect(resourceValues(host, { cpu_threads: '9' })).toBeNull();
  });
});
