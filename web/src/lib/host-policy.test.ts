import { describe, expect, it } from 'vitest';
import { parseHostPolicies, resourceInput, resourceValues } from './host-policy';

const row = { id: 'cpu', name: 'CPU', paused: false, draining: false, revision: 2,
  installed_jobs: ['convert'], allowed_jobs: ['convert'], disabled_jobs: [], resources: {},
  resource_bounds: { cpu_threads: { min: 1, max: 8 } }, effective_resources: { cpu_threads: 6 } };
describe('all-host policy', () => {
  it('accepts a non-evaluator, but never edits a missing revision or an unreported job kind', () => {
    expect(parseHostPolicies([row])?.[0].id).toBe('cpu');
    expect(parseHostPolicies([{ ...row, revision: null }])).toBeNull();
    expect(parseHostPolicies([{ ...row, disabled_jobs: ['evaluate'] }])).toBeNull();
  });
  it('does not lose an existing byte limit smaller than one GiB', () => {
    const host = parseHostPolicies([{ ...row, resource_bounds: {
      min_free_ram_bytes: { min: 1, max: 1073741824 },
    }, resources: { min_free_ram_bytes: 1 } }])![0];
    expect(resourceValues(host, { min_free_ram_bytes: resourceInput('min_free_ram_bytes', 1) })).toEqual({ min_free_ram_bytes: 1 });
  });
  it('keeps inherited resource limits absent, and rejects values outside controller bounds', () => {
    const host = parseHostPolicies([row])![0];
    expect(resourceValues(host, {})).toEqual({});
    expect(resourceValues(host, { cpu_threads: '4' })).toEqual({ cpu_threads: 4 });
    expect(resourceValues(host, { cpu_threads: '9' })).toBeNull();
  });
  it('converts GiB to bytes but still requires whole CPU counts and respects byte bounds', () => {
    const host = parseHostPolicies([{ ...row, resource_bounds: {
      ...row.resource_bounds, min_free_ram_bytes: { min: 1073741824, max: 2684354560 },
    } }])![0];
    expect(resourceValues(host, { min_free_ram_bytes: '1.5', cpu_threads: '4' })).toEqual({ min_free_ram_bytes: 1610612736, cpu_threads: 4 });
    expect(resourceValues(host, { min_free_ram_bytes: '2.51' })).toBeNull();
    expect(resourceValues(host, { cpu_threads: '4.5' })).toBeNull();
  });
});

it('does not offer an enable action without an explicit verified editor receipt', () => {
  expect(parseHostPolicies([{...row,weight_editor:{enabled:false,capabilities:{}}}])?.[0].weightEditor).toBeNull();
  const receipt={available:true,enabled:false,capabilities:{weight_inspect:false}};
  expect(parseHostPolicies([{...row,weight_editor:receipt}])?.[0].weightEditor).toEqual(receipt);
});
