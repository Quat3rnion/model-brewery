import { describe, expect, it } from 'vitest';
import { buildPut, endpointProblem, idProblem, parseHost, parseSettings, validateHost, effectiveReadiness, type HostSettings } from './settings';
import { adminActionAllowed, parseAdminState } from './admin';

const raw = {
  hosts: [
    {
      id: 'frame', name: 'Frame', managed: true, ready: false, reason: 'No benchmark target', revision: 3, approved_ports: [8000, 9000],
      approved_suites: [{ id: 'mmlu', label: 'MMLU' }, { id: 'arc' }],
      benchmark_targets: [{ id: 'local', label: 'Local vLLM', endpoint: 'http://127.0.0.1:8000/v1', model: 'qwen' }],
      enabled_suites: ['mmlu'], baseline_endpoints: [], evaluation_enabled: false,
    },
    { id: 'gpu2', managed: false, ready: false, reason: 'Not managed by the controller', revision: 0, approved_suites: [], benchmark_targets: [], enabled_suites: [], baseline_endpoints: [], evaluation_enabled: false },
    { bad: true },
  ],
};

describe('settings parsing', () => {
  it('reads hosts, skips malformed entries, keeps ready and reason as sent', () => {
    const s = parseSettings(raw);
    expect(s.map((h) => h.id)).toEqual(['frame', 'gpu2']);
    expect(s[0]).toMatchObject({ managed: true, ready: false, reason: 'No benchmark target', revision: 3, enabledSuites: ['mmlu'] });
    expect(parseSettings(null)).toEqual([]);
  });
});

describe('validation (controller rules)', () => {
  it('saves loopback target profiles without granting an execution port or suite', () => {
    const host=parseHost({...raw.hosts[1],evaluation:{status:'profile_only'},profile_binding:'fixture-binding',approved_ports:[]})!;
    host.benchmarkTargets=[{id:'draft',endpoint:'http://[::1]:80/v1',model:'test'}];
    expect(host.evaluationStatus).toBe('profile_only');
    expect(buildPut(host).profile_binding).toBe('fixture-binding');
    expect(buildPut({...host,evaluationStatus:'managed'})).not.toHaveProperty('profile_binding');
    expect(validateHost(host)).toEqual([]);
    for (const url of ['http://2130706433:8000/v1','http://127.0.0.1:080/v1','http://127.0.0.1:99999/v1','http://127.0.0.1:8000/ignored/../v1']) {
      expect(validateHost({...host,benchmarkTargets:[{...host.benchmarkTargets[0],endpoint:url}]})).not.toEqual([]);
    }
    expect(validateHost({...host,evaluationStatus:'managed'}).join(' ')).toContain('not approved');
    expect(validateHost({...host,enabledSuites:['mmlu'],evaluationEnabled:true})).toHaveLength(2);
  });
  it('IDs: [a-z0-9][a-z0-9_-]{0,63}', () => {
    expect(idProblem('local-1')).toBeNull();
    expect(idProblem('Local')).toMatch(/lowercase/);
    expect(idProblem('-x')).not.toBeNull();
    expect(idProblem('a'.repeat(65))).not.toBeNull();
  });
  it('endpoint: numeric loopback http with /v1, no credentials or query', () => {
    expect(endpointProblem('http://127.0.0.1:8000/v1')).toBeNull();
    expect(endpointProblem('http://[::1]:8000/v1')).toBeNull();
    expect(endpointProblem('http://localhost:8000/v1')).toMatch(/127\.0\.0\.1/);
    expect(endpointProblem('https://127.0.0.1:8000/v1')).not.toBeNull();
    expect(endpointProblem('http://user:pw@127.0.0.1:8000/v1')).not.toBeNull();
    expect(endpointProblem('http://127.0.0.1:8000/v1?key=x')).not.toBeNull();
    expect(endpointProblem('http://127.0.0.1/v1')).toMatch(/port/i);
    expect(endpointProblem('http://10.0.0.5:8000/v1')).not.toBeNull();
  });
  it('host form: duplicate IDs, enabled suites subset of approved, missing model', () => {
    const h = parseSettings(raw)[0];
    const edit: HostSettings = { ...h, benchmarkTargets: [...h.benchmarkTargets, { id: 'local', endpoint: 'http://127.0.0.1:9000/v1', model: '' }], enabledSuites: ['mmlu', 'gsm8k'] };
    const p = validateHost(edit);
    expect(p.join(' ')).toMatch(/Duplicate target ID local/);
    expect(p.join(' ')).toMatch(/model/);
    expect(p.join(' ')).toMatch(/gsm8k is not an approved suite/);
    expect(validateHost(h)).toEqual([]);
  });
});

describe('PUT body', () => {
  it('sends the read revision and only the editable fields', () => {
    const h = parseSettings(raw)[0];
    expect(buildPut(h)).toEqual({
      revision: 3,
      benchmark_targets: [{ id: 'local', label: 'Local vLLM', endpoint: 'http://127.0.0.1:8000/v1', model: 'qwen' }],
      enabled_suites: ['mmlu'],
      baseline_endpoints: [],
      evaluation_enabled: false,
    });
  });
});

describe('effective readiness vs saved', () => {
  it('reports controller capabilities separately from the saved configuration', () => {
    const h = parseSettings(raw)[0];
    const r = effectiveReadiness(h, { id: 'frame', capabilities: { benchmark: false, evaluate: true }, reasons: { benchmark: 'Target not applied yet' } });
    expect(r).toEqual([
      { kind: 'benchmark', what: 'Benchmarks', allowed: false, reason: 'Target not applied yet' },
      { kind: 'evaluate', what: 'GSM8K evaluation', allowed: true, reason: null },
      { kind: 'baseline', what: 'Prepare reference results', allowed: false, reason: expect.stringMatching(/did not report/) },
      { kind: 'generate', what: 'Generation', allowed: false, reason: expect.stringMatching(/did not report/) },
    ]);
  });
});

describe('admin', () => {
  const st = {
    workers: [{ id: 'frame', name: 'Frame', status: 'active', can_enroll: false, can_approve: false, can_deactivate: true, allow_pool: false, pool_ceiling: true, runtimes: [{ id: 'vllm', label: 'vLLM', enabled: false, available: true }, { id: 'tgi', enabled: false, available: false, reason: 'Not provisioned' }] }],
    audit: [{ id: 'a1', at_ms: 1, action: 'approve', worker: 'frame' }],
    unavailable: { install_runtime: 'Not implemented', host_provisioning: 'Not implemented', resource_controls: 'Only allow_pool' },
  };
  it('parses admin state and allows only actions the backend marks possible', () => {
    const s = parseAdminState(st)!;
    const w = s.workers[0];
    expect(adminActionAllowed(w, { action: 'deactivate', worker: 'frame' })).toBeNull();
    expect(adminActionAllowed(w, { action: 'approve', worker: 'frame' })).toMatch(/not possible/);
    expect(adminActionAllowed(w, { action: 'enable_runtime', worker: 'frame', runtime: 'vllm' })).toBeNull();
    expect(adminActionAllowed(w, { action: 'enable_runtime', worker: 'frame', runtime: 'tgi' })).toBe('Not provisioned');
    expect(adminActionAllowed(w, { action: 'set_resource_policy', worker: 'frame', allow_pool: true })).toBeNull();
    expect(adminActionAllowed({ ...w, pool_ceiling: false }, { action: 'set_resource_policy', worker: 'frame', allow_pool: true })).toMatch(/pool/);
    expect(s.unavailable).toHaveLength(3);
    expect(parseAdminState({})).toBeNull();
  });
});

describe('contract update: ports, evaluation_allowed, baseline identity', () => {
  const host = (over: Record<string, unknown> = {}) =>
    parseSettings({ hosts: [{ id: 'local-eval', managed: true, ready: false, revision: 1, approved_ports: [8080], evaluation_allowed: false, approved_suites: [], benchmark_targets: [], enabled_suites: [], baseline_endpoints: [], evaluation_enabled: false, ...over }] })[0];
  it('reads approved ports and evaluation_allowed', () => {
    const h = host();
    expect(h.approvedPorts).toEqual([8080]);
    expect(h.evaluationAllowed).toBe(false);
  });
  it('endpoint port must be approved for this host', () => {
    const h = host();
    const bad = { ...h, benchmarkTargets: [{ id: 't', endpoint: 'http://127.0.0.1:8000/v1', model: 'm' }] };
    expect(validateHost(bad).join(' ')).toMatch(/Port 8000 is not approved.*8080/);
    const good = { ...h, benchmarkTargets: [{ id: 't', endpoint: 'http://127.0.0.1:8080/v1', model: 'm' }] };
    expect(validateHost(good)).toEqual([]);
  });
  it('evaluation cannot be enabled when not allowed', () => {
    expect(validateHost({ ...host(), evaluationEnabled: true }).join(' ')).toMatch(/not allowed/);
  });
  it('baseline endpoints need source_sha256 and runtime_id and send them', () => {
    const h = host({ baseline_endpoints: [{ id: 'b', endpoint: 'http://127.0.0.1:8080/v1', model: 'm', source_sha256: 'A'.repeat(64), runtime_id: '' }] });
    expect(h.baselineEndpoints[0]).toMatchObject({ source_sha256: 'A'.repeat(64), runtime_id: '' });
    const p = validateHost(h).join(' ');
    expect(p).toMatch(/reference model SHA-256/);
    expect(p).toMatch(/runtime ID/);
    const ok = { ...h, baselineEndpoints: [{ ...h.baselineEndpoints[0], source_sha256: 'a'.repeat(64), runtime_id: 'halogen-0.18' }] };
    expect(validateHost(ok)).toEqual([]);
    expect(buildPut(ok).baseline_endpoints[0]).toEqual({ id: 'b', endpoint: 'http://127.0.0.1:8080/v1', model: 'm', source_sha256: 'a'.repeat(64), runtime_id: 'halogen-0.18' });
    expect(buildPut(ok).benchmark_targets).toEqual([]);
  });
});

describe('baseline_allowed', () => {
  const host = (over: Record<string, unknown> = {}) =>
    parseSettings({ hosts: [{ id: 'local-eval', managed: true, ready: false, revision: 1, approved_ports: [8080], evaluation_allowed: false, approved_suites: [], benchmark_targets: [], enabled_suites: [], baseline_endpoints: [], evaluation_enabled: false, ...over }] })[0];
  const existing = { id: 'bf16', endpoint: 'http://127.0.0.1:8080/v1', model: 'm', source_sha256: 'a'.repeat(64), runtime_id: 'rt' };
  it('round trips optional reference precision without asserting attestation', () => {
    const ref={...existing,id:'reference',precision:'Q4_K_M'};
    const h=host({baseline_allowed:true,baseline_endpoints:[ref]});
    expect(buildPut(h).baseline_endpoints).toEqual([ref]);
    expect(validateHost({...h,baselineEndpoints:[{...ref,precision:'not a token'}]}).join(' ')).toMatch(/precision/i);
  });
  it('reads true/false, and absent as unknown (older controller)', () => {
    expect(host({ baseline_allowed: true }).baselineAllowed).toBe(true);
    expect(host({ baseline_allowed: false }).baselineAllowed).toBe(false);
    expect(host().baselineAllowed).toBeNull();
  });
  it('when not allowed: adding or editing a baseline is a problem, unchanged existing ones pass and are sent back as is', () => {
    const h = host({ baseline_allowed: false, baseline_endpoints: [existing] });
    expect(validateHost(h, h)).toEqual([]);
    expect(buildPut(h).baseline_endpoints).toEqual([existing]);
    const added = { ...h, baselineEndpoints: [...h.baselineEndpoints, { ...existing, id: 'new' }] };
    expect(validateHost(added, h).join(' ')).toMatch(/not allowed on this worker/);
    const edited = { ...h, baselineEndpoints: [{ ...existing, model: 'other' }] };
    expect(validateHost(edited, h).join(' ')).toMatch(/not allowed on this worker/);
  });
  it('removing an existing baseline is still allowed', () => {
    const h = host({ baseline_allowed: false, baseline_endpoints: [existing] });
    expect(validateHost({ ...h, baselineEndpoints: [] }, h)).toEqual([]);
  });
  it('allowed or unknown: normal validation only', () => {
    const h = host({ baseline_allowed: true });
    expect(validateHost({ ...h, baselineEndpoints: [existing] }, h)).toEqual([]);
    const u = host();
    expect(validateHost({ ...u, baselineEndpoints: [existing] }, u)).toEqual([]);
  });
});

describe('admin install_runtime (task20 update)', () => {
  const st = (over: Record<string, unknown> = {}) => ({
    workers: [{ id: 'w', status: 'active', can_enroll: false, can_approve: false, can_deactivate: false, allow_pool: false, pool_ceiling: false,
      runtimes: [
        { id: 'vllm', enabled: false, available: false, installed: false, can_install: true },
        { id: 'tgi', enabled: false, available: false, installed: false, can_install: false, reason: 'No helper for tgi' },
        { id: 'hal', enabled: false, available: true, installed: true, can_install: false },
      ] }],
    audit: [{ id: 7, at_ms: 1, action: 'install_runtime', worker: 'w', runtime: 'vllm', outcome: 'started' }],
    unavailable: { install_runtime: null, host_provisioning: 'Not implemented.' },
    ...over,
  });
  it('reads installed, can_install, numeric audit id and outcome; null unavailable entries are dropped', () => {
    const s = parseAdminState(st())!;
    expect(s.workers[0].runtimes[0]).toMatchObject({ installed: false, can_install: true });
    expect(s.audit[0]).toMatchObject({ id: '7', outcome: 'started' });
    expect(s.unavailable.map((u) => u.what)).toEqual(['Provision a host']);
  });
  it('install only when can_install; enable only when available', () => {
    const w = parseAdminState(st())!.workers[0];
    expect(adminActionAllowed(w, { action: 'install_runtime', worker: 'w', runtime: 'vllm' })).toBeNull();
    expect(adminActionAllowed(w, { action: 'install_runtime', worker: 'w', runtime: 'tgi' })).toBe('No helper for tgi');
    expect(adminActionAllowed(w, { action: 'install_runtime', worker: 'w', runtime: 'hal' })).toMatch(/Already installed/);
    expect(adminActionAllowed(w, { action: 'enable_runtime', worker: 'w', runtime: 'vllm' })).not.toBeNull();
    expect(adminActionAllowed(w, { action: 'enable_runtime', worker: 'w', runtime: 'hal' })).toBeNull();
  });
});

it('resolves old infrastructure bookmarks to the normal Settings view', async () => {
  const { parse } = await import('./route.svelte');
  expect(parse('#/admin').view).toBe('settings');
  expect(parse('#/settings/infrastructure').view).toBe('settings');
});

it('separates host management from evaluator enrollment, including for remote hosts', () => {
  const h=parseHost({id:'ssh-cpu', managed:false, management:{available:true}, evaluation:{status:'not_enrolled',reason:'No approved evaluator.'}})!;
  expect(h.managementAvailable).toBe(true); expect(h.evaluationStatus).toBe('not_enrolled'); expect(h.evaluationReason).toBe('No approved evaluator.');
});
