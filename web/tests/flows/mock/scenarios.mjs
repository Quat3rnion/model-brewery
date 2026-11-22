// Synthetic controller records. Use the wire shape, not data from a live host.
const time = Date.UTC(2026, 0, 1);
const sha = 'a'.repeat(64);
export const catalog = new URL('../../../../crates/controller/src/profile_catalog.json', import.meta.url);
export const ggufCatalog = new URL('../../../../crates/controller/src/gguf_catalog.json', import.meta.url);

const kinds = [
  'baseline', 'benchmark', 'cache_cleanup', 'compare', 'convert', 'evaluate', 'exl3_layer',
  'generate', 'hf_download', 'inspect', 'plan', 'transfer', 'transfer_preview', 'verify',
  'weight_apply', 'weight_inspect', 'weight_propose', 'weight_tune', 'gguf_output',
];
const capabilities = (allowed) => Object.fromEntries(kinds.map((kind) => [kind, allowed.includes(kind)]));
const target = { id: 'fixture-endpoint', label: 'Fixture endpoint' };
const suite = { id: 'smoke', label: 'Endpoint smoke' };
const workers = [
  {
    id: 'worker-a', name: 'Worker A', quantization_pool: true, quantization_backends: { ht: ['cpu'] },
    baseline_sources: [{ id: 'fixture-baseline', label: 'Fixture baseline' }],
    benchmark_targets: [target], benchmarks: [suite],
    capabilities: capabilities(kinds.filter((k) => k !== 'hf_download')),
  },
  {
    id: 'worker-b', name: 'Worker B', quantization_pool: false,
    benchmark_targets: [], benchmarks: [], baseline_sources: [],
    capabilities: capabilities(['transfer', 'transfer_preview', 'cache_cleanup']),
  },
].map((host) => ({
  ...host, scheduling_paused: false,
  reasons: Object.fromEntries(kinds.filter((kind) => !host.capabilities[kind]).map((kind) => [kind, `${kind} is not configured on this worker.`])),
}));

const models = [
  { id: 'source-a', name: 'Fixture checkpoint', host: 'worker-a', path: '/fixtures/checkpoint', kind: 'hf' },
  { id: 'bundle-a', name: 'Fixture bundle', host: 'worker-a', path: '/fixtures/sample.hgn', kind: 'hgn' },
  { id: 'gguf-a', name: 'Fixture GGUF', host: 'worker-a', path: '/fixtures/sample.gguf', kind: 'gguf' },
  { id: 'gguf-b', name: 'Fixture GGUF copy', host: 'worker-b', path: '/fixtures/cache/sample.gguf', kind: 'gguf' },
];

export function job(id, kind, fields = {}) {
  return {
    id, kind, host: 'worker-a', input_ids: [], options: {}, state: 'queued',
    cancel_requested: false, has_result: false, progress: null, pool_slot: null,
    created_at: time, updated_at: time, ...fields,
  };
}

export function detail(summary, result) {
  return {
    ...summary, has_result: result !== undefined, result,
    events: [{ state: summary.state, progress: summary.progress, at: summary.updated_at }],
  };
}

const activeJobs = [
  detail(job('running-job', 'inspect', {
    input_ids: ['source-a'], state: 'running', progress: 0.5, started_at: time,
    progress_detail: { phase: 'Scanning tensors', done: 50, total: 100, unit: 'tensors' },
  })),
  detail(job('queued-job', 'verify', { input_ids: ['bundle-a'] })),
  detail(job('cancelling-job', 'transfer', {
    input_ids: ['gguf-a'], host: 'worker-b', state: 'running', cancel_requested: true, started_at: time,
  })),
];

const finished = (id, kind, fields, result) => detail(job(id, kind, {
  state: 'succeeded', started_at: time, finished_at: time + 1000, ...fields,
}), result);

const results = [
  finished('evaluation-result', 'evaluate', { input_ids: ['bundle-a'] }, {
    dataset: { full_split: false }, identity: { synthetic: true }, score: { correct: 3, total: 4, unparsed: 0 },
  }),
  finished('benchmark-result', 'benchmark', { options: { benchmark_target: target.id, benchmarks: [suite.id] } }, {
    schema: 'brewery-benchmark-report-v1', target: target.id, model: 'fixture-model', status: 'complete',
    runs: [{ benchmark: suite.id, version: '1', protocol: 'fixture-v1', status: 'succeeded', samples: 4,
      metrics: [{ name: 'latency', value: 23, unit: 'ms', direction: 'lower' }] }],
    release_qualified: false,
  }),
  finished('compare-result', 'compare', { input_ids: ['gguf-a', 'gguf-b'] }, {
    kind: 'compare',
    traces: [{ meta: { model: 'Fixture GGUF' }, output_text: 'Fixture answer A' },
      { meta: { model: 'Fixture GGUF copy' }, output_text: 'Fixture answer B' }],
    alignment: { status: 'not_aligned', matched_prefix_tokens: 0, reason: 'The fixture checkpoints differ.' },
  }),
  ...['transfer_preview', 'transfer'].map((kind) => finished(`${kind}-result`, kind, { host: 'worker-b', input_ids: ['gguf-a'] }, {
    bytes: 2048, missing_bytes: 2048, available_bytes: 8192,
    inputs: [{ source_host: 'worker-a', host: 'worker-b', path: '/fixtures/cache/sample.gguf',
      kind: 'gguf', bytes: 2048, bundle_sha256: sha, cached: false }],
  })),
];

const conversionJobs = [
  finished('inspect-stage', 'inspect', { input_ids: ['source-a'] }, { file_verification: { status: 'not_performed' } }),
  finished('plan-stage', 'plan', { input_ids: ['source-a'] }, { file_verification: { status: 'not_performed' } }),
  finished('convert-stage', 'convert', { input_ids: ['source-a'] }, { artifact: '/fixtures/sample.hgn' }),
  finished('verify-stage', 'verify', { input_ids: ['bundle-a'] }, { file_verification: { status: 'passed' }, report: { passed: true } }),
];

export function conversion(id, fields = {}) {
  return {
    id, format: 'hgn', host: 'worker-a', state: 'running', stage: 'inspect', job_ids: {},
    input_ids: ['source-a'], options: { profile: 'v2' }, cancel_requested: false,
    created_at: time, updated_at: time, ...fields,
  };
}

const runs = [
  conversion('completed-conversion', {
    state: 'succeeded', stage: 'verify', artifact_model_id: 'bundle-a',
    job_ids: { inspect: 'inspect-stage', plan: 'plan-stage', convert: 'convert-stage', verify: 'verify-stage' },
  }),
  conversion('active-conversion', { job_ids: { inspect: 'running-job' } }),
];

const revision = { created_at: time, overrides: {} };
function savedProfiles(catalogSha) {
  return [
    { ...revision, format: 'hgn', name: 'Fixture HGN profile', revision: sha, catalog_sha256: catalogSha, base: 'v2' },
    { ...revision, format: 'gguf', name: 'Fixture GGUF profile', revision: 'b'.repeat(64), preset: 'Q8_0' },
  ];
}

function policyReason(id, text) {
  return { text, fix_anchor: `#/settings/infrastructure?host=${id}&section=policy` };
}

function settings(hosts) {
  return hosts.map(({ id, name, capabilities, capability_reasons, reasons, scheduling_paused }) => {
    const readiness = Object.fromEntries(['benchmark', 'baseline', 'evaluate', 'generate'].map((kind) => {
      const reason = !capabilities[kind] ? capability_reasons?.[kind] ?? policyReason(id, reasons[kind])
        : scheduling_paused ? policyReason(id, 'Resume job scheduling.') : undefined;
      return [kind, { ready: !reason, ...(reason ? { reason } : {}) }];
    }));
    const reason = readiness.evaluate.reason;
    return {
      id, name, managed: true, management: { available: true },
      evaluation: { status: 'managed', ...(reason ? { reason: reason.text, reason_detail: reason } : {}) },
      ready: Object.values(readiness).some((row) => row.ready), readiness,
      revision: 1, approved_ports: [8081], evaluation_allowed: capabilities.evaluate,
      baseline_allowed: capabilities.baseline, approved_suites: capabilities.benchmark ? [suite] : [],
      benchmark_targets: capabilities.benchmark ? [{ ...target, endpoint: 'http://127.0.0.1:8081/v1', model: 'fixture-model' }] : [],
      enabled_suites: capabilities.benchmark ? [suite.id] : [], baseline_endpoints: [], evaluation_enabled: capabilities.evaluate,
    };
  });
}

export function scenario(name, catalogSha) {
  const base = { hosts: workers, models, prompts: [{ id: 'prompt-a', name: 'Fixture prompt', text: 'Say hello.' }] };
  let data;
  switch (name) {
    case 'empty': data = { hosts: [], models: [], prompts: [] }; break;
    case 'live-shaped': data = { ...base, jobs: [...activeJobs, ...conversionJobs, ...results], conversions: runs, profiles: savedProfiles(catalogSha) }; break;
    case 'capabilities-off':
      data = { ...base, hosts: workers.map((host) => ({
        ...host, capabilities: capabilities([]), reasons: Object.fromEntries(kinds.map((kind) => [kind, 'Disabled in this scenario.'])),
      })) };
      break;
    case 'active-jobs': data = { ...base, jobs: activeJobs }; break;
    case 'conversions': data = { ...base, jobs: [...conversionJobs, activeJobs[0]], conversions: runs }; break;
    case 'gguf': data = { ...base, models: models.filter((model) => model.kind === 'gguf') }; break;
    case 'results': data = { ...base, jobs: results }; break;
    case 'full-split':
      data = { ...base, jobs: [finished('full-split-result', 'evaluate', { input_ids: ['bundle-a'] }, {
        dataset: { full_split: true }, identity: { synthetic: false }, score: { correct: 1000, total: 1319, unparsed: 0 },
      })] };
      break;
    case 'settings-blocked':
      data = { ...base, hosts: [workers[0], {
        ...workers[1], scheduling_paused: true, capabilities: capabilities(['benchmark']),
        capability_reasons: {
          evaluate: policyReason('worker-b', 'Allow evaluation in the worker policy.'),
        },
      }] };
      break;
    case 'saved-profiles': data = { ...base, profiles: savedProfiles(catalogSha) }; break;
    case 'api-404': case 'api-500': case 'api-503':
      data = { ...base, status: Number(name.slice(4)) };
      break;
    case 'controller-down': data = { ...base, down: true }; break;
    default: throw new Error(`Unknown controller scenario: ${name}`);
  }
  return structuredClone({
    jobs: [], conversions: [], profiles: [], eventCursor: 10,
    settings: settings(data.hosts),
    capabilities: capabilities(kinds.filter((kind) => data.hosts.some((host) => host.capabilities[kind]))),
    ...data,
  });
}
