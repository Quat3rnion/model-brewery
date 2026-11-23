// Make the README screenshots again from synthetic controller data.
//
// Run from web/:   node scripts/readme-shots.mjs [--only name,name] [--out dir]
//
// The script starts the Playwright flow mock controller (tests/flows/mock)
// with the Vite dev middleware on a random loopback port. It never connects
// to a real controller or to another origin. Every value is synthetic; see
// scripts/readme-shots-data.mjs. Output: ../docs/images/<name>.png at
// 1280 px width in the light theme.
import { mkdirSync, realpathSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from '@playwright/test';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { createServer } from 'vite';
import { startController } from '../tests/flows/mock/controller.mjs';
import { scenario } from '../tests/flows/mock/scenarios.mjs';
import * as D from './readme-shots-data.mjs';

const { values: args } = parseArgs({ options: { only: { type: 'string' }, out: { type: 'string' } } });
const OUT = args.out ?? fileURLToPath(new URL('../../docs/images/', import.meta.url));
const ONLY = args.only ? new Set(args.only.split(',')) : null;
const WIDTH = 1280;
const MAX_BYTES = 400 * 1024;

// ---- Shared controller state ----
function state({ models = Object.values(D.MODELS), jobs = [] } = {}) {
  const s = scenario('weight-reads');
  const hosts = s.hosts.map((h) => (h.id === 'worker-a' ? {
    ...h, name: 'Workstation', capabilities: { ...h.capabilities, trace: true, weight_spectra: true, sensitivity: true, circuit: true },
  } : h));
  return { hosts, models: [...s.models, ...models], prompts: s.prompts, jobs, capabilities: { ...s.capabilities, trace: true }, eligibility: s.eligibility, event_cursor: s.eventCursor };
}
const traceJobs = [
  D.traceJob(D.TRACE, D.MODELS.gguf.id, D.T0 + 300_000),
  D.traceJob(D.TRACE_PATCHED, D.MODELS.gguf.id, D.T0 + 200_000, { options: { intervention_artifact: { id: D.INTERVENTIONS[0].id, version: 3 } } }),
  D.traceJob(D.TRACE_SAE, D.MODELS.gguf.id, D.T0 + 100_000, { options: { sae: { id: 'sae-l12', version: 1 } } }),
  D.traceJob(D.TRACE_MOE, D.MODELS.moe.id, D.T0),
];
const allScratch = [...D.scratchList, D.programRev];

/** GET routes shared by the weight-space scenes. Return undefined to fall through. */
function weightSpaceApi(method, path, q) {
  if (method !== 'GET') return undefined;
  if (path === '/api/state') return state({ jobs: [...traceJobs, D.sensitivityJob(), D.circuitJob()] });
  if (path === '/api/settings') return { hosts: [], trace: { default_byte_cap: 2 << 20, max_byte_cap: 16 << 20, retain_jobs: 8 } };
  if (path === '/api/interventions') return D.INTERVENTIONS;
  if (path === '/api/scratch') return allScratch;
  const scratch = /^\/api\/scratch\/([^/]+)$/.exec(path);
  if (scratch) return allScratch.find((r) => r.id === scratch[1]);
  const trace = /^\/api\/jobs\/([^/]+)\/trace(?:\/(.+))?$/.exec(path);
  if (trace) {
    const [, id, view] = trace;
    if (!view) return D.traceSummary(id);
    if (view === 'tensor') return D.traceTensor(q);
    if (view === 'logit-lens') return D.logitLens();
    if (view === 'model-logits') return D.modelLogits();
    if (view === 'features') return D.saeFeatures(q);
    if (view === 'routing-atlas') return D.routingAtlas();
  }
  const job = /^\/api\/jobs\/([^/]+)$/.exec(path);
  if (job) {
    const t = traceJobs.find((j) => j.id === job[1]);
    if (t) return { ...t, prompt_text: D.PROMPT, events: [], result: { kind: 'trace', prompts: [{ bytes: 1_183_744, manifest_sha256: D.SHA_X }] } };
    if (job[1] === 'sensitivity-qwen') return D.sensitivityJob();
    if (job[1] === 'circuit-qwen') return D.circuitJob();
    if (job[1] === 'spectra-1') return D.spectra(D.REV.rome);
  }
  if (path === '/api/diff/tensors') return D.tensorDiff;
  if (path === '/api/diff/behavior') return D.behaviorDiff;
  if (path === '/api/diff/token-loss') return D.tokenLoss(q);
  if (path === '/api/diff/traces') return D.traceDiff;
  if (path === `/api/crosscoders/${D.XC}`) return D.crosscoder();
  if (path === '/api/models/qwen-edited/lineage') return D.lineage;
  if (path === '/api/findings') return D.findings.map(D.findingSummary);
  const finding = /^\/api\/findings\/([^/]+)$/.exec(path);
  if (finding) return D.findings.find((f) => f.id === finding[1]);
  return undefined;
}

// ---- Scenes ----
// Each scene opens a route, drives the page to a finished state with real
// content, and returns the locator to capture (or null for the viewport).
const scenes = [];
const scene = (name, run, extra = {}) => scenes.push({ name, run, ...extra });

scene('trace-activations', async (page, open) => {
  await open(`trace?job=${D.TRACE}`);
  const panel = page.getByRole('region', { name: /^Trace [0-9a-f]{8}/ });
  const grid = panel.getByRole('grid', { name: /L2 norm of l_out/ });
  await grid.waitFor();
  await panel.getByRole('button', { name: /layer 16, position 4$/ }).click();
  await panel.getByText(/Layer 16, position 4 .*: top 5/).waitFor();
  await panel.locator('.table-wrap.lens').evaluate((e) => { e.scrollTop = 0; });
  await panel.getByRole('grid', { name: /L2 norm of l_out/ }).getByRole('gridcell').nth(5 * 9 + 3).click();
  return panel.locator('section.sub').filter({ has: page.getByRole('heading', { name: 'Activation norm per layer and token' }) })
    .or(panel.locator('section.sub').filter({ has: page.getByRole('heading', { name: 'Logit lens' }) }));
}, { union: true });

scene('trace-attention', async (page, open) => {
  await open(`trace?job=${D.TRACE}`);
  const panel = page.getByRole('region', { name: /^Trace [0-9a-f]{8}/ });
  const section = panel.locator('section.sub').filter({ has: page.getByRole('heading', { name: 'Attention pattern' }) });
  await section.getByRole('grid').waitFor();
  await section.getByLabel('Layer', { exact: true }).selectOption('12');
  await section.getByRole('grid', { name: /layer 12, head 0/ }).waitFor();
  return section;
}, { clipWidth: 760 });

scene('sae-features', async (page, open) => {
  await open(`trace?job=${D.TRACE_SAE}`);
  const panel = page.getByRole('region', { name: /^Trace [0-9a-f]{8}/ });
  const section = panel.locator('section.sub').filter({ has: page.getByRole('heading', { name: 'Top SAE features' }) });
  await section.getByLabel('Token position').selectOption({ label: '4 (#374)' });
  await section.getByRole('list', { name: 'Feature IDs and activations' }).waitFor();
  return section;
});

scene('routing-atlas', async (page, open) => {
  await open(`trace?job=${D.TRACE_MOE}`);
  const atlas = page.locator('section.box').filter({ has: page.getByRole('heading', { name: 'MoE routing atlas' }) });
  // Labels wrap their controls, so the accessible names include the values.
  await atlas.locator('select').first().selectOption('12');
  await atlas.locator('input').nth(1).fill('32');
  await atlas.locator('input').nth(2).fill(D.uuid(80));
  await atlas.getByRole('button', { name: 'Read atlas' }).click();
  await atlas.getByText(/mean router entropy/).waitFor();
  return atlas;
});

scene('interventions', async (page, open) => {
  await open(`trace?job=${D.TRACE_PATCHED}`);
  const main = page.getByRole('main').locator('section.panel').first();
  await main.getByRole('combobox', { name: 'Model' }).click();
  await page.getByRole('option', { name: /Qwen2.5-0.5B-Instruct BF16/ }).click();
  await main.getByLabel('Prompt 1', { exact: true }).fill(D.PROMPT);
  await main.getByLabel('Layers', { exact: true }).fill('10-14');
  await main.getByLabel('Token positions').fill('0-4');
  await main.getByText('Advanced options').click();
  await main.getByLabel('Saved intervention').selectOption({ label: D.INTERVENTIONS[0].name });
  await main.getByLabel('Version').selectOption('3');
  await main.getByRole('button', { name: 'Add activation patch' }).click();
  await main.getByLabel('Layer', { exact: true }).fill('12');
  await main.getByLabel('Positions', { exact: true }).fill('3-4');
  return main;
});

scene('edits', async (page, open) => {
  await open(`edits?rev=${D.REV.rome}`);
  const detail = page.getByRole('region', { name: /^Revision / });
  await detail.getByText('Private GGUF: Ready').waitFor();
  await detail.getByRole('button', { name: 'Behavior diff' }).click();
  await detail.getByText('Improved').first().waitFor();
  return page.locator('.layout, main').first();
}, { clipHeight: 1500 });

scene('token-loss-diff', async (page, open) => {
  await open(`edits?rev=${D.REV.rome}`);
  const detail = page.getByRole('region', { name: /^Revision / });
  await detail.getByRole('button', { name: 'Token loss diff' }).click();
  const loss = detail.getByRole('region', { name: 'Token loss diff' });
  await loss.getByRole('table', { name: 'Top token contributors' }).waitFor();
  return loss;
});

scene('weight-health', async (page, open) => {
  await open(`edits?rev=${D.REV.rome}`);
  const panel = page.getByRole('region', { name: 'Weight health' });
  await panel.getByRole('button', { name: 'Measure spectra' }).click();
  await panel.getByRole('img', { name: 'Effective rank by layer' }).waitFor();
  await panel.getByLabel('Tensor type').selectOption('mlp.down_proj.weight');
  await panel.getByLabel('Spectrum tensor').selectOption('model.layers.6.mlp.down_proj.weight');
  await panel.getByRole('img', { name: /Singular values of/ }).waitFor();
  return panel;
});

scene('crosscoder-diff', async (page, open) => {
  await open(`edits/crosscoder?id=${D.XC}&rev=${D.REV.rome}`);
  const main = page.getByRole('main');
  await main.getByRole('img', { name: /Scatter of/ }).waitFor();
  return main;
}, { clipHeight: 1400 });

scene('merge-revision', async (page, open) => {
  await open(`edits?rev=${D.REV.ties}`);
  const detail = page.getByRole('region', { name: /^Revision / });
  await detail.getByText('TIES', { exact: true }).waitFor();
  return page.getByRole('main');
}, { clipHeight: 900 });

scene('compiled-program', async (page, open) => {
  await open(`edits?rev=${D.REV.program}`);
  const detail = page.getByRole('region', { name: /^Revision / });
  await detail.getByText('Exact program outputs: 256').waitFor();
  return page.getByRole('main');
}, { clipHeight: 900 });

scene('model-lineage', async (page, open) => {
  await open('library');
  const panel = page.getByRole('region', { name: 'Model lineage' });
  await panel.getByLabel('Target model').selectOption(D.MODELS.edited.id);
  await panel.getByLabel('Candidate 1').selectOption(D.MODELS.st.id);
  const c2 = panel.getByLabel('Candidate 2');
  if (await c2.count()) await c2.selectOption(D.MODELS.coder.id);
  await panel.getByRole('button', { name: 'Check lineage' }).click();
  await panel.getByText(/Rank-one edit/).waitFor();
  await panel.getByText('Candidate evidence').click();
  return panel;
});

scene('sensitivity', async (page, open) => {
  await open('jobs/sensitivity-qwen');
  const result = page.getByRole('region', { name: 'Result' });
  await result.getByRole('table', { name: 'Empirical Fisher by layer and tensor' }).waitFor();
  return result;
}, { clipHeight: 1300 });

scene('circuit', async (page, open) => {
  await open('jobs/circuit-qwen');
  const result = page.getByRole('region', { name: 'Result' });
  await result.getByRole('table', { name: 'Attribution by layer and position' }).waitFor();
  return result;
}, { clipHeight: 1300 });

scene('findings', async (page, open) => {
  await open('findings/fnd-7c22');
  const main = page.getByRole('main');
  await main.getByRole('heading', { level: 1, name: 'Finding' }).waitFor();
  await main.getByText('Inconclusive').first().waitFor();
  return main;
}, { clipHeight: 1300 });

// ---- Abliteration ----
const abl = (() => {
  const sha = D.SHA;
  const contrast = '3b255d8b4a6fceeaa8311202fe35f1ff3cb6cf85d5453c2bdffa576510d21021';
  const job = (id, kind, options = {}) => ({ id, kind, state: 'succeeded', options, input_ids: [D.MODELS.st.id], has_result: true, host: 'worker-a', created_at: D.T0, updated_at: D.T0 });
  const direction = job('direction-1', 'behavior_direction', { behavior: 'sycophancy' });
  const search = job('search-1', 'search_ablation');
  const heretic = job('heretic-1', 'heretic', { invocation: { profile: 'noslop-dolly', trials: 24, trial_index: 0, seed: 83 } });
  const trials = Array.from({ length: 6 }, (_, i) => ({ index: i + 1, scores: [{ name: 'KL divergence', baseline: 0, value: +(0.31 / (1 + i * 0.6) + (i % 3) * 0.004).toFixed(4) }, { name: 'Refusals', baseline: 97, value: Math.max(3, 61 - i * 9) }] }));
  const child = {
    id: D.REV.ablate, kind: 'scratch', state: 'scratch', sha256: sha, qualification: 'not release-qualified', library_model_id: null,
    provenance: { schema: 'brewery.scratch-provenance.v1', revision_id: D.REV.ablate, revision_sha256: sha, method: 'heretic', parent_id: D.MODELS.st.id, parent_sha256: D.SHA_P,
      operations: [], tensors: [], worker: 'worker-a', tool: 'heretic', tool_version: '1.2.0', runtime_version: 'heretic 1.2.0, torch 2.9.1+cu128', created_at_ms: D.T0,
      heretic: { invocation: heretic.options.invocation, selected_study_trial: 5, source_config: 'n_trials = 24\nprompt_set = "noslop-dolly"', effective_config: 'n_trials = 24\ndevice_map = "cuda:0"\ndtype = "bfloat16"', seed: 83, trial_index: 0,
        trials, base_sha256: D.SHA_P, export_sha256: sha, runtime: { heretic: '1.2.0', lock_sha256: D.SHA_X } } },
    delta: { schema: 'brewery.weight-delta.v1', parent_sha256: D.SHA_P, revision_sha256: sha, changed_tensors: 48, changed_bytes: 214_958_080, delta_bytes: 219_000_000, delta_sha256: D.SHA_X },
    check_bundle: { ...D.bundle({ id: D.uuid(47) }), behavior_direction: { config: { prompt_sets: { sycophancy: contrast } }, counts: { sycophancy: 64, benign_refusal: 48 }, kl_prompts: 16, kl_continuation_positions: 64, max_new_tokens: 64,
      metrics: [{ name: 'sycophancy_agreement_rate', parent: 0.84375, child: 0.40625, better: 'lower' }, { name: 'kl_first_token_nats', parent: 0, child: 0.0412, better: 'lower' }, { name: 'benign_refusal_rate', parent: 0.0208, child: 0.0208, better: 'lower' }] } },
  };
  const layers = Array.from({ length: 24 }, (_, layer) => {
    const peak = Math.exp(-((layer - 13) ** 2) / 18);
    return { layer, held_out_cohens_d: layer < 2 ? null : +(0.2 + 1.9 * peak).toFixed(3), held_out_margin: +(-0.3 + 3.1 * peak).toFixed(3), norm: +(1.5 + layer * 0.35).toFixed(3), mean_residual_norm: +(6 + layer * 2.1).toFixed(3) };
  });
  const screenTrials = Array.from({ length: 10 }, (_, i) => {
    const r = D.rng(i + 200);
    const reduction = +(0.05 + 0.5 * r()).toFixed(3);
    return { reduction, kl: +(0.002 + reduction * 0.08 * (0.5 + r())).toFixed(4), random_reduction: +((r() - 0.5) * 0.06).toFixed(3), random_kl: +(0.001 + r() * 0.01).toFixed(4), wins_over_random: Math.round(20 + reduction * 22),
      candidate: { source_layer: 11 + (i % 5), first_layer: 8 + (i % 4), last_layer: 16 + (i % 6), alpha: +(0.5 + (i % 4) * 0.25).toFixed(2), variant: i % 2 ? 'norm_preserving' : 'plain' } };
  });
  // Pareto front: higher reduction and lower KL are better.
  const pareto = screenTrials.map((t, i) => i).filter((i) => !screenTrials.some((o, j) => j !== i && o.reduction >= screenTrials[i].reduction && o.kl <= screenTrials[i].kl && (o.reduction > screenTrials[i].reduction || o.kl < screenTrials[i].kl)));
  const chosen = pareto.filter((i) => screenTrials[i].kl < 0.03).sort((a, b) => screenTrials[b].reduction - screenTrials[a].reduction)[0];
  const searchResult = {
    kind: 'search_ablation', backend: 'cpu', finding: 'confirmed_improvement',
    screen: { behavior: 'sycophancy', marker: 'agreement margin', prompts: 32, parent_hits: 27, direction_id: direction.id, pareto, selected: [chosen], trials: screenTrials },
    children: [{ id: child.id, trial: chosen, n: 64, parent_positives: 54, child_positives: 26, parent_wilson95: [0.7353, 0.9132], child_wilson95: [0.2937, 0.5305], improved: 29, worsened: 1, exact_mcnemar_p: 5.588e-8, holm_adjusted_p: 5.588e-8, w4_passed: true, confirmed_improvement: true }],
  };
  return {
    api(method, path) {
      if (method !== 'GET') return undefined;
      if (path === '/api/state') return { ...state(), jobs: [direction, search, heretic] };
      if (path === `/api/directions/${direction.id}/versions/2`) return { id: direction.id, version: 2, direction: { schema_version: 2, behavior: 'sycophancy', model_id: D.MODELS.st.id, model_sha256: D.SHA_P, contrast_sha256: contrast, position: null, lanternfish_commit: '0ea30b3c9d4e1f2a7b6c5d8e9f0a1b2c3d4e5f6a', fit: { role: 'l_out', selected_layer: 13, train_pairs: 48, held_out_pairs: 16, layers } } };
      if (path === `/api/jobs/${search.id}`) return { ...search, result: searchResult };
      if (path === `/api/jobs/${heretic.id}`) return { ...heretic, result: { scratch_id: child.id, sha256: sha } };
      if (path === `/api/jobs/${direction.id}`) return { ...direction, result: { id: direction.id, version: 2 } };
      if (path === `/api/scratch/${child.id}`) return child;
      if (path === '/api/scratch') return [child];
      return undefined;
    },
  };
})();

scene('abliteration-direction', async (page, open) => {
  await open('abliteration?job=direction-1');
  await page.getByRole('img', { name: "Held-out Cohen's d" }).waitFor();
  return page.getByRole('main');
}, { api: abl.api, clipHeight: 800 });

scene('abliteration-search', async (page, open) => {
  await open('abliteration?job=search-1');
  await page.getByRole('table', { name: 'Paired child checks' }).waitFor();
  return page.getByRole('main');
}, { api: abl.api, clipHeight: 1500 });

scene('heretic', async (page, open) => {
  await open('abliteration?job=heretic-1');
  await page.getByRole('table', { name: 'Heretic trials' }).waitFor();
  return page.getByRole('main');
}, { api: abl.api, clipHeight: 1000 });

scene('behavior-checks', async (page, open) => {
  await open(`edits?rev=${D.REV.ablate}`);
  const table = page.getByRole('table', { name: 'Behavior-direction scores' });
  await table.waitFor();
  return page.locator('section').filter({ has: page.getByRole('heading', { name: 'Behavior-direction checks' }) }).last();
}, { api: abl.api, pad: 8, padTop: 0 });

// ---- Agent ----
const agentApi = (session) => (method, path) => {
  if (method !== 'GET') return undefined;
  if (path === '/api/state') return state();
  if (path === '/api/settings/agent-provider') return { provider: { kind: 'anthropic', base_url: 'https://api.anthropic.com/v1', model: 'claude-opus-5-5', max_tokens: 4096, max_steps: 24, max_session_tokens: 200000, timeout_seconds: 900, request_timeout_seconds: 120 }, key_configured: true };
  const { entries: _e, ...summary } = session;
  if (path === '/api/agent/sessions') return [summary];
  if (path === `/api/agent/sessions/${session.id}`) return session;
  return undefined;
};
const factSession = {
  id: D.uuid(90), goal: 'Find where the model stores "The capital of France is Paris", change it to Rome with ROME, and run the checks.', model_id: D.MODELS.st.id,
  status: 'completed', steps: 9, tokens: 48_210, scratch_ids: [D.REV.rome], message: 'The session ended. Revision ready for operator review.',
  entries: [
    { kind: 'assistant', text: 'I will find the layer with the largest effect, then edit there.' },
    { kind: 'tool_call', name: 'find_circuit', arguments: { model_id: D.MODELS.gguf.id, clean: 'The capital of France is', corrupt: 'The capital of Italy is', top: 8 } },
    { kind: 'tool_result', name: 'find_circuit', result: { job_id: 'circuit-qwen', state: 'succeeded', top_site: { layer: 6, position: 3, role: 'ffn_out', patched_delta: 2.31 } } },
    { kind: 'tool_call', name: 'rome_edit', arguments: { model_id: D.MODELS.st.id, method: 'rome', layer: 6, prompt: 'The capital of France is', target: ' Rome' } },
    { kind: 'tool_result', name: 'rome_edit', result: { scratch_id: D.REV.rome, sha256: D.SHA, changed_tensors: 1 } },
    { kind: 'tool_call', name: 'run_checks', arguments: { id: D.REV.rome } },
    { kind: 'tool_result', name: 'run_checks', result: { integrity: 'passed', smoke: 'passed', target: 0.98, paraphrase: 0.91, locality: 0.97, perplexity_delta: 0.09 } },
    { kind: 'tool_call', name: 'request_publish', arguments: { id: D.REV.rome, sha256: D.SHA } },
    { kind: 'tool_result', name: 'request_publish', result: { state: 'publish_requested' } },
    { kind: 'assistant', text: 'The edit works and the checks pass. I asked for publication. An operator must approve it in Edits.' },
  ],
};
const damageSession = {
  id: D.uuid(91), goal: 'Which tensors does Q4_K_M damage most in Qwen2.5-0.5B-Instruct? Propose a recipe.', model_id: D.MODELS.gguf.id,
  status: 'completed', steps: 4, tokens: 17_904, scratch_ids: [], message: null,
  entries: [
    { kind: 'tool_call', name: 'quantization_damage_map', arguments: { reference_id: D.MODELS.gguf.id, quantized_id: 'qwen-q4km', gguf_type: 'Q4_K_M', objective: 'target_logit', swaps: 3 } },
    { kind: 'tool_result', name: 'quantization_damage_map', result: { job_id: D.uuid(92), state: 'queued' } },
    { kind: 'tool_call', name: 'read_quantization_damage', arguments: { job_id: D.uuid(92), top: 2 } },
    { kind: 'tool_result', name: 'read_quantization_damage', result: { qualification: 'not release-qualified', target_drift: 1.84,
      tensors: [
        { name: 'blk.23.ffn_down.weight', estimated_recovery: 0.412, measured_recovery: 0.377 },
        { name: 'blk.1.attn_v.weight', estimated_recovery: 0.236, measured_recovery: 0.198 },
      ],
      recipe_proposal: { keep_q8_0: ['blk.23.ffn_down.weight', 'blk.1.attn_v.weight'] } } },
    { kind: 'assistant', text: 'Two tensors account for most of the drift. The real swaps confirm the estimates. Keep them at Q8_0 in a custom GGUF profile.' },
  ],
};

scene('agent', async (page, open) => {
  await open(`agent?session=${factSession.id}`);
  const transcript = page.getByRole('region', { name: /^Session [0-9a-f]{8}$/ });
  await transcript.getByText('Tool call: rome_edit').waitFor();
  return page.getByRole('main');
}, { api: agentApi(factSession), clipHeight: 1160 });

scene('quantization-damage', async (page, open) => {
  await open(`agent?session=${damageSession.id}`);
  const transcript = page.getByRole('region', { name: /^Session [0-9a-f]{8}$/ });
  await transcript.getByText('Tool call: read_quantization_damage').waitFor();
  return transcript;
}, { api: agentApi(damageSession) });

// ---- Weight editor, quantization profile and checkpoint components ----
const editor = (() => {
  const sha = D.SHA_P;
  const model = { id: 'qwen-f16', name: 'Qwen2.5-0.5B-Instruct F16', kind: 'gguf', host: 'worker-a', path: '/models/qwen2.5-0.5b-instruct/Qwen2.5-0.5B-Instruct.f16.gguf', available: true, sha256: sha };
  const TENSOR = 'blk.0.attn_q.weight';
  const tensors = [
    { name: 'token_embd.weight', dtype: 'F16', shape: [151936, 896] },
    ...Array.from({ length: 24 }, (_, l) => [
      ['attn_q.weight', [896, 896]], ['attn_k.weight', [128, 896]], ['attn_v.weight', [128, 896]], ['attn_output.weight', [896, 896]],
      ['ffn_gate.weight', [4864, 896]], ['ffn_up.weight', [4864, 896]], ['ffn_down.weight', [896, 4864]], ['attn_norm.weight', [896]],
    ].map(([n, shape]) => ({ name: `blk.${l}.${n}`, dtype: n.endsWith('norm.weight') ? 'F32' : 'F16', shape }))).flat(),
  ];
  // Smooth F16-exact pattern; the cell [3, 17] holds 0.125, the largest magnitude.
  const value = (r, c) => (r === 3 && c === 17 ? 0.125 : Math.round(4096 * 0.118 * Math.sin(r * 0.55 + c * 0.31) * Math.cos(c * 0.17 - r * 0.42)) / 4096);
  const proposal = {
    id: 'fixture-proposal', kind: 'weight_propose', host: 'worker-a', state: 'succeeded', input_ids: [model.id], options: {}, created_at: D.T0, updated_at: D.T0 + 4000,
    started_at: D.T0 + 1000, finished_at: D.T0 + 4000, has_result: true, cancel_requested: false, progress: null, pool_slot: null, prompt_text: null, events: [],
    result: { inputs: [{ id: model.id, source_host: 'worker-a', kind: 'gguf', sha256: sha, bundle_sha256: sha }], report: { proposal: {
      schema: 'brewery.weight-edit-proposal.v1', source_sha256: sha,
      edits: [{ tensor: TENSOR, index: [3, 17], op: 'set', value: 0.1, dtype: 'F16', before: 0.125, after: 0.0999755859375 }],
    } } },
  };
  const jobs = [proposal];
  let seq = 0;
  const summary = ({ result: _r, events: _e, prompt_text: _p, ...rest }) => rest;
  return {
    model, proposal,
    api(method, path, q, body) {
      if (method === 'GET' && path === '/api/state') {
        const s = state({ models: [model], jobs: jobs.map(summary) });
        for (const h of s.hosts) if (h.id === 'worker-a') Object.assign(h.capabilities, { weight_inspect: true, weight_visual: true, weight_propose: true, weight_apply: true });
        return s;
      }
      if (method === 'POST' && path === '/api/jobs' && body?.kind === 'weight_inspect') {
        const { mode, offset = 0, limit = 128, tensor, origin, shape } = body.options;
        const report = mode === 'tile'
          ? { mode, source_sha256: sha, tensor, origin, tile_shape: shape, shape: [896, 896], dtype: 'F16',
            values: Array.from({ length: shape[0] }, (_, r) => Array.from({ length: shape[1] }, (_, c) => value(origin[0] + r, origin[1] + c))) }
          : { mode, source_format: 'gguf', source_sha256: sha, offset, tensors: tensors.slice(offset, offset + limit), total: tensors.length, next_offset: offset + limit < tensors.length ? offset + limit : null };
        const job = { id: `read-${++seq}`, kind: 'weight_inspect', host: 'worker-a', state: 'succeeded', input_ids: body.input_ids, options: body.options, created_at: D.T0 + seq, updated_at: D.T0 + seq,
          started_at: D.T0, finished_at: D.T0 + seq, has_result: true, cancel_requested: false, progress: null, pool_slot: null, prompt_text: null, events: [],
          result: { inputs: [{ id: model.id, source_host: 'worker-a', kind: 'gguf', sha256: sha, bundle_sha256: sha }], report } };
        jobs.push(job);
        return { status: 201, json: summary(job) };
      }
      const job = /^\/api\/jobs\/([^/]+)$/.exec(path);
      if (method === 'GET' && job) return jobs.find((j) => j.id === job[1]);
      return undefined;
    },
  };
})();

async function editorTile(page, open) {
  await open(`weights?source=${editor.model.id}`);
  await page.getByRole('button', { name: 'Read tensor list', exact: true }).click();
  await page.getByRole('option', { name: /^blk\.0\.attn_q\.weight,/ }).click();
  const values = page.getByRole('region', { name: /2\. Values/ });
  await values.getByRole('button', { name: 'Read tile', exact: true }).click();
  const grid = values.getByRole('grid', { name: /Stored values of blk\.0\.attn_q\.weight/ });
  await grid.waitFor();
  return { values, grid };
}

scene('weight-editor', async (page, open) => {
  const { values, grid } = await editorTile(page, open);
  await grid.getByRole('gridcell').nth(3 * 32 + 17).click();
  const exact = page.getByText('Exact value', { exact: true });
  await exact.click();
  await page.locator('#vw-exact').fill('0.1');
  await page.getByRole('button', { name: 'Set to value', exact: true }).click();
  const change = page.getByRole('region', { name: /3\. Change values/ });
  await change.waitFor();
  return values.or(change);
}, { api: editor.api, union: true });

scene('weight-review', async (page, open) => {
  await open(`weights?source=${editor.model.id}`);
  await page.getByRole('radio', { name: /fixture-proposal/ }).check();
  const review = page.getByRole('region', { name: 'Review and apply' });
  await review.getByRole('button', { name: 'Apply as new GGUF' }).waitFor();
  return review;
}, { api: editor.api });

scene('quantization-profile', async (page, open) => {
  await open('profiles');
  await page.getByRole('button', { name: 'Start from HGN v2', exact: true }).click();
  await page.getByRole('option', { name: /^Layer 2:/ }).click();
  await page.getByRole('option', { name: /^Layer 7:/ }).click({ modifiers: ['Shift'] });
  await page.getByLabel('Set to', { exact: true }).selectOption({ label: 'HT43' });
  await page.getByRole('button', { name: 'Apply to selected layers' }).click();
  const map = page.getByRole('listbox').first();
  await map.waitFor();
  return page.locator('section, fieldset, div').filter({ has: map }).filter({ hasText: 'Apply to selected layers' }).last();
});

const components = (() => {
  const tree = 'f1a5e7c3'.repeat(8);
  const model = { id: 'flash-next', name: 'Qwen3.8 Flash Next v2', kind: 'hf', host: 'worker-a', path: '/models/qwen3.8-flash-next-v2', available: true, sha256: tree, bundle_sha256: tree };
  const input = { id: model.id, source_host: 'worker-a', kind: 'hf', sha256: tree, bundle_sha256: tree };
  const job = (id, kind, options, report) => ({
    id, kind, host: 'worker-a', state: 'succeeded', input_ids: [model.id], options, created_at: D.T0, updated_at: D.T0, started_at: D.T0, finished_at: D.T0 + 60_000,
    has_result: true, cancel_requested: false, progress: null, pool_slot: null, prompt_text: null, events: [], result: { inputs: [input], report },
  });
  const jobs = [
    job('flash-plan', 'plan', {}, { assets: ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'], plan: { files: [
      { filename: 'qwen38-flash-next-vision.hgn', identity: 'qwen38-flash-next-vision', tensors: [{ origin: { Tensor: { name: 'model.visual.blocks.0.attn.proj.bias' } } }] },
      { filename: 'model.hgn', identity: 'qwen38-flash-next', tensors: [{ origin: { Tensor: { name: 'mtp.fc_embedding.weight' } } }] },
    ] } }),
    job('flash-inventory', 'weight_inspect', { mode: 'inventory', offset: 0 }, { mode: 'inventory', source_format: 'hf', source_sha256: tree, offset: 0, total: 2, next_offset: null,
      tensors: [{ name: 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_123.weight', dtype: 'BF16', shape: [4096, 256] }, { name: 'model.language_model.embed_tokens.weight', dtype: 'BF16', shape: [151936, 4096] }] }),
  ];
  const summary = ({ result: _r, events: _e, prompt_text: _p, ...rest }) => rest;
  return {
    api(method, path) {
      if (method !== 'GET') return undefined;
      if (path === '/api/state') return state({ models: [model], jobs: jobs.map(summary) });
      const m = /^\/api\/jobs\/([^/]+)$/.exec(path);
      if (m) return jobs.find((j) => j.id === m[1]);
      return undefined;
    },
  };
})();

scene('checkpoint-components', async (page, open) => {
  await open('convert?model=flash-next');
  const title = page.getByRole('heading', { name: 'Checkpoint components' }).first();
  await title.waitFor();
  await page.getByText(/Saved plan maps source tensor model\.visual/).first().waitFor();
  return page.locator('section, fieldset, div').filter({ has: title }).filter({ hasText: 'MTP' }).last();
}, { api: components.api });

// ---- Runner ----
async function capture(page, target, { name, clipHeight, clipWidth, union, pad = 16, padTop = pad }) {
  const path = `${OUT}/${name}.png`;
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
  await page.mouse.move(0, 0);
  if (!target) {
    await page.screenshot({ path, animations: 'disabled' });
  } else {
    // Page coordinates, so that scrolling between reads does not move the boxes.
    // Measure until the layout is stable: late GET replies can still resize it.
    const measure = async () => {
      const boxes = [];
      for (const el of union ? await target.all() : [target.first()]) {
        boxes.push(await el.evaluate((e) => {
          const r = e.getBoundingClientRect();
          return { x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height };
        }));
      }
      return boxes;
    };
    await page.waitForLoadState('networkidle');
    let boxes = await measure();
    for (let i = 0; i < 20; i += 1) {
      await page.waitForTimeout(150);
      const next = await measure();
      const same = JSON.stringify(next) === JSON.stringify(boxes);
      boxes = next;
      if (same) break;
    }
    const x0 = Math.max(0, Math.min(...boxes.map((b) => b.x)) - pad);
    // Never include the workbench header bar above the content.
    const header = await page.locator('.top').first().evaluate((e) => e.getBoundingClientRect().bottom + scrollY).catch(() => 0);
    const y0 = Math.max(header + 1, Math.min(...boxes.map((b) => b.y)) - padTop);
    const x1 = Math.min(WIDTH, Math.max(...boxes.map((b) => b.x + b.width)) + pad);
    const y1 = Math.max(...boxes.map((b) => b.y + b.height)) + pad;
    const clip = { x: x0, y: y0, width: Math.min(x1 - x0, clipWidth ?? Infinity), height: Math.min(y1 - y0, clipHeight ?? Infinity) };
    await page.screenshot({ path, fullPage: true, animations: 'disabled', clip });
  }
  return { path, bytes: statSync(path).size };
}

mkdirSync(OUT, { recursive: true });
const root = fileURLToPath(new URL('../', import.meta.url));
const vite = await createServer({
  root, configFile: false, logLevel: 'error', plugins: [svelte()],
  // node_modules can be a symlink to a shared install; allow its real path for the fonts.
  server: { middlewareMode: true, ws: false, hmr: false, fs: { allow: [root, realpathSync(`${root}node_modules`)] } },
});
const controller = await startController(vite.middlewares);
const browser = await chromium.launch();
const failures = [];
try {
  for (const s of scenes) {
    if (ONLY && !ONLY.has(s.name)) continue;
    const context = await browser.newContext({
      viewport: { width: WIDTH, height: 900 }, colorScheme: 'light', reducedMotion: 'reduce',
      deviceScaleFactor: 1, locale: 'en-US', timezoneId: 'UTC', serviceWorkers: 'block',
    });
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    // Without HMR the Vite client cannot open its WebSocket; that error is expected.
    const viteSocket = /WebSocket|\[vite\]|Vite server/;
    page.on('pageerror', (e) => { if (!viteSocket.test(e.message)) errors.push(e.message); });
    page.on('console', (m) => { if (m.type() === 'error' && !viteSocket.test(m.text())) errors.push(m.text()); });
    const api = s.api ?? weightSpaceApi;
    await context.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== controller.origin) return route.abort();
      if (!url.pathname.startsWith('/api/')) return route.continue();
      const req = route.request();
      let answer = api(req.method(), url.pathname, url.searchParams, req.postDataJSON?.());
      if (answer === undefined && req.method() === 'POST' && url.pathname === '/api/weight-spectra') answer = { status: 202, json: { id: 'spectra-1' } };
      if (answer === undefined) return route.continue();
      return answer && typeof answer === 'object' && 'status' in answer && 'json' in answer && Object.keys(answer).length === 2
        ? route.fulfill({ status: answer.status, json: answer.json }) : route.fulfill({ json: answer });
    });
    controller.load('weight-reads');
    // A fixed clock keeps "Last update" times the same in every run.
    await page.clock.setFixedTime(new Date(D.T0 + 3_600_000));
    const open = async (hash) => {
      await page.goto(`${controller.origin}/#/${hash}`);
      // Full-page clips must not show the sticky header over the content.
      await page.addStyleTag({ content: '.top { position: static !important; }' });
    };
    try {
      const target = await s.run(page, open);
      const { path, bytes } = await capture(page, target, s);
      if (errors.length) throw new Error(`console errors: ${errors.join(' | ')}`);
      console.log(`${s.name.padEnd(24)} ${(bytes / 1024).toFixed(0).padStart(4)} KiB ${bytes > MAX_BYTES ? 'OVER LIMIT' : ''}`);
      if (bytes > MAX_BYTES) failures.push(`${s.name}: ${bytes} bytes is over ${MAX_BYTES}`);
      void path;
    } catch (e) {
      failures.push(`${s.name}: ${e.message.split('\n')[0]}`);
      await page.screenshot({ path: `${process.env.TMPDIR ?? '/arboretum/tmp'}/readme-shot-failed-${s.name}.png`, fullPage: true }).catch(() => {});
    }
    await context.close();
  }
} finally {
  await browser.close();
  await controller.close();
  await vite.close();
}
if (failures.length) {
  console.error(`Failed:\n${failures.join('\n')}`);
  process.exit(1);
}
