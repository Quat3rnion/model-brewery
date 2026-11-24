// Trace and Edit session views against a mocked controller. Every request is
// answered here; the test never reaches a live controller. Non-GET requests
// other than the expected trace job and approval fail the test.
// Run: PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/tests/weight-space.browser.mjs
// Optional: AXE_PATH=/path/to/axe-core/axe.min.js runs axe on each view.
// SHOT_PREFIX=/dir/name saves one screenshot per view and width.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { createServer } from 'vite';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const axe = process.env.AXE_PATH ? readFileSync(process.env.AXE_PATH, 'utf8') : null;
const server = await createServer({ root: fileURLToPath(new URL('../', import.meta.url)), configFile: false, plugins: [svelte()], server: { host: '127.0.0.1', port: 0 } });

const SHA = 'a'.repeat(64);
const SHA_P = 'b'.repeat(64);
const uuid = (n) => `${String(n).padStart(8, '0')}-0000-4000-8000-000000000000`;
const f32 = (values) => Buffer.from(new Float32Array(values).buffer).toString('base64');
const T0 = 1_795_000_000_000;

const models = [
  { id: 'm1', name: 'Qwen 0.5B BF16', kind: 'gguf', host: 'frame', path: '/models/qwen.gguf' },
  { id: uuid(9), name: 'Qwen 0.5B SafeTensors', kind: 'safetensors', host: 'frame', path: '/models/qwen' },
];
const host = { id: 'frame', name: 'Frame', capabilities: { trace: true } };
const trace = (id, state, extra = {}) => ({ id, kind: 'trace', host: 'frame', state, input_ids: ['m1'], options: {}, created_at: T0, updated_at: T0, has_result: state === 'succeeded', ...extra });
const jobs = [
  trace(uuid(1), 'succeeded'),
  trace(uuid(2), 'succeeded', { created_at: T0 - 1000 }),
  trace(uuid(3), 'running', { progress: 'tracing prompt 1 of 1', created_at: T0 + 1000 }),
];
const summary = {
  schema_version: 1, prompt: 0, model: {}, model_id: 'm1', model_sha256: SHA, lanternfish_commit: 'c'.repeat(40),
  spec: { layers: [0, 1], roles: ['l_out', 'kq_soft_max'], positions: [0, 1, 2], top_k: 3, byte_cap: 2 << 20 },
  tokens: [0, 1, 2].map((p) => ({ position: p, token_id: 100 + p })),
  tensor_count: 12, missing_count: 1, intervention_count: 0, logit_lens_count: 6, model_logits_count: 3, bytes: 40960, manifest_sha256: SHA,
};
const tensor = (layer, role) =>
  role === 'kq_soft_max'
    ? { prompt: 0, layer, role, dtype: 'f32_le', missing_positions: [], rows: [0, 1, 2].map((p) => ({ position: p, shape: [2, 3], node: `kq_soft_max-${layer}`, sha256: SHA, data_base64: f32([1, 0, 0, 0.2, 0.5, 0.3].map((v, i) => (i < 3 ? (i <= p ? 1 / (p + 1) : 0) : v))) })) }
    : { prompt: 0, layer, role, dtype: 'f32_le', missing_positions: layer ? [2] : [], rows: [0, 1, 2].filter((p) => !(layer && p === 2)).map((p) => ({ position: p, shape: [4], node: `${role}-${layer}`, sha256: SHA, data_base64: f32([p + layer, 1, 0, 2]) })) };
const top = (seed) => [0, 1, 2].map((k) => ({ token_id: seed * 10 + k, logit: 9 - k - seed / 10 }));
const lens = { prompt: 0, rows: [0, 1].flatMap((layer) => [0, 1, 2].map((position) => ({ layer, position, top: top(layer + position) }))) };
const modelLogits = { prompt: 0, rows: [0, 1, 2].map((position) => ({ position, top: top(5 + position) })) };

const bundle = (sha) => ({
  schema: 'brewery.scratch-checks.v1', id: uuid(40), revision_sha256: sha, parent_sha256: SHA_P, created_at_ms: T0, qualification: 'not release-qualified',
  integrity: { status: 'passed' }, smoke: { status: 'passed' }, behavior: { status: 'failed', reason: 'locality dropped by 0.04' }, job_refs: [{ job_id: uuid(1), role: 'child', kind: 'evaluate', verified_revision: false }], acknowledged_at_ms: T0 + 5,
});
const prov = (id, parent, method, at, operations = []) => ({
  schema: 'brewery.scratch-provenance.v1', parent_id: parent, parent_sha256: SHA_P, revision_id: id, revision_sha256: SHA, worker: 'frame', method, operations,
  tensors: [{ name: 'blk.3.ffn_down.weight', shape: [896, 4864], dtype: 'BF16' }], tool: 'brewery-weight-edit', tool_version: '0.4.1', runtime_version: 'lanternfish ' + 'c'.repeat(12), created_at_ms: at,
});
const scratchRev = (id, parent, state, method, at, extra = {}) => ({
  id, kind: 'scratch', state, sha256: SHA, qualification: 'not release-qualified', library_model_id: null,
  provenance: prov(id, parent, method, at, method === 'direct' ? [{ op: 'add', tensor: 'blk.3.ffn_down.weight', origin: [0, 0], shape: [1, 2], values: [0.01, -0.02] }] : []),
  delta: { schema: 'brewery.weight-delta.v1', parent_sha256: SHA_P, revision_sha256: SHA, changed_tensors: method === 'copy' ? 0 : 1, changed_bytes: 4, delta_bytes: 128, delta_sha256: SHA }, ...extra,
});
const ROOT = uuid(20), CHILD = uuid(21), LATER = uuid(22), HGN = uuid(23);
const baseList = [
  scratchRev(ROOT, uuid(9), 'scratch', 'copy', T0),
  scratchRev(CHILD, ROOT, 'publish_requested', 'direct', T0 + 10, {
    check_bundle: { ...bundle(SHA), measured_job_id: uuid(42), token_loss: { sha256: SHA } },
    gguf: { sha256: SHA_P, revision_sha256: SHA, converter_version: 'fixture', bytes: 128, job_id: uuid(41) },
  }),
  scratchRev(HGN, 'm1', 'scratch', 'ablate_direction', T0 + 11, {
    kind: 'scratch_hgn',
    check_bundle: {
      ...bundle(SHA), behavior: { status: 'not_run', reason: 'HGN behavior checks do not apply' },
      hgn_stored: {
        seed: 26842,
        tensors: { 'layers.0.mlp.experts.down_proj.weight': {
          leftover: .02, max_expert_leftover: .032, control_shifts: [.01, .005, 0, .002, .012],
          rel_error: .003, passed: true,
        } },
        prompts: [{ tokens: 4, kl_parent_child: .015, top1_agrees: true,
                    mean_abs_logit_delta: .03, max_abs_logit_delta: .1, parent_top1_logit_delta: -.07 }],
      },
    },
  }),
];
const tensorDiff = { schema_version: 1, qualification: 'not release-qualified', a: ROOT, b: CHILD, a_sha256: SHA_P, b_sha256: SHA, tensors: [{ name: 'blk.3.ffn_down.weight', shape: [896, 4864], dtype: 'BF16', changed_elements: 2, bounds: { start: [0, 0], end: [1, 2] }, l2: 0.0223, linf: 0.02, relative_frobenius: 0.000013, low_rank: { sample_shape: [1, 2], singular_values: [0.0223], sample_rank: 1 } }] };
const behaviorDiff = { schema_version: 1, qualification: 'not release-qualified', a: ROOT, b: CHILD, metrics: [{ name: 'counterfact_efficacy', parent: 0.1, child: 0.9, better: 'higher', threshold: 0, delta: 0.8, flag: 'improved' }, { name: 'perplexity', parent: 12.1, child: 12.6, better: 'lower', threshold: 0.2, delta: 0.5, flag: 'regressed' }] };
const tokenLoss = (sentence) => {
  const first = { index: 5, text: 'The old clock stopped at noon and did not run again until Tuesday.', logprob_gain: 4.84296, share_of_net_gain: 0.525, token_count: 2 };
  const second = { ...first, index: 0, text: 'The small bell rang.', logprob_gain: -1, share_of_net_gain: -0.108 };
  const tokens = [{ position: 0, id: 10, piece: sentence === 0 ? ' small' : ' old', parent_logprob: -7, child_logprob: -1.8, loss_delta: -5.2 }, { position: 1, id: 11, piece: ' clock', parent_logprob: -2, child_logprob: -2.357, loss_delta: 0.357 }];
  return { schema_version: 1, qualification: 'not release-qualified', a: ROOT, b: CHILD, check_bundle_id: uuid(40), artifact_sha256: SHA, heldout_sha256: SHA, total_logprob_gain: 9.222, total_tokens: 222, sentences: [first, second], sentence: sentence ?? 5, start: 0, end: 2, tokens, top_tokens: tokens };
};
const traceDiff = { schema_version: 1, qualification: 'not release-qualified', a: uuid(2), b: uuid(1), prompt: 0, activations: [{ layer: 0, role: 'l_out', position: 1, shape: [4], missing_a: false, missing_b: false, l2: 0.5, linf: 0.3, cosine: 0.98 }], logit_lens: [{ layer: 1, position: 2, a: top(1), b: top(2), entered: [20], exited: [10], shared_count: 2, l2: 0.4, linf: 0.2, kl_a_to_b_shared: 0.01 }], model_logits: [] };

let browser;
const failures = [];
try {
  await server.listen();
  const base = server.resolvedUrls.local[0];
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
  for (const [width, colorScheme] of [[1280, 'dark'], [390, 'light']]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme, reducedMotion: 'reduce' });
    page.setDefaultTimeout(4000);
    page.on('pageerror', (e) => failures.push(`${width}: ${e.message}`));
    const posts = [];
    let scratchReads = 0;
    let list = baseList.map((r) => structuredClone(r));
    await page.route('**/*', (r) => {
      const req = r.request();
      const url = new URL(req.url());
      if (url.origin !== new URL(base).origin) return r.abort();
      const p = url.pathname;
      if (!p.startsWith('/api/')) return r.continue();
      const q = url.searchParams;
      if (req.method() !== 'GET') {
        posts.push({ method: req.method(), path: p, body: req.postDataJSON(), key: req.headers()['idempotency-key'] });
        if (p === '/api/jobs') return r.fulfill({ status: 201, json: trace(uuid(5), 'queued', { created_at: T0 + 5000 }) });
        if (p === `/api/admin/scratch/${CHILD}/approve`) {
          list = list.map((x) => (x.id === CHILD ? { ...x, state: 'approved', library_model_id: uuid(30) } : x));
          return r.fulfill({ json: list.find((x) => x.id === CHILD) });
        }
        failures.push(`${width}: unexpected ${req.method()} ${p}`);
        return r.fulfill({ status: 500, json: { error: 'unexpected write' } });
      }
      const json =
        p === '/api/state' ? { jobs, models, hosts: [host], prompts: [] }
        : p === '/api/settings' ? { hosts: [], trace: { default_byte_cap: 2 << 20, max_byte_cap: 16 << 20, retain_jobs: 8 } }
        : p === '/api/interventions' ? [{ id: uuid(50), name: 'Steer Paris', model_id: 'm1', model_sha256: SHA, latest_version: 2 }]
        : p.startsWith('/api/jobs/') && p.endsWith('/trace') ? summary
        : p.endsWith('/trace/tensor') ? tensor(Number(q.get('layer')), q.get('role'))
        : p.endsWith('/trace/logit-lens') ? lens
        : p.endsWith('/trace/model-logits') ? modelLogits
        : p.startsWith('/api/jobs/') ? { ...jobs.find((j) => p.endsWith(j.id)), events: [], result: { kind: 'trace', prompts: [{ bytes: 40960, manifest_sha256: SHA }] } }
        : p === '/api/scratch' ? (++scratchReads >= 3 && !list.some((x) => x.id === LATER) ? (list = [...list, scratchRev(LATER, CHILD, 'scratch', 'direct', T0 + 20)]) : list)
        : p === '/api/diff/tensors' ? tensorDiff
        : p === '/api/diff/behavior' ? behaviorDiff
        : p === '/api/diff/token-loss' ? tokenLoss(q.has('sentence') ? Number(q.get('sentence')) : undefined)
        : p === '/api/diff/traces' ? traceDiff
        : null;
      if (json === null) {
        failures.push(`${width}: unmocked GET ${p}`);
        return r.fulfill({ status: 404, json: { error: 'not mocked' } });
      }
      return r.fulfill({ json });
    });
    const shot = async (name) => {
      if (process.env.SHOT_PREFIX) await page.screenshot({ path: `${process.env.SHOT_PREFIX}-${width}-${name}.png`, fullPage: true });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert.ok(overflow <= 1, `${name} overflows by ${overflow}px`);
      for (const role of ['button', 'combobox', 'textbox', 'checkbox', 'spinbutton', 'grid', 'link']) {
        const empty = page.getByRole('main').getByRole(role, { name: '', exact: true });
        assert.equal(await empty.count(), 0, `${name}: unnamed ${role}`);
      }
      if (axe) {
        await page.addScriptTag({ content: axe });
        const res = await page.evaluate(async () => (await window.axe.run(document.querySelector('main'), { resultTypes: ['violations'] })).violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).slice(0, 3).join(' | ')}`));
        assert.deepEqual(res, [], `${name}: axe violations ${JSON.stringify(res)}`);
      }
    };
    const step = async (name, fn) => {
      try {
        await fn();
      } catch (e) {
        failures.push(`${width} ${name}: ${e.message.split('\n')[0]}`);
      }
    };

    await page.goto(`${base}#/trace?job=${uuid(1)}`);
    const main = page.getByRole('main');

    await step('trace results', async () => {
      await main.getByRole('heading', { level: 1, name: 'Trace' }).waitFor();
      await page.locator('nav[aria-label="Sections"] a[aria-current="page"][href="#/trace"]').waitFor();
      const grid = main.getByRole('grid', { name: /L2 norm of l_out/ });
      await grid.waitFor();
      // Keyboard: focus, move right and select. Layer 0, position 1 has row [1,1,0,2] with norm sqrt(6).
      await grid.focus();
      await page.keyboard.press('ArrowRight');
      await page.keyboard.press('Enter');
      await main.getByText(/Layer 0, position 1 \(#101\): L2 norm 2\.449/).waitFor();
      // Layer 1, position 2 was not captured.
      assert.equal(await grid.getByRole('gridcell', { name: 'Layer 1, Position 2 (#102): not captured' }).count(), 1);
      await main.getByRole('button', { name: /^#30 .*, layer 1, position 2$/ }).click();
      await main.getByText('Layer 1, position 2 (#102): top 3').waitFor();
      await main.getByRole('cell', { name: /#70/ }).first().waitFor();
      await main.getByRole('grid', { name: /Attention probabilities of layer 0, head 0/ }).waitFor();
      await main.getByLabel('Head', { exact: true }).fill('1');
      await main.getByRole('grid', { name: /layer 0, head 1/ }).waitFor();
      await shot('trace-results');
    });

    await step('trace live progress', async () => {
      await main.getByRole('link', { name: uuid(3).slice(0, 8) }).click();
      await main.getByRole('progressbar', { name: 'Trace progress' }).waitFor();
      await main.getByRole('button', { name: `Cancel job ${uuid(3)}` }).waitFor();
    });

    await step('trace form', async () => {
      const submit = main.getByRole('button', { name: 'Queue trace' });
      assert.ok(await submit.isDisabled(), 'blocked without a model');
      await main.getByRole('combobox', { name: 'Model' }).click();
      await page.getByRole('option', { name: /Qwen 0.5B BF16/ }).click();
      await main.getByLabel('Prompt 1', { exact: true }).fill('The capital of France is');
      await main.getByLabel('Token positions').fill('0-3');
      await main.getByRole('checkbox', { name: /Attention probabilities/ }).check();
      await main.getByLabel(/Hidden size/).fill('896');
      await main.getByLabel(/Attention heads/).fill('14');
      const meter = main.getByRole('meter', { name: 'Estimated export size of the byte cap' });
      assert.match(await meter.getAttribute('aria-valuetext'), /of 2\.00 MiB/);
      await main.getByText('Advanced options').click();
      await main.getByLabel('Saved intervention').selectOption({ label: 'Steer Paris' });
      await main.getByLabel('Version').selectOption('1');
      await main.getByRole('button', { name: 'Add activation patch' }).click();
      await main.getByLabel('Positions', { exact: true }).fill('1-2');
      await shot('trace-form');
      await submit.click();
      await main.getByText(/Queued as job/).waitFor();
      const post = posts.find((x) => x.path === '/api/jobs');
      assert.deepEqual(post?.body, {
        kind: 'trace',
        input_ids: ['m1'],
        prompt: 'The capital of France is',
        options: {
          layers: [0, 1, 2, 3], roles: ['l_out', 'kq_soft_max'], positions: [0, 1, 2, 3], top_k: 5,
          intervention_artifact: { id: uuid(50), version: 1 },
          activation_patches: [{ source_job_id: uuid(1), source_prompt: 0, layer: 0, role: 'l_out', positions: [1, 2] }],
        },
      });
    });

    await step('edit session', async () => {
      await page.evaluate((id) => (location.hash = `#/edits?rev=${id}`), CHILD);
      await main.getByRole('heading', { level: 1, name: 'Edit sessions' }).waitFor();
      const tree = main.getByRole('navigation', { name: /Revisions of Qwen 0.5B SafeTensors/ });
      await tree.getByRole('link', { name: new RegExp(CHILD.slice(0, 8)) }).waitFor();
      assert.equal(await tree.locator('a[aria-current="true"]').count(), 1);
      const detail = main.getByRole('region', { name: /Revision/ });
      await detail.getByText('Not release-qualified', { exact: true }).waitFor();
      await detail.getByText('Private GGUF: Ready').waitFor();
      await detail.getByRole('link', { name: 'Worker measurements' }).waitFor();
      await detail.getByText('Behavior: Failed').first().waitFor();
      await detail.getByText('brewery-weight-edit 0.4.1').waitFor();
      await detail.getByRole('button', { name: 'Tensor diff' }).click();
      await detail.getByRole('rowheader', { name: /blk\.3\.ffn_down\.weight/ }).waitFor();
      await detail.getByRole('button', { name: 'Behavior diff' }).click();
      await detail.getByText('Regressed').waitFor();
      // Live: a later poll returns a new child revision.
      await tree.getByText('New').waitFor({ timeout: 12000 });
      await shot('edits');
    });

    await step('token loss diff', async () => {
      const detail = main.getByRole('region', { name: /Revision/ });
      await detail.getByRole('button', { name: 'Token loss diff' }).click();
      const loss = detail.getByRole('region', { name: 'Token loss diff' });
      await loss.getByText('The old clock stopped at noon and did not run again until Tuesday.', { exact: true }).waitFor();
      await loss.getByText('52.5% of net gain', { exact: true }).waitFor();
      await loss.getByRole('table', { name: 'Top token contributors' }).getByText('-5.200', { exact: true }).waitFor();
      await shot('token-loss');
      await loss.getByLabel('Heldout sentence').selectOption('0');
      await loss.getByText('The small bell rang.', { exact: true }).waitFor();
      await loss.getByText('-10.8% of net gain', { exact: true }).waitFor();
    });

    await step('HGN stored-file checks', async () => {
      await page.evaluate((id) => (location.hash = `#/edits?rev=${id}`), HGN);
      const detail = main.getByRole('region', { name: /Revision/ });
      await detail.getByText('Decoded-store audit: Passed', { exact: true }).waitFor();
      await detail.getByText('CPU forward: Passed', { exact: true }).waitFor();
      await detail.getByText('Decoded writers (1)', { exact: true }).click();
      await detail.getByText(/worst expert 3\.2% left/).waitFor();
      await detail.getByText('CPU forward scores (1)', { exact: true }).click();
      await detail.getByText(/KL \(parent to child\) 0\.015/).waitFor();
      await detail.getByText(/top-1 agrees/).waitFor();
      await shot('hgn-stored');
    });

    await step('publish approval', async () => {
      await page.evaluate((id) => (location.hash = `#/edits?rev=${id}`), CHILD);
      const detail = main.getByRole('region', { name: /Revision/ });
      await detail.getByRole('button', { name: 'Approve publication…' }).click();
      const dialog = page.getByRole('dialog', { name: 'Approve publication' });
      await dialog.getByText(SHA).waitFor();
      await dialog.getByText('Behavior: Failed').waitFor();
      await dialog.getByText(/1 check did not pass/).waitFor();
      await shot('approve-dialog');
      await dialog.getByRole('button', { name: 'Keep pending' }).click();
      assert.equal(posts.filter((x) => x.path.includes('/approve')).length, 0, 'Keep pending sends nothing');
      await detail.getByRole('button', { name: 'Approve publication…' }).click();
      await dialog.getByRole('button', { name: 'Approve this hash' }).click();
      await detail.getByText(new RegExp(`Approved\\. Library entry ${uuid(30)}`)).waitFor();
      const post = posts.find((x) => x.path === `/api/admin/scratch/${CHILD}/approve`);
      assert.deepEqual(post?.body, { sha256: SHA });
      assert.match(post?.key ?? '', /^approve-[0-9a-f]{32}$/);
    });

    await step('trace diff', async () => {
      await main.getByLabel('Trace A').selectOption(uuid(2));
      await main.getByLabel('Trace B').selectOption(uuid(1));
      await main.getByRole('button', { name: 'Compare traces' }).click();
      await main.getByRole('cell', { name: '[20]' }).waitFor();
      await shot('trace-diff');
    });

    await step('writes', async () => assert.deepEqual(posts.map((x) => x.path).sort(), ['/api/jobs', `/api/admin/scratch/${CHILD}/approve`].sort()));
    await page.close();
  }
  assert.deepEqual(failures, []);
  console.log('weight-space browser checks passed');
} finally {
  await browser?.close();
  await server.close();
}
