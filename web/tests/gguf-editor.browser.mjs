// Run with Chromium installed: PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/tests/gguf-editor.browser.mjs
// If Playwright is installed in this workspace, PLAYWRIGHT_MODULE is not needed.
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { createServer } from 'vite';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server = await createServer({
  root: fileURLToPath(new URL('../', import.meta.url)),
  configFile: false,
  plugins: [svelte()],
  server: { host: '127.0.0.1', port: 0 },
});

const model = (id, name, kind, host = 'worker') => ({ id, name, kind, host, path: `/models/${id}` });
const planSource = model('planned', 'Planned HF', 'hf');
const native = model('native', 'Unmapped HF', 'hf');
const slow = model('slow', 'Slow HF', 'hf');
const gguf = model('gguf', 'Existing GGUF', 'gguf');
const disabled = model('disabled', 'Unadmitted HGN', 'hgn');
const models = [planSource, native, slow, gguf, disabled];
const job = (id, kind, source, options = {}) => ({ id, kind, state: 'succeeded', input_ids: [source.id], host: source.host, options });
const sha = 'a'.repeat(64);
const plan = (id, source, name) => ({
  ...job(id, 'plan', source),
  result: {
    inputs: [{ id: source.id, kind: source.kind, sha256: sha, bundle_sha256: sha }],
    report: { source_format: source.kind, gguf_tensors: [{ name, shape: [32, 32] }] },
  },
});
const plans = [
  plan('other-plan', slow, 'blk.88.stale.weight'),
  plan('plan', planSource, 'blk.2.attn_q.weight'),
];
let releaseSlow;
const slowGate = new Promise((resolve) => { releaseSlow = resolve; });
const inventoryOptions = { mode: 'inventory', offset: 0, limit: 128 };
const tensor = (name) => ({ name, dtype: 'F16', shape: [32, 32], editable: true });
const inventory = (record, source, rows, total, next_offset) => ({
  ...record, result: {
    report: { mode: 'inventory', source_format: source.kind, source_sha256: sha, offset: record.options.offset, total, next_offset, tensors: rows },
  },
});
const details = new Map(plans.map((p) => [p.id, p]));
details.set('existing-read', inventory(job('existing-read', 'weight_inspect', gguf, inventoryOptions), gguf, [tensor('blk.3.ffn_down.weight')], 1, null));
const submissions = [];
const reads = [];
const fail = [];
let browser;
let page;
try {
  await server.listen();
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
  page = await browser.newPage();
  page.setDefaultTimeout(6000);
  page.on('pageerror', (error) => fail.push(error.message));
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    if (method === 'GET' && path === '/api/state') {
      const jobs = [...details.values()].map(({ result, ...summary }) => summary);
      return route.fulfill({ json: {
        hosts: [{ id: 'worker', capabilities: { weight_inspect: true, weight_visual: true, weight_safetensors: true } }],
        models, prompts: [], jobs,
      } });
    }
    if (method === 'GET' && path === '/api/profiles/gguf-catalog') {
      return route.fulfill({ json: { presets: [{ id: 'Q4_K_M' }], tensor_types: [{ id: 'F16', block_size: 1, block_bytes: 2 }] } });
    }
    if (method === 'GET' && path === '/api/profiles/catalog') return route.fulfill({ json: { schema: 1, tensors: [] } });
    if (method === 'GET' && path === '/api/profiles') return route.fulfill({ json: [] });
    if (method === 'GET' && path.startsWith('/api/jobs/')) {
      const id = path.split('/').at(-1);
      reads.push(id);
      if (id === 'other-plan') await slowGate;
      if (details.has(id)) return route.fulfill({ json: details.get(id) });
    }
    if (method === 'POST' && path === '/api/jobs') {
      const input = route.request().postDataJSON();
      submissions.push(input);
      const { offset } = input.options ?? {};
      if (input.kind === 'weight_inspect' && input.input_ids[0] === native.id) {
        const rows = offset === 0
          ? Array.from({ length: 128 }, (_, i) => tensor(`model.layers.${i}.self_attn.q_proj.weight`))
          : [tensor('model.final_native_only.weight')];
        const record = job(`native-${offset}`, input.kind, native, input.options);
        details.set(record.id, inventory(record, native, rows, 129, offset === 0 ? 128 : null));
        return route.fulfill({ json: record });
      }
    } else if (method !== 'GET') {
      submissions.push({ method, path });
    }
    return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } });
  });
  await page.goto(`${server.resolvedUrls.local[0]}#/profiles?format=gguf`);
  const source = page.getByRole('combobox', { name: 'Source model' });
  const output = (name) => page.getByRole('combobox', { name: `Output type for ${name}`, exact: true });
  await output('blk.2.attn_q.weight').waitFor();
  assert.deepEqual(reads, ['plan'], 'default model must not load other models’ receipts');
  assert.deepEqual(submissions, [], 'an exact plan needs no job or profile write');
  async function choose(name) {
    await source.click();
    // Support the workbench's searchable model picker.
    await page.getByRole('option', { name: new RegExp(name) }).click();
  }
  await Promise.all([page.waitForRequest((req) => req.url().endsWith('/api/jobs/other-plan')), choose('Slow HF')]);
  await choose('Unmapped HF');
  releaseSlow();
  await page.getByText(/129 .*source tensors/i).waitFor();
  const search = page.getByRole('searchbox', { name: 'Search tensors and overrides' });
  await search.fill('final_native_only');
  const nativeRow = page.getByRole('row', { name: /model\.final_native_only\.weight/ });
  await nativeRow.waitFor();
  assert.equal(await nativeRow.getByRole('combobox').count(), 0);
  assert.equal(await output('blk.88.stale.weight').count(), 0, 'late receipt must not overwrite the new model');
  assert.equal(await page.getByRole('group', { name: 'Layer range' }).count(), 0);
  assert.equal(await page.locator('datalist option').count(), 0, 'native names are not override suggestions');
  await page.getByRole('textbox', { name: 'GGUF tensor name' }).fill('model.final_native_only.weight');
  await page.getByRole('group', { name: 'Add an override by exact name' }).getByRole('combobox').selectOption('F16');
  assert.equal(await page.getByRole('button', { name: 'Add override' }).isDisabled(), true, 'known native names cannot be added manually');

  await search.fill('');
  await choose('Existing GGUF');
  await output('blk.3.ffn_down.weight').waitFor();
  await choose('Unadmitted HGN');
  await page.getByText(/weight_hgn support/).waitFor();
  await choose('Unmapped HF');
  await page.getByText(/129 .*source tensors/i).waitFor();
  await delay(1600); // Cross a state poll; switching models must not duplicate reads.
  assert.deepEqual(submissions, [
    { kind: 'weight_inspect', host: 'worker', input_ids: [native.id], options: inventoryOptions },
    { kind: 'weight_inspect', host: 'worker', input_ids: [native.id], options: { ...inventoryOptions, offset: 128, source_sha256: sha } },
  ]);
  assert.deepEqual(fail, []);
  console.log('GGUF editor model inventory regression passed.');
} catch (error) {
  console.error({ fail, reads, submissions, body: await page?.locator('body').innerText() });
  throw error;
} finally {
  releaseSlow();
  await browser?.close();
  await server.close();
}
