// PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/tests/hy4-plan.browser.mjs
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { createServer } from 'vite';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server = await createServer({ root: fileURLToPath(new URL('../', import.meta.url)),
  configFile: false, plugins: [svelte()], server: { host: '127.0.0.1', port: 0 } });
const models = ['ordinary', 'planner'].map(host => ({ id: `library-${host}`, kind: 'hf',
  host, name: `Checkpoint on ${host}`, path: `/models/${host}` }));
const writes = [], errors = [];
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
  const page = await browser.newPage();
  page.setDefaultTimeout(4000);
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/**', route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() === 'GET') return route.fulfill({ json: path === '/api/state' ? {
      hosts: [{ id: 'ordinary', capabilities: { plan: true } }, { id: 'planner', capabilities: { plan: true, hy4_plan: true } }],
      models, jobs: [], prompts: [],
    } : [] });
    writes.push({ path, key: req.headers()['idempotency-key'], body: req.postDataJSON() });
    return writes.length === 1 ? route.fulfill({ status: 503, json: { error: 'Response interrupted' } })
      : route.fulfill({ json: { id: 'plan-job', state: 'queued' } });
  });
  await page.goto(`${server.resolvedUrls.local[0]}#/convert`);
  async function capture(label) {
    if (!process.env.HY4_CAPTURE) return;
    for (const width of [390, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      await page.screenshot({ path: `${process.env.HY4_CAPTURE}/${label}-${width}.png`, fullPage: true });
    }
  }
  await page.getByRole('heading', { name: 'Convert', exact: true }).waitFor();
  const diagnostics = page.locator('summary').filter({ hasText: 'Individual steps (diagnostics)' });
  const disclosure = diagnostics.locator('..').locator('summary').filter({ hasText: 'Hy4 metadata plan' });
  await disclosure.waitFor({ state: 'hidden' });
  await capture('closed');
  await diagnostics.press('Enter');
  await disclosure.press('Enter');
  const panel = disclosure.locator('..');
  const source = panel.getByRole('combobox', { name: 'Hy4 source' });
  assert.equal(await source.inputValue(), '');
  const button = panel.getByRole('button', { name: 'Queue Hy4 plan' });
  assert(await button.isDisabled());
  for (const host of ['ordinary', 'planner']) {
    await source.click();
    await panel.getByRole('option', { name: new RegExp(`Checkpoint on ${host}`) }).click();
    assert.equal(await button.isDisabled(), host === 'ordinary');
  }
  for (const text of ['blk.1.ffn_down_exps.weight', 'Q8_0', '224,927,386,624', '/models/planner', 'b2498eba']) {
    assert((await panel.innerText()).includes(text), text);
  }
  assert.equal(writes.length, 0);
  await button.click();
  await page.getByRole('alert').filter({ hasText: 'Response interrupted' }).waitFor();
  await button.press('Enter');
  const link = panel.getByRole('link', { name: /job result/i });
  await link.waitFor();
  assert.equal(await link.getAttribute('href'), '#/jobs/plan-job');
  assert.equal(writes.length, 2);
  assert.match(writes[0].key, /^[A-Za-z0-9._-]{1,128}$/);
  assert.deepEqual(writes[0], writes[1]);
  assert.deepEqual(writes[1].body, { kind: 'plan', host: 'planner', input_ids: ['library-planner'],
    options: { hy4_overrides: { 'blk.1.ffn_down_exps.weight': 'Q8_0' } } });
  assert.equal(writes[1].path, '/api/jobs');
  await capture('opened');
  assert.deepEqual(errors, []);
  console.log('Hy4 Convert plan: blockers, metadata, exact keyed retry and result link passed.');
} finally {
  await browser?.close();
  await server.close();
}
