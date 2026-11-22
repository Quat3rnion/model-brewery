import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { cancelling, failedCompare, historicalPrompt, hy4Conversion, hy4Plan, inventory, tile } from './mock/jobs.mjs';
import { usageReport } from './mock/eval-bench-calib.mjs';
import { libraryWeights } from './mock/library-weights.mjs';
import { scenario } from './mock/scenarios.mjs';

const wcagAA = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];
const views = [
  ['library', 'library', 'Library', 'Fixture checkpoint'],
  ['queue', 'queue', 'Jobs', 'running-job'],
  ['transfers', 'transfers', 'Transfers', 'Recent previews and transfers'],
  ['compare', 'compare', 'Compare', 'New comparison'],
  ['convert', 'convert', 'Convert', 'Verified bundle ready'],
  ['evaluate', 'evaluate', 'Evaluate', 'Synthetic fixture'],
  ['benchmarks', 'benchmarks', 'Benchmarks', 'Scorecard'],
  ['calibration', 'calibration', 'Calibration inputs', 'No reports open.'],
  ['settings', 'settings', 'Settings', 'Evaluation readiness'],
  ['profiles', 'profiles', 'Conversion profiles', 'Fixture HGN profile'],
  ['weights', 'weights?source=gguf-a', 'Weight edits', '/fixtures/sample.gguf'],
  ['job', 'jobs/transfer-result', 'Transfer', 'Missing bytes fit in free space'],
];

for (const [view, route, heading, ready] of views) {
  test(`${view}: WCAG A/AA`, async ({ page, controller }, info) => {
    await controller.open(page, route);
    const main = page.getByRole('main');
    await expect(main.getByRole('heading', { level: 1, name: heading, exact: true })).toBeVisible();
    await expect(main.getByText(ready, { exact: true })).toBeVisible();
    await scan(page, info);
  });
}

async function scan(page, info) {
  await expect(page.locator('html')).toHaveAttribute('data-theme', info.project.use.colorScheme);
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  const { violations } = await new AxeBuilder({ page }).withTags(wcagAA).analyze();
  if (violations.length) {
    await info.attach('axe-violations', { body: JSON.stringify(violations, null, 2), contentType: 'application/json' });
  }
  expect(violations.flatMap(({ id, nodes }) => nodes.map(({ target }) => ({ id, target })))).toEqual([]);
}

for (const view of ['compare', 'convert', 'evaluate', 'benchmarks', 'transfers', 'weights']) {
  test(`${view}: all capabilities off`, async ({ page, controller }, info) => {
    controller.load('capabilities-off');
    await controller.open(page, view === 'weights' ? 'weights?source=gguf-a' : view);
    if (view === 'weights') await expect(page.getByText('Inspect: not enabled')).toBeVisible();
    else await expect(page.getByRole('region', { name: 'Setup required' }).first()).toBeVisible();
    await scan(page, info);
  });
}

const scanJob = (id, ready, setup) => [id, async (page, controller) => {
  if (typeof setup === 'string') controller.load(setup);
  else if (setup) controller.respond('GET', `/api/jobs/${id}`, { json: setup });
  await controller.open(page, `jobs/${id}`);
  await expect(page.getByRole('main').getByText(ready, { exact: true }).first()).toBeVisible();
}];

const supplementary = [
  ['Library: empty', async (page, c) => { c.load('empty'); await c.open(page); await expect(page.getByText('No saved prompts.')).toBeVisible(); }],
  ['Library: identity', async (page, c) => { libraryWeights(c); await c.open(page); await expect(page.getByText('Verified GGUF', { exact: true })).toBeVisible(); }],
  ['Library: Remove confirmation', async (page, c) => {
    libraryWeights(c); await c.open(page);
    await page.getByRole('button', { name: 'Remove Verified GGUF from library' }).click();
    await expect(page.getByRole('group', { name: 'Confirm removal' })).toBeVisible();
  }],
  ['Jobs: active', async (page, c) => { c.load('active-jobs'); await c.open(page, 'queue'); await expect(page.getByRole('main').getByText('running-job')).toBeVisible(); }],
  scanJob(tile.id, 'Tile shape', tile),
  scanJob(inventory.id, 'blk.0.ffn_down.weight', inventory),
  scanJob(cancelling.id, 'Cancelling', cancelling),
  scanJob(hy4Plan.id, 'The job did not verify the tensor payload.', hy4Plan),
  scanJob(hy4Conversion.id, 'The result does not show that the model can run.', hy4Conversion),
  scanJob(failedCompare.id, failedCompare.error, failedCompare),
  scanJob('missing-job', 'This job does not exist or was removed.'),
  scanJob('full-split-result', 'Not release-qualified', 'full-split'),
  scanJob(historicalPrompt.id, historicalPrompt.prompt_text, historicalPrompt),
  ['Job detail: transient result recovery', async (page, c) => {
    let attempts = 0;
    const data = scenario('results');
    const result = data.jobs.find((job) => job.id === 'transfer-result');
    c.respond('GET', '/api/jobs/transfer-result', () => ++attempts === 1
      ? { status: 503, json: { error: 'Result not available yet.' } } : { json: result });
    await c.open(page, 'jobs/transfer-result');
    await expect(page.getByRole('alert')).toContainText('Result not available yet.');
    await expect(page.getByText('Missing bytes fit in free space')).toBeVisible();
  }],
  ['Compare: Advanced', async (page, c) => { await c.open(page, 'compare'); await page.getByText('Advanced options', { exact: true }).click(); await expect(page.getByLabel('CPU threads (optional)')).toBeVisible(); }],
  ['Convert: diagnostic Convert tab', async (page, c) => { await c.open(page, 'convert?step=convert&model=source-a'); await expect(page.getByRole('tab', { name: 'Convert' })).toHaveAttribute('aria-selected', 'true'); }],
  ['Evaluate: full-split GSM8K', async (page, c) => { c.load('full-split'); await c.open(page, 'evaluate'); await expect(page.getByText('Standalone GSM8K score', { exact: true })).toBeVisible(); }],
  ['Benchmarks: empty', async (page, c) => { c.load('empty'); await c.open(page, 'benchmarks'); await expect(page.getByText('No benchmark jobs yet.')).toBeVisible(); }],
  ['Calibration: loaded report and opened expert metrics', async (page, c) => {
    await c.open(page, 'calibration');
    await page.getByLabel('Report files (JSON)').setInputFiles({ name: 'usage.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(usageReport)) });
    await page.getByRole('button', { name: 'Experts for layer 0' }).click();
    await expect(page.getByRole('table', { name: 'Prefill: most selected experts' })).toBeVisible();
  }],
  ['Settings: Add SSH', async (page, c) => { await c.open(page, 'settings'); await page.getByRole('button', { name: 'Add SSH host' }).click(); await expect(page.getByLabel('Host address')).toBeVisible(); }],
  ['Settings: Remove token', async (page, c) => {
    c.respond('GET', '/api/settings/hf-tokens', { json: { hosts: [{ id: 'backend-hf', configured: true }] } });
    await c.open(page, 'settings'); await page.getByRole('button', { name: 'Remove token', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Confirm removal' })).toBeVisible();
  }],
  ['Profiles: empty saved list', async (page, c) => { c.load('empty'); await c.open(page, 'profiles'); await expect(page.getByText('No saved profiles yet.', { exact: false })).toBeVisible(); }],
  ['Profiles: HGN editor', async (page, c) => { await c.open(page, 'profiles'); await page.getByRole('button', { name: 'Start from HGN v2', exact: true }).click(); await expect(page.getByRole('listbox', { name: 'Trunk layers' })).toBeVisible(); }],
  ['Profiles: GGUF editor', async (page, c) => { await c.open(page, 'profiles?format=gguf'); await expect(page.getByRole('combobox', { name: 'Source model' })).toBeVisible(); }],
  ['Weights: blocked source', async (page, c) => { libraryWeights(c); await c.open(page, 'weights?source=missing'); await expect(page.getByText('The source file does not exist.', { exact: true })).toBeVisible(); }],
  ...[false, true].map((withTile) => [`Weights: ${withTile ? 'tile' : 'inventory'} editor`, async (page, c) => {
    c.load('weight-reads');
    await c.open(page, 'weights?source=gguf-a');
    await page.getByRole('button', { name: 'Read tensor list', exact: true }).click();
    const tensor = page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ });
    await expect(tensor).toBeVisible();
    if (withTile) {
      await tensor.click();
      await page.getByRole('button', { name: 'Read tile', exact: true }).click();
      await expect(page.getByRole('gridcell').first()).toBeVisible();
    }
  }]),
  ['Shell: missing route', async (page, c) => { await c.open(page, 'not-a-view'); await expect(page.getByRole('heading', { name: 'Not found' })).toBeVisible(); }],
  ...['controller-down', 'api-503'].map((state) => [`Shell: startup ${state}`, async (page, c) => {
    c.load(state); await c.open(page); await expect(page.getByRole('heading', { name: 'Controller not reachable' })).toBeVisible();
  }]),
];
for (const [name, open] of supplementary) {
  test(`${name}: WCAG A/AA`, async ({ page, controller }, info) => {
    await open(page, controller);
    await scan(page, info);
  });
}

for (const [view, ready] of [['library', 'No saved prompts.'], ['convert', 'Convert'], ['evaluate', 'Evaluate']]) {
  test(`${view}: shell states release qualification`, async ({ page, controller }) => {
    controller.load(view === 'library' ? 'empty' : 'live-shaped');
    await controller.open(page, view);
    await expect(page.getByRole('main').getByText(ready, { exact: true }).first()).toBeVisible();
    await expect(page.getByRole('banner').getByText('This workbench is not release-qualified.', { exact: true })).toBeVisible();
  });
}
