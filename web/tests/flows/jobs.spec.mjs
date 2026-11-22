import { test, expect } from './fixtures.mjs';
import { scenario } from './mock/scenarios.mjs';
import { cancelling, failedCompare, hy4Conversion, hy4Plan, interruptedTransfer, inventory, tile } from './mock/jobs.mjs';

const jobPath = (id) => `/api/jobs/${id}`;
const setJob = (controller, job) => controller.respond('GET', jobPath(job.id), { json: job });
const go = (page, id) => page.evaluate((jobId) => { location.hash = `#/jobs/${jobId}`; }, id);

test('Job detail clears a previous result before the new id resolves', async ({ page, controller }) => {
  controller.load('results');
  let release;
  controller.respond('GET', jobPath('next-job'), () => new Promise((resolve) => { release = resolve; }));
  await controller.open(page, 'jobs/transfer-result');
  await expect(page.getByText('Missing bytes fit in free space')).toBeVisible();
  try {
    await go(page, 'next-job');
    await expect.poll(() => typeof release).toBe('function');
    await expect(page.getByRole('heading', { name: 'Job next-job' })).toBeVisible();
    await expect(page.getByText('Missing bytes fit in free space')).toHaveCount(0);
  } finally {
    release?.({ status: 404, json: { error: 'job not found' } });
  }
});

test('A missing job id stops polling after a 404', async ({ page, controller }) => {
  await page.clock.install();
  await controller.open(page, 'jobs/does-not-exist');
  await expect(page.getByRole('alert')).toBeVisible();
  await page.clock.runFor(6500);
  expect.soft(controller.requests('GET', jobPath('does-not-exist'))).toHaveLength(1);
  await expect(page.getByRole('alert')).toHaveText('This job does not exist or was removed.');
  await page.getByRole('link', { name: /All jobs/ }).click();
  await expect(page.getByRole('heading', { name: 'Jobs', exact: true })).toBeVisible();
});

test('Inspection results support inventory and tile modes', async ({ page, controller }) => {
  setJob(controller, inventory);
  setJob(controller, tile);
  await controller.open(page, `jobs/${inventory.id}`);
  await expect(page.getByRole('table', { name: 'Tensor inventory' }).getByText('blk.0.ffn_down.weight', { exact: true })).toBeVisible();
  await go(page, tile.id);
  const result = page.getByRole('region', { name: 'Result' });
  await expect(result.getByText('Tile shape', { exact: true })).toBeVisible();
  await expect(result.getByText('blk.0.ffn_down.weight', { exact: true })).toBeVisible();
  await expect(result.getByRole('link', { name: 'Open source in Weights' })).toHaveAttribute('href', '#/weights?source=gguf-a');
});

test('Jobs filters by kind and links once per result with a distinct job name', async ({ page, controller }) => {
  const data = scenario('results');
  controller.respond('GET', '/api/state', { json: { ...data, jobs: [...data.jobs, failedCompare] } });
  await controller.open(page, 'queue');
  const main = page.getByRole('main');
  await main.getByRole('combobox', { name: 'Job kind' }).selectOption('compare');
  await expect(main.getByRole('listitem')).toHaveCount(2);
  const row = main.getByRole('listitem').filter({ hasText: '981c4194-104' });
  const link = row.getByRole('link', { name: /^Fixture GGUF.*Prompt comparison.*981c4194-104/ });
  await expect(link).toHaveAccessibleDescription(failedCompare.error);
  await expect(main.getByRole('listitem').filter({ hasText: 'compare-result' }).getByRole('link')).toHaveCount(1);
});

test('Job detail shows progress when cancellation is pending', async ({ page, controller }) => {
  setJob(controller, cancelling);
  await controller.open(page, `jobs/${cancelling.id}`);
  const main = page.getByRole('main');
  await expect(main.getByText('Cancelling', { exact: true }).first()).toBeVisible();
  await expect(main.getByRole('progressbar', { name: 'Job progress' })).toHaveAttribute('aria-valuenow', '50');
  await expect(main.getByRole('button', { name: /Cancel job/ })).toHaveCount(0);
});

test('Failed comparison explains the fix and opens a prefilled form without queuing work', async ({ page, controller }) => {
  setJob(controller, failedCompare);
  await controller.open(page, `jobs/${failedCompare.id}`);
  const log = page.getByRole('region', { name: 'Event log' });
  await expect(log.getByText('Failed', { exact: true })).toBeVisible();
  await expect(log.getByText(/Failed · failed/i)).toHaveCount(0);
  await expect(page.getByText('Free space on the host before you try again.', { exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Open in Compare' }).click();
  await expect(page.getByRole('textbox', { name: 'Prompt text' })).toHaveValue(failedCompare.prompt);
  await expect(page.getByRole('combobox', { name: 'Model A' })).toHaveValue('Fixture GGUF');
  await expect(page.getByRole('combobox', { name: 'Model B' })).toHaveValue('Fixture GGUF copy');
  const tokens = page.getByRole('spinbutton', { name: 'Decode tokens' });
  await expect(tokens).toHaveValue('32');
  await tokens.fill('129');
  await tokens.press('Tab');
  await expect(tokens).toHaveAttribute('aria-invalid', 'true');
  await expect(tokens).toHaveAccessibleDescription('Enter a whole number from 0 to 128.');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Hy4 plan and conversion explain their evidence and link a registered output', async ({ page, controller }) => {
  setJob(controller, hy4Plan);
  setJob(controller, hy4Conversion);
  await controller.open(page, `jobs/${hy4Plan.id}`);
  const result = page.getByRole('region', { name: 'Result' });
  await expect(result.getByText('2.00 GiB', { exact: true })).toBeVisible();
  await expect(result.getByText('The job did not verify the tensor payload.')).toBeVisible();
  await expect(result.getByText('Not release-qualified', { exact: true })).toBeVisible();
  await go(page, hy4Conversion.id);
  await expect(result.getByText('The result does not show that the model can run.')).toBeVisible();
  await expect(result.getByRole('link', { name: 'View in Library' })).toHaveAttribute('href', '#/library?entry=gguf-a');
});

test('A verified conversion requires an explicit Library registration action', async ({ page, controller }) => {
  const output = { ...hy4Conversion.result.artifact, path: '/fixtures/new-output.gguf' };
  setJob(controller, { ...hy4Conversion, result: { ...hy4Conversion.result, artifact: output } });
  await controller.open(page, `jobs/${hy4Conversion.id}`);
  const register = page.getByRole('button', { name: 'Register in Library' });
  await expect(register).toBeVisible();
  expect(controller.requests('POST', '/api/models')).toHaveLength(0);
  await register.click();
  await expect(page.getByRole('link', { name: 'View in Library' })).toBeVisible();
  expect(controller.requests('POST', '/api/models').map(({ body }) => body)).toEqual([
    { host: 'worker-a', path: output.path, kind: 'gguf', name: 'new-output.gguf' },
  ]);
  await expect(page.getByText('Not release-qualified', { exact: true })).toBeVisible();
});

test('An interrupted transfer opens the prior destination and entries without starting a copy', async ({ page, controller }) => {
  setJob(controller, interruptedTransfer);
  await controller.open(page, `jobs/${interruptedTransfer.id}`);
  await page.getByRole('link', { name: 'Open in Transfers' }).click();
  await expect(page.getByRole('combobox', { name: 'Destination host' })).toHaveValue('Worker B');
  await expect(page.getByRole('checkbox', { name: /^Fixture GGUF/ })).toBeChecked();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Transfers keeps entry names readable at narrow widths and hides ineligible destinations', async ({ page, controller }) => {
  const data = scenario('live-shaped');
  const name = 'Pinned BF16 source with a very long file name (local-model-worker)';
  data.models[0].name = name;
  data.hosts.push({ id: 'worker-c', name: 'Worker C', capabilities: { transfer: false, transfer_preview: false } });
  controller.respond('GET', '/api/state', { json: data });
  await controller.open(page, 'transfers');
  await page.getByRole('combobox', { name: 'Destination host' }).click();
  await expect.soft(page.getByRole('option', { name: /Worker C/ })).toHaveCount(0);
  await page.getByRole('option', { name: /^Worker B/ }).click();
  const entry = page.getByText(name, { exact: true });
  await expect(entry).toBeVisible();
  expect(await entry.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize().width);
});

test('When Compare has no eligible host, reports can still be opened locally', async ({ page, controller }) => {
  controller.load('capabilities-off');
  await controller.open(page, 'compare');
  await expect(page.getByRole('region', { name: 'Setup required' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Queue comparison' })).toHaveCount(0);
  await page.getByLabel('Report files (JSON)').setInputFiles({
    name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('not json'),
  });
  await expect(page.getByRole('region', { name: 'Not opened invalid.json' }).getByRole('alert')).toBeVisible();
  expect(controller.requests().filter(({ method }) => method !== 'GET')).toEqual([]);
});

test('Older controllers without an eligibility snapshot still gate unavailable workflows', async ({ page, controller }) => {
  const { eligibility, ...data } = scenario('capabilities-off');
  controller.respond('GET', '/api/state', { json: data });
  await controller.open(page, 'compare');
  const setup = page.getByRole('region', { name: 'Setup required' });
  await expect(setup.getByRole('link', { name: /Worker A.*compare/i })).toHaveAttribute('href', '#/settings/infrastructure?host=worker-a&section=policy');
  await expect(page.getByRole('combobox', { name: 'Model A' })).toHaveCount(0);
  await page.getByRole('link', { name: 'Transfers', exact: true }).click();
  await expect(setup).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Destination host' })).toHaveCount(0);
});
