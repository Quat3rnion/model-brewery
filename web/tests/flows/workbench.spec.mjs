import { test, expect } from './fixtures.mjs';

async function pick(page, field, option) {
  await page.getByRole('combobox', { name: field, exact: true }).click();
  await page.getByRole('option', { name: option }).click();
}

const jobsPosted = (controller) => controller.requests('POST', '/api/jobs').map(({ body }) => body);
const inventoryRead = (id, offset = 0) => ({
  kind: 'weight_inspect', host: 'worker-a', input_ids: [id],
  options: { mode: 'inventory', offset, limit: 128, ...(offset ? { source_sha256: 'a'.repeat(64) } : {}) },
});

async function expectNoNewJobs(page, controller, count = 0) {
  // Let reactive work and the next controller refresh finish before checking for a POST.
  await page.waitForResponse((res) => new URL(res.url()).pathname === '/api/state' && res.ok());
  expect(jobsPosted(controller)).toHaveLength(count);
}

test('keyboard navigation focuses the section heading and the skip target', async ({ page, controller }) => {
  await controller.open(page);
  const nav = page.getByRole('navigation', { name: 'Sections' });
  const compare = nav.getByRole('link', { name: 'Compare' });
  await compare.press('Enter');
  await expect(page.getByRole('heading', { name: 'Compare', level: 1 })).toBeFocused();
  await expect(nav.locator('[aria-current="page"]')).toHaveCount(1);
  await expect(compare).toHaveAttribute('aria-current', 'page');
  await page.getByRole('link', { name: 'Skip to content' }).press('Enter');
  await expect(page.getByRole('main')).toBeFocused();
  await expect(page).toHaveURL(/#\/compare$/);
});

test('Library filters GGUF entries and opens the weight editor on that source', async ({ page, controller }) => {
  controller.load('gguf');
  await controller.open(page);
  await page.getByRole('searchbox', { name: 'Filter entries' }).fill('/fixtures/sample.gguf');
  const table = page.getByRole('table', { name: /Library entries/ });
  await expect(table.getByText('Fixture GGUF', { exact: true })).toBeVisible();
  await expect(table.getByText('Fixture GGUF copy', { exact: true })).toHaveCount(0);
  await table.getByRole('link', { name: /Edit weights/ }).click();
  await expect(page.getByRole('heading', { name: 'Weight edits' })).toBeVisible();
  await expect(page.getByText('/fixtures/sample.gguf', { exact: true })).toBeVisible();
  await expect(page.getByText('Inspect: enabled', { exact: true })).toBeVisible();
});

test('Jobs shows running, queued and cancelling work and confirms cancellation', async ({ page, controller }) => {
  controller.load('active-jobs');
  await controller.open(page, 'queue');
  await page.getByRole('button', { name: /^Active / }).click();
  const main = page.getByRole('main');
  for (const [id, status] of [['running-job', 'Running'], ['queued-job', 'Queued'], ['cancelling-job', 'Cancelling']]) {
    await expect(main.getByRole('listitem').filter({ hasText: id }).getByText(status, { exact: true })).toBeVisible();
  }
  const cancelPath = '/api/jobs/queued-job/cancel';
  await main.getByRole('button', { name: 'Cancel job queued-job' }).click();
  await expect(main.getByRole('button', { name: 'Confirm cancel' })).toBeVisible();
  expect(controller.requests('POST', cancelPath)).toHaveLength(0);
  await main.getByRole('button', { name: 'Confirm cancel' }).click();
  await expect(main.getByText('queued-job', { exact: true })).toHaveCount(0);
  expect(controller.requests('POST', cancelPath)).toHaveLength(1);
  await main.getByRole('button', { name: /^Finished / }).click();
  await expect(main.getByText('Cancelled', { exact: true })).toBeVisible();
});

test('job detail shows a transfer result and links back to the queue', async ({ page, controller }) => {
  controller.load('results');
  await controller.open(page, 'jobs/transfer-result');
  const result = page.getByRole('region', { name: 'Result' });
  await expect(result.getByText('Missing bytes fit in free space')).toBeVisible();
  await expect(result.getByText('worker-b:/fixtures/cache/sample.gguf', { exact: false })).toBeVisible();
  await page.getByRole('link', { name: /All jobs/ }).click();
  await expect(page.getByRole('heading', { name: 'Jobs', exact: true })).toBeVisible();
});

test('Job detail does not mark a verification that was not performed as Failed', async ({ page, controller }) => {
  controller.load('conversions');
  await controller.open(page, 'jobs/plan-stage');
  const result = page.getByRole('region', { name: 'Result' });
  await expect(result.getByRole('heading', { name: 'File verification' })).toBeVisible();
  await expect(result.getByText('Not performed', { exact: true })).toHaveClass(/\bmuted\b/);
  await expect(result.getByText('Failed', { exact: true })).toHaveCount(0);
  await controller.open(page, 'jobs/verify-stage');
  await expect(result.getByText('Passed', { exact: true })).toHaveClass(/\bok\b/);
});

test('a missing job deep link shows an error without a result', async ({ page, controller }) => {
  await controller.open(page, 'jobs/missing');
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Result' })).toHaveCount(0);
});

test('a singular job link opens the job and an unknown route shows Not found', async ({ page, controller }) => {
  controller.load('results');
  await controller.open(page, 'job/transfer-result');
  await expect(page.getByRole('region', { name: 'Result' })).toContainText('Missing bytes fit in free space');
  await page.evaluate(() => { location.hash = '#/missing-section'; });
  await expect(page.getByRole('heading', { name: 'Not found', exact: true })).toBeFocused();
  await page.getByRole('main').getByRole('link', { name: 'Open Library' }).click();
  await expect(page.getByRole('heading', { name: 'Library', exact: true })).toBeVisible();
});

test('Library follows entry links without leaving the Library view', async ({ page, controller }) => {
  await controller.open(page, 'library?entry=gguf-a');
  const table = page.getByRole('table', { name: /Library entries/ });
  await expect(table.getByText('Fixture GGUF', { exact: true })).toBeVisible();
  await page.evaluate(() => { location.hash = '#/library?entry=gguf-b'; });
  await expect(table.getByText('Fixture GGUF copy', { exact: true })).toBeVisible();
  await expect(table.getByText('Fixture GGUF', { exact: true })).toHaveCount(0);
});

test('Transfers filters source entries and queues only a preview', async ({ page, controller }) => {
  await controller.open(page, 'transfers');
  await pick(page, 'Destination host', /^Worker B/);
  await page.getByRole('searchbox', { name: 'Filter entries to copy' }).fill('Fixture GGUF');
  await page.getByRole('checkbox', { name: /^Fixture GGUF/ }).check();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await page.getByRole('button', { name: 'Preview', exact: true }).click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body)).toEqual([
    { kind: 'transfer_preview', host: 'worker-b', input_ids: ['gguf-a'] },
  ]);
});

test('Compare accepts a typed prompt and requires both models', async ({ page, controller }) => {
  await controller.open(page, 'compare');
  const typed = page.getByRole('radio', { name: 'Type a prompt' });
  await expect(typed).toHaveAttribute('name', 'compare-prompt');
  await typed.check();
  await page.getByRole('textbox', { name: 'Prompt text' }).fill('Compare these responses.');
  await pick(page, 'Model A', /^Fixture GGUF\b(?! copy)/);
  const queue = page.getByRole('button', { name: 'Queue comparison' });
  const modelB = page.getByRole('combobox', { name: 'Model B' });
  const fixes = page.getByRole('list', { name: 'Changes needed' });
  await expect(fixes).toHaveCount(0);
  await expect(modelB).not.toHaveAttribute('aria-invalid', 'true');
  await queue.click();
  await expect(fixes.getByRole('listitem')).toHaveText(['Select model B.']);
  await expect(modelB).toHaveAttribute('aria-invalid', 'true');
  await expect(modelB).toHaveAccessibleDescription('Select model B.');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await pick(page, 'Model B', /^Fixture GGUF copy/);
  await expect(modelB).not.toHaveAttribute('aria-invalid', 'true');
  await expect(fixes).toHaveCount(0);
  await queue.click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body.input_ids)).toEqual([['gguf-a', 'gguf-b']]);
});

test('Compare explains missing capability before selection', async ({ page, controller }) => {
  controller.load('capabilities-off');
  await controller.open(page, 'compare');
  await expect(page.getByRole('heading', { name: 'Compare', level: 1 })).toBeVisible();
  const notice = page.getByRole('region', { name: 'Setup required' });
  await expect(notice).toContainText('No host can run this workflow.');
  await expect(notice.getByRole('link', { name: /Worker B/ })).toHaveAttribute('href', '#/settings/infrastructure?host=worker-b&section=policy');
  await expect(page.getByRole('combobox', { name: 'Model A' })).toHaveCount(0);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Compare marks an ineligible host and links the fix for an ineligible default host', async ({ page, controller }) => {
  await controller.open(page, 'compare');
  const host = page.getByRole('combobox', { name: 'Run on' });
  await host.click();
  const unavailable = page.getByRole('option', { name: /Worker B.*compare is not configured/ });
  await expect(unavailable).toHaveAttribute('aria-disabled', 'true');
  await host.press('Escape');
  await pick(page, 'Model A', /^Fixture GGUF copy/);
  await host.focus();
  await host.press('Tab');
  await expect(host).toHaveAttribute('aria-invalid', 'true');
  const fixes = page.getByRole('list', { name: 'Changes needed' });
  await expect(fixes.getByRole('link', { name: /Worker B.*compare is not configured/ })).toHaveAttribute('href', '#/settings/infrastructure?host=worker-b&section=policy');
  await pick(page, 'Run on', /^Worker A/);
  await expect(host).not.toHaveAttribute('aria-invalid', 'true');
});

test('Compare keeps a typed draft if navigation is cancelled', async ({ page, controller }) => {
  await controller.open(page, 'compare');
  await page.getByRole('radio', { name: 'Type a prompt' }).check();
  const prompt = page.getByRole('textbox', { name: 'Prompt text' });
  await prompt.fill('Draft prompt.');
  const library = page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: 'Library' });
  const refused = page.waitForEvent('dialog');
  const firstClick = library.click();
  const stay = await refused;
  expect(stay.type()).toBe('confirm');
  expect(stay.message()).toBe('Discard unsaved changes?');
  await stay.dismiss();
  await firstClick;
  await expect(page).toHaveURL(/#\/compare$/);
  await expect(prompt).toHaveValue('Draft prompt.');
  const accepted = page.waitForEvent('dialog');
  const secondClick = library.click();
  await (await accepted).accept();
  await secondClick;
  await expect(page.getByRole('heading', { name: 'Library', level: 1 })).toBeFocused();
});

test('Compare can return to the original history entry after declining Back', async ({ page, controller }) => {
  await controller.open(page);
  await page.getByRole('link', { name: 'Compare', exact: true }).click();
  await page.getByRole('radio', { name: 'Type a prompt' }).check();
  const prompt = page.getByRole('textbox', { name: 'Prompt text' });
  await prompt.fill('Draft prompt.');
  const refused = page.waitForEvent('dialog');
  await page.evaluate(() => { history.back(); });
  await (await refused).dismiss();
  await expect(page).toHaveURL(/#\/compare$/);
  await expect(prompt).toHaveValue('Draft prompt.');
  const accepted = page.waitForEvent('dialog');
  await page.evaluate(() => { history.back(); });
  await (await accepted).accept();
  await expect(page.getByRole('heading', { name: 'Library', level: 1 })).toBeFocused();
});

test('Compare warns before the browser unloads a typed draft', async ({ page, controller }) => {
  await controller.open(page, 'compare');
  await page.getByRole('radio', { name: 'Type a prompt' }).check();
  const prompt = page.getByRole('textbox', { name: 'Prompt text' });
  await prompt.fill('Draft prompt.');
  const confirmed = page.waitForEvent('dialog');
  await page.evaluate(() => { setTimeout(() => { location.search = '?test-unload'; }, 0); });
  const dialog = await confirmed;
  expect(dialog.type()).toBe('beforeunload');
  await dialog.dismiss();
  await expect(prompt).toHaveValue('Draft prompt.');
  await expect(page).toHaveURL(/\/#\/compare$/);
});

test('Compare returns focus to the file input when a report closes', async ({ page, controller }) => {
  await controller.open(page, 'compare');
  const files = page.getByLabel('Report files (JSON)');
  await files.setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{bad') });
  await page.getByRole('button', { name: 'Close bad.json' }).press('Enter');
  await expect(page.getByRole('button', { name: 'Close bad.json' })).toHaveCount(0);
  await expect(files).toBeFocused();
});

test('Evaluate and Benchmarks explain when no worker can run those jobs', async ({ page, controller }) => {
  controller.load('capabilities-off');
  await controller.open(page, 'evaluate');
  await expect(page.getByRole('region', { name: 'GSM8K evaluation', exact: true }).getByText('No host can run this workflow.')).toBeVisible();
  await controller.open(page, 'benchmarks');
  await expect(page.getByRole('region', { name: 'Run benchmarks' }).getByText('No host can run this workflow.')).toBeVisible();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Compare result distinguishes side-by-side output from aligned traces', async ({ page, controller }) => {
  controller.load('results');
  await controller.open(page, 'jobs/compare-result');
  await expect(page.getByRole('heading', { name: 'Prompt comparison' })).toBeVisible();
  await expect(page.getByText('Side by side only', { exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Model A' })).toContainText('Fixture answer A');
  await expect(page.getByRole('region', { name: 'Model B' })).toContainText('Fixture answer B');
});

test('Convert shows stage links and starts one managed conversion', async ({ page, controller }) => {
  controller.load('conversions');
  await controller.open(page, 'convert');
  await expect(page.getByText('Verified bundle ready', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Verify job verify-stage' })).toHaveAttribute('href', '#/jobs/verify-stage');
  const convert = page.getByRole('button', { name: 'Start conversion', exact: true });
  await expect(convert).toBeDisabled();
  await pick(page, 'Source checkpoint', /^Fixture checkpoint/);
  await page.getByRole('region', { name: 'Convert a checkpoint' }).getByLabel('Profile', { exact: true }).selectOption('v2');
  await convert.click();
  await expect(page.getByText(/Started conversion conversion-mock-/)).toBeVisible();
  const requests = controller.requests('POST', '/api/conversions');
  expect(requests).toHaveLength(1);
  expect(requests[0].body.input_ids).toEqual(['source-a']);
  expect(requests[0].headers['idempotency-key']).toMatch(/^conv-/);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Profiles shows a saved HGN revision and carries it into Convert', async ({ page, controller }) => {
  controller.load('saved-profiles');
  await controller.open(page, 'profiles');
  await expect(page.getByRole('rowheader', { name: 'Fixture HGN profile' })).toBeVisible();
  await page.getByRole('button', { name: /Use in Convert: Fixture HGN profile revision/ }).click();
  await expect(page).toHaveURL(/#\/convert$/);
  await expect(page.getByRole('region', { name: 'Convert a checkpoint' }).getByLabel('Profile', { exact: true })).toHaveValue(`rev:${'a'.repeat(64)}`);
});

test('Profiles keeps GGUF revisions separate from HGN revisions', async ({ page, controller }) => {
  controller.load('saved-profiles');
  await controller.open(page, 'profiles?format=gguf');
  await expect(page.getByRole('table', { name: 'Saved GGUF profile revisions' })
    .getByRole('rowheader', { name: 'Fixture GGUF profile' })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'Fixture HGN profile' })).toHaveCount(0);
});

test('Convert reads recipe layers only on request and reuses the completed read', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'convert');
  await page.getByRole('radio', { name: 'Layer recipe' }).check();
  await expectNoNewJobs(page, controller);
  await page.locator('#recipe-source').click();
  await page.getByRole('option', { name: /^Fixture checkpoint/ }).click();
  await expectNoNewJobs(page, controller);
  await expect(page.getByText('This queues a weight_inspect job on Worker A.')).toBeVisible();
  await page.getByRole('button', { name: 'Read layers', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Layer read job' })).toHaveAttribute('href', '#/jobs/job-mock-1');
  await expect(page.getByLabel('Layer 0 type')).toBeVisible();
  expect(jobsPosted(controller)).toEqual([inventoryRead('source-a')]);

  await page.reload();
  await page.getByRole('radio', { name: 'Layer recipe' }).check();
  await page.locator('#recipe-source').click();
  await page.getByRole('option', { name: /^Fixture checkpoint/ }).click();
  await expect(page.getByLabel('Layer 0 type')).toBeVisible();
  await expectNoNewJobs(page, controller, 1);
});

test('Profiles opens the GGUF editor with no source selected and no job', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  const source = page.getByRole('combobox', { name: 'Source model' });
  await expect(source).toBeEnabled();
  await expectNoNewJobs(page, controller);
  await expect(source).toHaveValue('');
});

test('Profiles reads GGUF editor tensors only on request', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  // This source has no saved tensor names, so it needs a new read.
  await pick(page, 'Source model', /^Fixture checkpoint/);
  await expectNoNewJobs(page, controller);
  await expect(page.getByText('This queues a weight_inspect job on Worker A.')).toBeVisible();
  await page.getByRole('button', { name: 'Read tensor list', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Tensor list read job' })).toHaveAttribute('href', '#/jobs/job-mock-1');
  await expect(page.getByRole('rowheader', { name: 'model.layers.0.mlp.down_proj.weight' })).toBeVisible();
  expect(jobsPosted(controller)).toEqual([inventoryRead('source-a')]);
});

test('Weights does not submit inspection jobs until explicitly requested', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'weights');
  await pick(page, 'Source model', /^Fixture GGUF\b(?! copy)/);
  await expectNoNewJobs(page, controller);
  await page.reload(); // The URL now includes the selected source.
  await expect(page.getByRole('heading', { name: '1. Tensor' })).toBeVisible();
  await expectNoNewJobs(page, controller);

  await expect(page.getByText('This queues a weight_inspect job on Worker A.')).toBeVisible();
  await page.getByRole('button', { name: 'Read tensor list', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Tensor list read job' })).toHaveAttribute('href', '#/jobs/job-mock-1');
  const next = page.getByRole('button', { name: 'Read next tensors', exact: true });
  await expect(next).toBeEnabled();
  await expectNoNewJobs(page, controller, 1);
  expect(jobsPosted(controller)).toEqual([inventoryRead('gguf-a')]);
  await next.click();
  await expect(page.getByText(/129 of 129 tensors/)).toBeVisible();
  expect(jobsPosted(controller)).toEqual([inventoryRead('gguf-a'), inventoryRead('gguf-a', 128)]);

  const tensor = 'blk.0.ffn_down.weight';
  await page.getByRole('option', { name: new RegExp(`^${tensor},`) }).click();
  const values = page.getByRole('region', { name: /2\. Values/ });
  const columns = values.getByLabel('Columns');
  await columns.selectOption('1');
  await expectNoNewJobs(page, controller, 2);
  await expect(values.getByText('This queues a weight_inspect job on Worker A.')).toBeVisible();
  await values.getByRole('button', { name: 'Read tile', exact: true }).click();
  await expect(values.getByRole('link', { name: 'Tile read job' })).toHaveAttribute('href', '#/jobs/job-mock-3');
  const grid = values.getByRole('grid', { name: /Stored values of blk\.0\.ffn_down\.weight/ });
  await expect(grid).toBeVisible();
  expect(jobsPosted(controller)).toEqual([inventoryRead('gguf-a'), inventoryRead('gguf-a', 128), {
    kind: 'weight_inspect', host: 'worker-a', input_ids: ['gguf-a'],
    options: { mode: 'tile', tensor, origin: [0, 32], shape: [2, 1], source_sha256: 'a'.repeat(64) },
  }]);

  await columns.selectOption('0');
  await expect(values.getByRole('button', { name: 'Read tile', exact: true })).toBeEnabled();
  await columns.selectOption('1');
  await expect(grid).toBeVisible();
  await expectNoNewJobs(page, controller, 3);
});

test('Evaluate presents synthetic scores without treating them as release proof', async ({ page, controller }) => {
  controller.load('results');
  await controller.open(page, 'evaluate');
  const bundle = page.getByRole('row').filter({ hasText: 'Fixture bundle' });
  await expect(bundle.getByText('Synthetic fixture', { exact: true })).toBeVisible();
  await expect(bundle).toContainText('3 / 4 correct (75.00%)');
  await expect(bundle).toContainText('Not release-qualified.');
});

test('Evaluate and job detail mark even full-split GSM8K scores as not release-qualified', async ({ page, controller }) => {
  controller.load('full-split');
  await controller.open(page, 'evaluate');
  const bundle = page.getByRole('row').filter({ hasText: 'Fixture bundle' });
  await expect(bundle.getByText('Standalone GSM8K score', { exact: true })).toBeVisible();
  await expect(bundle).toContainText('Not release-qualified.');
  await bundle.getByRole('link', { name: 'GSM8K score job for Fixture bundle' }).click();
  const result = page.getByRole('region', { name: 'Result' });
  await expect(result.getByText('Full pinned split', { exact: true })).toBeVisible();
  await expect(result.getByText('Not release-qualified', { exact: true })).toBeVisible();
});

test('Evaluate asks for an approved runtime, not a qualified host', async ({ page, controller }) => {
  await controller.open(page, 'evaluate');
  await expect(page.getByText('Needs a worker with an approved Halogen runtime.', { exact: false })).toBeVisible();
});

test('Benchmarks loads an endpoint result and restricts runs to advertised choices', async ({ page, controller }) => {
  await controller.open(page, 'benchmarks');
  await expect(page.getByText('latency', { exact: true })).toBeVisible();
  const queue = page.getByRole('button', { name: 'Queue benchmark run' });
  await expect(queue).toBeDisabled();
  await pick(page, 'Host', /^Worker A/);
  await pick(page, 'Endpoint target', /^Fixture endpoint/);
  await page.getByRole('checkbox', { name: /^Endpoint smoke/ }).check();
  await expect(queue).toBeEnabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Calibration opens a local report, isolates invalid JSON and closes the report', async ({ page, controller }) => {
  await controller.open(page, 'calibration');
  const report = { schema: 'brewery-halogen-speculation-v1', requests: 4, total_with_speculation: 3, missing_speculation: 1,
    methods: { head: { rounds: 5, drafted: 10, accepted: 7, acceptance_fraction: 0.7 } } };
  await page.getByLabel('Report files (JSON)').setInputFiles([
    { name: 'spec.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(report)) },
    { name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{bad') },
  ]);
  const opened = page.getByRole('region', { name: 'Speculation acceptance spec.json' });
  await expect(opened).toBeVisible();
  await expect(page.getByRole('region', { name: 'Not opened bad.json' }).getByRole('alert')).toContainText('JSON');
  await opened.getByRole('button', { name: 'Close spec.json' }).click();
  await expect(opened).toHaveCount(0);
  expect(controller.requests().filter(({ method }) => method !== 'GET')).toEqual([]);
});

test('Settings saves an evaluation edit without starting a job', async ({ page, controller }) => {
  await controller.open(page, 'settings');
  const gsm8k = page.getByRole('checkbox', { name: 'Allow GSM8K evaluation on this worker' });
  const save = page.getByRole('button', { name: 'Save evaluation settings' });
  await expect(gsm8k).toBeChecked();
  await gsm8k.uncheck();
  await expect(page.getByText('You have unsaved edits;', { exact: false })).toBeVisible();
  await save.click();
  await expect(page.getByText('Evaluation settings saved on worker-a. No job was started.')).toBeVisible();
  const saves = controller.requests('PUT', '/api/settings/worker-a');
  expect(saves).toHaveLength(1);
  expect(saves[0].body).toMatchObject({ revision: 1, evaluation_enabled: false });
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Settings readiness does not contradict live capabilities', async ({ page, controller }) => {
  controller.load('settings-blocked');
  await controller.open(page, 'settings');
  const readiness = page.getByRole('region', { name: 'Evaluation readiness' });
  await expect(readiness.getByText('Configuration ready', { exact: true })).toBeVisible();
  await expect(readiness.getByRole('row', { name: /^Benchmarks/ })).toContainText('Available');
  await page.getByRole('navigation', { name: 'Workers' }).getByRole('button', { name: /Worker B/ }).click();
  await expect(readiness.getByText('Configuration not ready', { exact: true })).toBeVisible();
  await expect.soft(readiness.getByText('Not available', { exact: true })).toHaveCount(4);
  await expect.soft(readiness.getByRole('row', { name: /^Benchmarks/ })).toContainText('Resume job scheduling.');
  await expect.soft(readiness.getByRole('row', { name: /^GSM8K evaluation/ })).toContainText('Allow evaluation in the worker policy.');
  const gsm8k = page.getByRole('checkbox', { name: 'Allow GSM8K evaluation on this worker' });
  await expect(gsm8k).toBeDisabled();
  await expect(gsm8k).toHaveAccessibleDescription('Allow evaluation in the worker policy.');
});

// Audit regressions for later tasks. Replace each placeholder with an assertion
// of the corrected behavior; do not lock in the current broken behavior.
