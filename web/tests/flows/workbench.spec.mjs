import { test, expect } from './fixtures.mjs';

async function pick(page, field, option) {
  await page.getByRole('combobox', { name: field, exact: true }).click();
  await page.getByRole('option', { name: option }).click();
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

test('a missing job deep link shows an error without a result', async ({ page, controller }) => {
  await controller.open(page, 'jobs/missing');
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Result' })).toHaveCount(0);
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
  await page.getByRole('radio', { name: 'Type a prompt' }).check();
  await page.getByRole('textbox', { name: 'Prompt text' }).fill('Compare these responses.');
  await pick(page, 'Model A', /^Fixture GGUF\b(?! copy)/);
  const queue = page.getByRole('button', { name: 'Queue comparison' });
  await expect(queue).toBeDisabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await pick(page, 'Model B', /^Fixture GGUF copy/);
  await expect(queue).toBeEnabled();
});

test('Evaluate and Benchmarks explain when no worker can run those jobs', async ({ page, controller }) => {
  controller.load('capabilities-off');
  await controller.open(page, 'evaluate');
  await expect(page.getByText('No worker can run GSM8K evaluation yet.', { exact: false })).toBeVisible();
  await controller.open(page, 'benchmarks');
  await expect(page.getByText('No worker can run benchmarks yet.', { exact: false })).toBeVisible();
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
  await page.getByLabel('Output profile', { exact: true }).selectOption('v2');
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
  await page.getByRole('button', { name: /Use Fixture HGN profile revision .* in Convert/ }).click();
  await expect(page).toHaveURL(/#\/convert$/);
  await expect(page.getByLabel('Output profile', { exact: true })).toHaveValue(`rev:${'a'.repeat(64)}`);
});

test('Profiles keeps GGUF revisions separate from HGN revisions', async ({ page, controller }) => {
  controller.load('saved-profiles');
  await controller.open(page, 'profiles?format=gguf');
  await expect(page.getByRole('table', { name: 'Saved GGUF profile revisions' })
    .getByRole('rowheader', { name: 'Fixture GGUF profile' })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'Fixture HGN profile' })).toHaveCount(0);
});

test('Evaluate presents synthetic scores without treating them as release proof', async ({ page, controller }) => {
  controller.load('results');
  await controller.open(page, 'evaluate');
  const bundle = page.getByRole('row').filter({ hasText: 'Fixture bundle' });
  await expect(bundle.getByText('Synthetic fixture', { exact: true })).toBeVisible();
  await expect(bundle).toContainText('3 / 4 correct (75.00%)');
  await expect(bundle).toContainText('does not qualify a release');
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

// Audit regressions for later tasks. Replace each placeholder with an assertion
// of the corrected behavior; do not lock in the current broken behavior.
for (const finding of [
  'Library identity stays Not yet hashed because its report fields are absent',
  'Weights does not submit inspection jobs until explicitly requested',
  'Convert requires every managed conversion stage, not just convert',
  'GGUF conversion does not offer a destination it will later reject',
  'Profiles warns before discarding unsaved editor changes',
  'Job detail clears a previous result when the id changes',
  'A missing job id stops polling after a 404',
  'Inspection results support inventory and tile modes',
  'Job detail does not mark a verification that was not performed as Failed',
  'Transfers keeps entry names readable at narrow widths',
  'Compare explains missing capability before selection',
  'Evaluate explains disabled generation and baseline before selection',
  'Calibration keeps reports during navigation',
  'Settings warns before discarding unsaved evaluation or policy edits',
  'Settings readiness does not contradict live capabilities',
  'Settings host selection also selects that infrastructure host',
]) {
  test.fixme(finding, async () => {});
}
