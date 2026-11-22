import { test, expect } from './fixtures.mjs';

test('conversion eligibility checks the destination without starting work', async ({ request, controller }) => {
  controller.load('gguf');
  const origin = controller.origin;
  const endpoint = `${origin}/api/conversions/eligibility`;
  const body = { format: 'gguf', input_ids: ['gguf-a'], options: { preset: 'Q8_0' } };
  const accepted = await request.post(endpoint, { data: { ...body, host: 'worker-a' } });
  expect(accepted.ok()).toBe(true);
  expect(await accepted.json()).toEqual({ eligible: true, reasons: [] });
  const blocked = await request.post(endpoint, { data: { ...body, host: 'worker-b' } });
  expect(blocked.ok()).toBe(true);
  const result = await blocked.json();
  expect(result.eligible).toBe(false);
  expect(result.reasons[0].fix_anchor).toBe('#/settings/infrastructure?host=worker-b&section=policy');
  const state = await (await request.get(`${origin}/api/state`)).json();
  expect(state.eligibility.hosts['worker-b'].gguf_output_by_kind.gguf.reasons).toContainEqual(result.reasons[0]);
  expect(state.jobs).toEqual([]);
  expect(await (await request.get(`${origin}/api/conversions`)).json()).toEqual([]);
});

test('opens an empty Library', async ({ page, controller }) => {
  controller.load('empty');
  await controller.open(page, 'library');
  await expect(page.getByRole('heading', { name: 'Library', exact: true })).toBeVisible();
  await expect(page.getByText('No entries yet.', { exact: false })).toBeVisible();
});

test('registers a model and records exactly one request', async ({ page, controller }) => {
  await controller.open(page, 'library');
  await expect(page.getByText('Fixture checkpoint', { exact: true })).toBeVisible();
  const model = { name: 'New sample', host: 'worker-a', kind: 'gguf', path: '/fixtures/new.gguf' };
  const form = page.getByRole('region', { name: 'Register a source' });
  await form.getByRole('combobox', { name: 'Host', exact: true }).click();
  await page.getByRole('option', { name: 'Worker A', exact: false }).click();
  await form.getByRole('combobox', { name: 'Path on host' }).fill(model.path);
  await form.getByRole('combobox', { name: 'Name (optional)' }).fill(model.name);
  await form.getByRole('button', { name: 'Register', exact: true }).click();
  await expect(page.getByText('New sample', { exact: true })).toBeVisible();
  expect(controller.requests('POST', '/api/models').map(({ body }) => body)).toEqual([model]);
});

for (const status of [404, 500, 503]) {
  test(`shows a startup error for HTTP ${status}`, async ({ page, controller }) => {
    controller.load(`api-${status}`);
    await controller.open(page);
    await expect(page.getByRole('heading', { name: 'Controller not reachable' })).toBeVisible();
    await expect(page.getByText(`Fixture HTTP ${status}`, { exact: false })).toBeVisible();
    expect(controller.requests('GET', '/api/state').length).toBeGreaterThan(0);
  });
}

test('keeps the last Library during an outage and recovers', async ({ page, controller }) => {
  await controller.open(page);
  await expect(page.getByText('Fixture checkpoint', { exact: true })).toBeVisible();
  controller.load('controller-down');
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByRole('alert')).toContainText('Lost contact with the controller (The controller is not reachable.)');
  await expect(page.getByText('Fixture checkpoint', { exact: true })).toBeVisible();
  controller.load('empty');
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText('No entries yet.', { exact: false })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('retries Settings after an endpoint error without disconnecting the shell', async ({ page, controller }) => {
  controller.respond('GET', '/api/settings', { status: 503, json: { error: 'Fixture settings outage' } });
  await controller.open(page, 'settings');
  await expect(page.getByRole('alert')).toContainText('Fixture settings outage');
  await expect(page.getByText('Connected', { exact: true })).toBeVisible();
  controller.load('live-shaped');
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Evaluation readiness' })).toBeVisible();
  expect(controller.requests('GET', '/api/settings')).toHaveLength(2);
});
