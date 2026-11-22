import { test, expect } from './fixtures.mjs';

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
