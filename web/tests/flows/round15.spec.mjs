import { test, expect } from './fixtures.mjs';
import { job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (page, name) => page.getByRole('button', { name, exact: true });
const key = request => request.headers['idempotency-key'];
const posts = c => c.requests('POST', '/api/jobs');
const choose = async (page, field, name) => {
  await page.locator(field).click();
  await page.getByRole('option', { name }).click();
};
const refresh = page => page.evaluate(() => window.dispatchEvent(new Event('online')));
const post = page => page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.request().method() === 'POST');
async function send(page, label) {
  const pending = post(page);
  await button(page, label).press('Enter');
  return pending;
}
async function benchmarks(page) {
  await choose(page, '#bm-host', /^Worker A/);
  await choose(page, '#bm-target', /^Fixture endpoint/);
  await page.getByRole('checkbox', { name: /^Endpoint smoke/ }).check();
}

test('Jobs and Back matches an accepted benchmark; only Queue another opts out', async ({ page, controller }) => {
  const stale = await (await page.request.get(`${controller.origin}/api/state`)).json();
  controller.respond('GET', '/api/state', { json: stale });
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  let accepted;
  await page.route('**/api/jobs', async route => {
    const response = await route.fetch();
    expect(response.status()).toBe(201);
    accepted = (await response.json()).id;
    await route.abort();
  }, { times: 1 });
  await button(page, 'Queue benchmark run').press('Enter');
  await expect(page.getByRole('alert').filter({ hasText: 'Check Jobs' })).toBeVisible();
  await page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: /^Jobs\b/ }).click();
  await expect(page.getByRole('heading', { name: 'Jobs', exact: true })).toBeVisible();
  await page.goBack();
  await benchmarks(page);
  const matched = await send(page, 'Queue benchmark run');
  expect(matched.status()).toBe(200);
  expect(matched.headers()['brewery-duplicate']).toBe(accepted);
  expect(key(posts(controller)[0])).not.toBe(key(posts(controller)[1]));
  const status = page.getByText('This request is already queued as job', { exact: false });
  await expect(status.getByRole('link')).toHaveAttribute('href', `#/jobs/${accepted}`);
  await page.locator('#bm-target').press('Enter');
  expect(posts(controller)).toHaveLength(2);
  const newJob = await send(page, 'Queue another');
  expect(newJob.status()).toBe(201);
  expect((await newJob.json()).id).not.toBe(accepted);
  const [first, matchedPost, repeat] = posts(controller);
  expect(repeat.body).toEqual(first.body);
  expect(repeat.headers['brewery-allow-duplicate']).toBe('1');
  expect(new Set([key(first), key(matchedPost), key(repeat)]).size).toBe(3);
});

test('an exact JSON proxy 403 cannot discard an unconfirmed key', async ({ page, controller }) => {
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  await page.route('**/api/jobs', async route => {
    expect((await route.fetch()).status()).toBe(201);
    await route.abort();
  }, { times: 1 });
  await button(page, 'Queue benchmark run').press('Enter');
  const alert = page.getByRole('alert').filter({ hasText: 'Check Jobs' });
  await expect(alert).toBeVisible();
  await page.route('**/api/jobs', route => route.fulfill({ status: 403, json: { error: 'Gateway access policy denied this request.' } }), { times: 1 });
  expect((await send(page, 'Queue benchmark run')).status()).toBe(403);
  await expect(alert).toContainText('Gateway access policy');
  const matched = await send(page, 'Queue benchmark run');
  expect(matched.status()).toBe(200);
  expect(matched.headers()['brewery-duplicate']).toBeUndefined(); // Key replay wins.
  expect(key(posts(controller)[1])).toBe(key(posts(controller)[0]));
});

test('a keyboard Generate rejection scrolls into view without losing retry focus', async ({ page, controller }) => {
  controller.respond('POST', '/api/jobs', { status: 400, json: { error: 'The prompt exceeds 65536 bytes.' } });
  await controller.open(page, 'evaluate');
  await choose(page, '#g-bundle', /^Fixture bundle/);
  await page.getByRole('radio', { name: 'Type a prompt', exact: true }).check();
  await page.locator('#g-text').fill('x'.repeat(65_537));
  const action = button(page, 'Queue generation');
  await action.evaluate(el => el.scrollIntoView({ block: 'end' }));
  expect((await send(page, 'Queue generation')).status()).toBe(400);
  const alert = page.getByText('The prompt exceeds 65536 bytes.', { exact: true });
  await expect(alert).toBeInViewport({ ratio: 1 });
  await expect(action).toBeFocused();
});

const editors = [
  ['Layer recipe', 'convert?format=recipe&output=gguf', '#recipe-source', /^Fixture checkpoint/, 'Read layers', 'Read next layers', 'source-a'],
  ['GGUF Profiles', 'profiles?format=gguf', '[role=combobox][aria-labelledby$="-label"]', /^Fixture GGUF GGUF/, 'Read tensor list', 'Read next tensors', 'gguf-a'],
  ['Visual Weights', 'weights?source=gguf-a', null, null, 'Read tensor list', 'Read next tensors', 'gguf-a'],
];

for (const [name, route, field, option, firstLabel, nextLabel, sourceId] of editors) {
  async function open(page, controller) {
    await controller.open(page, route);
    if (name === 'GGUF Profiles') {
      await page.getByRole('combobox', { name: 'Source model', exact: true }).click();
      await page.getByRole('option', { name: option }).click();
    } else if (field) await choose(page, field, option);
  }
  test(`${name} restarts at offset zero after a page-two digest failure`, async ({ page, controller }) => {
    const state = scenario('weight-reads');
    const model = state.models.find(m => m.id === sourceId);
    const changed = 'b'.repeat(64);
    controller.respond('GET', '/api/state', { json: state });
    controller.respond('POST', '/api/jobs', ({ body }) => {
      const count = posts(controller).length;
      const id = `inventory-${count}`;
      const result = weightRead(id, body, model.kind === 'hf' ? { ...model, kind: 'gguf' } : model);
      result.result.report.source_format = model.kind;
      result.result.inputs[0].kind = model.kind;
      if (count === 2) Object.assign(result, { state: 'failed', error: 'visual weight source SHA-256 changed', has_result: false });
      if (count > 2) {
        result.result.report.source_sha256 = changed;
        Object.assign(result.result.inputs[0], { sha256: changed, bundle_sha256: changed });
      }
      state.jobs.push(result);
      controller.respond('GET', `/api/jobs/${id}`, { json: result });
      return { status: 201, json: result };
    });
    await open(page, controller);
    await send(page, firstLabel);
    await send(page, nextLabel);
    const error = page.getByRole('alert').filter({ hasText: 'visual weight source SHA-256 changed' });
    await expect(error).toBeVisible();
    await send(page, 'Read from page 1');
    expect(posts(controller)[2].body.options).toMatchObject({ offset: 0 });
    expect(posts(controller)[2].body.options.source_sha256).toBeUndefined();
    await expect(error).toHaveCount(0);
    await send(page, nextLabel);
    expect(posts(controller)[3].body.options.source_sha256).toBe(changed);
  });

  test(`${name} shows receipt GET failures and recovers without another POST`, async ({ page, controller }) => {
    controller.load('weight-reads');
    controller.respond('GET', '/api/jobs/job-mock-1', { status: 503, json: { error: 'Fixture detail outage.' } });
    await open(page, controller);
    await send(page, firstLabel);
    await expect(page.getByRole('alert').filter({ hasText: 'Fixture detail outage.' })).toBeVisible();
    controller.load('weight-reads');
    const state = scenario('weight-reads');
    const model = state.models.find(m => m.id === sourceId);
    const read = weightRead('job-mock-1', posts(controller)[0].body, model);
    state.jobs.push(read);
    controller.respond('GET', '/api/state', { json: state });
    controller.respond('GET', '/api/jobs/job-mock-1', { json: read });
    await refresh(page);
    await expect(page.getByText('Fixture detail outage.', { exact: false })).toHaveCount(0);
    expect(posts(controller)).toHaveLength(1);
  });
}

test('70 distinct Visual Weights tile reads can be sent despite lost replies', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  const source = state.models.find(m => m.id === 'gguf-a');
  const cached = weightRead('cached-inventory', { kind: 'weight_inspect', input_ids: [source.id], host: source.host,
    options: { mode: 'inventory', offset: 0, limit: 128 } }, source);
  cached.result.report.tensors[0] = { ...cached.result.report.tensors[0], shape: [70, 2, 33] };
  state.jobs.push(cached);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', '/api/jobs/cached-inventory', { json: cached });
  await page.route('**/api/jobs', async route => {
    expect((await route.fetch()).status()).toBe(201);
    await route.fulfill({ status: 502, json: { error: 'Tile reply lost.' } });
  });
  await controller.open(page, 'weights?source=gguf-a');
  await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
  for (let i = 0; i < 70; i++) {
    await page.locator('#vw-lead-0').selectOption(`${i}`);
    await button(page, 'Read tile').press('Enter');
    await expect(page.getByRole('alert').filter({ hasText: 'Tile reply lost.' })).toBeVisible({ timeout: 1500 });
    expect(posts(controller)).toHaveLength(i + 1);
  }
  expect(posts(controller)).toHaveLength(70);
});

test('a resumed active layer read reports failure and uses the current retry policy', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  const source = state.models.find(m => m.id === 'source-a');
  state.models.push({ ...source, id: 'other-source', name: 'Other checkpoint' });
  const active = job('resumed-read', 'weight_inspect', { input_ids: [source.id], host: source.host,
    options: { mode: 'inventory', offset: 0, limit: 128 }, state: 'running' });
  state.jobs.push(active);
  controller.respond('GET', '/api/state', { json: state });
  await controller.open(page, 'convert?format=recipe&output=gguf');
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  await expect(page.getByRole('link', { name: 'Layer read job' })).toHaveAttribute('href', '#/jobs/resumed-read');
  await choose(page, '#recipe-source', /^Other checkpoint/);
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  Object.assign(active, { state: 'failed', error: 'Resumed fixture read failed.' });
  const host = state.hosts.find(h => h.id === source.host);
  host.capability_reasons = { weight_inspect: { text: 'Inventory paused by test policy.', fix_anchor: '#/settings' } };
  host.capabilities.weight_inspect = false;
  await refresh(page);
  await expect(page.getByText(active.error, { exact: true })).toBeVisible();
  await expect(page.getByText('Inventory paused by test policy.', { exact: true })).toBeVisible();
  await expect(button(page, 'Read layers')).toHaveCount(0);
});
