import { test, expect } from './fixtures.mjs';
import { conversion, detail, job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (page, name) => page.getByRole('button', { name, exact: true });
const key = request => request.headers['idempotency-key'];
const choose = async (page, field, name) => {
  await page.locator(field).click();
  await page.getByRole('option', { name }).click();
};
function gate() {
  let release;
  const wait = new Promise(resolve => { release = resolve; });
  return { wait, release };
}

async function open(page, controller, prior, view) {
  controller.respond('GET', `/api/jobs/${prior.id}`, { json: prior });
  await controller.open(page, `jobs/${prior.id}`);
  await page.getByRole('link', { name: `Open in ${view}`, exact: true }).click();
}

async function submit(page, label, path = '/api/jobs') {
  const pending = page.waitForResponse(r => r.url().endsWith(path) && r.request().method() === 'POST');
  await button(page, label).click();
  const response = await pending;
  return [response.status(), (await response.json()).id];
}

test('unavailable retry history blocks queue until reloaded or explicitly left', async ({ page, controller }) => {
  const prior = detail(job('failed-benchmark', 'benchmark', { state: 'failed',
    options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } }));
  await open(page, controller, prior, 'Benchmarks');
  const [accepted, id] = await submit(page, 'Queue benchmark run');
  expect(accepted).toBe(201);
  controller.respond('GET', `/api/jobs/${prior.id}`, { status: 503, json: { error: 'History unavailable.' } });
  await page.reload();
  const reload = button(page, 'Reload previous request');
  await expect(reload).toBeVisible();
  await expect(button(page, 'Queue benchmark run')).toBeDisabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(1);
  controller.respond('GET', `/api/jobs/${prior.id}`, { json: prior });
  await reload.click();
  expect(await submit(page, 'Queue benchmark run')).toEqual([200, id]);
  await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
  await button(page, 'Start a new run').click();
  await expect(page).toHaveURL(/#\/benchmarks$/);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(2);
  const [fresh, freshId] = await submit(page, 'Queue benchmark run');
  expect(fresh).toBe(201);
  expect(freshId).not.toBe(id);
  const [first, match, last] = controller.requests('POST', '/api/jobs');
  expect(last.body).toEqual(first.body);
  expect(key(match)).toBe(key(first));
  expect(key(last)).toMatch(/^conv-/);

  controller.respond('GET', `/api/jobs/${prior.id}`, { status: 503, json: { error: 'History unavailable.' } });
  await controller.open(page, `benchmarks?retry=${prior.id}`);
  await button(page, 'Start a new run instead').click();
  await expect(page).toHaveURL(/#\/benchmarks$/);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(3);
});

test('managed custom stage and parent retry match the same workflow', async ({ page, controller }) => {
  const revision = 'a'.repeat(64);
  const stage = detail(job('failed-stage', 'verify', { state: 'failed', profile_revision: revision,
    input_ids: ['source-a', 'bundle-a'], options: { decode_backend: 'cpu' } }));
  const parent = conversion('failed-parent', { state: 'failed', stage: 'verify', job_ids: { verify: stage.id },
    profile_revision: revision, options: { threads: 8 }, decode_backend: 'cpu' });
  controller.respond('GET', `/api/jobs/${stage.id}`, { json: stage });
  controller.respond('GET', '/api/conversions', { json: [parent] });
  controller.respond('GET', `/api/conversions/${parent.id}`, { json: parent });
  await controller.open(page, `convert?retry=${stage.id}`);
  await expect(page.locator('#wf-profile')).toHaveValue(`rev:${revision}`);
  const [accepted, id] = await submit(page, 'Start conversion', '/api/conversions');
  expect(accepted).toBe(201);
  await expect(page.getByText(/^Started conversion /)).toBeVisible();
  await controller.open(page, `jobs/${stage.id}`);
  await expect(page.getByRole('link', { name: 'Run again', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Open in Convert', exact: true })).toHaveCount(0);
  await page.getByRole('link', { name: 'Run again', exact: true }).click();
  expect(await submit(page, 'Start conversion', '/api/conversions')).toEqual([200, id]);
  const [first, second] = controller.requests('POST', '/api/conversions');
  expect(second.body).toEqual(first.body);
  expect(key(second)).toBe(key(first));
  expect(key(first)).toMatch(/^r-failed-parent-convert.managed-/);
});

for (const kind of ['transfer', 'benchmark']) {
  test(`${kind} rechecking the same entries keeps the wire body and key`, async ({ page, controller }) => {
    const state = scenario('live-shaped');
    state.hosts[0].benchmarks.push({ id: 'extra', label: 'Extra suite' });
    controller.respond('GET', '/api/state', { json: state });
    const transfer = kind === 'transfer';
    const prior = detail(job(`failed-${kind}`, kind, { state: 'failed', ...(transfer
      ? { host: 'worker-b', input_ids: ['gguf-a', 'source-a'] }
      : { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke', 'extra'] } }) }));
    const label = transfer ? 'Transfer' : 'Queue benchmark run';
    await open(page, controller, prior, transfer ? 'Transfers' : 'Benchmarks');
    const [accepted, id] = await submit(page, label);
    expect(accepted).toBe(201);
    const entry = page.getByRole('checkbox', { name: transfer ? /^Fixture GGUF / : /^Endpoint smoke/ });
    await entry.uncheck();
    await expect(button(page, label)).toBeDisabled();
    await entry.check();
    expect(await submit(page, label)).toEqual([200, id]);
    const [first, second] = controller.requests('POST', '/api/jobs');
    expect(second.body).toEqual(first.body);
    expect(key(second)).toBe(key(first));
    if (!transfer) expect(second.body.options.benchmarks).toEqual(['smoke', 'extra']);
  });
}

for (const kind of ['hy4', 'generate']) {
  test(`a matched ${kind} can be deliberately repeated with the same options`, async ({ page, controller }) => {
    const hy4 = kind === 'hy4';
    const state = scenario('live-shaped');
    state.hosts[0].capabilities.hy4_encode = true;
    state.hosts[0].capabilities.hy4_plan = true;
    controller.respond('GET', '/api/state', { json: state });
    const prior = detail(job(`failed-${kind}`, hy4 ? 'convert' : 'generate', { state: 'failed', ...(hy4
      ? { input_ids: ['source-a'], options: { hy4_overrides: { 'blk.1.attn_gate.weight': 'Q8_0' } } }
      : { input_ids: ['bundle-a'], prompt_text: 'Keep the output short.', options: { max_tokens: 7 } }) }));
    const label = hy4 ? 'Queue Hy4 conversion' : 'Queue generation';
    await open(page, controller, prior, hy4 ? 'Convert' : 'Evaluate');
    const [accepted, id] = await submit(page, label);
    expect(accepted).toBe(201);
    expect(await submit(page, label)).toEqual([200, id]);
    if (hy4) {
      await page.getByLabel('Hy4 operation').selectOption({ label: 'Metadata plan' });
      await expect(page.getByText('Result of the previous submission', { exact: true })).toBeVisible();
    } else {
      const limit = page.getByRole('spinbutton', { name: 'Maximum output tokens' });
      await expect(limit).toHaveValue('7');
      await limit.fill('11');
    }
    await button(page, 'Start a new run').click();
    await expect(page).not.toHaveURL(/retry=/);
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(2);
    const [fresh, newId] = await submit(page, label);
    expect(fresh).toBe(201);
    expect(newId).not.toBe(id);
    const [first, , last] = controller.requests('POST', '/api/jobs');
    expect(last.body).toEqual(first.body);
    expect(key(last)).toMatch(/^conv-/);
  });
}

test('cleanup replay restores focus when the confirmation is removed', async ({ page, controller }) => {
  const prior = detail(job('failed-cleanup', 'cache_cleanup', { state: 'failed', host: 'worker-b',
    options: { bundle_sha256: 'a'.repeat(64) } }));
  await open(page, controller, prior, 'Transfers');
  const clean = button(page, 'Clean cache…');
  await clean.press('Enter');
  await submit(page, 'Queue cleanup');
  await clean.press('Enter');
  await button(page, 'Queue cleanup').press('Enter');
  await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
  await expect(clean).toBeFocused();
});

test('a late cleanup match does not move focus back from another field', async ({ page, controller }) => {
  const prior = detail(job('failed-cleanup', 'cache_cleanup', { state: 'failed', host: 'worker-b',
    options: { bundle_sha256: 'a'.repeat(64) } }));
  await open(page, controller, prior, 'Transfers');
  await button(page, 'Clean cache…').press('Enter');
  await submit(page, 'Queue cleanup');
  const held = gate();
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const response = await route.fetch();
    await held.wait;
    await route.fulfill({ response });
  });
  try {
    await button(page, 'Clean cache…').press('Enter');
    await button(page, 'Queue cleanup').press('Enter');
    const digest = page.locator('#cl-digest');
    await digest.focus();
    held.release();
    await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
    await expect(digest).toBeFocused();
  } finally { held.release(); }
});

test('a new replacement component stays selected and blocks when it disappears', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'new-vision', name: 'New vision', kind: 'vision', host: 'worker-a', path: '/fixtures/vision' });
  controller.respond('GET', '/api/state', () => ({ json: state }));
  await open(page, controller, detail(job('failed-plan', 'plan', { state: 'failed',
    input_ids: ['source-a'], options: { profile: 'v2' } })), 'Convert');
  const vision = page.locator('#cv-comp-vision');
  await vision.selectOption('new-vision');
  await expect(button(page, 'Queue plan')).toBeEnabled();
  state.models = state.models.filter(model => model.id !== 'new-vision');
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText(/The vision input new-vision is not available/)).toBeVisible();
  await expect(vision).toHaveValue('new-vision');
  await expect(button(page, 'Queue plan')).toBeDisabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await vision.selectOption('');
  expect((await submit(page, 'Queue plan'))[0]).toBe(201);
});

test('late plain HGN acceptance does not mark changed inputs clean', async ({ page, controller }) => {
  const prior = conversion('earlier-hgn', { state: 'failed', input_ids: ['source-a'], options: { profile: 'v2' } });
  controller.respond('GET', `/api/conversions/${prior.id}`, { json: prior });
  const held = gate();
  await page.route('**/api/conversions', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const response = await route.fetch();
    await held.wait;
    await route.fulfill({ response });
  });
  await controller.open(page, 'convert');
  await choose(page, '#wf-source', /^Fixture checkpoint/);
  const profile = page.locator('#wf-profile');
  await profile.selectOption('v2');
  try {
    await button(page, 'Start conversion').click();
    await expect.poll(() => controller.requests('POST', '/api/conversions').length).toBe(1);
    await profile.selectOption('ht43');
    held.release();
    await expect(page.getByText(/^Started conversion /)).toBeVisible();
    await expect(page.getByText('Result of the previous submission', { exact: true })).toBeVisible();
    await page.evaluate(id => { location.hash = `#/convert?retry_run=${id}`; }, prior.id);
    await expect(button(page, 'Use previous inputs')).toBeVisible();
    await expect(profile).toHaveValue('ht43');
  } finally { held.release(); }
});

for (const mode of ['inventory', 'tile']) {
  test(`visual ${mode} matches the exact previous request and reserves its ID before a poll`, async ({ page, controller }) => {
    controller.load('weight-reads');
    const state = scenario('weight-reads');
    const source = state.models.find(model => model.id === 'gguf-a');
    const inventory = { mode: 'inventory', offset: 0, limit: 128 };
    if (mode === 'tile') {
      const cached = weightRead('cached-inventory', { kind: 'weight_inspect', input_ids: [source.id], host: source.host, options: inventory }, source);
      state.jobs.push(cached);
      controller.respond('GET', `/api/jobs/${cached.id}`, { json: cached });
    }
    controller.respond('GET', '/api/state', { json: state });
    const options = mode === 'tile' ? {
      mode, tensor: 'blk.0.ffn_down.weight', origin: [0, 0], shape: [2, 32], source_sha256: 'a'.repeat(64),
    } : inventory;
    const prior = detail(job('failed-weight-read', 'weight_inspect', { state: 'failed', input_ids: [source.id], host: source.host, options }));
    await open(page, controller, prior, 'Weights');
    const failed = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.status() === 502);
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      const response = await route.fetch();
      expect(response.status()).toBe(201);
      controller.respond('GET', `/api/jobs/${(await response.json()).id}`, { status: 503, json: { error: 'Not ready.' } });
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
    await button(page, 'Queue previous request').click();
    await failed;
    if (mode === 'tile') await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
    const label = mode === 'tile' ? 'Read tile' : 'Read tensor list';
    const [status, id] = await submit(page, label);
    expect(status).toBe(200);
    await expect(page.getByRole('link', { name: mode === 'tile' ? 'Tile read job' : 'Tensor list read job' })).toHaveAttribute('href', `#/jobs/${id}`);
    await expect(button(page, label)).toHaveCount(0);
    const [first, second] = controller.requests('POST', '/api/jobs');
    expect(key(first)).toMatch(/^r-failed-weight-read-/);
    expect(key(second)).toBe(key(first));
    expect(second.body).toEqual(first.body);
  });
}

test('a matched layer inventory recovers after a transient detail GET failure', async ({ page, controller }) => {
  controller.load('weight-reads');
  const state = scenario('weight-reads');
  const source = state.models.find(model => model.id === 'source-a');
  controller.respond('GET', '/api/state', () => ({ json: state }));
  await controller.open(page, 'convert?format=recipe&output=gguf');
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  let accepted;
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const response = await route.fetch();
    accepted = weightRead((await response.json()).id, route.request().postDataJSON(), source);
    await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
  }, { times: 1 });
  await button(page, 'Read layers').click();
  await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toBeVisible();
  controller.respond('GET', `/api/jobs/${accepted.id}`, { status: 503, json: { error: 'Detail temporarily unavailable.' } });
  expect(await submit(page, 'Read layers')).toEqual([200, accepted.id]);
  await expect.poll(() => controller.requests('GET', `/api/jobs/${accepted.id}`).length).toBeGreaterThan(0);
  state.jobs.push(accepted);
  controller.respond('GET', `/api/jobs/${accepted.id}`, { json: accepted });
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText('Choose a default type and optional layer overrides.')).toBeVisible();
  await expect(page.getByText('Detail temporarily unavailable.')).toHaveCount(0);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(2);
});

test('a layer recipe match belongs to the submitted recipe, not the edited draft', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  Object.assign(state.hosts[0].capabilities, { gguf_layer_recipe: true, gguf_hf_output: true });
  const source = state.models.find(model => model.id === 'source-a');
  const inventory = weightRead('recipe-inventory', { kind: 'weight_inspect', host: source.host, input_ids: [source.id],
    options: { mode: 'inventory', offset: 0, limit: 128 } }, source);
  state.jobs.push(inventory);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
  const prior = detail(job('failed-recipe-stage', 'convert', { state: 'failed', format: 'gguf', input_ids: [source.id], options: {
    layer_recipe: { schema: 'brewery.layer-quant.v1', source_sha256: 'a'.repeat(64), default_type: 'F16', layers: { 0: 'Q8_0' } },
  } }));
  const parent = conversion('failed-recipe-run', { state: 'failed', format: 'gguf', job_ids: { convert: prior.id },
    input_ids: prior.input_ids, options: prior.options });
  controller.respond('GET', '/api/conversions', { json: [parent] });
  await open(page, controller, prior, 'Convert');
  const [accepted, id] = await submit(page, 'Run layer recipe', '/api/conversions');
  expect(accepted).toBe(201);
  expect(await submit(page, 'Run layer recipe', '/api/conversions')).toEqual([200, id]);
  const type = page.getByRole('combobox', { name: 'Default type', exact: true });
  await type.selectOption('BF16');
  await expect(page.getByText('Result of the previous submission', { exact: true })).toBeVisible();
  await button(page, 'Start a new run').click();
  await expect(page).toHaveURL(/#\/convert\?format=recipe&output=gguf$/);
  await expect(type).toHaveValue('F16');
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(2);
  expect((await submit(page, 'Run layer recipe', '/api/conversions'))[0]).toBe(201);
  const [first, second, fresh] = controller.requests('POST', '/api/conversions');
  expect(key(first)).toMatch(/^r-failed-recipe-run-convert.recipe-/);
  expect(key(second)).toBe(key(first));
  expect(fresh.body).toEqual(first.body);
  expect(key(fresh)).toMatch(/^conv-/);
});
