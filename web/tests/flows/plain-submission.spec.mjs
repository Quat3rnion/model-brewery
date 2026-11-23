import { test, expect } from './fixtures.mjs';
import { scenario } from './mock/scenarios.mjs';

const button = (page, name) => page.getByRole('button', { name, exact: true });
const check = (page, name) => page.getByRole('checkbox', { name });
const posts = c => c.requests('POST', '/api/jobs');
const choose = async (page, field, name) => {
  await field.click();
  await page.getByRole('option', { name }).click();
};
const refresh = page => page.evaluate(() => window.dispatchEvent(new Event('online')));
const response = (page, path = '/api/jobs') =>
  page.waitForResponse(r => r.url().endsWith(path) && r.request().method() === 'POST');

async function benchmarks(page) {
  await choose(page, page.locator('#bm-host'), /^Worker A/);
  await choose(page, page.locator('#bm-target'), /^Fixture endpoint/);
  await check(page, /^Endpoint smoke/).check();
}

async function transfers(page) {
  await choose(page, page.locator('#tx-host'), /^Worker B/);
  await check(page, /^Fixture GGUF /).check();
}

async function generate(page) {
  await choose(page, page.locator('#g-bundle'), /^Fixture bundle/);
  await page.getByRole('radio', { name: 'Type a prompt', exact: true }).check();
  await page.locator('#g-text').fill('Fixture prompt');
}

async function compare(page) {
  const model = name => page.getByRole('combobox', { name, exact: true });
  await choose(page, model('Model A'), /^Fixture GGUF GGUF/);
  await choose(page, model('Model B'), /^Fixture GGUF copy/);
  await page.getByRole('radio', { name: 'Type a prompt', exact: true }).check();
  await page.getByLabel('Prompt text').fill('Compare fixture answers.');
}

for (const [view, label, setup] of [
  ['benchmarks', 'Queue benchmark run', benchmarks],
  ['transfers', 'Preview', transfers],
  ['transfers', 'Transfer', transfers],
  ['evaluate', 'Queue generation', generate],
  ['compare', 'Queue comparison', compare],
]) {
  test(`${label} ignores the second click of a native double-click`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    const action = button(page, label);
    const first = response(page);
    await action.dblclick();
    expect((await first).status()).toBe(201);
    await expect(action).toBeEnabled();
    expect(posts(controller)).toHaveLength(1);
  });

  test(`${label} ignores a multi-click after a fast reply but accepts a new click`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    const action = button(page, label);
    await action.scrollIntoViewIfNeeded();
    const cdp = await page.context().newCDPSession(page);
    async function click(clickCount) {
      const { x, y, width, height } = await action.boundingBox();
      const point = { x: x + width / 2, y: y + height / 2, button: 'left', clickCount };
      await cdp.send('Input.dispatchMouseEvent', { ...point, type: 'mousePressed', buttons: 1 });
      await cdp.send('Input.dispatchMouseEvent', { ...point, type: 'mouseReleased' });
    }
    const first = response(page);
    await click(1);
    expect((await first).status()).toBe(201);
    await expect(action).toBeEnabled();
    await action.scrollIntoViewIfNeeded();
    // Force the fast-reply ordering: the browser delivers detail=2 after the
    // first request has finished. With no guard, this sends a new request.
    const post = r => r.method() === 'POST' && r.url().endsWith('/api/jobs');
    const premature = page.waitForRequest(post, { timeout: 500 }).catch(() => null);
    await click(2);
    expect(await premature).toBeNull();
    expect(posts(controller)).toHaveLength(1);
    const next = response(page);
    await click(1);
    expect((await next).status()).toBe(201);
    expect(posts(controller)).toHaveLength(2);
  });

  test(`${label} focuses the job link after an accepted reply`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    const pending = response(page);
    await button(page, label).press('Enter');
    const res = await pending;
    expect(res.status()).toBe(201);
    const { id } = await res.json();
    await expect(page.getByRole('link', { name: id, exact: true })).toBeFocused();
  });
}

for (const moved of [false, true]) {
  test(`a matched benchmark reply ${moved ? 'does not take' : 'moves'} focus`, async ({ page, controller }) => {
    await controller.open(page, 'benchmarks');
    await benchmarks(page);
    let id;
    await page.route('**/api/jobs', async route => {
      const accepted = await route.fetch();
      id = (await accepted.json()).id;
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
    const action = button(page, 'Queue benchmark run');
    await action.press('Enter');
    await expect(page.getByRole('alert')).toContainText('Reply lost.');
    let release;
    const held = new Promise(resolve => { release = resolve; });
    await page.route('**/api/jobs', async route => {
      const accepted = await route.fetch();
      await held;
      await route.fulfill({ response: accepted });
    }, { times: 1 });
    try {
      await action.press('Enter');
      const other = page.locator('#bm-target');
      if (moved) await other.focus();
      release();
      await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
      await expect(moved ? other : page.getByRole('link', { name: id, exact: true })).toBeFocused();
    } finally { release(); }
  });
}

test('GGUF Profiles matches the original inventory job when the reply is lost', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  await choose(page, page.getByRole('combobox', { name: 'Source model', exact: true }), /^Fixture checkpoint/);
  const read = button(page, 'Read tensor list');
  const snapshot = await (await page.request.get(controller.origin + '/api/state')).json();
  const freeze = route => route.fulfill({ json: snapshot });
  await page.route('**/api/state', freeze);
  let firstId;
  await page.route('**/api/jobs', async route => {
    const accepted = await route.fetch();
    expect(accepted.status()).toBe(201);
    firstId = (await accepted.json()).id;
    await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
  }, { times: 1 });
  await read.press('Enter');
  await expect(page.getByText('Reply lost.', { exact: false })).toBeVisible();
  const pending = response(page);
  await read.press('Enter');
  const matched = await pending;
  expect([matched.status(), (await matched.json()).id]).toEqual([200, firstId]);
  const [a, b] = posts(controller);
  expect(b.body).toEqual(a.body);
  expect(a.headers['idempotency-key']).toBeTruthy();
  expect(b.headers['idempotency-key']).toBe(a.headers['idempotency-key']);
  await expect(page.getByRole('link', { name: 'Tensor list read job' })).toBeFocused();
  await page.unroute('**/api/state', freeze);
  await refresh(page);
  await expect(page.getByText('model.layers.0.mlp.down_proj.weight', { exact: true })).toBeVisible();
});

for (const transfer of [true, false]) {
  const view = transfer ? 'transfers' : 'benchmarks';
  test(`${view} requires review when a selected item disappears`, async ({ page, controller }) => {
    const data = scenario('live-shaped');
    data.hosts[0].benchmarks.push({ id: 'extra', label: 'Additional suite' });
    controller.respond('GET', '/api/state', () => ({ json: data }));
    await controller.open(page, view);
    await (transfer ? transfers : benchmarks)(page);
    const label = transfer ? 'Fixture checkpoint' : 'Additional suite';
    const extra = check(page, new RegExp(`^${label}`));
    await extra.check();
    const action = button(page, transfer ? 'Transfer' : 'Queue benchmark run');
    await expect(action).toBeEnabled();
    if (transfer) data.models = data.models.filter(m => m.id !== 'source-a');
    else data.hosts[0].benchmarks = data.hosts[0].benchmarks.filter(s => s.id !== 'extra');
    await refresh(page);
    await expect(extra).toHaveCount(0);
    await expect(action).toBeDisabled();
    if (transfer) await expect(button(page, 'Preview')).toBeDisabled();
    await expect(page.getByRole('alert')).toContainText(label);
    expect(posts(controller)).toHaveLength(0);
    await button(page, 'Keep remaining selection').press('Enter');
    await expect(action).toBeEnabled();
    const pending = response(page);
    await action.press('Enter');
    expect((await pending).status()).toBe(201);
    expect(transfer ? posts(controller)[0].body.input_ids : posts(controller)[0].body.options.benchmarks)
      .toEqual(transfer ? ['gguf-a'] : ['smoke']);
  });
}

test('Managed Convert does not promise the same result after the request changes', async ({ page, controller }) => {
  await controller.open(page, 'convert');
  await choose(page, page.locator('#wf-source'), /^Fixture checkpoint/);
  await page.locator('#wf-profile').selectOption('v2');
  await page.getByRole('region', { name: 'Convert a checkpoint', exact: true }).getByText('Advanced options', { exact: true }).click();
  const seed = page.locator('#wf-seed');
  await seed.fill('7');
  const start = button(page, 'Start conversion');
  await expect(start).toBeEnabled();
  let firstId;
  await page.route('**/api/conversions', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const accepted = await route.fetch();
    expect(accepted.status()).toBe(201);
    firstId = (await accepted.json()).id;
    await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
  }, { times: 1 });
  await start.press('Enter');
  const warning = page.getByRole('alert').filter({ hasText: 'Reply lost.' });
  await expect(warning).toBeVisible();
  await seed.fill('11');
  await expect(start).toBeEnabled();
  await expect(warning).not.toContainText('it will not start a second conversion');
  const pending = response(page, '/api/conversions');
  await start.press('Enter');
  const next = await pending;
  expect(next.status()).toBe(201);
  expect((await next.json()).id).not.toBe(firstId);
  const [first, second] = controller.requests('POST', '/api/conversions');
  expect([first.body.options.seed, second.body.options.seed]).toEqual([7, 11]);
  expect(first.headers['idempotency-key']).not.toBe(second.headers['idempotency-key']);
});
