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
const twoFrames = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const freezeState = async (page, controller) => {
  const snapshot = await (await page.request.get(controller.origin + '/api/state')).json();
  await page.route('**/api/state', route => route.fulfill({ json: snapshot }));
  return snapshot;
};
const maskAcceptedRead = async (page) => {
  await page.route('**/api/jobs', async route => {
    expect((await route.fetch()).status()).toBe(201);
    await route.fulfill({ status: 502, json: { error: 'Read reply lost.' } });
  }, { times: 1 });
};

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

async function gguf(page) {
  await choose(page, page.locator('#gg-source'), /^Fixture GGUF GGUF/);
  await page.locator('#gg-choice').selectOption('preset:Q8_0');
}

async function clickPoint(cdp, { x, y }, clickCount) {
  const point = { x, y, button: 'left', clickCount };
  await cdp.send('Input.dispatchMouseEvent', { ...point, type: 'mousePressed', buttons: 1 });
  await cdp.send('Input.dispatchMouseEvent', { ...point, type: 'mouseReleased' });
}

async function center(action) {
  const { x, y, width, height } = await action.boundingBox();
  return { x: x + width / 2, y: y + height / 2 };
}

for (const [view, label, setup, path = '/api/jobs'] of [
  ['benchmarks', 'Queue benchmark run', benchmarks],
  ['transfers', 'Preview', transfers],
  ['transfers', 'Transfer', transfers],
  ['evaluate', 'Queue generation', generate],
  ['compare', 'Queue comparison', compare],
  ['convert?format=gguf', 'Start GGUF conversion', gguf, '/api/conversions'],
]) {
  test(`${label} ignores the second click of a native double-click`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    const action = button(page, label);
    const first = response(page, path);
    await action.dblclick();
    expect((await first).status()).toBe(201);
    await expect(action).toBeEnabled();
    expect(controller.requests('POST', path)).toHaveLength(1);
  });

  test(`${label} ignores a multi-click after a fast reply and sends a fresh click for matching`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    const action = button(page, label);
    await action.scrollIntoViewIfNeeded();
    const cdp = await page.context().newCDPSession(page);
    const click = async detail => clickPoint(cdp, await center(action), detail);
    const first = response(page, path);
    await click(1);
    expect((await first).status()).toBe(201);
    await expect(action).toBeEnabled();
    await action.scrollIntoViewIfNeeded();
    // Force the fast-reply ordering: the browser delivers detail=2 after the
    // first request has finished. With no guard, this sends a new request.
    const post = r => r.method() === 'POST' && r.url().endsWith(path);
    const premature = page.waitForRequest(post, { timeout: 850 }).catch(() => null);
    await click(2);
    expect(await premature).toBeNull();
    expect(controller.requests('POST', path)).toHaveLength(1);
    const next = response(page, path);
    await click(1);
    const match = await next;
    expect(match.status()).toBe(200);
    const { id: matchedId, duplicate_of } = await match.json();
    expect(duplicate_of).toBe(true);
    expect(match.headers()['brewery-duplicate']).toBe(matchedId);
    expect(controller.requests('POST', path)).toHaveLength(2);
    const explicit = response(page, path);
    await button(page, 'Queue another').press('Enter');
    const repeated = await explicit;
    expect(repeated.status()).toBe(201);
    expect((await repeated.json()).id).not.toBe(matchedId);
    const [a, b, c] = controller.requests('POST', path);
    expect(c.body).toEqual(a.body);
    expect(c.headers['brewery-allow-duplicate']).toBe('1');
    expect(new Set([a, b, c].map(r => r.headers['idempotency-key'])).size).toBe(3);
  });

  test(`${label} focuses the job link after an accepted reply`, async ({ page, controller }) => {
    await controller.open(page, view);
    await setup(page);
    let release;
    if (setup === gguf) {
      const held = new Promise(resolve => { release = resolve; });
      await page.route(`**${path}`, async route => {
        if (route.request().method() !== 'POST') return route.continue();
        const accepted = await route.fetch();
        await held;
        await route.fulfill({ response: accepted });
      }, { times: 1 });
    }
    const pending = response(page, path);
    try {
      await button(page, label).press('Enter');
      if (release) {
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        release();
      }
      const res = await pending;
      expect(res.status()).toBe(201);
      const { id } = await res.json();
      await expect(page.getByRole('link', { name: id, exact: true })).toBeFocused();
    } finally { release?.(); }
  });
}

test('the first enabled activation submits even with a native click count of two', async ({ page, controller }) => {
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  const suite = check(page, /^Endpoint smoke/);
  const action = button(page, 'Queue benchmark run');
  await suite.uncheck();
  await expect(action).toBeDisabled();
  await action.scrollIntoViewIfNeeded();
  const cdp = await page.context().newCDPSession(page);
  await clickPoint(cdp, await center(action), 1);
  // Change eligibility without starting a new pointer sequence.
  await suite.evaluate(el => el.click());
  await expect(action).toBeEnabled();
  const pending = response(page);
  await clickPoint(cdp, await center(action), 2);
  expect((await pending).status()).toBe(201);
});

test('a repeat mouse press that starts disabled stays suppressed if enabled before release', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.hosts[0].benchmarks.push({ id: 'second', label: 'Second suite' });
  controller.respond('GET', '/api/state', { json: state });
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  const action = button(page, 'Queue benchmark run');
  await action.scrollIntoViewIfNeeded();
  const rect = await action.boundingBox();
  const at = { x: rect.x + 30, y: rect.y + rect.height / 2 };
  const cdp = await page.context().newCDPSession(page);
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/api/jobs', async route => {
    const accepted = await route.fetch();
    await held;
    await route.fulfill({ response: accepted });
  }, { times: 1 });
  const first = response(page);
  try {
    await clickPoint(cdp, at, 1);
    await expect(button(page, 'Submitting…')).toBeDisabled();
    expect(await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.closest('button')?.matches(':disabled'), at)).toBe(true);
    await cdp.send('Input.dispatchMouseEvent', {
      ...at, type: 'mousePressed', button: 'left', buttons: 1, clickCount: 2,
    });
    // Use a different body so only the native click rule can suppress this.
    await check(page, /^Second suite/).evaluate(el => el.click());
    await expect(check(page, /^Second suite/)).toBeChecked();
    release();
    expect((await first).status()).toBe(201);
    await expect(action).toBeEnabled();
    await action.scrollIntoViewIfNeeded();
    const ready = await action.boundingBox();
    const up = { x: ready.x + 30, y: ready.y + ready.height / 2 };
    await page.evaluate(() => {
      window.repeatDetail = null;
      window.addEventListener('click', event => {
        window.repeatDetail = [event.detail, event.target.closest('button')?.textContent.trim()];
      }, { once: true, capture: true });
    });
    const duplicate = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
      { timeout: 500 }).catch(() => null);
    await cdp.send('Input.dispatchMouseEvent', { ...up, type: 'mouseReleased', button: 'left', clickCount: 2 });
    expect(await page.evaluate(() => window.repeatDetail)).toEqual([2, 'Queue benchmark run']);
    expect(await duplicate).toBeNull();
    expect(posts(controller)).toHaveLength(1);
  } finally { release(); }
});

test('a disabled first click resets an older native submit sequence', async ({ page, controller }) => {
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  const suite = check(page, /^Endpoint smoke/);
  const action = button(page, 'Queue benchmark run');
  await action.scrollIntoViewIfNeeded();
  const point = await center(action);
  const cdp = await page.context().newCDPSession(page);
  const first = response(page);
  await clickPoint(cdp, point, 1);
  expect((await first).status()).toBe(201);
  await expect(action).toBeEnabled();
  await new Promise(resolve => setTimeout(resolve, 850));
  await suite.evaluate(el => el.click());
  await expect(action).toBeDisabled();
  await clickPoint(cdp, point, 1);
  await suite.evaluate(el => el.click());
  await expect(action).toBeEnabled();
  expect(await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.closest('button')?.textContent.trim(), point))
    .toBe('Queue benchmark run');
  const second = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.request().method() === 'POST',
    { timeout: 2000 }).catch(() => null);
  await clickPoint(cdp, point, 2);
  expect((await second)?.status()).toBe(200);
  expect(posts(controller)).toHaveLength(2);
});

test('a fast GGUF inventory reply does not put its job link under the second click', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  await choose(page, page.getByRole('combobox', { name: 'Source model', exact: true }), /^Fixture checkpoint/);
  const read = button(page, 'Read tensor list');
  await read.scrollIntoViewIfNeeded();
  const cdp = await page.context().newCDPSession(page);
  const rect = await read.boundingBox();
  // This point overlaps the link when the launcher row collapses.
  const at = { x: rect.x + 35, y: rect.y + rect.height / 2 };
  const pending = response(page);
  await clickPoint(cdp, at, 1);
  expect((await pending).status()).toBe(201);
  await expect(page.getByRole('link', { name: 'Tensor list read job' })).toBeVisible();
  await clickPoint(cdp, at, 2);
  await expect(page).toHaveURL(/#\/profiles\?format=gguf$/);
  expect(posts(controller)).toHaveLength(1);
});

test('GGUF inventory lets another source submit and keeps the first source key', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  const source = page.getByRole('combobox', { name: 'Source model', exact: true });
  const snapshot = await (await page.request.get(controller.origin + '/api/state')).json();
  await page.route('**/api/state', route => route.fulfill({ json: snapshot }));
  let release;
  const held = new Promise(resolve => { release = resolve; });
  const accepted = new Promise(resolve => {
    void page.route('**/api/jobs', async route => {
      const result = await route.fetch();
      resolve(result.status());
      await held;
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
  });
  const read = button(page, 'Read tensor list');
  try {
    await choose(page, source, /^Fixture checkpoint/);
    await read.press('Enter');
    expect(await accepted).toBe(201);
    await choose(page, source, /^Fixture GGUF GGUF/);
    await expect(read).toBeEnabled();
    const next = response(page);
    await read.press('Enter');
    expect((await next).status()).toBe(201);
    release();
    await choose(page, source, /^Fixture checkpoint/);
    await expect(page.getByText('Reply lost.', { exact: false })).toBeVisible();
    const retry = response(page);
    await read.press('Enter');
    expect((await retry).status()).toBe(200);
    const [a, b, again] = posts(controller);
    expect([a.body.input_ids, b.body.input_ids]).toEqual([['source-a'], ['gguf-a']]);
    expect(again.headers['idempotency-key']).toBe(a.headers['idempotency-key']);
    expect(b.headers['idempotency-key']).not.toBe(a.headers['idempotency-key']);
  } finally { release(); }
});

test('GGUF inventory can queue the next page while the first reply is held', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  await choose(page, page.getByRole('combobox', { name: 'Source model', exact: true }), /^Fixture GGUF GGUF/);
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/api/jobs', async route => {
    const accepted = await route.fetch();
    await held;
    await route.fulfill({ response: accepted });
  }, { times: 1 });
  try {
    await button(page, 'Read tensor list').press('Enter');
    await expect.poll(() => posts(controller).length).toBe(1);
    await refresh(page);
    const next = button(page, 'Read next tensors');
    await expect(next).toBeEnabled();
    const pending = response(page);
    await next.press('Enter');
    expect((await pending).status()).toBe(201);
    expect(posts(controller).map(r => r.body.options.offset)).toEqual([0, 128]);
  } finally { release(); }
});

test('a repeat click on a newly ready GGUF inventory page does not queue the next read', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  await choose(page, page.getByRole('combobox', { name: 'Source model', exact: true }), /^Fixture GGUF GGUF/);
  const read = button(page, 'Read tensor list');
  await read.scrollIntoViewIfNeeded();
  const point = await center(read);
  const cdp = await page.context().newCDPSession(page);
  await clickPoint(cdp, point, 1);
  await expect(button(page, 'Read next tensors')).toBeEnabled();
  expect(await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.closest('button')?.textContent.trim(), point))
    .toBe('Read next tensors');
  const repeat = response(page).catch(() => null);
  // Synchronize on input handling, not a fixed double-click time window.
  await clickPoint(cdp, point, 2);
  await twoFrames(page);
  expect(posts(controller).map(r => r.body.options.offset)).toEqual([0]);
  await clickPoint(cdp, point, 1);
  expect((await repeat).status()).toBe(201);
  expect(posts(controller).map(r => r.body.options.offset)).toEqual([0, 128]);
});

test('a Library digest update does not unlock a pending identical Profile wire request', async ({ page, controller }) => {
  controller.load('weight-reads');
  const state = scenario('weight-reads');
  state.hosts[0].capabilities.weight_hgn = true;
  const hgn = { id: 'managed-hgn', name: 'Managed HGN', kind: 'hgn', host: 'worker-a', path: '/fixtures/managed.hgn', available: true };
  state.models.push(hgn);
  controller.respond('GET', '/api/state', () => ({ json: state }));
  await controller.open(page, 'profiles?format=gguf');
  const chooseSource = page.getByRole('combobox', { name: 'Source model', exact: true });
  await choose(page, chooseSource, /^Managed HGN/);
  const read = button(page, 'Read tensor list');
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/api/jobs', async route => {
    const accepted = await route.fetch();
    await held;
    await route.fulfill({ status: 502, json: { error: 'Read reply lost.' } });
    expect(accepted.status()).toBe(201);
  }, { times: 1 });
  try {
    await read.press('Enter');
    await expect.poll(() => posts(controller).length).toBe(1);
    hgn.sha256 = 'a'.repeat(64);
    // A changed display label confirms the refreshed model is present.
    hgn.name = 'Verified HGN';
    await refresh(page);
    await expect(chooseSource).toHaveValue('Verified HGN');
    await expect(read).toBeDisabled();
    release();
    await expect(page.getByText(/Read reply lost/)).toBeVisible();
    const retry = response(page);
    await read.press('Enter');
    expect((await retry).status()).toBe(200);
    const [a, b] = posts(controller);
    expect(b.body).toEqual(a.body);
    expect(b.headers['idempotency-key']).toBe(a.headers['idempotency-key']);
  } finally { release(); }
});

for (const kind of ['recipe inventory', 'visual inventory', 'visual tile']) {
  const recipe = kind === 'recipe inventory';
  const tile = kind === 'visual tile';
  const label = recipe ? 'Read layers' : tile ? 'Read tile' : 'Read tensor list';
  const link = recipe ? 'Layer read job' : tile ? 'Tile read job' : 'Tensor list read job';
  const view = recipe ? 'convert?format=recipe&output=gguf' : 'weights?source=gguf-a';
  const setup = async (page, controller) => {
    controller.load('weight-reads');
    await controller.open(page, view);
    const picker = page.locator(recipe ? '#recipe-source' : '#w-source');
    if (recipe) await choose(page, picker, /^Fixture checkpoint/);
    if (tile) {
      await button(page, 'Read tensor list').press('Enter');
      await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
    }
  };
  test(`${kind} does not follow its result link on the second click`, async ({ page, controller }) => {
    await setup(page, controller);
    const action = button(page, label);
    await action.scrollIntoViewIfNeeded();
    const rect = await action.boundingBox();
    const point = { x: rect.x + 36, y: rect.y + rect.height / 2 };
    const cdp = await page.context().newCDPSession(page);
    const route = page.url();
    const before = posts(controller).length;
    await clickPoint(cdp, point, 1);
    const result = page.getByRole('link', { name: link, exact: true });
    await expect(result).toBeVisible();
    await clickPoint(cdp, point, 2);
    await twoFrames(page);
    expect(page.url()).toBe(route);
    expect(posts(controller)).toHaveLength(before + 1);
    await result.click();
    await expect(page).toHaveURL(/#\/jobs\//);
  });

  test(`${kind} keeps the retry key after a different read`, async ({ page, controller }) => {
    await setup(page, controller);
    if (recipe) {
      await page.request.post(`${controller.origin}/api/models`, { data: {
        name: 'Second checkpoint', kind: 'hf', host: 'worker-a', path: '/fixtures/second-checkpoint',
      } });
      await refresh(page);
    }
    const picker = page.locator(recipe ? '#recipe-source' : '#w-source');
    await freezeState(page, controller);
    const count = posts(controller).length;
    await maskAcceptedRead(page);
    const first = response(page);
    await button(page, label).press('Enter');
    expect((await first).status()).toBe(502);
    await expect(page.getByText(/Read reply lost/)).toBeVisible();
    const select = async again => {
      if (tile) await page.locator('#vw-tc').selectOption(again ? '0' : '1');
      else await choose(page, picker, again ? (recipe ? /^Fixture checkpoint/ : /^Fixture GGUF GGUF/) :
        (recipe ? /^Second checkpoint/ : /^Fixture checkpoint/));
    };
    await select(false);
    const other = response(page);
    await button(page, label).press('Enter');
    expect((await other).status()).toBe(201);
    if (tile) await expect(page.getByRole('gridcell').first()).toBeVisible();
    await select(true);
    const retry = response(page);
    await button(page, label).press('Enter');
    expect((await retry).status()).toBe(200);
    const [a, b, again] = posts(controller).slice(count);
    expect(b.body).not.toEqual(a.body);
    expect(b.headers['idempotency-key']).not.toBe(a.headers['idempotency-key']);
    expect(again.body).toEqual(a.body);
    expect(again.headers['idempotency-key']).toBe(a.headers['idempotency-key']);
  });

  if (!tile) test(`${kind} shows an accepted read received while another source is selected`, async ({ page, controller }) => {
    await setup(page, controller);
    if (recipe) {
      await page.request.post(`${controller.origin}/api/models`, { data: {
        name: 'Second checkpoint', kind: 'hf', host: 'worker-a', path: '/fixtures/second-checkpoint',
      } });
      await refresh(page);
    }
    const picker = page.locator(recipe ? '#recipe-source' : '#w-source');
    await freezeState(page, controller);
    let release, report;
    const held = new Promise(resolve => { release = resolve; });
    const stored = new Promise(resolve => { report = resolve; });
    await page.route('**/api/jobs', async route => {
      const accepted = await route.fetch();
      report(await accepted.json());
      await held;
      await route.fulfill({ response: accepted });
    }, { times: 1 });
    const pending = response(page);
    try {
      await button(page, label).press('Enter');
      await stored;
      await choose(page, picker, recipe ? /^Second checkpoint/ : /^Fixture checkpoint/);
      release();
      expect((await pending).status()).toBe(201);
      await twoFrames(page);
      await choose(page, picker, recipe ? /^Fixture checkpoint/ : /^Fixture GGUF GGUF/);
      const { id } = await stored;
      await expect(page.getByRole('link', { name: link, exact: true })).toHaveAttribute('href', `#/jobs/${id}`);
      await expect(button(page, label)).toHaveCount(0);
      expect(posts(controller)).toHaveLength(1);
    } finally { release(); }
  });
}

for (const isTransfer of [true, false]) {
  const view = isTransfer ? 'transfers' : 'benchmarks';
  test(`${view} restores focus when changing omissions without accepting a held Space`, async ({ page, controller }) => {
    const state = scenario('live-shaped');
    const models = state.models;
    const suites = [...state.hosts[0].benchmarks, { id: 'extra-a', label: 'Fixture suite A' }, { id: 'extra-b', label: 'Fixture suite B' }];
    controller.respond('GET', '/api/state', () => ({ json: state }));
    const update = async ids => {
      state.models = models.filter(m => !ids.includes(m.id));
      state.hosts[0].benchmarks = suites.filter(s => !ids.includes(s.id));
      await refresh(page);
    };
    await update([]);
    await controller.open(page, view);
    await (isTransfer ? transfers : benchmarks)(page);
    const labels = isTransfer ? ['Fixture checkpoint', 'Fixture GGUF'] : ['Fixture suite A', 'Fixture suite B'];
    const ids = isTransfer ? ['source-a', 'gguf-a'] : ['extra-a', 'extra-b'];
    for (const label of labels) await check(page, new RegExp(`^${label}`)).check();
    await update(ids.slice(0, 1));
    const review = button(page, 'Keep remaining selection');
    const alert = page.getByRole('alert');
    await expect(alert).toContainText(labels[0]);
    await review.focus();
    await page.keyboard.down('Space');
    await update(ids);
    await expect(alert).toContainText(labels[1]);
    await page.keyboard.up('Space');
    await expect(alert).toContainText(labels[1]);
    await expect.soft(review).toBeFocused({ timeout: 1000 });
    await review.focus();
    await page.keyboard.down('Space');
    await update([]);
    const restored = check(page, new RegExp(`^${labels[0]}`));
    await expect(restored).toBeChecked();
    await expect(review).toHaveCount(0);
    expect.soft(await page.locator('input[type=checkbox]:focus').count()).toBe(0);
    await page.keyboard.up('Space');
    await expect(restored).toBeChecked();
    await expect(check(page, new RegExp(`^${labels[1]}`))).toBeChecked();
    await expect(page.locator('legend:focus')).toHaveText(isTransfer ? 'Entries to copy' : /Suites/);
    await expect(button(page, isTransfer ? 'Transfer' : 'Queue benchmark run')).toBeEnabled();
    expect(posts(controller)).toHaveLength(0);
  });
}

test('GGUF Convert names a missing optional matrix and waits for review', async ({ page, controller }) => {
  const data = scenario('live-shaped');
  data.models.push({ id: 'matrix-a', name: 'Fixture imatrix', kind: 'imatrix', host: 'worker-a', path: '/fixtures/a.imatrix' });
  controller.respond('GET', '/api/state', () => ({ json: data }));
  await controller.open(page, 'convert?format=gguf');
  await gguf(page);
  await page.locator('#gg-choice').selectOption('preset:IQ2_XXS');
  await choose(page, page.locator('#gg-imatrix'), /^Fixture imatrix/);
  await page.locator('#gg-choice').selectOption('preset:Q8_0');
  const start = button(page, 'Start GGUF conversion');
  await expect(start).toBeEnabled();
  data.models.pop();
  await refresh(page);
  await expect(page.getByRole('alert')).toContainText('Fixture imatrix');
  await expect(start).toBeDisabled();
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  await button(page, 'Keep remaining selection').press('Enter');
  await expect(page.locator('#gg-choice')).toBeFocused();
  await expect(start).toBeEnabled();
  const pending = response(page, '/api/conversions');
  await start.press('Enter');
  expect((await pending).status()).toBe(201);
  expect(controller.requests('POST', '/api/conversions')[0].body.input_ids).toEqual(['gguf-a']);
});

test('Transfers blocks backend files and more than 32 inputs', async ({ page, controller }) => {
  const data = scenario('live-shaped');
  data.models.push(
    { id: 'backend', name: 'Backend fixture', kind: 'gguf', host: 'backend-hf', path: '/backend/a.gguf', available: true },
    ...Array.from({ length: 33 }, (_, i) => ({ id: `copy-${i}`, name: `Copy fixture ${i}`, kind: 'gguf', host: 'worker-a', path: `/fixtures/${i}.gguf` })),
  );
  controller.respond('GET', '/api/state', () => ({ json: data }));
  await controller.open(page, 'transfers');
  await choose(page, page.locator('#tx-host'), /^Worker B/);
  const backend = check(page, /Backend fixture/);
  await backend.check();
  const preview = button(page, 'Preview');
  const transfer = button(page, 'Transfer');
  await expect.soft(preview).toBeDisabled();
  await expect.soft(transfer).toBeDisabled();
  await expect.soft(page.getByText('Backend files are only available for local weight jobs.')).toBeVisible();
  await backend.uncheck();
  for (const box of await check(page, /Copy fixture/).all()) await box.check();
  await expect.soft(preview).toBeDisabled();
  await expect.soft(transfer).toBeDisabled();
  await expect.soft(page.getByText('Select at most 32 entries to copy.')).toBeVisible();
  expect(posts(controller)).toHaveLength(0);
});

test('Benchmarks counts omitted suites until review and shows each new omission', async ({ page, controller }) => {
  const data = scenario('live-shaped');
  data.hosts[0].benchmarks.push(...Array.from({ length: 8 }, (_, i) => ({ id: `suite-${i}`, label: `Fixture suite ${i}` })));
  controller.respond('GET', '/api/state', () => ({ json: data }));
  await controller.open(page, 'benchmarks');
  await benchmarks(page);
  for (let i = 0; i < 7; i++) await check(page, new RegExp(`Fixture suite ${i}`)).check();
  const spare = check(page, /Fixture suite 7/);
  await expect(spare).toBeDisabled();
  data.hosts[0].benchmarks = data.hosts[0].benchmarks.filter(s => s.id !== 'suite-0');
  await refresh(page);
  const alert = page.getByRole('alert');
  const keep = button(page, 'Keep remaining selection');
  await expect(alert).toContainText('Fixture suite 0');
  await expect.soft(spare).toBeDisabled();
  // Begin activation on the first review, then change the set before release.
  await keep.scrollIntoViewIfNeeded();
  const at = await center(keep);
  await page.mouse.move(at.x, at.y);
  await page.mouse.down();
  data.hosts[0].benchmarks = data.hosts[0].benchmarks.filter(s => s.id !== 'suite-1');
  await refresh(page);
  await expect(alert).toContainText('Fixture suite 1');
  await page.mouse.up();
  await expect(alert).toContainText('Fixture suite 1');
  await expect(button(page, 'Queue benchmark run')).toBeDisabled();
  await keep.press('Enter');
  await expect(spare).toBeEnabled();
  data.hosts[0].benchmarks = data.hosts[0].benchmarks.filter(s => s.id !== 'suite-2');
  await refresh(page);
  await expect(alert).toContainText('Fixture suite 2');
  await expect(button(page, 'Queue benchmark run')).toBeDisabled();
});

for (const cleanup of [false, true]) {
  test(`${cleanup ? 'Cleanup' : 'Benchmark'} returns focus after a delayed lost reply`, async ({ page, controller }) => {
    await controller.open(page, cleanup ? 'transfers' : 'benchmarks');
    if (cleanup) {
      await choose(page, page.locator('#cl-host'), /^Worker A/);
      await button(page, 'Clean cache…').click();
    } else await benchmarks(page);
    let release;
    const held = new Promise(resolve => { release = resolve; });
    await page.route('**/api/jobs', async route => {
      await route.fetch();
      await held;
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
    const launch = button(page, cleanup ? 'Queue cleanup' : 'Queue benchmark run');
    try {
      await launch.press('Enter');
      await expect(button(page, 'Submitting…')).toBeDisabled();
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      release();
      await expect(page.getByRole('alert')).toContainText('Reply lost.');
      await expect(launch).toBeFocused();
    } finally { release(); }
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
    await expect(check(page, transfer ? /^Fixture GGUF / : /^Endpoint smoke/)).toBeFocused();
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
