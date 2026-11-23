import { test, expect } from './fixtures.mjs';
import { conversion, detail, job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (page, name) => page.getByRole('button', { name, exact: true });
const nav = (page, name) => page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name, exact: true });
const posts = c => c.requests('POST', '/api/jobs');
const key = request => request.headers['idempotency-key'];
const choose = async (page, field, name) => {
  await page.locator(field).click();
  await page.getByRole('option', { name }).click();
};

function gate() {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  return { pending, release };
}

async function submit(page, label, path = '/api/jobs') {
  const pending = page.waitForResponse(r => r.url().endsWith(path) && r.request().method() === 'POST');
  await button(page, label).click();
  const response = await pending;
  return [response.status(), (await response.json()).id];
}

test('Each confirmed Inspect press queues new work', async ({ page, controller: c }) => {
  const accepted = new Map();
  c.respond('POST', '/api/jobs', ({ body, headers }) => {
    const key = headers['idempotency-key'];
    if (!accepted.has(key)) accepted.set(key, job(`inspect-${accepted.size + 1}`, body.kind, body));
    return { json: accepted.get(key) };
  });
  await c.open(page, 'convert?step=inspect&model=source-a&profile=v2');
  const queue = button(page, 'Queue inspect');
  for (let i = 1; i <= 2; i++) {
    await queue.click();
    await expect.poll(() => posts(c).length).toBe(i);
    await expect(queue).toBeEnabled();
  }
  expect(posts(c)[0].body).toEqual(posts(c)[1].body);
  expect([...accepted.values()].map(j => j.id)).toEqual(['inspect-1', 'inspect-2']);
});

test('Transfer preview resends with its unconfirmed key', async ({ page: p, controller: c }) => {
  await c.open(p, 'transfers');
  await p.locator('#tx-host').click();
  await p.getByRole('option', { name: /^Worker B/ }).click();
  await p.getByRole('checkbox', { name: /^Fixture GGUF / }).check();
  let lost = true;
  await p.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST' || !lost) return route.continue();
    lost = false;
    await route.fetch();
    await route.abort();
  });
  await button(p, 'Preview').click();
  await expect(p.getByRole('alert')).toContainText(/without reloading/);
  await button(p, 'Preview').click();
  await expect.poll(() => posts(c).length).toBe(2);
  const [a, b] = posts(c);
  expect(b.body).toEqual(a.body);
  expect(key(a)).toMatch(/^conv-/);
  expect(key(b)).toBe(key(a));
  await expect(p.getByText('No second job was queued.', { exact: false })).toBeVisible();
});

async function openCleanup(page, controller) {
  await controller.open(page, 'transfers');
  await choose(page, '#cl-host', /^Worker A/);
  await page.locator('#cl-digest').fill('a'.repeat(64));
  await button(page, 'Clean cache…').press('Enter');
}

for (const moveFocus of [false, true]) {
  test(`resending cleanup after a lost reply ${moveFocus ? 'preserves' : 'restores'} focus`, async ({ page, controller }) => {
    await openCleanup(page, controller);
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      await route.fetch();
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
    const queue = button(page, 'Queue cleanup');
    await queue.press('Enter');
    await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toBeVisible();
    const held = gate();
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      const response = await route.fetch();
      await held.pending;
      await route.fulfill({ response });
    });
    try {
      await queue.press('Enter');
      const target = moveFocus ? page.locator('#cl-digest') : button(page, 'Clean cache…');
      if (moveFocus) await target.focus();
      held.release();
      await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
      await expect(target).toBeFocused();
      const [first, second] = posts(controller);
      expect(first.body).toEqual(second.body);
      expect(key(first)).toMatch(/^conv-/);
      expect(key(first)).toBe(key(second));
    } finally { held.release(); }
  });
}

test('keeping the cache returns keyboard focus without queueing cleanup', async ({ page, controller }) => {
  await openCleanup(page, controller);
  const clean = button(page, 'Clean cache…');
  await expect(button(page, 'Keep cache')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(clean).toBeFocused();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

for (const [kind, action] of [
  ['weight_inspect', 'Inspect'],
  ['weight_propose', 'Create proposal'],
  ['weight_tune', 'Tune selected weights'],
]) test(`${kind} editor reuses the key after an ambiguous reply`, async ({ page, controller: c }) => {
  const tensor = 'blk.0.ffn_down.weight';
  const state = scenario('weight-reads');
  state.hosts[0].capabilities.weight_visual = false;
  for (const [id, name] of [['fit', 'Fit split'], ['held', 'Held-out split']])
    state.models.push({ id, name, host: 'worker-a', kind: 'weight-calibration', path: `/fixtures/${id}.json` });
  c.respond('GET', '/api/state', { json: state });
  c.respond('POST', '/api/jobs', () => posts(c).length === 1
    ? { status: 502, json: { error: 'Reply lost.' } } : { json: job('accepted', kind, posts(c).at(-1).body) });
  await c.open(page, 'weights?source=gguf-a');
  if (kind === 'weight_inspect') {
    const lookup = page.getByRole('group', { name: /Look up a stored value/ });
    await lookup.getByLabel('Tensor', { exact: true }).fill(tensor);
    await lookup.getByLabel('Coordinate', { exact: true }).fill('0, 1');
  } else if (kind === 'weight_propose') {
    await page.getByLabel('Tensor, edit 1').fill(tensor);
    await page.getByLabel('Coordinate, edit 1').fill('0, 1');
    await page.getByLabel('Value, edit 1').fill('0.25');
    await button(page, 'Add to proposal').click();
  } else {
    await page.getByRole('radio', { name: 'Automatic' }).check();
    await page.getByRole('combobox', { name: 'Calibration data', exact: true }).selectOption({ label: 'Fit split' });
    await page.getByRole('combobox', { name: 'Held-out data', exact: true }).selectOption({ label: 'Held-out split' });
    await page.locator('#w-tune-t').fill(tensor);
    await page.locator('#w-tune-m').fill('0.02');
    await page.locator('#w-tune-c').fill('0, 1');
  }
  await button(page, action).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toBeVisible();
  await button(page, action).click();
  await expect.poll(() => posts(c).length).toBe(2);
  const [first, second] = posts(c);
  expect(second.body).toEqual(first.body);
  expect(first.headers['idempotency-key']).toBeTruthy();
  expect(second.headers['idempotency-key']).toBe(first.headers['idempotency-key']);
});

test('Weight Apply resends after a lost reply and does not apply a completed proposal twice', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  const proposalId = '6ecf76ee-0473-4544-ac77-106a37c05ddd';
  const proposal = detail(job(proposalId, 'weight_propose', { state: 'succeeded', input_ids: ['gguf-a'] }), {
    report: { proposal: { schema: 'brewery.weight-edit-proposal.v1', source_sha256: 'a'.repeat(64),
      edits: [{ tensor: 'blk.0.ffn_down.weight', index: [0, 1], op: 'set', value: .125, dtype: 'F32', before: .5, after: .125 }] } },
  });
  state.jobs.push(proposal);
  controller.respond('GET', `/api/jobs/${proposalId}`, { json: proposal });
  controller.respond('GET', '/api/state', () => ({ json: state }));
  let finished;
  await page.route('**/api/jobs', async route => {
    const response = await route.fetch();
    expect(response.status()).toBe(201);
    finished = detail(job((await response.json()).id, 'weight_apply', {
      ...route.request().postDataJSON(), state: 'succeeded',
    }));
    await route.fulfill({ status: 502, json: { error: 'Accepted reply lost.' } });
  }, { times: 1 });
  await controller.open(page, 'weights?source=gguf-a');
  await page.getByRole('radio', { name: new RegExp(proposalId) }).check();
  const review = page.getByRole('checkbox', { name: /I reviewed all/ });
  await review.check();
  const action = 'Apply as new GGUF';
  await button(page, action).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Accepted reply lost.' })).toBeVisible();
  expect(await submit(page, action)).toEqual([200, finished.id]);
  await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
  const [first, second] = posts(controller);
  expect(first.body).toEqual(second.body);
  expect(key(first)).toMatch(/^conv-/);
  expect(key(first)).toBe(key(second));
  state.jobs.push(finished);
  controller.respond('GET', `/api/jobs/${finished.id}`, { json: finished });
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText(/^Applied by/).getByRole('link')).toHaveText(finished.id);
  await expect(button(page, action)).toHaveCount(0);
  expect(posts(controller)).toHaveLength(2);
});

for (const mode of ['inventory', 'tile']) {
  test(`visual ${mode} resends with the same key and reserves its ID before a poll`, async ({ page, controller }) => {
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
    await controller.open(page, 'weights?source=gguf-a');
    if (mode === 'tile') await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
    const label = mode === 'tile' ? 'Read tile' : 'Read tensor list';
    const failed = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.status() === 502);
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      const response = await route.fetch();
      expect(response.status()).toBe(201);
      controller.respond('GET', `/api/jobs/${(await response.json()).id}`, { status: 503, json: { error: 'Not ready.' } });
      await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
    }, { times: 1 });
    await button(page, label).click();
    await failed;
    await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toBeVisible();
    const [status, id] = await submit(page, mode === 'tile' ? 'Read tile again' : label);
    expect(status).toBe(200);
    await expect(page.getByRole('link', { name: mode === 'tile' ? 'Tile read job' : 'Tensor list read job' })).toHaveAttribute('href', `#/jobs/${id}`);
    await expect(button(page, label)).toHaveCount(0);
    const [first, second] = controller.requests('POST', '/api/jobs');
    expect(key(first)).toMatch(/^conv-/);
    expect(key(second)).toBe(key(first));
    expect(second.body).toEqual(first.body);
  });
}

test('Library shows a download accepted after focus moves', async ({ page: p, controller: c }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[0], id: 'backend-hf', capabilities: { hf_download: true } });
  c.respond('GET', '/api/state', { json: state });
  const hold = gate();
  await p.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const response = await route.fetch();
    await hold.pending;
    await route.fulfill({ response });
  });
  try {
    await c.open(p);
    await p.getByText('Download from Hugging Face', { exact: true }).click();
    await p.locator('#hf-repo').fill('old/model');
    await p.locator('#hf-filename').fill('model.gguf');
    const url = p.url();
    await button(p, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(1);
    await p.locator('#hf-repo').press('Tab');
    await expect(p.locator('#hf-revision')).toBeFocused();
    hold.release();
    await expect(p.getByText(/^Queued as job /).getByRole('link')).toHaveText('job-mock-1');
    await expect(p.locator('#hf-revision')).toBeFocused();
    await expect(p).toHaveURL(url);
    expect(posts(c)).toHaveLength(1);
    await button(p, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(2);
    const [a, b] = posts(c);
    expect(b.body).toEqual(a.body);
    expect(key(b)).not.toBe(key(a));
    await expect(p).toHaveURL(/#\/jobs\/job-mock-2$/);
  } finally { hold.release(); }
});

test('late download success does not take over a later Library visit', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[0], id: 'backend-hf', capabilities: { hf_download: true } });
  c.respond('GET', '/api/state', { json: state });
  const wait = gate();
  const dialogs = [];

  c.respond('POST', '/api/jobs', async () => {
    await wait.pending;
    return { json: job('accepted-download', 'hf_download') };
  });
  try {
    await c.open(page);
    await page.getByText('Download from Hugging Face', { exact: true }).click();
    await page.locator('#hf-repo').fill('old/model');
    await page.locator('#hf-filename').fill('old.gguf');
    await button(page, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(1);
    page.once('dialog', dialog => dialog.accept());
    await nav(page, 'Transfers').click();
    await expect(page).toHaveURL(/#\/transfers$/);
    await nav(page, 'Library').click();
    await page.getByText('Download from Hugging Face', { exact: true }).click();
    await expect(page.getByLabel('Repository ID')).toHaveValue('');
    const refreshed = page.waitForResponse(r => r.url().endsWith('/api/state'));
    wait.release();
    await refreshed;
    await expect(button(page, 'Start download')).toBeVisible();
    await expect(page).toHaveURL(/#\/library$/);
    page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
    await nav(page, 'Transfers').click();
    await expect(page).toHaveURL(/#\/transfers$/);
    expect(dialogs).toEqual([]);
  } finally { wait.release(); }
});

test('a delayed conversion result does not take focus from another control', async ({ page, controller: c }) => {
  const response = gate();
  const run = conversion('slow-linked-run', { state: 'failed' });
  c.respond('GET', '/api/conversions', async () => { await response.pending; return { json: [run] }; });
  try {
    await c.open(page, `convert?run=${run.id}`);
    await expect.poll(() => c.requests('GET', '/api/conversions').length).toBe(1);
    const profile = page.locator('#wf-profile');
    await profile.press('Escape');
    await expect(profile).toBeFocused();
    response.release();
    await expect(page.locator(`#run-${run.id}`)).toBeVisible();
    await expect(profile).toBeFocused();
  } finally { response.release(); }
});

for (const paused of [true, false]) {
  test(`Hy4 encode cannot queue on a ${paused ? 'paused' : 'convert-disabled'} worker`, async ({ page, controller }) => {
    const state = scenario('live-shaped');
    const host = state.hosts[0];
    host.capabilities.hy4_encode = true;
    host.capabilities.convert = paused;
    host.scheduling_paused = paused;
    state.eligibility.hosts[host.id].convert = { eligible: false, reasons: [{ text: paused ? 'Resume job scheduling.' : 'Enable conversion.', fix_anchor: '#/settings' }] };
    controller.respond('GET', '/api/state', { json: state });
    await controller.open(page, 'convert?step=convert');
    await page.getByText('Hy4 metadata plan', { exact: true }).click();
    await page.getByLabel('Hy4 operation').selectOption({ label: 'Gate conversion' });
    await choose(page, '#hy4-source', /^Fixture checkpoint/);
    await expect(page.getByRole('combobox', { name: 'Hy4 source' })).toHaveValue('Fixture checkpoint');
    await expect(page.getByRole('button', { name: 'Queue Hy4 conversion' })).toBeDisabled();
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  });
}

test('a selected component blocks Plan when it disappears', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'new-vision', name: 'New vision', kind: 'vision', host: 'worker-a', path: '/fixtures/vision' });
  controller.respond('GET', '/api/state', () => ({ json: state }));
  await controller.open(page, 'convert?step=plan&model=source-a&profile=v2');
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

test('changing Recipe output keeps the edited recipe and source', async ({ page, controller: c }) => {
  const state = scenario('weight-reads');
  const model = state.models.find(m => m.id === 'source-a');
  const inventory = weightRead('recipe-inventory', { kind: 'weight_inspect', host: model.host, input_ids: [model.id],
    options: { mode: 'inventory', offset: 0, limit: 128 } }, model);
  state.jobs.push(inventory);
  c.respond('GET', '/api/state', { json: state });
  c.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
  await c.open(page, 'convert?format=recipe&output=gguf');
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  const layer = page.getByLabel('Layer 0 type');
  await layer.selectOption('F16');
  await page.getByLabel('Recipe output').selectOption('hgn');
  await expect(page.locator('#recipe-source')).toHaveValue('Fixture checkpoint');
  await expect(layer).toHaveValue('F16');
});

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

test('a layer recipe resends with its unconfirmed key', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  Object.assign(state.hosts[0].capabilities, { gguf_layer_recipe: true, gguf_hf_output: true });
  const source = state.models.find(model => model.id === 'source-a');
  const inventory = weightRead('recipe-inventory', { kind: 'weight_inspect', host: source.host, input_ids: [source.id],
    options: { mode: 'inventory', offset: 0, limit: 128 } }, source);
  state.jobs.push(inventory);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
  await controller.open(page, 'convert?format=recipe&output=gguf');
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  await page.route('**/api/conversions', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    await route.fetch();
    await route.fulfill({ status: 502, json: { error: 'Reply lost.' } });
  }, { times: 1 });
  await button(page, 'Run layer recipe').click();
  await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toBeVisible();
  expect((await submit(page, 'Run layer recipe', '/api/conversions'))[0]).toBe(200);
  await expect(page.getByText('No second conversion was started.', { exact: false })).toBeVisible();
  const [first, second] = controller.requests('POST', '/api/conversions');
  expect(first.body).toEqual(second.body);
  expect(key(first)).toMatch(/^conv-/);
  expect(key(first)).toBe(key(second));
});

test('a reference request can submit a GSM8K evaluation artifact', async ({ page: p, controller: c }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'gsm-results', name: 'Fixture evaluation', kind: 'gsm8k-evaluation', host: 'worker-a', path: '/fixtures/eval.json' });
  c.respond('GET', '/api/state', { json: state });
  await c.open(p, 'evaluate');
  await choose(p, '#b-file', /^Fixture evaluation/);
  await expect(p.locator('#b-file')).toHaveValue(/Fixture evaluation/);
  await button(p, 'Queue reference results').click();
  await expect.poll(() => posts(c).map(r => r.body.input_ids)).toEqual([['gsm-results']]);
});
