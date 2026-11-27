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
  await button(page, label).press('Enter');
  const response = await pending;
  return [response.status(), (await response.json()).id];
}

test('Separate Inspect presses outside the duplicate window queue new work', async ({ page, controller: c }) => {
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
    if (i === 1) await new Promise(resolve => setTimeout(resolve, 850));
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
  await button(p, 'Preview').press('Enter');
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
  await button(page, action).press('Enter');
  await expect.poll(() => posts(c).length).toBe(2);
  const [first, second] = posts(c);
  expect(second.body).toEqual(first.body);
  expect(first.headers['idempotency-key']).toBeTruthy();
  expect(second.headers['idempotency-key']).toBe(first.headers['idempotency-key']);
});

for (const [lost, activation] of [[true, 'Enter'], [false, 'Enter'], [false, 'Space']]) test(`Weight Apply does not requeue after ${activation} receives ${lost ? 'a matched retry' : 'a 201'}`, async ({ page, controller }) => {
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
  if (lost) await page.route('**/api/jobs', async route => {
    const response = await route.fetch();
    expect(response.status()).toBe(201);
    finished = detail(job((await response.json()).id, 'weight_apply', {
      ...route.request().postDataJSON(), state: 'succeeded',
    }));
    await route.fulfill({ status: 502, json: { error: 'Accepted reply lost.' } });
  }, { times: 1 });
  await controller.open(page, 'weights?source=gguf-a');
  if (activation === 'Space') {
    // Exercise fractional line bounds without depending on the host's fonts.
    await page.evaluate(() => { document.body.style.zoom = '1.15'; });
  }
  await page.getByRole('radio', { name: new RegExp(proposalId) }).check();
  const review = page.getByRole('checkbox', { name: /I reviewed all/ });
  await review.check();
  const action = 'Apply as new GGUF';
  if (lost) {
    await button(page, action).click();
    await expect(page.getByRole('alert').filter({ hasText: 'Accepted reply lost.' })).toBeVisible();
  }
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.route('**/api/jobs', async route => {
    const accepted = await route.fetch();
    await held;
    await route.fulfill({ response: accepted });
  }, { times: 1 });
  try {
    const match = page.waitForResponse(r => r.url().endsWith('/api/jobs') && r.request().method() === 'POST');
    await button(page, action).press(activation);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    release();
    const accepted = await match;
    if (lost) expect([accepted.status(), (await accepted.json()).id]).toEqual([200, finished.id]);
    else {
      expect(accepted.status()).toBe(201);
      finished = detail({ ...(await accepted.json()), state: 'succeeded' });
    }
    const status = page.getByText(/^(?:This request was already queued|Queued) as job /);
    const result = status.getByRole('link', { name: finished.id, exact: true });
    await expect(result).toBeFocused();
    await expect(button(page, action)).toBeDisabled({ timeout: 1500 });
    await expect(result).toBeInViewport({ ratio: 1, timeout: 1500 });
  } finally { release(); }
  const [first, second] = posts(controller);
  expect(key(first)).toMatch(/^conv-/);
  if (lost) {
    await expect(page.getByText('No second job was queued.', { exact: false })).toBeVisible();
    expect(first.body).toEqual(second.body);
    expect(key(first)).toBe(key(second));
  }
  if (activation === 'Enter' && !lost) {
    state.jobs.push({ ...finished, state: 'queued' });
    state.jobs.push(...Array.from({ length: 20 }, (_, i) => job(`newer-${i}`, 'weight_propose', {
      input_ids: ['gguf-a'], created_at: '2026-12-01T00:00:00Z',
    })));
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await expect(page.getByRole('radio', { name: /newer-19/ })).toBeVisible();
    const focused = page.locator(':focus');
    await expect(focused).toHaveText(new RegExp(`^(?:${finished.id}|Review and apply)$`), { timeout: 1500 });
    const index = state.jobs.indexOf(proposal);
    state.jobs.splice(index, 1);
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await expect(page.getByRole('heading', { name: 'Review and apply' })).toBeFocused({ timeout: 1500 });
    state.jobs.splice(index, 0, proposal);
    state.jobs.splice(-20);
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await page.getByRole('radio', { name: new RegExp(proposalId) }).check();
  }
  state.jobs = [...state.jobs.filter(job => job.id !== finished.id), finished];
  controller.respond('GET', `/api/jobs/${finished.id}`, { json: finished });
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText(/^Applied by/).getByRole('link')).toHaveText(finished.id);
  await expect(button(page, action)).toHaveCount(0);
  expect(posts(controller)).toHaveLength(lost ? 2 : 1);
});

test('visual reread keeps the loaded inventory after suppression, POST failure, or terminal failure', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'weights?source=gguf-a');
  const tensor = page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ });
  expect((await submit(page, 'Read tensor list'))[0]).toBe(201);
  await expect(tensor).toBeVisible();
  expect((await submit(page, 'Read next tensors'))[0]).toBe(201);
  const reread = button(page, 'Read tensor list again');
  await reread.press('Enter');
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  expect(posts(controller)).toHaveLength(2);
  expect(await tensor.isVisible()).toBe(true);
  await page.waitForTimeout(850); // A deliberate new keyboard press.
  await page.route('**/api/jobs', route => route.fulfill({ status: 403, body: 'Forbidden' }), { times: 1 });
  await reread.press('Enter');
  await expect(page.getByRole('alert').filter({ hasText: 'Forbidden' })).toBeVisible();
  expect(await tensor.isVisible()).toBe(true);
  const state = await (await page.request.get(`${controller.origin}/api/state`)).json();
  const message = 'The refreshed file could not be opened.';
  const editor = page.getByRole('region', { name: '1. Tensor' });
  controller.respond('GET', '/api/state', () => ({ json: state }));
  controller.respond('POST', '/api/jobs', ({ body }) => {
    const failed = job('failed-reread', 'weight_inspect', { ...body, state: 'failed', error: message });
    state.jobs.push(failed);
    return { status: 201, json: failed };
  });
  expect(await submit(page, 'Read tensor list again')).toEqual([201, 'failed-reread']);
  await expect(editor.getByRole('alert').filter({ hasText: message })).toBeVisible();
  await expect(reread).toBeEnabled();
  expect(await tensor.isVisible()).toBe(true);
  await choose(page, '#w-source', /^Fixture checkpoint/);
  await choose(page, '#w-source', /^Fixture GGUF GGUF/);
  await expect(tensor).toBeVisible();
  await expect(editor.getByText(message)).toHaveCount(0);
});

test('a failed layer inventory leaves a retry available and is not retained across source switches', async ({ page, controller }) => {
  controller.load('weight-reads');
  const state = scenario('weight-reads');
  state.models.push({ ...state.models.find(m => m.id === 'source-a'), id: 'source-b', name: 'Second checkpoint' });
  controller.respond('GET', '/api/state', () => ({ json: state }));
  await controller.open(page, 'convert?format=recipe&output=gguf');
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  await page.route('**/api/jobs', async route => {
    const response = await route.fetch();
    await route.fulfill({ status: 201, json: { ...await response.json(), state: 'queued' } });
  }, { times: 1 });
  const [status, id] = await submit(page, 'Read layers');
  expect(status).toBe(201);
  const message = 'Fixture read failed after the worker could not open the path.';
  state.jobs.push(job(id, 'weight_inspect', { ...posts(controller)[0].body, state: 'failed', error: message }));
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByRole('alert').filter({ hasText: message })).toBeVisible();
  await expect(button(page, 'Read layers')).toBeEnabled({ timeout: 1500 });
  await choose(page, '#recipe-source', /^Second checkpoint/);
  await choose(page, '#recipe-source', /^Fixture checkpoint/);
  await expect(page.getByText(message)).toHaveCount(0);
  await page.waitForTimeout(850);
  const [matchStatus, matchId] = await submit(page, 'Read layers');
  expect([matchStatus, matchId]).toEqual([200, id]);
  const [retryStatus, retryId] = await submit(page, 'Queue another');
  expect(retryStatus).toBe(201);
  expect(retryId).not.toBe(id);
  expect(key(posts(controller)[0])).not.toBe(key(posts(controller)[1]));
  expect(posts(controller)[2].headers['brewery-allow-duplicate']).toBe('1');
});

test('Profiles GGUF retries a transient inventory detail error on the next state poll', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'profiles?format=gguf');
  const picker = page.getByRole('combobox', { name: 'Source model', exact: true });
  await picker.click();
  await page.getByRole('option', { name: /^Fixture GGUF GGUF/ }).click();
  const message = 'Fixture temporary receipt outage.';
  const path = '/api/jobs/job-mock-1';
  await page.route(`**${path}`, async route => {
    await route.fetch();
    await route.fulfill({ status: 503, json: { error: message } });
  }, { times: 1 });
  expect(await submit(page, 'Read tensor list')).toEqual([201, 'job-mock-1']);
  await expect(page.getByText(message, { exact: true })).toBeVisible();
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(button(page, 'Read next tensors')).toBeEnabled({ timeout: 1500 });
  await expect(page.getByText(message)).toHaveCount(0);
  expect(controller.requests('GET', path)).toHaveLength(2);
  expect(posts(controller)).toHaveLength(1);
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
    await p.locator('#hf-filename').fill('second.gguf');
    await button(p, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(2);
    const [a, b] = posts(c);
    expect(b.body.options).toEqual({ ...a.body.options, filename: 'second.gguf' });
    expect(key(b)).not.toBe(key(a));
    await expect(p).toHaveURL(/#\/jobs\/job-mock-2$/);
  } finally { hold.release(); }
});

for (const moved of [false, true]) test(`Job detail ${moved ? 'keeps the operator’s focus' : 'keeps heading focus'} after a delayed download detail`, async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[0], id: 'backend-hf', capabilities: { hf_download: true } });
  controller.respond('GET', '/api/state', { json: state });
  const hold = gate();
  await page.route('**/api/jobs/job-mock-1', async route => {
    const response = await route.fetch();
    await hold.pending;
    await route.fulfill({ response });
  });
  await controller.open(page);
  await page.getByText('Download from Hugging Face', { exact: true }).click();
  await page.locator('#hf-repo').fill('fixture/model');
  await page.locator('#hf-filename').fill('model.gguf');
  try {
    await button(page, 'Start download').press(moved ? 'Space' : 'Enter');
    await expect(page.getByRole('heading', { name: 'Job job-mock-1', exact: true })).toBeFocused();
    const library = page.getByRole('navigation').getByRole('link', { name: 'Library' });
    if (moved) await library.focus();
    hold.release();
    const ready = page.getByRole('heading', { name: 'Download from Hugging Face', exact: true });
    await expect(ready).toBeVisible();
    await expect(moved ? library : ready).toBeFocused({ timeout: 1500 });
    expect(posts(controller)).toHaveLength(1);
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
