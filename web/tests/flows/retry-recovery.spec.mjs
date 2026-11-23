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
    await expect(page.getByText('Result of the previous submission', { exact: true })).toBeVisible();
    await expect(page.getByText(/submitted inputs, not the current form/)).toBeVisible();
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

test('keeping the cache returns keyboard focus without queueing cleanup', async ({ page, controller }) => {
  await open(page, controller, detail(job('failed-cleanup', 'cache_cleanup', { state: 'failed', host: 'worker-b',
    options: { bundle_sha256: 'a'.repeat(64) } })), 'Transfers');
  const clean = button(page, 'Clean cache…');
  await clean.press('Enter');
  await button(page, 'Keep cache').press('Enter');
  await expect(clean).toBeFocused();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

for (const kind of ['transfer', 'benchmark']) {
  test(`${kind} repeat checks unavailable items from the submitted request`, async ({ page, controller }) => {
    const state = scenario('live-shaped');
    const transfer = kind === 'transfer';
    const extra = { id: 'repeat-extra', label: 'Repeat extra' };
    state.models.push({ ...state.models.find(model => model.id === 'gguf-a'), id: extra.id, name: extra.label });
    state.hosts[0].benchmarks.push(extra);
    controller.respond('GET', '/api/state', () => ({ json: state }));
    const label = transfer ? 'Transfer' : 'Queue benchmark run';
    const prior = detail(job(`failed-${kind}`, kind, { state: 'failed', ...(transfer
      ? { host: 'worker-b', input_ids: ['gguf-a'] }
      : { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } }) }));
    await open(page, controller, prior, transfer ? 'Transfers' : 'Benchmarks');
    const choice = page.getByRole('checkbox', { name: /^Repeat extra/ });
    await choice.check();
    const [status, id] = await submit(page, label);
    expect(status).toBe(201);
    expect(await submit(page, label)).toEqual([200, id]);
    state.models = state.models.filter(model => model.id !== extra.id);
    state.hosts[0].benchmarks = state.hosts[0].benchmarks.filter(suite => suite.id !== extra.id);
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await expect(choice).toHaveCount(0);
    await button(page, 'Start a new run').click();
    await expect(page).not.toHaveURL(/retry=/);
    const queue = button(page, label);
    await expect(page.getByText(transfer ? 'repeat-extra: Not in Library.' : 'Suite repeat-extra is not available on this host.',
      { exact: true })).toBeVisible();
    await expect(queue).toBeDisabled();
    if (transfer) {
      await expect(button(page, 'Preview')).toBeDisabled();
      await page.getByRole('checkbox', { name: 'Continue without these entries' }).check();
    } else await button(page, 'Acknowledge omitted suites').click();
    const requests = () => controller.requests('POST', '/api/jobs');
    expect(requests()).toHaveLength(2);
    expect((await submit(page, label))[0]).toBe(201);
    expect(requests()[2].body).toEqual(transfer
      ? { ...requests()[0].body, input_ids: ['gguf-a'] }
      : { ...requests()[0].body, options: { ...requests()[0].body.options, benchmarks: ['smoke'] } });
  });
}

for (const kind of ['generate', 'benchmark', 'transfer', 'cache_cleanup', 'hf_download']) {
  test(`${kind} attributes a delayed match and preserves the edited inputs for a new run`, async ({ page, controller }) => {
    const sha = 'a'.repeat(64);
    const state = scenario('live-shaped');
    state.hosts[0].benchmarks.push({ id: 'repeat-extra', label: 'Repeat extra' });
    state.hosts.push({ ...state.hosts[1], id: 'backend-hf', capabilities: { hf_download: true } });
    controller.respond('GET', '/api/state', { json: state });
    const fixture = {
      generate: { view: 'Evaluate', label: 'Queue generation', field: '#g-max-tokens', edit: '11',
        request: { input_ids: ['bundle-a'], prompt_text: 'Keep it short.', options: { max_tokens: 7 } } },
      benchmark: { view: 'Benchmarks', label: 'Queue benchmark run', choice: /^Repeat extra/,
        request: { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } } },
      transfer: { view: 'Transfers', label: 'Transfer', choice: /^Fixture checkpoint/,
        request: { host: 'worker-b', input_ids: ['gguf-a'] } },
      cache_cleanup: { view: 'Transfers', label: 'Queue cleanup', field: '#cl-digest', edit: 'b'.repeat(64),
        request: { host: 'worker-b', options: { bundle_sha256: sha } } },
      hf_download: { view: 'Library', label: 'Start download', field: '#hf-revision', edit: 'r2',
        request: { host: 'backend-hf', options: { repo_id: 'fixture/weights', filename: 'fixture.gguf', revision: 'r1' } } },
    }[kind];
    const { view, label, request } = fixture;
    await open(page, controller, detail(job(`failed-${kind}`, kind, { state: 'failed', ...request })), view);
    const field = fixture.field ? page.locator(fixture.field) : page.getByRole('checkbox', { name: fixture.choice });
    const prepare = async () => { if (kind === 'cache_cleanup') await button(page, 'Clean cache…').click(); };
    // Keep the first HF result on the form: moving focus while the reply is
    // pending prevents a successful plain response from taking over the visit.
    const firstHeld = gate(), firstAccepted = gate();
    if (kind === 'hf_download') await page.route('**/api/jobs', async route => {
      const response = await route.fetch();
      firstAccepted.release();
      await firstHeld.wait;
      await route.fulfill({ response });
    }, { times: 1 });
    await prepare();
    const first = submit(page, label);
    if (kind === 'hf_download') {
      try {
        await firstAccepted.wait;
        await field.press('Escape');
      } finally { firstHeld.release(); }
    }
    const [status, id] = await first;
    expect(status).toBe(201);
    await expect(page.getByText(/^Queued as job/)).toContainText(id);
    const held = gate(), accepted = gate();
    await page.route('**/api/jobs', async route => {
      const response = await route.fetch();
      accepted.release();
      await held.wait;
      await route.fulfill({ response });
    }, { times: 1 });
    const requests = () => controller.requests('POST', '/api/jobs');
    try {
      await prepare();
      await button(page, label).click();
      await accepted.wait;
      if (fixture.edit) await field.fill(fixture.edit);
      else await field.check();
      held.release();
      await expect(page.getByText(/^This request was already queued as job/)).toContainText(id);
      await expect(page.getByText('Result of the previous submission', { exact: true })).toBeVisible();
      await expect(page.getByText(/submitted inputs, not the current form/)).toBeVisible();
      await expect(field).toBeFocused();
      await field.press('Enter');
      await expect(page).toHaveURL(/retry=/);
      expect(requests()).toHaveLength(2);
      await button(page, 'Start a new run with the current form').click();
      await expect(page).not.toHaveURL(/retry=/);
      if (fixture.edit) await expect(field).toHaveValue(fixture.edit);
      else await expect(field).toBeChecked();
      expect(requests()).toHaveLength(2);
      await prepare();
      expect((await submit(page, label))[0]).toBe(201);
      const original = requests()[0].body;
      const edited = {
        generate: { options: { ...original.options, max_tokens: 11 } },
        benchmark: { options: { ...original.options, benchmarks: ['smoke', 'repeat-extra'] } },
        transfer: { input_ids: ['gguf-a', 'source-a'] },
        cache_cleanup: { options: { ...original.options, bundle_sha256: fixture.edit } },
        hf_download: { options: { ...original.options, revision: fixture.edit } },
      }[kind];
      expect(requests()[2].body).toEqual({ ...original, ...edited });
      expect(key(requests()[2])).toMatch(/^conv-/);
    } finally { held.release(); }
  });
}

test('a succeeded Weight Apply can be repeated after a lost reply', async ({ page, controller }) => {
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
  await open(page, controller, detail(job('failed-apply', 'weight_apply', { state: 'failed',
    input_ids: ['gguf-a'], options: { proposal_job_id: proposalId, reviewed: true } })), 'Weights');
  const review = page.getByRole('checkbox', { name: /I reviewed all/ });
  await review.check();
  const action = 'Apply as new GGUF';
  await button(page, action).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Accepted reply lost.' })).toBeVisible();
  state.jobs.push(finished);
  controller.respond('GET', `/api/jobs/${finished.id}`, { json: finished });
  await page.reload();
  await expect(page.getByText(/^Applied by/).getByRole('link')).toHaveText(finished.id);
  await expect(button(page, 'Start a new run')).toBeVisible();
  const requests = () => controller.requests('POST', '/api/jobs');
  expect(requests()).toHaveLength(1);
  await button(page, 'Start a new run').click();
  await expect(page).not.toHaveURL(/retry=/);
  await expect(review).not.toBeChecked();
  await expect(button(page, action)).toBeDisabled();
  expect(requests()).toHaveLength(1);
  await review.check();
  const [again, repeatId] = await submit(page, action);
  expect(again).toBe(201);
  expect(key(requests()[1])).toMatch(/^conv-/);
  expect(requests()[1].body).toEqual(requests()[0].body);

  // Plain proposal review offers the same repeat, but not while it is active.
  const active = job(repeatId, 'weight_apply', { ...requests()[1].body, state: 'running' });
  state.jobs.push(active);
  await controller.open(page, 'weights?source=gguf-a');
  await page.getByRole('radio', { name: new RegExp(proposalId) }).check();
  await expect(page.getByText(/^Applied by/).getByRole('link')).toHaveText(finished.id);
  await expect(button(page, 'Start a new run')).toHaveCount(0);
  active.state = 'succeeded';
  active.updated_at++;
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await button(page, 'Start a new run').click();
  await expect(review).not.toBeChecked();
  await expect(button(page, action)).toBeDisabled();
  expect(requests()).toHaveLength(2);
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
