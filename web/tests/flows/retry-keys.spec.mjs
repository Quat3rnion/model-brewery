import { test, expect } from './fixtures.mjs';
import { conversion, detail, job, scenario } from './mock/scenarios.mjs';

const button = (p, name) => p.getByRole('button', { name, exact: true });
const nav = (p, name) => p.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name, exact: true });
const posts = (c, path = '/api/jobs') => c.requests('POST', path);
const key = r => r.headers['idempotency-key'];
const benchmark = { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } };
const generate = { input_ids: ['bundle-a'], prompt_text: 'Retry prompt.' };
function gate() {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  return { pending, release };
}
async function open(p, c, kind, view, fields, onRead = () => {}) {
  const prior = detail(job(`failed-${kind}`, kind, { state: 'failed', ...fields }));
  let reads = 0;
  c.respond('GET', `/api/jobs/${prior.id}`, async () => {
    await onRead(++reads);
    return { json: prior };
  });
  await c.open(p, `jobs/${prior.id}`);
  await p.getByRole('link', { name: `Open in ${view}` }).click();
  return prior;
}
const queued = p => p.getByText(/^Queued as job /);
const matched = p => p.getByText(/^This request was already queued as job /);

test('mock controller matches canonically identical submissions and rejects conflicting keys', async ({ request, controller: c }) => {
  const post = data => request.post(`${c.origin}/api/jobs`, { data, headers: { 'Idempotency-Key': 'mock-check' } });
  const a = await post({ kind: 'inspect', input_ids: ['source-a'], options: { seed: 1, threads: 2 } });
  const b = await post({ options: { threads: 2, seed: 1 }, input_ids: ['source-a'], kind: 'inspect' });
  const d = await post({ kind: 'inspect', input_ids: ['source-a'], options: { seed: 2, threads: 2 } });
  expect(a.status()).toBe(201);
  expect(b.status()).toBe(200);
  expect(await b.json()).toEqual({ id: (await a.json()).id });
  expect(d.status()).toBe(409);
});

test('confirmed benchmark retry keeps its key and result after reload', async ({ page: p, controller: c }) => {
  await open(p, c, 'benchmark', 'Benchmarks', benchmark);
  const queue = button(p, 'Queue benchmark run');
  await queue.click();
  const id = await queued(p).getByRole('link').textContent();
  await p.reload();
  expect(posts(c)).toHaveLength(1);
  await queue.click();
  const [a, b] = posts(c);
  expect(b.body).toEqual(a.body);
  expect(key(b)).toBe(key(a));
  await expect(matched(p).getByRole('link')).toHaveText(id);
});

for (const [kind, name] of [['transfer', 'Preview'], ['transfer_preview', 'Transfer']]) {
  test(`${name} on a ${kind} retry does not replay the other action`, async ({ page: p, controller: c }) => {
    await open(p, c, kind, 'Transfers', { host: 'worker-b', input_ids: ['gguf-a'] });
    for (let n = 1; n <= 2; n++) {
      await button(p, name).click();
      await expect.poll(() => posts(c).length).toBe(n);
      await expect(button(p, name)).toBeEnabled();
    }
    const [a, b] = posts(c);
    expect(b.body).toEqual(a.body);
    expect(key(b)).not.toBe(key(a));
  });
}

test('plain Preview neither inherits retry identity nor loses its own unconfirmed key', async ({ page: p, controller: c }) => {
  const retry = { host: 'worker-b', input_ids: ['gguf-a'] };
  await open(p, c, 'transfer', 'Transfers', retry);
  await p.getByRole('checkbox', { name: /^Fixture checkpoint/ }).check();
  await button(p, 'Preview').click();
  await expect(queued(p)).toBeVisible();
  await nav(p, 'Transfers').click();
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
  await expect.poll(() => posts(c).length).toBe(3);
  const [a, b, d] = posts(c);
  expect(b.body).toEqual(a.body);
  expect(key(b)).not.toBe(key(a));
  expect(key(b)).toBe(key(d));
  expect(key(b)).not.toMatch(/^r-/);
});

for (const [kind, view, action, fields, dirty] of [
  ['generate', 'Evaluate', 'Queue generation', generate, async p => {
    await p.getByRole('radio', { name: 'Type a prompt' }).check();
    await p.locator('#g-text').fill('Unsaved prompt.');
  }],
  ['benchmark', 'Benchmarks', 'Queue benchmark run', benchmark, async p => {
    await p.locator('#bm-host').click();
    await p.getByRole('option', { name: /^Worker A/ }).click();
  }],
]) test(`${view} reload and explicit restoration reuse an accepted key after a lost reply`, async ({ page: p, controller: c }) => {
  const wait = gate();
  let hold = false;
  await open(p, c, kind, view, fields, reads => hold && reads > 2 ? wait.pending : undefined);
  let lose = true;
  await p.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST' || !lose) return route.continue();
    lose = false;
    await route.fetch();
    await route.abort();
  });
  const queue = button(p, action);
  try {
    await queue.click();
    await expect(p.getByRole('alert').filter({ hasText: 'Check Jobs' })).toBeVisible();
    hold = true;
    await p.reload({ waitUntil: 'domcontentloaded' });
    await dirty(p);
    wait.release();
    await button(p, 'Use previous inputs').click();
    await queue.click();
    await expect.poll(() => posts(c).length).toBe(2);
    const [a, b] = posts(c);
    expect(b.body).toEqual(a.body);
    expect(key(b)).toBe(key(a));
    await expect(matched(p)).toBeVisible();
  } finally { wait.release(); }
});

for (const [kind, view, action, fields] of [
  ['generate', 'Evaluate', 'Queue generation', generate],
  ['benchmark', 'Benchmarks', 'Queue benchmark run', benchmark],
]) test(`${view} resends the same hand-built request after applying the offered inputs`, async ({ page: p, controller: c }) => {
  const pending = gate();
  await open(p, c, kind, view, fields, reads => reads > 1 ? pending.pending : undefined);
  let lose = true;
  await p.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST' || !lose) return route.continue();
    lose = false;
    await route.fetch();
    await route.abort();
  });
  const choose = async (field, name) => {
    await field.click();
    await p.getByRole('option', { name }).click();
  };
  try {
    await expect(button(p, action)).toBeDisabled();
    if (kind === 'generate') {
      await choose(p.locator('#g-bundle'), /^Fixture bundle/);
      await choose(p.getByRole('region', { name: 'Generate' }).getByRole('combobox', { name: 'Run on' }), /^Worker A/);
      await p.getByRole('radio', { name: 'Type a prompt' }).check();
      await p.locator('#g-text').fill(fields.prompt_text);
    } else {
      await choose(p.locator('#bm-host'), /^Worker A/);
      await choose(p.locator('#bm-target'), /^Fixture endpoint/);
      await p.getByRole('checkbox', { name: /^Endpoint smoke/ }).check();
    }
    await expect(button(p, action)).toBeDisabled();
    await expect(p.getByRole('status').filter({ hasText: 'Loading the previous job options.' })).toBeVisible();
    expect(posts(c)).toHaveLength(0);
    pending.release();
    await expect(button(p, 'Use previous inputs')).toBeVisible();
    await button(p, action).click();
    await expect(p.getByRole('alert').filter({ hasText: 'Check Jobs' })).toBeVisible();
    await button(p, 'Use previous inputs').click();
    await button(p, action).click();
    await expect.poll(() => posts(c).length).toBe(2);
    const [a, b] = posts(c);
    expect(b.body).toEqual(a.body);
    expect(key(b)).toBe(key(a));
    expect(key(a)).toMatch(new RegExp(`^r-failed-${kind}-`));
    await expect(matched(p).getByRole('link')).toHaveText('job-mock-1');
  } finally { pending.release(); }
});

test('Inspect rebuilt on a fresh plain Convert entry is new work', async ({ page: p, controller: c }) => {
  await open(p, c, 'inspect', 'Convert', { input_ids: ['source-a'], options: { profile: 'v2' } });
  const queue = button(p, 'Queue inspect');
  await queue.click();
  await expect(queued(p)).toBeVisible();
  await nav(p, 'Convert').click();
  await expect(p).toHaveURL(/#\/convert$/);
  await expect(queue).toBeHidden();
  await p.locator('summary').filter({ hasText: 'Individual steps (diagnostics)' }).click();
  await p.locator('#cv-source').click();
  await p.getByRole('option', { name: /^Fixture checkpoint/ }).click();
  await p.getByRole('tabpanel', { name: 'Inspect' }).getByRole('combobox', { name: 'Run on' }).click();
  await p.getByRole('option', { name: /^Worker A/ }).click();
  await p.locator('#cv-profile').selectOption('v2');
  await queue.click();
  await expect.poll(() => posts(c).length).toBe(2);
  const [a, b] = posts(c);
  expect(a.body).toEqual(b.body);
  expect(key(a)).not.toBe(key(b));
  await expect(queued(p).getByRole('link')).toHaveText('job-mock-2');
});

test('managed retry accepted after Escape is visible and reload matches it', async ({ page: p, controller: c }) => {
  const failed = conversion('failed-hgn', { state: 'failed', stage: 'verify', options: { profile: 'ht43', seed: 23 } });
  c.respond('GET', `/api/conversions/${failed.id}`, { json: failed });
  const hold = gate();
  await p.route('**/api/conversions', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    const response = await route.fetch();
    await hold.pending;
    await route.fulfill({ response });
  });
  try {
    await c.open(p, `convert?retry_run=${failed.id}`);
    const start = button(p, 'Start conversion');
    await start.click();
    await expect.poll(() => posts(c, '/api/conversions').length).toBe(1);
    const profile = p.locator('#wf-profile');
    await profile.press('Escape');
    hold.release();
    const id = await p.getByText(/^Started conversion /).getByRole('link').textContent();
    await expect(profile).toBeFocused();
    await expect(p).toHaveURL(/retry_run=/);
    await p.reload();
    await start.click();
    await expect.poll(() => posts(c, '/api/conversions').length).toBe(2);
    const [a, b] = posts(c, '/api/conversions');
    expect(a.body).toEqual(b.body);
    expect(key(b)).toBe(key(a));
    await expect(matched(p).getByRole('link')).toHaveText(id);
  } finally { hold.release(); }
});

for (const isRetry of [false, true]) test(`Library shows ${isRetry ? 'a retry' : 'a plain'} download accepted after focus moves`, async ({ page: p, controller: c }) => {
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
    if (isRetry) await open(p, c, 'hf_download', 'Library', { host: 'backend-hf', options: { repo_id: 'old/model', filename: 'model.gguf' } });
    else {
      await c.open(p);
      await p.getByText('Download from Hugging Face', { exact: true }).click();
      await p.locator('#hf-repo').fill('old/model');
      await p.locator('#hf-filename').fill('model.gguf');
    }
    const url = p.url();
    await button(p, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(1);
    await p.locator('#hf-repo').press('Tab');
    await expect(p.locator('#hf-revision')).toBeFocused();
    hold.release();
    await expect(queued(p).getByRole('link')).toHaveText('job-mock-1');
    await expect(p.locator('#hf-revision')).toBeFocused();
    await expect(p).toHaveURL(url);
    expect(posts(c)).toHaveLength(1);
    if (isRetry) await p.reload();
    await button(p, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(2);
    const [a, b] = posts(c);
    expect(b.body).toEqual(a.body);
    if (isRetry) {
      expect(key(b)).toBe(key(a));
      await expect(matched(p).getByRole('link')).toHaveText('job-mock-1');
    } else {
      expect(key(b)).not.toBe(key(a));
      await expect(p).toHaveURL(/#\/jobs\/job-mock-2$/);
    }
  } finally { hold.release(); }
});

test('Plan clears a deleted optional component after the inventory refreshes on Back', async ({ page: p, controller: c }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'vision-a', name: 'Fixture vision', kind: 'vision', host: 'worker-a', path: '/fixtures/vision' });
  const hold = gate();
  let deleted = false;
  c.respond('GET', '/api/state', async () => {
    if (deleted) await hold.pending;
    return { json: state };
  });
  c.respond('DELETE', '/api/models/vision-a', () => {
    deleted = true;
    state.models = state.models.filter(m => m.id !== 'vision-a');
    return { status: 204 };
  });
  try {
    await open(p, c, 'plan', 'Convert', { input_ids: ['source-a', 'vision-a'], options: { profile: 'v2', vision: 1 } });
    await expect(button(p, 'Queue plan')).toBeEnabled();
    await nav(p, 'Library').click();
    await button(p, 'Remove Fixture vision from library').click();
    await button(p, 'Remove entry').click();
    await expect.poll(() => deleted).toBe(true);
    await p.goBack();
    const vision = p.locator('#cv-comp-vision');
    await expect(vision).toHaveValue('vision-a');
    hold.release();
    await button(p, 'Acknowledge omitted inputs').click();
    await expect(vision).toHaveValue('');
    await button(p, 'Queue plan').click();
    await expect.poll(() => posts(c).map(r => r.body.input_ids)).toEqual([['source-a']]);
  } finally { hold.release(); }
});

test('a baseline retry can restore and submit a GSM8K evaluation artifact', async ({ page: p, controller: c }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'gsm-results', name: 'Fixture evaluation', kind: 'gsm8k-evaluation', host: 'worker-a', path: '/fixtures/eval.json' });
  c.respond('GET', '/api/state', { json: state });
  await open(p, c, 'baseline', 'Evaluate', { input_ids: ['gsm-results'] });
  await expect(p.locator('#b-file')).toHaveValue(/Fixture evaluation/);
  await button(p, 'Queue reference results').click();
  await expect.poll(() => posts(c).map(r => r.body.input_ids)).toEqual([['gsm-results']]);
});

for (const kind of ['benchmark', 'transfer']) test(`${kind} acknowledgment expires when the omission set changes and returns`, async ({ page: p, controller: c }) => {
  const isBench = kind === 'benchmark';
  const state = scenario('live-shaped');
  state.hosts[0].benchmarks.push({ id: 'extra', label: 'Extra suite' });
  c.respond('GET', '/api/state', { json: state });
  await open(p, c, kind, isBench ? 'Benchmarks' : 'Transfers', isBench
    ? { options: { ...benchmark.options, benchmarks: ['smoke', 'extra'] } }
    : { host: 'worker-b', input_ids: ['gguf-a', 'source-a'] });
  const queue = button(p, isBench ? 'Queue benchmark run' : 'Transfer');
  await expect(queue).toBeEnabled();
  const item = p.getByRole('checkbox', { name: isBench ? /^Extra suite/ : /^Fixture checkpoint/ });
  await item.uncheck();
  await expect(queue).toBeDisabled();
  if (isBench) await button(p, 'Acknowledge omitted suites').click();
  else await p.getByRole('checkbox', { name: 'Continue without these entries' }).check();
  await expect(queue).toBeEnabled();
  await item.check();
  await expect(queue).toBeEnabled();
  await item.uncheck();
  await expect(queue).toBeDisabled();
});
