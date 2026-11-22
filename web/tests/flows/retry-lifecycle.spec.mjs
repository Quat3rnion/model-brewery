import { test, expect } from './fixtures.mjs';
import { conversion, detail, job, scenario } from './mock/scenarios.mjs';

function delayedGet(controller, path, json, immediate = 0) {
  let release, reads = 0;
  const wait = new Promise(resolve => { release = resolve; });
  controller.respond('GET', path, async () => {
    if (++reads > immediate) await wait;
    return { json };
  });
  return release;
}

async function openRetry(page, controller, prior, view) {
  controller.respond('GET', `/api/jobs/${prior.id}`, { json: prior });
  await controller.open(page, `jobs/${prior.id}`);
  await page.getByRole('link', { name: `Open in ${view}` }).click();
}

const nav = (page, name) => page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name });
const noQueue = (c) => {
  expect(c.requests('POST', '/api/jobs')).toHaveLength(0);
  expect(c.requests('POST', '/api/conversions')).toHaveLength(0);
};

test('plain Transfers clears the partial selection and its warning', async ({ page, controller: c }) => {
  await openRetry(page, c, detail(job('partial', 'transfer', {
    state: 'interrupted', host: 'worker-b', input_ids: ['gguf-a', 'deleted-entry'],
  })), 'Transfers');
  const pick = page.getByRole('checkbox', { name: /^Fixture GGUF/ });
  const transfer = page.getByRole('button', { name: 'Transfer', exact: true });
  await expect(pick).toBeChecked();
  await expect(page.getByText('deleted-entry: Not in Library.')).toBeVisible();
  await nav(page, 'Transfers').click();
  await expect(pick).toHaveCount(0);
  await expect(transfer).toBeDisabled();
  await expect(page.getByRole('combobox', { name: 'Destination host' })).toHaveValue('');
  await page.goBack();
  await expect(pick).toBeChecked();
  await expect(page.getByText('deleted-entry: Not in Library.')).toBeVisible();
  await page.goForward();
  await expect(pick).toHaveCount(0);
  noQueue(c);
});

test('leaving remote download retry keeps edited repository but clears the old host and filename', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[0], id: 'backend-hf', name: 'Backend downloads',
    capabilities: { hf_download: true } });
  c.respond('GET', '/api/state', { json: state });
  await openRetry(page, c, detail(job('remote', 'hf_download', {
    state: 'failed', host: 'worker-a', options: { repo_id: 'old/model', filename: 'old.gguf' },
  })), 'Library');
  const repo = page.getByLabel('Repository ID');
  const file = page.getByLabel('Model filename');
  await expect(repo).toHaveValue('old/model');
  await expect(page.getByText('Download on Worker A.')).toBeVisible();
  await repo.fill('new/model');
  page.once('dialog', d => d.accept());
  await nav(page, 'Library').click();
  await expect(repo).toHaveValue('new/model');
  await expect(file).toHaveValue('');
  await expect(page.getByText('Download on Worker A.')).toHaveCount(0);
  await file.fill('new.gguf');
  await file.press('Enter');
  noQueue(c);
  await page.getByRole('button', { name: 'Start download' }).click();
  await expect.poll(() => c.requests('POST', '/api/jobs').map(r => r.body)).toEqual([
    { kind: 'hf_download', input_ids: [], options: { repo_id: 'new/model', filename: 'new.gguf' } },
  ]);
});

test('plain Generate never inherits the retry token limit', async ({ page, controller: c }) => {
  await openRetry(page, c, detail(job('capped', 'generate', {
    state: 'failed', input_ids: ['bundle-a'], prompt_text: 'Old prompt.', options: { max_tokens: 1 },
  })), 'Evaluate');
  const form = page.getByRole('region', { name: 'Generate', exact: true });
  const cap = form.getByText('Maximum output tokens from previous request: 1.');
  await expect(cap).toBeVisible();
  await nav(page, 'Evaluate').click();
  await expect(cap).toHaveCount(0);
  await form.getByRole('combobox', { name: 'Bundle' }).click();
  await form.getByRole('option', { name: /Fixture bundle/ }).click();
  await form.getByRole('radio', { name: 'Type a prompt' }).check();
  await form.getByLabel('Prompt text').fill('New prompt.');
  noQueue(c);
  await form.getByRole('button', { name: 'Queue generation' }).click();
  await expect.poll(() => c.requests('POST', '/api/jobs').map(r => r.body)).toEqual([
    expect.objectContaining({ kind: 'generate', prompt: 'New prompt.' }),
  ]);
  expect(c.requests('POST', '/api/jobs')[0].body.options).toBeUndefined();
});

test('a delayed download retry does not overwrite draft fields or steal focus', async ({ page, controller: c }) => {
  const old = detail(job('slow-download', 'hf_download', {
    state: 'failed', host: 'worker-a', options: { repo_id: 'old/model', revision: 'old-tag', filename: 'old.gguf' },
  }));
  const release = delayedGet(c, `/api/jobs/${old.id}`, old, 1);
  try {
    await c.open(page, `jobs/${old.id}`);
    await page.getByRole('link', { name: 'Open in Library' }).click();
    const repo = page.getByLabel('Repository ID');
    const rev = page.getByLabel('Revision (optional)');
    const file = page.getByLabel('Model filename');
    await repo.fill('new/model');
    await rev.fill('new-tag');
    await file.fill('new.gguf');
    await file.press('Enter');
    release();
    const use = page.getByRole('button', { name: 'Use previous inputs' });
    await expect(use).toBeVisible();
    await expect(repo).toHaveValue('new/model');
    await expect(rev).toHaveValue('new-tag');
    await expect(file).toHaveValue('new.gguf');
    await expect(file).toBeFocused();
    await use.press('Enter');
    await expect(repo).toHaveValue('old/model');
    await expect(repo).toBeFocused();
    noQueue(c);
  } finally { release(); }
});

test('a delayed managed retry does not overwrite profile and seed or steal focus', async ({ page, controller: c }) => {
  const prior = conversion('slow-run', { state: 'failed', options: { profile: 'ht43', seed: 23 } });
  c.respond('GET', '/api/conversions', { json: [prior] });
  const release = delayedGet(c, `/api/conversions/${prior.id}`, prior);
  try {
    await c.open(page, 'convert');
    await page.getByRole('link', { name: 'Run again', exact: true }).click();
    await expect(page.getByText('Loading the previous conversion.')).toBeVisible();
    const profile = page.locator('#wf-profile');
    const seed = page.locator('#wf-seed');
    await profile.selectOption('v2');
    await page.getByText('Advanced options', { exact: false }).first().click();
    await seed.fill('91');
    await seed.press('Enter');
    release();
    const use = page.getByRole('button', { name: 'Use previous inputs' });
    await expect(use).toBeVisible();
    await expect(profile).toHaveValue('v2');
    await expect(seed).toHaveValue('91');
    await expect(seed).toBeFocused();
    await use.press('Enter');
    await expect(seed).toHaveValue('23');
    await expect(page.locator('#wf-source')).toBeFocused();
    noQueue(c);
  } finally { release(); }
});

test('managed retry reuses a lost-reply key after Back and Forward', async ({ page, controller: c }) => {
  const prior = conversion('old-run', { state: 'failed', requested_host: 'auto', options: { profile: 'ht43' } });
  c.respond('GET', '/api/conversions', { json: [prior] });
  c.respond('GET', `/api/conversions/${prior.id}`, { json: prior });
  c.respond('POST', '/api/conversions', () => c.requests('POST', '/api/conversions').length === 1
    ? { status: 502, json: { error: 'The gateway lost the controller response.' } }
    : { json: conversion('accepted'), status: 201 });
  await c.open(page, 'convert');
  await page.getByRole('link', { name: 'Run again', exact: true }).click();
  const start = page.getByRole('button', { name: 'Start conversion', exact: true });
  await start.click();
  await expect(page.getByRole('alert').filter({ hasText: /Pressing Start again/ })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/#\/convert$/);
  await page.goForward();
  await expect(page.locator('#wf-profile')).toHaveValue('ht43');
  await start.click();
  await expect.poll(() => c.requests('POST', '/api/conversions').length).toBe(2);
  const [first, second] = c.requests('POST', '/api/conversions');
  expect(second.body).toEqual(first.body);
  expect(first.headers['idempotency-key']).toBeTruthy();
  expect(second.headers['idempotency-key']).toBe(first.headers['idempotency-key']);
});

test('managed retry can submit a request with no explicit profile', async ({ page, controller: c }) => {
  const prior = conversion('default-profile', { state: 'failed', options: {} });
  c.respond('GET', '/api/conversions', { json: [prior] });
  c.respond('GET', `/api/conversions/${prior.id}`, { json: prior });
  await c.open(page, 'convert');
  await page.getByRole('link', { name: 'Run again', exact: true }).click();
  await expect(page.locator('#wf-profile').locator('option:checked')).toHaveText('Converter default profile');
  noQueue(c);
  await page.getByRole('button', { name: 'Start conversion', exact: true }).click();
  await expect.poll(() => c.requests('POST', '/api/conversions').length).toBe(1);
  expect(c.requests('POST', '/api/conversions')[0].body.options?.profile).toBeUndefined();
});

test('benchmark omission stays visible when a previously omitted suite becomes available', async ({ page, controller: c }) => {
  await openRetry(page, c, detail(job('partial-bench', 'benchmark', {
    state: 'failed', options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke', 'second'] },
  })), 'Benchmarks');
  const warning = page.getByText(/Suite second (is not available|was not restored)/);
  await expect(warning).toBeVisible();
  const state = scenario('live-shaped');
  state.hosts[0].benchmarks.push({ id: 'second', label: 'Second suite' });
  c.respond('GET', '/api/state', { json: state });
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  const second = page.getByRole('checkbox', { name: /Second suite/ });
  await expect(second).toBeVisible();
  await expect(second).not.toBeChecked();
  await expect(warning).toBeVisible({ timeout: 1000 });
  await second.check();
  await expect(warning).toHaveCount(0);
  noQueue(c);
});

for (const missing of ['deleted', 'unavailable']) {
  const source = missing === 'deleted' ? 'source-a' : 'gguf-a';
  test(`GGUF retry from ${source} explains the omitted ${missing} optional imatrix`, async ({ page, controller: c }) => {
    const state = scenario('live-shaped');
    if (missing === 'unavailable') state.models.push({
      id: 'matrix', name: 'Previous matrix', kind: 'imatrix', host: 'worker-a', path: '/matrix', available: false,
    });
    c.respond('GET', '/api/state', { json: state });
    const parent = conversion('matrix-run', {
      state: 'failed', format: 'gguf', input_ids: [source, 'matrix'],
      job_ids: { convert: 'matrix-job' }, options: { preset: 'Q4_K_M' },
    });
    c.respond('GET', '/api/conversions', { json: [parent] });
    await openRetry(page, c, detail(job('matrix-job', 'convert', {
      state: 'failed', format: 'gguf', input_ids: parent.input_ids, options: parent.options,
    })), 'Convert');
    await expect(page.getByText(/Previous importance matrix matrix was not restored/)).toBeVisible();
    noQueue(c);
    await expect(page.getByRole('button', { name: 'Start GGUF conversion' })).toBeDisabled();
    await page.getByRole('button', { name: 'Acknowledge omitted matrix' }).click();
    await page.getByRole('button', { name: 'Start GGUF conversion' }).click();
    await expect.poll(() => c.requests('POST', '/api/conversions').map(r => r.body.input_ids)).toEqual([[source]]);
  });
}

test('terminal child gains Run again when the scheduler later fails the parent', async ({ page, controller: c }) => {
  const parent = conversion('parent', { state: 'running', stage: 'convert', job_ids: { convert: 'child' } });
  c.respond('GET', '/api/conversions', () => ({ json: [parent] }));
  c.respond('GET', '/api/jobs/child', { json: detail(job('child', 'convert', { state: 'succeeded' })) });
  await c.open(page, 'jobs/child');
  await expect.poll(() => c.requests('GET', '/api/conversions').length).toBeGreaterThan(0);
  await expect(page.getByRole('link', { name: 'Run again' })).toHaveCount(0);
  parent.state = 'failed';
  await expect(page.getByRole('link', { name: 'Run again' })).toBeVisible({ timeout: 6000 });
  noQueue(c);
});
