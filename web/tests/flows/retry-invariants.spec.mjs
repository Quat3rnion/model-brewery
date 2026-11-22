import { readFileSync } from 'node:fs';
import { test, expect } from './fixtures.mjs';
import { conversion, detail, ggufCatalog, job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (page, name) => page.getByRole('button', { name, exact: true });
const nav = (page, name) => page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name, exact: true });
const posts = c => c.requests('POST', '/api/jobs');
const choose = async (page, field, name) => {
  await page.locator(field).click();
  await page.getByRole('option', { name }).click();
};
async function openRetry(page, c, kind, view, fields, respond) {
  const prior = detail(job(`prior-${kind}`, kind, { state: 'failed', ...fields }));
  c.respond('GET', `/api/jobs/${prior.id}`, respond ? () => respond(prior) : { json: prior });
  await c.open(page, `jobs/${prior.id}`);
  await page.getByRole('link', { name: `Open in ${view}` }).click();
}
function gate() {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  return { pending, release };
}

test('edited partial transfer still requires acknowledgment on the plain route', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[1], id: 'worker-c', name: 'Worker C' });
  state.eligibility.hosts['worker-c'] = state.eligibility.hosts['worker-b'];
  c.respond('GET', '/api/state', { json: state });
  await openRetry(page, c, 'transfer', 'Transfers', { host: 'worker-b', input_ids: ['gguf-a', 'deleted-entry'] });
  const warning = page.getByText('deleted-entry: Not in Library.');
  await expect(warning).toBeVisible();
  await choose(page, '#tx-host', /^Worker C/);
  await page.getByRole('checkbox', { name: /^Fixture checkpoint/ }).check();
  await nav(page, 'Transfers').click();
  await expect(warning).toBeVisible();
  for (const name of ['Preview', 'Transfer']) await expect(button(page, name)).toBeDisabled();
  expect(posts(c)).toHaveLength(0);
  await page.getByRole('checkbox', { name: 'Continue without these entries' }).check();
  await button(page, 'Transfer').click();
  await expect.poll(() => posts(c).map(r => r.body.input_ids)).toEqual([['source-a', 'gguf-a']]);
});

for (const d of [
  { kind: 'baseline', view: 'Evaluate', queue: 'Queue reference results', missing: 'ref-removed',
    fields: { input_ids: ['ref-removed'] }, field: '#b-file', replacement: 'gsm8k-baseline' },
  { kind: 'plan', view: 'Convert', queue: 'Queue plan', missing: 'vision-removed',
    fields: { input_ids: ['source-a', 'vision-removed'], options: { profile: 'v2', vision: 1 } },
    field: '#cv-comp-vision', replacement: 'vision' },
]) test(`${d.kind} identifies and blocks a deleted input until replaced`, async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'replacement', name: 'Replacement file', kind: d.replacement, host: 'worker-a', path: '/fixtures/replacement' });
  c.respond('GET', '/api/state', { json: state });
  await openRetry(page, c, d.kind, d.view, d.fields);
  await expect(button(page, d.queue)).toBeDisabled();
  await expect(page.getByText(new RegExp(d.missing))).toBeVisible();
  expect(posts(c)).toHaveLength(0);
  if (d.kind === 'plan') await page.locator(d.field).selectOption('replacement');
  else await choose(page, d.field, /^Replacement file/);
  await expect(button(page, d.queue)).toBeEnabled();
});

test('benchmark blocks an omitted suite until acknowledgment', async ({ page, controller: c }) => {
  await openRetry(page, c, 'benchmark', 'Benchmarks', { options: {
    benchmark_target: 'fixture-endpoint', benchmarks: ['smoke', 'deleted-suite'],
  } });
  const queue = button(page, 'Queue benchmark run');
  const ack = button(page, 'Acknowledge omitted suites');
  await expect(ack).toBeVisible();
  await expect(queue).toBeDisabled();
  expect(posts(c)).toHaveLength(0);
  await ack.click();
  await expect(page.getByText('Suite deleted-suite is not available on this host.')).toBeVisible();
  await queue.click();
  await expect.poll(() => posts(c).map(r => r.body.options.benchmarks)).toEqual([['smoke']]);
});

test('GGUF revalidates the matrix on Back and blocks the new omission', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'matrix', name: 'Fixture matrix', kind: 'imatrix', host: 'worker-a', path: '/matrix' });
  c.respond('GET', '/api/state', () => ({ json: state }));
  await openRetry(page, c, 'convert', 'Convert', {
    format: 'gguf', input_ids: ['source-a', 'matrix'], options: { preset: 'Q4_K_M' },
  });
  const queue = button(page, 'Start GGUF conversion');
  await expect(page.locator('#gg-imatrix')).toHaveValue('Fixture matrix');
  await expect(queue).toBeEnabled();
  await nav(page, 'Library').click();
  state.models = state.models.filter(m => m.id !== 'matrix');
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await expect(page.getByText('Fixture matrix', { exact: false })).toHaveCount(0);
  await page.goBack();
  await expect(button(page, 'Acknowledge omitted matrix')).toBeVisible();
  await expect(queue).toBeDisabled();
  await button(page, 'Acknowledge omitted matrix').click();
  await expect(page.getByText(/Previous importance matrix matrix was not restored/)).toBeVisible();
  await queue.click();
  await expect.poll(() => c.requests('POST', '/api/conversions').map(r => r.body.input_ids)).toEqual([['source-a']]);
});

for (const [kind, action, old] of [
  ['weight_inspect', 'Inspect', { tensor: 'blk.0.ffn_down.weight', index: [0, 0] }],
  ['weight_propose', 'Create proposal', { edits: [{ tensor: 'blk.0.ffn_down.weight', index: [0, 0], op: 'set', value: .125 }] }],
  ['weight_tune', 'Tune selected weights', { tensor: 'blk.0.ffn_down.weight', indices: [[0, 0]], max_update: .01 }],
]) test(`${kind} editor reuses the key after an ambiguous reply`, async ({ page, controller: c }) => {
  const tensor = 'blk.0.ffn_down.weight';
  const state = scenario('weight-reads');
  state.hosts[0].capabilities.weight_visual = false;
  for (const [id, name] of [['fit', 'Fit split'], ['held', 'Held-out split']])
    state.models.push({ id, name, host: 'worker-a', kind: 'weight-calibration', path: `/fixtures/${id}.json` });
  c.respond('GET', '/api/state', { json: state });
  c.respond('POST', '/api/jobs', () => posts(c).length === 1
    ? { status: 502, json: { error: 'Reply lost.' } } : { json: job('accepted', kind, posts(c).at(-1).body) });
  await openRetry(page, c, kind, 'Weights', { input_ids: kind === 'weight_tune' ? ['gguf-a', 'fit', 'held'] : ['gguf-a'], options: old });
  await expect(page.getByRole('region', { name: 'Previous weight request' })).toBeVisible();
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
  expect(first.body.options).not.toEqual(old);
  expect(second.body).toEqual(first.body);
  expect(first.headers['idempotency-key']).toBeTruthy();
  expect(second.headers['idempotency-key']).toBe(first.headers['idempotency-key']);
});

test('ambiguous retry reply says to check Jobs or retry without reloading', async ({ page, controller: c }) => {
  c.respond('POST', '/api/jobs', { status: 502, json: { error: 'Reply lost.' } });
  await openRetry(page, c, 'generate', 'Evaluate', { input_ids: ['bundle-a'], prompt_text: 'Retry this.' });
  await button(page, 'Queue generation').click();
  await expect(page.getByRole('alert').filter({ hasText: 'Reply lost.' })).toContainText(/check Jobs.+without reloading/i);
});

for (const [kind, view, action, fields] of [
  ['generate', 'Evaluate', 'Queue generation', { input_ids: ['bundle-a'], prompt_text: 'Earlier prompt.' }],
  ['benchmark', 'Benchmarks', 'Queue benchmark run', { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } }],
]) test(`${view} explains why Queue is disabled during retry loading`, async ({ page, controller: c }) => {
  const wait = gate();
  let reads = 0;
  try {
    await openRetry(page, c, kind, view, fields, async prior => {
      if (++reads > 1) await wait.pending;
      return { json: prior };
    });
    await expect(button(page, action)).toBeDisabled();
    await expect(page.getByRole('status').filter({ hasText: /Loading the previous/ })).toBeVisible();
  } finally { wait.release(); }
  await expect(button(page, action)).toBeEnabled();
  expect(posts(c)).toHaveLength(0);
});

test('Compare uses snapshot text after the saved prompt changes and the operator goes Back', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  c.respond('GET', '/api/state', () => ({ json: state }));
  await openRetry(page, c, 'compare', 'Compare', { input_ids: ['gguf-a', 'gguf-b'], prompt_id: 'prompt-a', prompt_text: 'Say hello.' });
  await expect(page.getByRole('combobox', { name: 'Saved prompt' })).toHaveValue('Fixture prompt');
  await nav(page, 'Library').click();
  state.prompts[0].text = 'New Library text.';
  const refresh = page.waitForResponse(r => r.url().endsWith('/api/state'));
  await page.evaluate(() => window.dispatchEvent(new Event('online')));
  await refresh;
  await page.goBack();
  await expect(page.getByLabel('Prompt text')).toHaveValue('Say hello.');
  await button(page, 'Queue comparison').click();
  await expect.poll(() => posts(c).map(r => r.body.prompt)).toEqual(['Say hello.']);
});

test('cache cleanup on a transfer retry route does not replay a prior cleanup', async ({ page, controller: c }) => {
  await openRetry(page, c, 'transfer', 'Transfers', { host: 'worker-b', input_ids: ['gguf-a'] });
  await choose(page, '#cl-host', /^Worker A/);
  for (let i = 1; i <= 2; i++) {
    await button(page, 'Clean cache…').click();
    await button(page, 'Queue cleanup').click();
    await expect.poll(() => posts(c).length).toBe(i);
  }
  const [first, second] = posts(c);
  const key = first.headers['idempotency-key'];
  if (key) expect(second.headers['idempotency-key']).not.toBe(key);
});

test('changing Recipe output keeps the edited recipe and source', async ({ page, controller: c }) => {
  const state = scenario('weight-reads');
  const model = state.models.find(m => m.id === 'source-a');
  const inventory = weightRead('recipe-inventory', { kind: 'weight_inspect', host: model.host, input_ids: [model.id],
    options: { mode: 'inventory', offset: 0, limit: 128 } }, model);
  state.jobs.push(inventory);
  c.respond('GET', '/api/state', { json: state });
  c.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
  await openRetry(page, c, 'convert', 'Convert', { format: 'gguf', input_ids: [model.id], options: {
    layer_recipe: { schema: 'brewery.layer-quant.v1', source_sha256: 'a'.repeat(64), default_type: 'F16', layers: { 0: 'Q8_0' } },
  } });
  const layer = page.getByLabel('Layer 0 type');
  await expect(layer).toHaveValue('Q8_0');
  await layer.selectOption('F16');
  await page.getByLabel('Recipe output').selectOption('hgn');
  await expect(page.locator('#recipe-source')).toHaveValue('Fixture checkpoint');
  await expect(layer).toHaveValue('F16');
});

test('late download success does not navigate away or dirty the plain Library', async ({ page, controller: c }) => {
  const state = scenario('live-shaped');
  state.hosts.push({ ...state.hosts[0], id: 'backend-hf', capabilities: { hf_download: true } });
  c.respond('GET', '/api/state', { json: state });
  const wait = gate();
  const dialogs = [];
  page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
  c.respond('POST', '/api/jobs', async () => {
    await wait.pending;
    return { json: job('accepted-download', 'hf_download') };
  });
  try {
    await openRetry(page, c, 'hf_download', 'Library', { requested_host: 'backend-hf', options: { repo_id: 'old/model', filename: 'old.gguf' } });
    await button(page, 'Start download').click();
    await expect.poll(() => posts(c).length).toBe(1);
    await nav(page, 'Library').click();
    await expect(page.getByLabel('Repository ID')).toHaveValue('');
    const refreshed = page.waitForResponse(r => r.url().endsWith('/api/state'));
    wait.release();
    await refreshed;
    await expect(button(page, 'Start download')).toBeVisible();
    await expect(page).toHaveURL(/#\/library$/);
    await nav(page, 'Transfers').click();
    await expect(page).toHaveURL(/#\/transfers$/);
    expect(dialogs).toEqual([]);
  } finally { wait.release(); }
});

test('explicit GGUF restore focuses Source when its catalog mounts', async ({ page, controller: c }) => {
  const catalog = gate(), jobRead = gate();
  let reads = 0;
  c.respond('GET', '/api/profiles/gguf-catalog', async () => {
    await catalog.pending;
    return { json: JSON.parse(readFileSync(ggufCatalog, 'utf8')) };
  });
  try {
    await openRetry(page, c, 'convert', 'Convert', { format: 'gguf', input_ids: ['source-a'], options: { preset: 'Q4_K_M' } }, async prior => {
      if (++reads > 1) await jobRead.pending;
      return { json: prior };
    });
    await page.getByText('Individual steps (diagnostics)', { exact: true }).click();
    await page.getByRole('tab', { name: 'Plan', exact: true }).click();
    jobRead.release();
    await button(page, 'Use previous inputs').click();
    await expect(page.locator('#gg-source')).toHaveCount(0);
    catalog.release();
    await expect(page.locator('#gg-source')).toBeFocused();
  } finally { jobRead.release(); catalog.release(); }
});

test('Compare keeps an edited inline prompt visible after leaving the retry', async ({ page, controller: c }) => {
  await openRetry(page, c, 'compare', 'Compare', { input_ids: ['gguf-a', 'gguf-b'], prompt_id: 'prompt-a', prompt_text: 'Say hello.' });
  await expect(page.getByRole('combobox', { name: 'Saved prompt' })).toHaveValue('Fixture prompt');
  await page.getByRole('radio', { name: 'Type a prompt' }).check();
  const prompt = page.getByLabel('Prompt text');
  await prompt.fill('Edited text.');
  await page.getByRole('radio', { name: 'Saved prompt' }).check();
  page.once('dialog', d => d.accept());
  await nav(page, 'Compare').click();
  await expect(page).toHaveURL(/#\/compare$/);
  await expect(prompt).toHaveValue('Edited text.');
  expect(posts(c)).toHaveLength(0);
});
