import { test, expect } from './fixtures.mjs';
import { conversion, detail, job, scenario, weightRead } from './mock/scenarios.mjs';
import { libraryWeights } from './mock/library-weights.mjs';

async function retry(page, controller, kind, view, fields) {
  const prior = detail(job(`retry-${kind}`, kind, { state: 'failed', ...fields }));
  controller.respond('GET', `/api/jobs/${prior.id}`, { json: prior });
  await controller.open(page, `jobs/${prior.id}`);
  await page.getByRole('link', { name: `Open in ${view}` }).click();
  return prior;
}

test('Evaluate restores a bundle, host and reference file', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'ref', name: 'Reference scores', kind: 'gsm8k-baseline', host: 'worker-a', path: '/reference.json' });
  controller.respond('GET', '/api/state', { json: state });
  await retry(page, controller, 'evaluate', 'Evaluate', { input_ids: ['bundle-a', 'ref'] });
  const form = page.getByRole('region', { name: 'GSM8K evaluation' });
  await expect(form.getByRole('combobox', { name: 'Bundle' })).toHaveValue('Fixture bundle');
  await expect(form.getByRole('combobox', { name: 'Run on' })).toHaveValue('Worker A');
  await expect(form.getByRole('combobox', { name: 'Reference results' })).toHaveValue('Reference scores');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Generate restores the exact text sent to the worker even if the saved prompt changed', async ({ page, controller }) => {
  await retry(page, controller, 'generate', 'Evaluate', {
    input_ids: ['bundle-a'], prompt_id: 'prompt-a', prompt_text: 'The saved text at submission.',
  });
  const form = page.getByRole('region', { name: 'Generate', exact: true });
  await expect(form.getByRole('combobox', { name: 'Bundle' })).toHaveValue('Fixture bundle');
  await expect(form.getByRole('combobox', { name: 'Run on' })).toHaveValue('Worker A');
  await expect(form.getByRole('textbox', { name: 'Prompt text' })).toHaveValue('The saved text at submission.');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Evaluate blocks a removed bundle until the operator chooses a replacement', async ({ page, controller }) => {
  await retry(page, controller, 'evaluate', 'Evaluate', { input_ids: ['removed-bundle'] });
  const form = page.getByRole('region', { name: 'GSM8K evaluation' });
  const queue = form.getByRole('button', { name: 'Queue full GSM8K evaluation' });
  await expect(form.getByText(/Previous bundle removed-bundle is not available/)).toBeVisible();
  await expect(queue).toBeDisabled();
  await form.getByRole('combobox', { name: 'Bundle' }).click();
  await form.getByRole('option', { name: /Fixture bundle/ }).click();
  await expect(queue).toBeEnabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Generate offers delayed previous inputs without replacing a typed draft', async ({ page, controller }) => {
  const old = detail(job('delayed-gen', 'generate', { state: 'failed', input_ids: ['bundle-a'], prompt_text: 'Earlier prompt.' }));
  let release;
  const wait = new Promise((resolve) => { release = resolve; });
  let count = 0;
  controller.respond('GET', `/api/jobs/${old.id}`, async () => {
    if (++count > 1) await wait;
    return { json: old };
  });
  await controller.open(page, `jobs/${old.id}`);
  await page.getByRole('link', { name: 'Open in Evaluate' }).click();
  const form = page.getByRole('region', { name: 'Generate', exact: true });
  await form.getByRole('radio', { name: 'Type a prompt' }).check();
  const text = form.getByRole('textbox', { name: 'Prompt text' });
  await text.fill('New draft.');
  release();
  const use = form.getByRole('button', { name: 'Use previous inputs' });
  await expect(use).toBeVisible();
  await expect(text).toHaveValue('New draft.');
  await use.click();
  await expect(text).toHaveValue('Earlier prompt.');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Generate resubmits the previous API token limit', async ({ page, controller }) => {
  await retry(page, controller, 'generate', 'Evaluate', {
    input_ids: ['bundle-a'], prompt_text: 'Earlier prompt.', options: { max_tokens: 73 },
  });
  const submit = page.getByRole('button', { name: 'Queue generation' });
  await expect(submit).toBeEnabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await submit.click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body.options?.max_tokens)).toEqual([73]);
});

test('Transfer names omitted entries and waits for acknowledgment before copying', async ({ page, controller }) => {
  await retry(page, controller, 'transfer', 'Transfers', {
    state: 'interrupted', host: 'worker-b', input_ids: ['gguf-a', 'deleted-entry'],
  });
  const queue = page.getByRole('button', { name: 'Transfer', exact: true });
  await expect(page.getByText('deleted-entry: Not in Library.')).toBeVisible();
  await expect(queue).toBeDisabled();
  await page.getByRole('checkbox', { name: 'Continue without these entries' }).check();
  await expect(queue).toBeEnabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await queue.click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body.input_ids)).toEqual([['gguf-a']]);
});

test('Reference restores its prior endpoint and host', async ({ page, controller }) => {
  await retry(page, controller, 'baseline', 'Evaluate', { options: { baseline_source: 'fixture-baseline' } });
  const form = page.getByRole('region', { name: 'Prepare reference results' });
  await expect(form.getByRole('radio', { name: 'Use a reference endpoint' })).toBeChecked();
  await expect(form.getByRole('combobox', { name: 'Host' })).toHaveValue('Worker A');
  await expect(form.getByRole('combobox', { name: 'Reference endpoint' })).toHaveValue('Fixture baseline');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Benchmarks restores the prior host, target and suites', async ({ page, controller }) => {
  await retry(page, controller, 'benchmark', 'Benchmarks', { options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke'] } });
  await expect(page.getByRole('combobox', { name: 'Host' })).toHaveValue('Worker A');
  await expect(page.getByRole('combobox', { name: 'Endpoint target' })).toHaveValue('Fixture endpoint');
  await expect(page.getByRole('checkbox', { name: /Endpoint smoke/ })).toBeChecked();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Benchmarks names each unavailable suite from the previous request', async ({ page, controller }) => {
  await retry(page, controller, 'benchmark', 'Benchmarks', {
    options: { benchmark_target: 'fixture-endpoint', benchmarks: ['smoke', 'disabled-suite'] },
  });
  await expect(page.getByRole('checkbox', { name: /Endpoint smoke/ })).toBeChecked();
  await expect(page.getByText('Suite disabled-suite is not available on this host.')).toBeVisible();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('A failed managed HGN stage offers both the step and full request', async ({ page, controller }) => {
  const parent = conversion('failed-hgn', {
    state: 'failed', requested_host: 'auto', stage: 'verify',
    input_ids: ['source-a'], job_ids: { verify: 'failed-verify' },
    options: { profile: 'ht43', threads: 7, seed: 23 }, decode_backend: 'cpu',
  });
  controller.respond('GET', '/api/conversions', { json: [parent] });
  controller.respond('GET', `/api/conversions/${parent.id}`, { json: parent });
  controller.respond('GET', '/api/jobs/failed-verify', { json: detail(job('failed-verify', 'verify', {
    state: 'failed', input_ids: ['source-a', 'bundle-a'], options: { decode_backend: 'cpu' },
  })) });
  await controller.open(page, 'jobs/failed-verify');
  await expect(page.getByRole('link', { name: 'Open in Convert' })).toBeVisible();
  await page.getByRole('link', { name: 'Run again', exact: true }).click();
  await expect(page.getByRole('radio', { name: 'Managed HGN' })).toBeChecked();
  await expect(page.locator('#wf-source')).toHaveValue('Fixture checkpoint');
  await expect(page.locator('#wf-profile')).toHaveValue('ht43');
  await expect(page.getByRole('checkbox', { name: 'Any eligible pool worker' })).toBeChecked();
  const expected = { host: 'auto', input_ids: ['source-a'], options: parent.options, decode_backend: 'cpu' };
  await expect.poll(() => controller.requests('POST', '/api/conversions/eligibility').map(({ body }) => body)).toEqual([expected]);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  await page.getByRole('button', { name: 'Start conversion', exact: true }).click();
  await expect.poll(() => controller.requests('POST', '/api/conversions').map(({ body }) => body)).toEqual([expected]);
});

test('A failed managed HGN run with a successful child offers Run again', async ({ page, controller }) => {
  const parent = conversion('failed-after-success', {
    state: 'failed', stage: 'convert', job_ids: { convert: 'successful-last-step' },
    options: { profile: 'ht43' }, error: 'Missing result artifact.',
  });
  controller.respond('GET', '/api/conversions', { json: [parent] });
  controller.respond('GET', `/api/conversions/${parent.id}`, { json: parent });
  controller.respond('GET', '/api/jobs/successful-last-step', { json: detail(job('successful-last-step', 'convert', {
    state: 'succeeded', input_ids: parent.input_ids, options: parent.options,
  }), {}) });
  await controller.open(page, 'convert');
  const run = page.getByRole('article', { name: /Fixture checkpoint.*HT43/ });
  await run.getByRole('link', { name: 'Run again', exact: true }).click({ timeout: 5000 });
  await expect(page.locator('#wf-profile')).toHaveValue('ht43');
  await run.getByRole('link', { name: 'Convert job successful-last-step' }).click();
  await expect(page.getByRole('link', { name: 'Run again', exact: true })).toBeVisible();
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
});

test('Run again starts a new managed conversion after an identical failed submission', async ({ page, controller }) => {
  const failed = conversion('failed-on-page', {
    state: 'failed', requested_host: 'auto', stage: 'convert', options: { profile: 'ht43' },
  });
  const path = '/api/conversions';
  controller.respond('GET', path, () => ({ json: controller.requests('POST', path).length ? [failed] : [] }));
  controller.respond('GET', `${path}/${failed.id}`, { json: failed });
  controller.respond('POST', path, { json: failed, status: 201 });
  await controller.open(page, 'convert');
  await page.locator('#wf-source').click();
  await page.getByRole('option', { name: /^Fixture checkpoint/ }).click();
  await page.locator('#wf-profile').selectOption('ht43');
  await page.getByRole('checkbox', { name: 'Any eligible pool worker' }).check();
  const start = page.getByRole('button', { name: 'Start conversion', exact: true });
  await start.click();
  await page.getByRole('link', { name: 'Run again', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Convert a checkpoint' })).toBeFocused();
  expect(controller.requests('POST', path)).toHaveLength(1);
  await start.click();
  await expect.poll(() => controller.requests('POST', path).length).toBe(2);
  const [first, second] = controller.requests('POST', path);
  expect(second.body).toEqual(first.body);
  expect(second.headers['idempotency-key']).not.toBe(first.headers['idempotency-key']);
});

test('Custom HGN stage restores explicit CPU decode without an effect loop', async ({ page, controller }) => {
  const errors = [];
  page.on('pageerror', (error) => {
    if (error.message.includes('effect_update_depth_exceeded')) errors.push(error.message);
  });
  const revision = 'a'.repeat(64);
  const parent = conversion('custom-hgn', {
    state: 'failed', stage: 'verify', job_ids: { verify: 'retry-verify' },
    profile_revision: revision, options: { threads: 8 }, decode_backend: 'cpu',
  });
  controller.respond('GET', '/api/conversions', { json: [parent] });
  await retry(page, controller, 'verify', 'Convert', {
    profile_revision: revision, input_ids: ['source-a', 'bundle-a'], options: { decode_backend: 'cpu' },
  });
  await expect(page.locator('#wf-profile')).toHaveValue(`rev:${revision}`);
  await expect(page.locator('#wf-dec')).toHaveValue('cpu');
  await expect.poll(() => controller.requests('POST', '/api/conversions/eligibility').length + errors.length).toBeGreaterThan(0);
  expect(errors).toEqual([]);
  expect(controller.requests('POST', '/api/conversions/eligibility').at(-1).body.decode_backend).toBe('cpu');
});

test('A failed GGUF stage opens the GGUF form with its parent conversion options', async ({ page, controller }) => {
  const previous = conversion('gguf-run', {
    format: 'gguf', state: 'failed', job_ids: { convert: 'retry-convert' },
    input_ids: ['gguf-a'], options: { preset: 'Q8_0', threads: 9 },
  });
  controller.respond('GET', '/api/conversions', { json: [] });
  await controller.open(page, 'convert');
  await expect(page.getByText('No conversions yet.', { exact: false })).toBeVisible();
  controller.respond('GET', '/api/conversions', async () => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { json: [previous] };
  });
  const failed = detail(job('retry-convert', 'convert', {
    state: 'failed', format: 'gguf', input_ids: ['gguf-a'], options: { preset: 'Q8_0' },
  }));
  controller.respond('GET', `/api/jobs/${failed.id}`, { json: failed });
  await page.evaluate(() => { location.hash = '#/jobs/retry-convert'; });
  await page.getByRole('link', { name: 'Open in Convert' }).click();
  await expect(page.locator('#gg-source')).toHaveValue('Fixture GGUF');
  await expect(page.locator('#gg-choice')).toHaveValue('preset:Q8_0');
  await expect(page.locator('#gg-threads')).toHaveValue('9');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
});

test('GGUF retains a valid optional imatrix when retry uses Q4_K_M', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'matrix-a', name: 'Fixture imatrix', kind: 'imatrix', host: 'worker-a', path: '/fixtures/imatrix' });
  controller.respond('GET', '/api/state', { json: state });
  const previous = conversion('gguf-matrix', {
    format: 'gguf', state: 'failed', job_ids: { convert: 'retry-convert' },
    input_ids: ['gguf-a', 'matrix-a'], options: { preset: 'Q4_K_M' },
  });
  controller.respond('GET', '/api/conversions', { json: [previous] });
  await retry(page, controller, 'convert', 'Convert', { format: 'gguf', input_ids: ['gguf-a'], options: previous.options });
  await expect(page.locator('#gg-choice')).toHaveValue('preset:Q4_K_M');
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  await page.getByRole('button', { name: 'Start GGUF conversion' }).click();
  await expect.poll(() => controller.requests('POST', '/api/conversions').map(({ body }) => body.input_ids)).toEqual([previous.input_ids]);
});

for (const [kind, tensor, action] of [
  ['plan', 'blk.1.ffn_down_exps.weight', 'Queue Hy4 plan'],
  ['convert', 'blk.1.attn_gate.weight', 'Queue Hy4 conversion'],
]) {
  test(`A failed Hy4 ${kind} opens its diagnostic with the exact override`, async ({ page, controller }) => {
    await retry(page, controller, kind, 'Convert', {
      input_ids: ['source-a'], options: { hy4_overrides: { [tensor]: 'Q8_0' } },
    });
    await expect(page.getByRole('combobox', { name: 'Hy4 source' })).toHaveValue('Fixture checkpoint');
    await expect(page.getByText(tensor, { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: action })).toBeVisible();
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  });
}

for (const paused of [true, false]) {
  test(`Hy4 encode cannot queue on a ${paused ? 'paused' : 'convert-disabled'} worker`, async ({ page, controller }) => {
    const state = scenario('live-shaped');
    const host = state.hosts[0];
    host.capabilities.hy4_encode = true;
    host.capabilities.convert = paused;
    host.scheduling_paused = paused;
    state.eligibility.hosts[host.id].convert = { eligible: false, reasons: [{ text: paused ? 'Resume job scheduling.' : 'Enable conversion.', fix_anchor: '#/settings' }] };
    controller.respond('GET', '/api/state', { json: state });
    await retry(page, controller, 'convert', 'Convert', {
      input_ids: ['source-a'], options: { hy4_overrides: { 'blk.1.attn_gate.weight': 'Q8_0' } },
    });
    await expect(page.getByRole('combobox', { name: 'Hy4 source' })).toHaveValue('Fixture checkpoint');
    await expect(page.getByRole('button', { name: 'Queue Hy4 conversion' })).toBeDisabled();
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  });
}

test('A failed recipe stage restores the output format, source and layer types', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  const model = state.models.find((m) => m.id === 'source-a');
  const inv = weightRead('recipe-inventory', { kind: 'weight_inspect', host: model.host, input_ids: [model.id],
    options: { mode: 'inventory', offset: 0, limit: 128 } }, model);
  state.jobs.push(inv);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', `/api/jobs/${inv.id}`, { json: inv });
  await retry(page, controller, 'convert', 'Convert', {
    format: 'gguf', input_ids: ['source-a'], options: { layer_recipe: {
      schema: 'brewery.layer-quant.v1', source_sha256: 'a'.repeat(64), default_type: 'F16', layers: { 0: 'Q8_0' },
    } },
  });
  await expect(page.getByLabel('Recipe output')).toHaveValue('gguf');
  await expect(page.locator('#recipe-source')).toHaveValue('Fixture checkpoint');
  await expect(page.getByLabel('Default type')).toHaveValue('F16');
  await expect(page.getByLabel('Layer 0 type')).toHaveValue('Q8_0');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Layer recipe uses the replacement source digest, not the failed source digest', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  Object.assign(state.hosts[0].capabilities, { gguf_layer_recipe: true, gguf_hf_output: true });
  const replacement = { ...state.models[0], id: 'replacement', name: 'Replacement checkpoint', path: '/fixtures/other' };
  state.models.push(replacement);
  const inventory = weightRead('replacement-inventory', {
    kind: 'weight_inspect', host: replacement.host, input_ids: [replacement.id],
    options: { mode: 'inventory', offset: 0, limit: 128 },
  }, replacement);
  state.jobs.push(inventory);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
  await retry(page, controller, 'convert', 'Convert', {
    format: 'gguf', input_ids: ['source-a'], options: { layer_recipe: {
      schema: 'brewery.layer-quant.v1', source_sha256: 'b'.repeat(64), default_type: 'F16', layers: { 0: 'Q8_0' },
    } },
  });
  const source = page.locator('#recipe-source');
  await expect(source).toHaveValue('Fixture checkpoint');
  await source.click();
  await page.getByRole('option', { name: /^Replacement checkpoint/ }).click();
  const start = page.getByRole('button', { name: 'Run layer recipe' });
  await expect(start).toBeEnabled();
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  await start.click();
  await expect.poll(() => controller.requests('POST', '/api/conversions').map(({ body }) => body.options.layer_recipe.source_sha256)).toEqual(['a'.repeat(64)]);
});

test('A failed HGN diagnostic restores its advanced options and component inputs', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.models.push({ id: 'vision-a', name: 'Fixture vision', kind: 'vision', host: 'worker-a', path: '/fixtures/vision' });
  controller.respond('GET', '/api/state', { json: state });
  await retry(page, controller, 'convert', 'Convert', {
    input_ids: ['source-a', 'vision-a'], options: { profile: 'ht43', threads: 7, seed: 23, vision: 1 },
  });
  const form = page.getByRole('tabpanel');
  await expect(form.getByLabel('Profile', { exact: false })).toHaveValue('ht43');
  await expect(form.getByLabel(/Threads/)).toHaveValue('7');
  await expect(form.getByLabel(/Seed/)).toHaveValue('23');
  await expect(form.getByLabel('Vision tower')).toHaveValue('vision-a');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

for (const filename of ['nested/weights.gguf', undefined]) {
  test(`HF download restores the prior ${filename ? 'file' : 'whole repository'} request`, async ({ page, controller }) => {
    libraryWeights(controller);
    await retry(page, controller, 'hf_download', 'Library', {
      host: 'backend-hf', options: { repo_id: 'owner/test-model', revision: 'test-tag', filename },
    });
    await expect(page.getByLabel('Repository ID')).toHaveValue('owner/test-model');
    await expect(page.getByLabel('Revision (optional)')).toHaveValue('test-tag');
    await expect(page.getByLabel('Download scope')).toHaveValue(filename ? 'file' : 'repo');
    if (filename) await expect(page.getByLabel('Model filename')).toHaveValue(filename);
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  });
}

for (const direction of ['back', 'forward']) {
  test(`HF download retains an unsent draft when ${direction} is declined`, async ({ page, controller }) => {
    libraryWeights(controller);
    const sections = page.getByRole('navigation', { name: 'Sections' });
    if (direction === 'back') {
      await controller.open(page, 'queue');
      await sections.getByRole('link', { name: 'Library' }).click();
    } else {
      await controller.open(page, 'library');
      await sections.getByRole('link', { name: 'Jobs' }).click();
      await page.goBack();
    }
    await page.getByText('Download from Hugging Face', { exact: true }).click();
    await page.getByLabel('Repository ID').fill('fixture/model');
    await page.getByLabel('Revision (optional)').fill('selected-tag');
    await page.getByLabel('Model filename').fill('nested/test.gguf');
    const confirmation = page.waitForEvent('dialog', { timeout: 5000 }).then((dialog) => dialog.dismiss());
    await page.evaluate((direction) => { history[direction](); }, direction);
    await confirmation;
    await expect(page).toHaveURL(/#\/library$/);
    await expect(page.getByLabel('Repository ID')).toHaveValue('fixture/model');
    await expect(page.getByLabel('Revision (optional)')).toHaveValue('selected-tag');
    await expect(page.getByLabel('Model filename')).toHaveValue('nested/test.gguf');
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  });
}

test('HF download verifies files before adding a Library entry', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page, 'library');
  await page.getByText('Download from Hugging Face', { exact: true }).click();
  await expect(page.getByText('The download service verifies the files before it adds the entry to Library.', { exact: false })).toBeVisible();
});

test('HF download retries on the previous remote host', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  state.hosts[0].capabilities.hf_download = true;
  controller.respond('GET', '/api/state', { json: state });
  await retry(page, controller, 'hf_download', 'Library', {
    host: 'worker-a', options: { repo_id: 'owner/test-model', revision: 'test-tag', filename: 'test.gguf' },
  });
  const submit = page.getByRole('button', { name: 'Start download' });
  await expect(submit).toBeEnabled();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await submit.click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body.host)).toEqual(['worker-a']);
});

const sha = 'a'.repeat(64);
const tensor = 'blk.0.ffn_down.weight';
const edit = { tensor, index: [0, 1], op: 'set', value: 0.125 };
test('Weights can inspect an exact GGUF value with inspect and tune only', async ({ page, controller }) => {
  const state = scenario('live-shaped');
  const host = state.hosts[0];
  Object.assign(host.capabilities, { weight_inspect: true, weight_tune: true, weight_visual: false, weight_propose: false });
  controller.respond('GET', '/api/state', { json: state });
  await controller.open(page, 'weights?source=gguf-a');
  const lookup = page.getByRole('group', { name: /Look up a stored value/ });
  await expect(lookup).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Tensor, edit 1' })).toHaveCount(0);
  await lookup.getByLabel('Tensor', { exact: true }).fill(tensor);
  await lookup.getByLabel('Coordinate').fill('0, 1');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await lookup.getByRole('button', { name: 'Inspect', exact: true }).click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body)).toEqual([
    { kind: 'weight_inspect', host: 'worker-a', input_ids: ['gguf-a'], options: { tensor, index: [0, 1] } },
  ]);
});

const proposal = detail(job('old-proposal', 'weight_propose', { state: 'succeeded', input_ids: ['gguf-a'] }), {
  report: { proposal: { schema: 'brewery.weight-edit-proposal.v1', source_sha256: sha,
    edits: [{ ...edit, dtype: 'F32', before: 0.5, after: 0.125 }] } },
});

for (const [name, kind, input_ids, options, check] of [
  ['value inspection', 'weight_inspect', ['gguf-a'], { tensor, index: [0, 1] }, 'index'],
  ['inventory', 'weight_inspect', ['gguf-a'], { mode: 'inventory', offset: 128, limit: 128, source_sha256: sha }, '128'],
  ['tile', 'weight_inspect', ['gguf-a'], { mode: 'tile', tensor, origin: [0, 32], shape: [2, 1], source_sha256: sha }, tensor],
  ['manual proposal', 'weight_propose', ['gguf-a'], { edits: [edit] }, 'edits'],
  ['automatic tuning', 'weight_tune', ['gguf-a', 'fit', 'held-out'], {
    tensor, indices: [[0, 1]], max_update: 0.01, source_sha256: sha,
  }, 'max update'],
]) {
  test(`Weights restores the prior ${name} request and queues it only on request`, async ({ page, controller }) => {
    const state = scenario('weight-reads');
    for (const id of ['fit', 'held-out']) state.models.push({ id, name: id, host: 'worker-a', kind: 'weight-calibration', path: `/fixtures/${id}.json` });
    controller.respond('GET', '/api/state', { json: state });
    await retry(page, controller, kind, 'Weights', { input_ids, options });
    const previous = page.getByRole('region', { name: 'Previous weight request' });
    await expect(previous.getByText(check, { exact: true }).first()).toBeVisible();
    expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
    await previous.getByRole('button', { name: 'Queue previous request' }).click();
    await expect.poll(() => controller.requests('POST', '/api/jobs').map(({ body }) => body)).toEqual([
      { kind, host: 'worker-a', input_ids, options },
    ]);
  });
}

test('Weights opens the prior proposal for apply but does not reuse its approval', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  state.jobs.push(proposal);
  controller.respond('GET', '/api/state', { json: state });
  controller.respond('GET', `/api/jobs/${proposal.id}`, { json: proposal });
  await retry(page, controller, 'weight_apply', 'Weights', {
    input_ids: ['gguf-a'], options: { proposal_job_id: proposal.id, reviewed: true },
  });
  await expect(page.getByRole('radio', { name: /old-proposal/ })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: /reviewed/ })).not.toBeChecked();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Weights keeps a draft when Forward is declined after Back between identical URLs', async ({ page, controller }) => {
  const state = scenario('weight-reads');
  state.hosts[1].capabilities = state.hosts[0].capabilities;
  const applied = detail(job('applied-proposal', 'weight_apply', {
    state: 'succeeded', input_ids: ['gguf-a'], options: { proposal_job_id: proposal.id, reviewed: true },
  }), { library_model_id: 'gguf-b', report: { schema: 'brewery.weight-edit-result.v1', source_sha256: sha,
    output_sha256: 'b'.repeat(64), edit_count: 1, verified: true } });
  state.jobs.push(proposal, applied);
  controller.respond('GET', '/api/state', { json: state });
  for (const detail of [proposal, applied]) controller.respond('GET', `/api/jobs/${detail.id}`, { json: detail });
  await controller.open(page, 'weights?source=gguf-a');
  await page.getByRole('radio', { name: /old-proposal/ }).check();
  await page.getByRole('link', { name: 'Fixture GGUF copy', exact: true }).click();
  const picker = page.locator('#w-source');
  await expect(picker).toHaveValue('Fixture GGUF copy');
  await picker.click();
  await page.getByRole('option', { name: /^Fixture GGUF\b(?! copy)/ }).click();
  await expect(page).toHaveURL(/source=gguf-a$/);
  // The URL stays the same, but the browser has moved to a different entry.
  const back = page.evaluate(() => new Promise((resolve) => {
    window.addEventListener('popstate', () => resolve(), { once: true });
    history.back();
  }));
  await back;
  await picker.click();
  await page.getByRole('option', { name: /^Fixture GGUF copy/ }).click();
  await page.getByText('Text entry: type tensor names and coordinates').click();
  const draft = page.getByRole('textbox', { name: 'Tensor, edit 1' });
  await draft.fill(tensor);
  const confirm = page.waitForEvent('dialog', { predicate: (d) => d.type() === 'confirm' }).then((d) => d.dismiss());
  await page.evaluate(() => { history.forward(); });
  await confirm;
  await expect(page).toHaveURL(/source=gguf-b$/);
  await expect(draft).toHaveValue(tensor);
});
