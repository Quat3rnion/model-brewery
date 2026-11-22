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

const sha = 'a'.repeat(64);
const tensor = 'blk.0.ffn_down.weight';
const edit = { tensor, index: [0, 1], op: 'set', value: 0.125 };
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
    expect(controller.requests('POST', '/api/jobs').map(({ body }) => body)).toEqual([
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
