import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { libraryWeights, verifiedHash } from './mock/library-weights.mjs';
import { detail, job, scenario } from './mock/scenarios.mjs';

test('Library shows controller identity and does not link unusable weights to the editor', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page);
  const entry = (name) => page.getByRole('row').filter({ has: page.getByText(name, { exact: true }) });
  const verified = entry('Verified GGUF');
  await expect(verified.getByText(verifiedHash, { exact: true })).toBeVisible();
  await expect(verified.getByText('2.00 KiB', { exact: true })).toBeVisible();
  await expect(verified.getByText('Available', { exact: true })).toBeVisible();
  await expect(verified.getByRole('link', { name: 'Output of job download-job' })).toHaveAttribute('href', '#/jobs/download-job');
  await expect(verified.getByRole('link', { name: 'Edit weights of Verified GGUF' })).toBeVisible();
  for (const [name, status, reason] of [
    ['Fixture GGUF copy', 'Unknown', 'The worker has not checked this path.'],
    ['Missing GGUF', 'Unavailable', 'The source file does not exist.'],
  ]) {
    await expect(entry(name).getByText(status, { exact: true })).toBeVisible();
    await expect(entry(name)).toContainText(reason);
    await expect(entry(name).getByRole('link', { name: /Edit weights/ })).toHaveCount(0);
  }
  await expect(entry('Fixture GGUF copy')).toContainText('weight_propose is not configured on this worker.');
});

test('Register excludes the download host and explains what is needed before input', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page);
  const form = page.getByRole('region', { name: 'Register an entry' });
  await expect(form.getByRole('button', { name: 'Register', exact: true })).toHaveAccessibleDescription(/Select a host/);
  await form.getByRole('combobox', { name: 'Host', exact: true }).click();
  await expect(form.getByRole('option', { name: /Worker A/ })).toBeVisible();
  await expect(form.getByRole('option', { name: /Backend downloads/ })).toHaveCount(0);
});

test('Library warns before removing a download and returns focus after Keep and Remove', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page);
  const remove = page.getByRole('button', { name: 'Remove Verified GGUF from library' });
  const group = page.getByRole('group', { name: 'Confirm removal' });
  await remove.press('Enter');
  await expect(group).toContainText('You cannot register it again by hand.');
  await expect(group).toContainText('Download the source again');
  const keep = group.getByRole('button', { name: 'Keep' });
  await expect(keep).toBeFocused();
  await keep.press('Enter');
  await expect(remove).toBeFocused();
  expect(controller.requests('DELETE', '/api/models/downloaded')).toHaveLength(0);
  await remove.press('Enter');
  await group.getByRole('button', { name: 'Remove entry' }).press('Enter');
  await expect(remove).toHaveCount(0);
  await expect(page.getByRole('searchbox', { name: 'Filter entries' })).toBeFocused();
});

test('Library distinguishes backend weight revision and calibration batch removal', async ({ page, controller }) => {
  const data = scenario('weight-reads');
  data.jobs.push(job('apply-a', 'weight_apply', { host: 'backend-hf', state: 'succeeded' }));
  data.models.push({
    id: 'edited-a', name: 'Edited HF', host: 'backend-hf', kind: 'hf',
    path: '/fixtures/edits/apply-a/edited.hf', job_id: 'apply-a', available: true,
  });
  data.models.push({
    id: 'split-a', name: 'Fixture split', host: 'backend-hf', kind: 'weight-calibration',
    path: '/fixtures/split.json', available: true,
  });
  controller.respond('GET', '/api/state', { json: data });
  await controller.open(page);
  await page.getByRole('button', { name: 'Remove Edited HF from library' }).press('Enter');
  const group = page.getByRole('group', { name: 'Confirm removal' });
  await expect(group).toContainText('The files stay on the host.');
  await expect.soft(group).toContainText('You cannot restore this revision');
  await expect.soft(group).toContainText('A new Apply needs the input files and creates a new entry');
  await expect(group).not.toContainText('download');
  await group.getByRole('button', { name: 'Keep' }).press('Enter');
  await page.getByRole('button', { name: 'Remove Fixture split from library' }).press('Enter');
  await expect(group).toContainText('Upload the calibration batch again in Weights');
});

test('Library updates a saved prompt and shows the controller job conflict before deletion', async ({ page, controller }) => {
  const { allowPromptDeletion } = libraryWeights(controller);
  await controller.open(page);
  const saved = page.getByRole('region', { name: /Saved prompts/ });
  await saved.getByRole('button', { name: /Original prompt/ }).click();
  await saved.getByRole('button', { name: 'Edit prompt' }).click();
  const editor = page.getByRole('region', { name: 'Edit prompt', exact: true });
  await editor.getByLabel('Name', { exact: true }).fill('Updated prompt');
  await editor.getByLabel('Prompt text').fill(' Text with spaces. \n');
  await editor.getByRole('button', { name: 'Save changes' }).click();
  await expect(saved.getByRole('button', { name: /Updated prompt/ })).toBeVisible();
  expect(controller.requests('PUT', '/api/prompts/editable').map((r) => r.body)).toEqual([
    { name: 'Updated prompt', text: ' Text with spaces. \n' },
  ]);
  await saved.getByRole('button', { name: 'Delete prompt', exact: true }).click();
  const confirm = saved.getByRole('group', { name: 'Confirm prompt deletion' });
  await confirm.getByRole('button', { name: 'Delete prompt', exact: true }).click();
  await expect(saved.getByRole('alert')).toHaveText('Job job-fixture-42 uses this prompt.');
  allowPromptDeletion();
  await confirm.getByRole('button', { name: 'Delete prompt', exact: true }).click();
  await expect(saved.getByRole('button', { name: /Updated prompt/ })).toHaveCount(0);
  expect(controller.requests('DELETE', '/api/prompts/editable')).toHaveLength(2);
});

test('Library keeps a new prompt draft when the operator declines to leave', async ({ page, controller }) => {
  await controller.open(page);
  const name = page.getByLabel('Name', { exact: true }).last();
  await name.fill('New draft');
  await page.getByLabel('Prompt text').fill('Keep this text.');
  const jobs = page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: /^Jobs/ });
  const dialog = page.waitForEvent('dialog', { timeout: 2000 }).then((d) => d.dismiss());
  await jobs.click();
  await dialog;
  await expect(name).toHaveValue('New draft');
  await expect(page.getByLabel('Prompt text')).toHaveValue('Keep this text.');
  const leave = page.waitForEvent('dialog').then((d) => d.accept());
  await jobs.click();
  await leave;
  await expect(page.getByRole('heading', { name: 'Jobs', exact: true })).toBeVisible();
  expect(controller.requests('POST', '/api/prompts')).toHaveLength(0);
});

test('Weights labels unusable entries and blocks a missing entry opened by a deep link', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page, 'weights');
  await expect(page.getByRole('radiogroup', { name: 'Editing mode' })).toHaveCount(0);
  const picker = page.getByRole('combobox', { name: 'Source entry' });
  await picker.click();
  await expect(page.getByRole('option', { name: /Missing GGUF.*The source file does not exist/ })).toHaveAttribute('aria-disabled', 'true');
  await expect(page.getByRole('option', { name: /Fixture GGUF copy.*weight_propose/ })).toHaveAttribute('aria-disabled', 'true');
  await controller.open(page, 'weights?source=missing');
  await expect(page.getByText('The source file does not exist.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /Read tensor|Create proposal|Tune selected weights/ })).toHaveCount(0);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Inspect and visual capabilities permit explicit read-only inspection without edit permission', async ({ page, controller }) => {
  controller.load('weight-reads');
  const state = scenario('weight-reads');
  const host = state.hosts[0];
  host.capabilities.weight_propose = false;
  host.capabilities.weight_tune = false;
  controller.respond('GET', '/api/state', { json: state });
  await controller.open(page, 'weights?source=gguf-a');
  const read = page.getByRole('button', { name: 'Read tensor list', exact: true });
  await expect(read).toBeEnabled();
  await expect(page.getByText(/Read-only:.*weight_propose/i)).toBeVisible();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await read.click();
  await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
  await page.getByRole('button', { name: 'Read tile', exact: true }).click();
  await expect(page.getByRole('gridcell').first()).toBeVisible();
  await expect(page.getByRole('button', { name: /Raise by|Create proposal|Tune selected weights/ })).toHaveCount(0);
  expect(controller.requests('POST', '/api/jobs').map(({ body }) => body.kind)).toEqual(['weight_inspect', 'weight_inspect']);
});

test('GGUF combines text and visual edits in one proposal and validates text on blur', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'weights?source=gguf-a');
  await page.getByText('Text entry: type tensor names and coordinates', { exact: true }).click();
  const tensor = page.getByRole('textbox', { name: 'Tensor, edit 1' });
  const coordinate = page.getByRole('textbox', { name: 'Coordinate, edit 1' });
  await tensor.fill('blk.0.ffn_down.weight');
  await coordinate.fill('-1');
  await coordinate.press('Tab');
  await expect(coordinate).toHaveAccessibleDescription('Coordinates are whole numbers from 0, separated by commas.');
  await coordinate.fill('0, 1');
  await page.getByRole('textbox', { name: 'Value, edit 1' }).fill('0.25');
  const picker = page.getByRole('combobox', { name: 'Source entry' });
  await picker.click();
  page.once('dialog', (dialog) => dialog.dismiss());
  await page.getByRole('option', { name: /^Fixture checkpoint/ }).click();
  await expect(picker).toHaveValue('Fixture GGUF');
  await expect(page).toHaveURL(/#\/weights\?source=gguf-a$/);
  await expect(coordinate).toHaveValue('0, 1');
  await page.getByRole('button', { name: 'Add to proposal' }).click();
  await page.getByRole('button', { name: 'Read tensor list', exact: true }).click();
  await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
  await page.getByRole('button', { name: 'Read tile', exact: true }).click();
  await page.getByRole('gridcell').first().click();
  await page.getByRole('button', { name: 'Raise by 1 stored step', exact: true }).click();
  const draft = page.getByRole('region', { name: /Proposal \(2 of 256 edits\)/ });
  await expect(draft.getByRole('row')).toHaveCount(3);
  await expect(page.getByRole('button', { name: 'Create proposal', exact: true })).toHaveCount(1);
  await draft.getByRole('button', { name: 'Create proposal' }).click();
  await expect.poll(() => controller.requests('POST', '/api/jobs').map((r) => r.body).filter((b) => b.kind === 'weight_propose')).toEqual([{
    kind: 'weight_propose', host: 'worker-a', input_ids: ['gguf-a'],
    options: { edits: [
      { tensor: 'blk.0.ffn_down.weight', index: [0, 1], op: 'set', value: 0.25 },
      { tensor: 'blk.0.ffn_down.weight', index: [0, 0], op: 'set', value: 0.5000000596046448 },
    ], source_sha256: 'a'.repeat(64) },
  }]);
});

test('Weights states the upload and worker calibration limits together', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page, 'weights?source=downloaded');
  await page.getByRole('radio', { name: 'Automatic' }).check();
  const copy = page.getByText(/1,024 rows/);
  await expect(copy).toHaveCount(1);
  await expect(copy).toContainText('16,384 columns');
  await expect(copy).toContainText('256 KiB');
});

test('Library and Weights use consistent entry and calibration labels', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page);
  await expect.soft(page.getByRole('heading', { name: 'Register an entry' })).toBeVisible();
  await expect.soft(page.getByRole('heading', { name: /^Library entries/ })).toBeVisible();
  await controller.open(page, 'weights?source=downloaded');
  await expect.soft(page.getByRole('combobox', { name: 'Source entry' })).toHaveValue('Verified GGUF');
  await page.getByRole('radio', { name: 'Automatic' }).check();
  await expect(page.getByLabel('Upload calibration data (JSON)')).toBeVisible();
});

test('Running jobs have AA contrast and can collapse on mobile', async ({ page, controller }, info) => {
  await controller.open(page);
  const strip = page.getByRole('region', { name: 'Running jobs' });
  const running = strip.getByText('Running', { exact: true });
  await expect(running).toBeVisible();
  const { violations } = await new AxeBuilder({ page })
    .include('[aria-label="Running jobs"]').withRules(['color-contrast']).analyze();
  expect.soft(violations.map((v) => v.id)).toEqual([]);
  if (info.project.use.viewport.width > 760) return;
  const toggle = strip.getByRole('button', { name: 'Job details' });
  await toggle.press('Enter');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(running).toBeHidden();
  await expect(strip).toContainText('2 running');
  await toggle.press('Enter');
  await expect(running).toBeVisible();
});

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
