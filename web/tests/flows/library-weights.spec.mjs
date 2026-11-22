import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { libraryWeights, verifiedHash } from './mock/library-weights.mjs';

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
  const form = page.getByRole('region', { name: 'Register a source' });
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

test('Weights labels unusable entries and blocks a missing entry opened by a deep link', async ({ page, controller }) => {
  libraryWeights(controller);
  await controller.open(page, 'weights');
  await expect(page.getByRole('radiogroup', { name: 'Editing mode' })).toHaveCount(0);
  const picker = page.getByRole('combobox', { name: 'Source model' });
  await picker.click();
  await expect(page.getByRole('option', { name: /Missing GGUF.*The source file does not exist/ })).toHaveAttribute('aria-disabled', 'true');
  await expect(page.getByRole('option', { name: /Fixture GGUF copy.*weight_propose/ })).toHaveAttribute('aria-disabled', 'true');
  await controller.open(page, 'weights?source=missing');
  await expect(page.getByText('The source file does not exist.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /Read tensor|Create proposal|Tune selected weights/ })).toHaveCount(0);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
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
  const picker = page.getByRole('combobox', { name: 'Source model' });
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
