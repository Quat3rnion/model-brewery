import { test, expect } from './fixtures.mjs';
import { convertState } from './mock/convert.mjs';

async function pick(page, id, option) {
  await page.locator(id).click();
  await page.getByRole('option', { name: option }).click();
}

test('managed conversion requires all four stages, including when the pool is empty', async ({ page, controller }) => {
  controller.respond('GET', '/api/state', { json: convertState() });
  await controller.open(page, 'convert');
  await pick(page, '#wf-source', /^Fixture GGUF copy/);
  await page.locator('#wf-profile').selectOption('v2');
  await expect(page.getByRole('button', { name: 'Start conversion', exact: true })).toBeDisabled();
  await expect(page.getByRole('link', { name: /Enable Verify on Worker B/ })).toBeVisible();
  await page.locator('#wf-host').click();
  await expect(page.getByRole('option', { name: /^Worker B/ })).toBeDisabled();
  await page.keyboard.press('Escape');
  await page.getByRole('checkbox', { name: 'Any eligible pool worker' }).check();
  await expect(page.getByRole('button', { name: 'Start conversion', exact: true })).toBeDisabled();
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
});

test('managed conversion keeps diagnostics visible when no host supports the workflow', async ({ page, controller }) => {
  controller.load('capabilities-off');
  await controller.open(page, 'convert');
  await expect(page.getByRole('heading', { name: 'Setup required' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start conversion', exact: true })).toHaveCount(0);
  await expect(page.getByRole('link', { name: /verify is not configured/i }).first())
    .toHaveAttribute('href', /#\/settings\/infrastructure\?host=/);
  await expect(page.getByText('Individual steps (diagnostics)', { exact: true })).toBeVisible();
});

test('GGUF uses only the source host; mode selections stay separate and the URL stores the mode', async ({ page, controller }) => {
  controller.respond('GET', '/api/state', { json: convertState() });
  await controller.open(page, 'convert');
  await pick(page, '#wf-source', /^Fixture checkpoint/);
  await page.getByRole('radio', { name: 'GGUF', exact: true }).check();
  await expect(page).toHaveURL(/convert\?format=gguf$/);
  await expect(page.locator('#gg-source')).toHaveValue('');
  await pick(page, '#gg-source', /^Fixture GGUF copy/);
  await expect(page.getByText('Worker B', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('link', { name: 'Open Transfers' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start GGUF conversion' })).toBeDisabled();
  await expect(page.getByRole('combobox', { name: 'Run on' })).toHaveCount(0);
  await page.getByRole('radio', { name: 'Layer recipe' }).check();
  await expect(page.locator('#recipe-source')).toHaveValue('');
  await page.getByRole('radio', { name: 'Managed HGN' }).check();
  await expect(page.locator('#wf-source')).toHaveValue('Fixture checkpoint');
  await page.getByRole('radio', { name: 'GGUF', exact: true }).check();
  await expect(page.locator('#gg-source')).toHaveValue('Fixture GGUF copy');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('layer recipes put host setup first and show one geometry warning after an explicit read', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page, 'convert?format=recipe');
  await pick(page, '#recipe-source', /^Fixture checkpoint/);
  const hostReason = page.getByRole('link', { name: /Worker A:.*hgn_layer_recipe/i });
  const supported = page.getByText(/^Supported architectures: dense Llama and Qwen2/);
  await expect(hostReason).toBeVisible();
  await expect(supported).toBeVisible();
  const first = await hostReason.elementHandle();
  expect(await supported.evaluate((node, earlier) => !!(earlier.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING), first)).toBeTruthy();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await page.getByRole('button', { name: 'Read layers', exact: true }).click();
  await expect(page.getByText(/HT3 and HT4 are not available for some layers/)).toHaveCount(1);
  await expect(page.getByLabel('Layer 0 type')).toHaveAttribute('aria-describedby', 'recipe-ht-help');
  await page.getByLabel('Recipe output').selectOption('gguf');
  await expect(page).toHaveURL(/convert\?format=recipe&output=gguf$/);
  await expect(page.getByText(/^Supported architectures: GPT-2 and Qwen2/)).toBeVisible();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(1);
});

test('custom profiles mark unknown source hashes and check the exact selection without queuing', async ({ page, controller }) => {
  let checks = 0;
  controller.respond('POST', '/api/conversions/eligibility', () => {
    checks++;
    return { json: { eligible: false, reasons: [{ text: 'Fixture exact admission problem', fix_anchor: '#/library' }] } };
  });
  await controller.open(page, 'convert');
  await pick(page, '#wf-source', /^Fixture checkpoint/);
  await page.locator('#wf-profile').selectOption(`rev:${'a'.repeat(64)}`);
  await expect(page.getByText('Source checkpoint unverified.', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: /Fixture exact admission problem/ })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start conversion', exact: true })).toBeDisabled();
  expect(checks).toBe(1);
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('Hy4 is inside diagnostics and does not use the managed source', async ({ page, controller }) => {
  await controller.open(page, 'convert');
  await pick(page, '#wf-source', /^Fixture checkpoint/);
  await expect(page.getByText('Hy4 metadata plan', { exact: true })).toBeHidden();
  await page.getByText('Individual steps (diagnostics)', { exact: true }).click();
  await page.getByText('Hy4 metadata plan', { exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Hy4 source' })).toHaveValue('');
});

test('a job can reopen its diagnostic form without queuing a replacement', async ({ page, controller }) => {
  await controller.open(page, 'convert?step=plan&model=source-a&host=worker-a&profile=ht43');
  const diagnostics = page.getByRole('tabpanel', { name: 'Plan' });
  await expect(diagnostics.getByRole('combobox', { name: 'Source checkpoint' })).toHaveValue('Fixture checkpoint');
  await expect(diagnostics.getByRole('combobox', { name: 'Run on' })).toHaveValue('Worker A');
  await expect(diagnostics.getByLabel('Profile')).toHaveValue('ht43');
  await expect(page.locator('#wf-source')).toHaveValue('');
  expect(controller.requests('POST')).toHaveLength(0);
});

test('the success message links to the submitted conversion', async ({ page, controller }) => {
  await controller.open(page, 'convert');
  await pick(page, '#wf-source', /^Fixture checkpoint/);
  await page.locator('#wf-profile').selectOption('v2');
  await page.getByRole('button', { name: 'Start conversion', exact: true }).click();
  const run = page.getByRole('link', { name: 'conversion-mock-1', exact: true });
  await expect(run).toHaveAttribute('href', '#/convert?run=conversion-mock-1');
  await run.click();
  await expect(page.locator('#run-conversion-mock-1')).toBeFocused();
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(1);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('running and finished managed conversions stay distinct from Convert step jobs', async ({ page, controller }) => {
  await controller.open(page, 'convert');
  await expect(page.getByText('Verified bundle ready', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Verify job verify-stage' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Inspect job running-job' })).toBeVisible();
  await expect(page.getByRole('link', { name: /Convert step/ })).toBeVisible();
  await page.getByRole('button', { name: 'Cancel conversion active-conversion' }).click();
  await page.getByRole('button', { name: 'Keep running' }).click();
  await expect(page.getByRole('button', { name: 'Cancel conversion active-conversion' })).toBeFocused();
  expect(controller.requests('POST')).toHaveLength(0);
});

test('Profiles asks before Close, Start from, Edit or switching format', async ({ page, controller }) => {
  await controller.open(page, 'profiles');
  await page.getByRole('button', { name: 'Start from HGN v2' }).click();
  const name = page.getByLabel('Profile name', { exact: true });
  await name.fill('Draft name only');
  let prompts = 0;
  page.on('dialog', async (dialog) => {
    expect(dialog.message()).toBe('Discard unsaved changes?');
    if (++prompts === 5) await dialog.accept();
    else await dialog.dismiss();
  });
  await page.getByRole('button', { name: 'Close editor' }).click();
  await page.getByRole('button', { name: 'Start from HT43' }).click();
  await page.getByRole('button', { name: /Edit Fixture HGN profile/ }).click();
  await page.getByRole('radio', { name: /GGUF/ }).click();
  await expect(name).toHaveValue('Draft name only');
  await expect(page).toHaveURL(/#\/profiles$/);
  expect(prompts).toBe(4);
  await page.getByRole('button', { name: 'Close editor' }).click();
  await expect(name).toBeHidden();
  await expect(page.getByRole('button', { name: 'Start from HGN v2' })).toBeFocused();
  await page.getByRole('radio', { name: /HGN/ }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page).toHaveURL(/#\/profiles\?format=gguf$/);
  await expect(page.getByRole('radio', { name: /GGUF/ })).toBeChecked();
});

test('GGUF Profiles keeps a draft when Edit is dismissed', async ({ page, controller }) => {
  await controller.open(page, 'profiles?format=gguf');
  const name = page.getByLabel('Profile name', { exact: true });
  await name.fill('Unsaved GGUF name');
  page.once('dialog', (dialog) => dialog.dismiss());
  await page.getByRole('button', { name: /Edit Fixture GGUF profile/ }).click();
  await expect(name).toHaveValue('Unsaved GGUF name');
});
