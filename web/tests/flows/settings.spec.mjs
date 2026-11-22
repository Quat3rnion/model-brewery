import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { adminState, fingerprint, gib, sshHost } from './mock/settings.mjs';

const workers = (page) => page.getByRole('navigation', { name: 'Workers' });
const select = (page) => page.getByLabel('Defined host');
const pause = (page) => page.getByRole('checkbox', { name: 'Pause new work' });
const gsm8k = (page) => page.getByLabel('Allow GSM8K evaluation on this worker');

test('Settings uses one host selection and focuses fix links', async ({ page, controller }) => {
  await controller.open(page, 'settings/evaluation?host=worker-b');
  await expect(workers(page).getByRole('button', { name: /Worker B/ })).toHaveAttribute('aria-current', 'true');
  await expect(page.getByRole('heading', { name: 'Evaluation configuration' })).toBeFocused();
  await expect(select(page)).toHaveValue('worker-b');
  await expect(page.getByRole('group', { name: 'Benchmark targets' })).toHaveCount(0);
  await select(page).selectOption('worker-a');
  await expect(workers(page).getByRole('button', { name: /Worker A/ })).toHaveAttribute('aria-current', 'true');
  await page.getByRole('link', { name: 'Admission and resource limits' }).click();
  await expect(page.getByRole('heading', { name: 'Admission and resource limits' })).toBeFocused();
  await page.evaluate(() => { location.hash = '#/settings/infrastructure?host=worker-b&section=policy'; });
  await expect(select(page)).toHaveValue('worker-b');
  await expect(page.getByRole('heading', { name: 'Admission and resource limits' })).toBeFocused();
  await page.evaluate(() => { location.hash = '#/settings?host=worker-a&section=policy'; });
  await expect(select(page)).toHaveValue('worker-a');
  await expect(page.getByRole('heading', { name: 'Admission and resource limits' })).toBeFocused();
});

test('Settings guards navigation, reload, host selection and refresh and keeps host drafts', async ({ page, controller }) => {
  controller.respond('GET', '/api/admin/state', { json: adminState() });
  await controller.open(page, 'settings');
  await gsm8k(page).uncheck();
  await pause(page).check();
  const dialogs = [];
  let accept = false;
  page.on('dialog', async (dialog) => {
    dialogs.push(dialog.type());
    await (accept ? dialog.accept() : dialog.dismiss());
  });
  await page.getByRole('link', { name: 'Library', exact: true }).click();
  await expect(page).toHaveURL(/#\/settings$/);
  await page.evaluate(() => { location.hash = '#/evaluate'; });
  await expect.poll(() => dialogs.length).toBe(2);
  await expect(page).toHaveURL(/#\/settings$/);
  await page.evaluate(() => { location.reload(); });
  await expect.poll(() => dialogs.at(-1)).toBe('beforeunload');
  await select(page).selectOption('worker-b');
  await expect(select(page)).toHaveValue('worker-a');
  const refresh = page.getByRole('button', { name: 'Refresh infrastructure' });
  await refresh.click();
  await expect.poll(() => dialogs.length).toBe(5);
  expect(controller.requests('GET', '/api/admin/state')).toHaveLength(1);
  accept = true;
  controller.respond('GET', '/api/admin/state', { json: adminState(2) });
  await refresh.click();
  await expect.poll(() => controller.requests('GET', '/api/admin/state').length).toBe(2);
  await expect(pause(page)).toBeChecked();
  await select(page).selectOption('worker-b');
  await expect(pause(page)).not.toBeChecked();
  await workers(page).getByRole('button', { name: /Worker A/ }).click();
  await expect(pause(page)).toBeChecked();
  await expect(gsm8k(page)).not.toBeChecked();
});

test('Settings reviews pool changes, checks resource fields and describes audited limits', async ({ page, controller }) => {
  controller.respond('GET', '/api/admin/state', { json: adminState() });
  controller.respond('POST', '/api/admin/actions', { json: adminState(2, true) });
  await controller.open(page, 'settings');
  const ram = page.getByLabel('Minimum free RAM (GiB)');
  await expect(ram).not.toHaveAttribute('aria-invalid', 'true');
  await ram.fill('17');
  await ram.blur();
  await expect(ram).toHaveAttribute('aria-invalid', 'true');
  await expect(ram).toHaveAccessibleDescription(/Minimum free RAM.*16 GiB/);
  await ram.fill('1.5');
  await page.getByRole('button', { name: 'Review resource changes' }).click();
  await expect(page.getByRole('button', { name: 'Confirm resource limits' })).toBeFocused();
  await page.getByRole('button', { name: 'Back to editing' }).click();
  await expect.soft(page.getByRole('region', { name: 'Audit log' }).getByRole('cell', { name: /Minimum free RAM.*2 GiB/ })).toBeVisible();
  await page.getByRole('button', { name: 'Review pool membership' }).click();
  expect(controller.requests('POST', '/api/admin/actions')).toHaveLength(0);
  await page.getByRole('button', { name: 'Confirm pool membership' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Pool membership saved' })).toBeVisible();
  expect(controller.requests('POST', '/api/admin/actions')[0].body).toEqual({ action: 'set_resource_policy', worker: 'worker-a', allow_pool: true });
  await page.getByRole('button', { name: 'Review resource changes' }).click();
  await page.getByRole('button', { name: 'Confirm resource limits' }).click();
  expect(controller.requests('POST', '/api/admin/actions')[1].body).toMatchObject({ resources: { min_free_ram_bytes: 1.5 * gib } });
});

test('Settings clears the SSH pin for a changed address and probes the new address', async ({ page, controller }) => {
  controller.respond('GET', '/api/admin/ssh-hosts', { json: { hosts: [sshHost], available: true } });
  controller.respond('POST', '/api/admin/ssh-host-key', ({ body }) => ({
    json: { ...body, fingerprint, algorithm: 'ssh-ed25519', public_key: `ssh-ed25519 ${'A'.repeat(68)}` },
  }));
  await controller.open(page, 'settings');
  await page.getByRole('button', { name: 'Edit definition' }).click();
  const pin = page.getByLabel('Host key fingerprint');
  await expect(pin).toHaveValue(fingerprint);
  await page.getByLabel('Host address').fill('new.example');
  await expect(pin).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Save definition' })).toBeDisabled();
  await page.getByRole('button', { name: 'Probe host key' }).click();
  await page.getByRole('button', { name: 'Accept fingerprint' }).click();
  await expect(pin).toHaveValue(fingerprint);
  await page.getByLabel('SSH port').fill('2222');
  await expect(pin).toHaveValue('');
  expect(controller.requests('POST', '/api/admin/ssh-host-key')[0].body).toEqual({ host: 'new.example', port: 22 });
});

test('Settings rejects whitespace-only tokens without a write', async ({ page, controller }) => {
  controller.respond('GET', '/api/settings/hf-tokens', { json: { hosts: [{ id: 'backend-hf', configured: false }] } });
  controller.respond('PUT', '/api/settings/backend-hf/hf-token', { json: { id: 'backend-hf', configured: true } });
  await controller.open(page, 'settings');
  await page.getByLabel('New access token').fill('   ');
  await page.getByRole('button', { name: 'Set token' }).click();
  await expect(page.getByText('Enter an access token.', { exact: true })).toBeVisible();
  expect(controller.requests('PUT', '/api/settings/backend-hf/hf-token')).toHaveLength(0);
  await page.getByLabel('New access token').fill('  synthetic-token  ');
  await page.getByRole('button', { name: 'Set token' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Token saved.' })).toBeVisible();
  expect(controller.requests('PUT', '/api/settings/backend-hf/hf-token')[0].body).toEqual({ token: 'synthetic-token' });
});

test('Settings only shows endpoint errors for fields the user leaves', async ({ page, controller }) => {
  await controller.open(page, 'settings');
  const targets = page.getByRole('group', { name: 'Benchmark targets' });
  await targets.getByRole('button', { name: 'Add target' }).click();
  const id = targets.getByRole('textbox', { name: 'ID', exact: true }).last();
  const model = targets.getByRole('textbox', { name: 'Model name' }).last();
  await expect(page.getByRole('list', { name: 'Changes needed' })).toHaveCount(0);
  await expect(id).not.toHaveAttribute('aria-invalid', 'true');
  await id.focus();
  await id.blur();
  await expect(id).toHaveAttribute('aria-invalid', 'true');
  await expect(model).not.toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByRole('list', { name: 'Changes needed' })).not.toContainText('enter the model name');
});

test('Settings keeps a profile-only worker distinct from a host that never evaluates', async ({ page, controller }) => {
  const profile = { managed: false, evaluation: { status: 'profile_only' }, profile_binding: 'fixture', revision: 1 };
  controller.respond('GET', '/api/settings', { json: { hosts: [
    { ...profile, id: 'worker-a', name: 'Worker A' },
    { ...profile, id: 'backend-hf', name: 'Verified downloads', reason: 'This worker does not run evaluation.' },
  ] } });
  await controller.open(page, 'settings/evaluation?host=backend-hf');
  await expect(page.getByRole('heading', { name: 'Evaluation configuration' })).toBeFocused();
  await expect(page.getByRole('button', { name: 'Add target' })).toHaveCount(0);
  await workers(page).getByRole('button', { name: /Worker A/ }).click();
  await expect(page.getByRole('button', { name: 'Add target' })).toBeVisible();
});

test('Settings controls and populated audit log have no WCAG A/AA violations', async ({ page, controller }) => {
  controller.respond('GET', '/api/admin/state', { json: adminState() });
  await controller.open(page, 'settings');
  await expect(page.getByRole('cell', { name: /Minimum free RAM.*2 GiB/ })).toBeVisible();
  const editor = page.getByRole('region', { name: 'Weight editing', exact: true });
  await expect(editor.getByText('The controller did not report how to set up weight editing on this host.')).toBeVisible();
  await expect(editor.getByRole('textbox')).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: 'Add a directory' })).toHaveCount(1);
  await page.getByRole('button', { name: 'Review pool membership' }).click();
  const { violations } = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze();
  expect(violations.map(({ id, nodes }) => ({ id, targets: nodes.map(n => n.target) }))).toEqual([]);
});
