import AxeBuilder from '@axe-core/playwright';
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures.mjs';

const wcag = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];
const SETTINGS = '/api/settings/agent-provider';
const KEY = '/api/settings/agent-provider/key';
const SESSIONS = '/api/agent/sessions';
const SECRET = 'sk-synthetic-provider-key-0001';
const provider = {
  kind: 'anthropic', base_url: 'https://api.anthropic.com/v1', model: 'claude-fixture',
  max_tokens: 1024, max_steps: 12, max_session_tokens: 20000, timeout_seconds: 180, request_timeout_seconds: 30,
};

async function axe(page) {
  const { violations } = await new AxeBuilder({ page }).include('main').withTags(wcag).analyze();
  expect(violations.map(({ id, nodes }) => ({ id, targets: nodes.map((n) => n.target) }))).toEqual([]);
}

async function shot(page, info, name) {
  const dir = process.env.W15_SHOTS;
  if (!dir) return;
  mkdirSync(dir, { recursive: true });
  await page.screenshot({ path: `${dir}/${name}-${info.project.name}.png`, fullPage: true });
}

/** Stateful provider settings on top of the mock controller. */
function providerStore(controller, initial = { provider: null, key_configured: false }) {
  const state = structuredClone(initial);
  controller.respond('GET', SETTINGS, () => ({ json: state }));
  controller.respond('PUT', SETTINGS, ({ body }) => {
    if (state.provider && (state.provider.kind !== body.kind || state.provider.base_url !== body.base_url)) state.key_configured = false;
    state.provider = body;
    return { json: state };
  });
  controller.respond('PUT', KEY, () => {
    if (!state.provider) return { status: 409, json: { error: 'Save a provider before you set its key' } };
    state.key_configured = true;
    return { json: state };
  });
  controller.respond('DELETE', KEY, () => { state.key_configured = false; return { json: state }; });
  return state;
}

test('Settings saves the provider and keeps the API key write-only', async ({ page, controller }, info) => {
  providerStore(controller);
  await controller.open(page, 'settings');
  const panel = page.getByRole('region', { name: 'Agent model provider' });
  await expect(panel.getByText('Not configured', { exact: true })).toBeVisible();
  await expect(panel.getByRole('button', { name: 'Set key' })).toBeDisabled();
  await expect(panel.getByText('Save the provider before you set its API key.')).toBeVisible();

  await panel.getByLabel('Model ID').fill('claude-fixture');
  await panel.getByRole('button', { name: 'Save provider' }).click();
  await expect(panel.getByRole('status').filter({ hasText: 'Provider saved.' })).toBeVisible();
  expect(controller.requests('PUT', SETTINGS).map((r) => r.body)).toEqual([provider]);

  const keyField = panel.getByLabel('New API key');
  await expect(keyField).toHaveAttribute('type', 'password');
  await keyField.fill(SECRET);
  await panel.getByRole('button', { name: 'Set key' }).click();
  await expect(panel.getByRole('status').filter({ hasText: 'Key saved.' })).toBeVisible();
  await expect(keyField).toHaveValue('');
  expect(controller.requests('PUT', KEY).map((r) => r.body)).toEqual([{ key: SECRET }]);
  await expect(panel.getByText('Configured', { exact: true })).toBeVisible();
  expect(await page.content()).not.toContain(SECRET);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }))).not.toContain(SECRET);
  await shot(page, info, 'settings-provider');
  await axe(page);

  await panel.getByRole('button', { name: 'Remove key' }).click();
  expect(controller.requests('DELETE', KEY)).toHaveLength(0);
  await panel.getByRole('button', { name: 'Confirm removal' }).click();
  await expect(panel.getByRole('status').filter({ hasText: 'Key removed.' })).toBeVisible();
  expect(controller.requests('DELETE', KEY)).toHaveLength(1);
  await expect(panel.getByText('Not configured', { exact: true })).toBeVisible();
});

test('Settings warns that an endpoint change removes the key and blocks bad limits', async ({ page, controller }) => {
  providerStore(controller, { provider, key_configured: true });
  await controller.open(page, 'settings');
  const panel = page.getByRole('region', { name: 'Agent model provider' });
  await expect(panel.getByLabel('Model ID')).toHaveValue('claude-fixture');
  await panel.getByLabel('Provider kind').selectOption('openai');
  await expect(panel.getByLabel('Base URL')).toHaveValue('https://api.openai.com/v1');
  await expect(panel.getByText(/the controller removes the saved API key/)).toBeVisible();

  const steps = panel.getByLabel('Maximum requests');
  await steps.fill('40');
  await steps.blur();
  await expect(steps).toHaveAttribute('aria-invalid', 'true');
  await panel.getByRole('button', { name: 'Save provider' }).click();
  await expect(panel.getByRole('list', { name: 'Changes needed' })).toContainText('1 to 32');
  expect(controller.requests('PUT', SETTINGS)).toHaveLength(0);

  await steps.fill('8');
  await panel.getByRole('button', { name: 'Save provider' }).click();
  await expect(panel.getByRole('status').filter({ hasText: 'Provider saved. Set the API key again.' })).toBeVisible();
  expect(controller.requests('PUT', SETTINGS)[0].body).toMatchObject({ kind: 'openai', base_url: 'https://api.openai.com/v1', max_steps: 8 });
  await expect(panel.getByText('Not configured', { exact: true })).toBeVisible();
});

test('Agent starts a session, follows it live, opens scratch edits and cancels it', async ({ page, controller }, info) => {
  providerStore(controller, { provider, key_configured: true });
  const id = '00000000-0000-4000-8000-000000000001';
  const detail = {
    id, goal: 'Lower refusals on layer 12', model_id: 'source-a', status: 'running', steps: 0, tokens: 0,
    scratch_ids: [], message: null, entries: [],
  };
  const sessions = [];
  controller.respond('GET', SESSIONS, () => ({ json: sessions.map(({ entries, ...s }) => s) }));
  controller.respond('POST', SESSIONS, ({ body, headers }) => {
    expect(headers['idempotency-key']).toMatch(/^[A-Za-z0-9._-]+$/);
    Object.assign(detail, body);
    sessions.unshift(detail);
    return { status: 201, json: detail };
  });
  let reads = 0;
  controller.respond('GET', `${SESSIONS}/${id}`, () => {
    reads += 1;
    if (reads === 2) {
      detail.steps = 2;
      detail.tokens = 830;
      detail.entries = [
        { kind: 'assistant', text: 'I will trace layer 12 first.' },
        { kind: 'tool_call', name: 'apply_edit', arguments: { layer: 12, method: 'rank_one' } },
        { kind: 'tool_result', name: 'apply_edit', result: { id: 'rev-fixture-1' } },
      ];
      detail.scratch_ids = ['rev-fixture-1'];
    }
    return { json: detail };
  });
  controller.respond('POST', `${SESSIONS}/${id}/cancel`, () => {
    detail.status = 'cancelled';
    detail.message = 'The operator cancelled the session.';
    return { json: detail };
  });

  await controller.open(page, 'agent');
  await expect(page.getByRole('heading', { name: 'Agent', level: 1 })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: 'Agent' })).toHaveAttribute('aria-current', 'page');
  await expect(page.getByText('Provider: Anthropic, claude-fixture.')).toBeVisible();
  await page.getByRole('button', { name: 'Start session' }).click();
  await expect(page.getByRole('list', { name: 'Changes needed' })).toContainText('Enter a goal');
  expect(controller.requests('POST', SESSIONS)).toHaveLength(0);

  await page.getByLabel('Goal').fill('Lower refusals on layer 12');
  await page.getByRole('combobox', { name: 'Target model', exact: true }).click();
  await page.getByRole('option', { name: /Fixture checkpoint/ }).click();
  await page.getByRole('button', { name: 'Start session' }).click();
  await expect.poll(() => controller.requests('POST', SESSIONS).map((r) => r.body)).toEqual([
    { goal: 'Lower refusals on layer 12', model_id: 'source-a' },
  ]);
  await expect(page).toHaveURL(new RegExp(`#/agent\\?session=${id}$`));

  const transcript = page.getByRole('region', { name: /^Session [0-9a-f]{8}$/ });
  await expect(transcript.getByText('Running', { exact: true })).toBeVisible();
  await expect(transcript.getByText('I will trace layer 12 first.')).toBeVisible({ timeout: 10_000 });
  await expect(transcript.getByText('Tool call: apply_edit')).toBeVisible();
  await expect(transcript.getByText('"method": "rank_one"')).toBeVisible();
  await expect(transcript.getByText('2 of 12 requests')).toBeVisible();
  const link = transcript.getByRole('link', { name: /Open revision rev-fixt/ });
  await expect(link).toHaveAttribute('href', '#/edits?rev=rev-fixture-1');
  await expect(transcript.getByText(/Only an operator approves a publish request in Edits/)).toBeVisible();
  await shot(page, info, 'agent-running');
  await axe(page);

  await transcript.getByRole('button', { name: 'Cancel session' }).click();
  expect(controller.requests('POST', `${SESSIONS}/${id}/cancel`)).toHaveLength(0);
  await transcript.getByRole('button', { name: 'Confirm cancel' }).click();
  await expect(transcript.getByText('Cancelled', { exact: true })).toBeVisible();
  await expect(transcript.getByText('The operator cancelled the session.')).toBeVisible();
  expect(controller.requests('POST', `${SESSIONS}/${id}/cancel`).map((r) => r.body)).toEqual([{}]);
  await expect(transcript.getByRole('button', { name: 'Cancel session' })).toHaveCount(0);

  await link.click();
  await expect(page.getByRole('heading', { name: 'Edit sessions', level: 1 })).toBeVisible();
  await expect(page).toHaveURL(/#\/edits\?rev=rev-fixture-1$/);
});

test('Agent tells the operator to configure the provider and shows a controller rejection', async ({ page, controller }, info) => {
  controller.respond('POST', SESSIONS, { status: 409, json: { error: 'Set a provider and its API key in Settings' } });
  await controller.open(page, 'agent');
  await expect(page.getByRole('link', { name: 'Set the agent provider in Settings' })).toHaveAttribute('href', '#/settings');
  await expect(page.getByText('No agent sessions.')).toBeVisible();
  await expect(page.getByText('The controller sends the goal and tool results to the hosted provider. Do not include secrets.')).toBeVisible();
  await page.getByLabel('Goal').fill('Find the refusal direction');
  await page.getByRole('combobox', { name: 'Target model', exact: true }).click();
  // GGUF targets are allowed for traces.
  await page.getByRole('option', { name: /^Fixture GGUF\b/ }).first().click();
  await page.getByRole('button', { name: 'Start session' }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Set a provider and its API key in Settings' })).toBeVisible();
  expect(controller.requests('POST', SESSIONS).map((r) => r.body)).toEqual([{ goal: 'Find the refusal direction', model_id: 'gguf-a' }]);
  await shot(page, info, 'agent-empty');
  await axe(page);
});
