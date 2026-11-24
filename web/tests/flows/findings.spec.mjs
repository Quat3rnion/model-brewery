import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';

const wcagAA = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];
const SHA = 'e'.repeat(64);
const HYP = 'Layer 6 MLP stores the capital-city fact for the fixture prompt.';
const model = { id: 'gguf-a', sha256: SHA };
const base = { hypothesis: HYP, created_at_ms: 1_790_000_000_000, model };

const experiments = [
  { spec: { kind: 'trace', layers: [6] }, artifact: { id: 'trace-1', job_id: 'trace-1', sha256: SHA, api_url: '/api/jobs/trace-1', view_url: '#/jobs/trace-1' }, metrics: [{ path: 'logit_diff', value: 2.5 }, { path: 'top1_prob', value: 0.61 }] },
  { spec: { kind: 'activation_patch', layer: 6 }, artifact: { id: 'patch-1', job_id: 'patch-1', sha256: SHA, api_url: '/api/jobs/patch-1', view_url: '#/jobs/patch-1' }, metrics: [{ path: 'recovered_fraction', value: 0.8 }] },
];
const provenance = { agent: { kind: 'builtin', id: 'session-9' }, tool_version: '0.9.0' };
const v1 = { ...base, id: 'fnd-1', version: 1, parent_id: null, state: 'ready', verdict: { status: 'supported', reasoning: 'Patching layer 6 recovers 80% of the logit difference.' }, experiments, provenance, comparison: [] };
const replay = {
  ...base, id: 'fnd-2', version: 2, parent_id: 'fnd-1', replay_of: 'fnd-1', state: 'ready',
  verdict: { status: 'inconclusive', reasoning: 'The replay recovers only 60%. The effect is weaker than the original.' },
  experiments: experiments.map((e, i) => (i === 1 ? { ...e, metrics: [{ path: 'recovered_fraction', value: 0.6 }] } : e)),
  provenance: { agent: { kind: 'external_mcp', id: 'claude-desktop' }, tool_version: '0.9.0' },
  comparison: [
    { experiment: 1, path: 'logit_diff', original: 2.5, replay: 2.5 },
    { experiment: 1, path: 'top1_prob', original: 0.61, replay: 0.61 },
    { experiment: 2, path: 'recovered_fraction', original: 0.8, replay: 0.6 },
  ],
};
const running = { ...base, id: 'fnd-3', version: 1, hypothesis: 'Head 4.2 copies the subject token.', parent_id: null, state: 'running', verdict: null, experiments: [], provenance, comparison: [] };
const failed = { ...base, id: 'fnd-4', version: 1, hypothesis: 'A very long hypothesis '.repeat(8).trim(), parent_id: null, state: 'failed', verdict: null, experiments: [], provenance, comparison: [], error: 'trace job trace-9 failed: worker exited' };
const all = [v1, replay, running, failed];
const summaryOf = ({ experiments: _e, provenance: _p, comparison: _c, error: _x, ...rest }) => rest;

function mock(controller) {
  controller.respond('GET', '/api/findings', { json: all.map(summaryOf) });
  for (const f of all) controller.respond('GET', `/api/findings/${f.id}`, { json: f });
}

async function scan(page, info) {
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  const { violations } = await new AxeBuilder({ page }).withTags(wcagAA).analyze();
  if (violations.length) await info.attach('axe-violations', { body: JSON.stringify(violations, null, 2), contentType: 'application/json' });
  expect(violations.flatMap(({ id, nodes }) => nodes.map(({ target }) => ({ id, target })))).toEqual([]);
}

test('Findings lists findings and opens a finding with evidence, verdict and provenance', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'findings');
  const main = page.getByRole('main');
  await expect(main.getByRole('heading', { level: 1, name: 'Findings' })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: 'Findings' })).toHaveAttribute('aria-current', 'page');
  await expect(main.getByText('Not release-qualified').first()).toBeVisible();

  const list = main.getByRole('table', { name: 'Findings' });
  await expect(list.getByRole('row')).toHaveCount(5);
  const row2 = list.getByRole('row').filter({ hasText: 'fnd-2' });
  await expect(row2).toContainText('Inconclusive');
  await expect(row2).toContainText('Replay of');
  await expect(row2.getByRole('link', { name: 'fnd-1' })).toHaveAttribute('href', '#/findings/fnd-1');
  await expect(list.getByRole('row').filter({ hasText: 'fnd-3' })).toContainText('Running');
  await expect(list.getByRole('row').filter({ hasText: 'fnd-3' })).toContainText('No verdict');
  await expect(list.getByRole('row').filter({ hasText: 'fnd-4' })).toContainText('Failed');
  await scan(page, info);
  await page.screenshot({ path: info.outputPath('list.png'), fullPage: true });

  await list.getByRole('row').filter({ hasText: 'First version' }).filter({ hasText: 'Supported' }).getByRole('link', { name: HYP }).click();
  await expect(page).toHaveURL(/#\/findings\/fnd-1$/);
  await expect(main.getByRole('heading', { level: 1, name: 'Finding' })).toBeFocused();
  await expect(page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: 'Findings' })).toHaveAttribute('aria-current', 'page');
  const verdict = main.getByRole('region', { name: 'Verdict' });
  await expect(verdict).toContainText('Supported');
  await expect(verdict).toContainText('recovers 80%');
  const timeline = main.getByRole('list', { name: 'Experiments' });
  await expect(timeline.getByRole('listitem')).toHaveCount(2);
  await expect(timeline.getByRole('listitem').first()).toContainText('trace');
  await expect(timeline.getByRole('listitem').first()).toContainText('logit_diff');
  await expect(timeline.getByRole('link', { name: /Open trace-1/ })).toHaveAttribute('href', '#/jobs/trace-1');
  const prov = main.getByRole('region', { name: 'Provenance' });
  await expect(prov).toContainText('Built-in agent');
  await expect(prov.getByRole('link', { name: 'session-9' })).toHaveAttribute('href', '#/agent?session=session-9');
  await expect(main.getByRole('region', { name: 'Replay comparison' })).toHaveCount(0);
  await scan(page, info);
  await page.screenshot({ path: info.outputPath('detail.png'), fullPage: true });
  expect(controller.requests().filter((r) => r.method !== 'GET')).toEqual([]);
});

test('a replay finding shows original and replay values side by side', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'findings/fnd-2');
  const main = page.getByRole('main');
  await expect(main.getByText('Replay of', { exact: false }).first()).toBeVisible();
  const table = main.getByRole('table', { name: /Original and replay values/ });
  await expect(table.getByRole('row')).toHaveCount(4);
  const changed = table.getByRole('row').filter({ hasText: 'recovered_fraction' });
  await expect(changed).toContainText('0.8');
  await expect(changed).toContainText('0.6');
  await expect(changed).toContainText('Changed');
  await expect(table.getByRole('row').filter({ hasText: 'logit_diff' })).toContainText('Same');
  await expect(main.getByText('1 of 3 values changed')).toBeVisible();
  const prov = main.getByRole('region', { name: 'Provenance' });
  await expect(prov).toContainText('External agent (MCP)');
  await expect(prov.getByRole('link', { name: 'claude-desktop' })).toHaveCount(0);
  await scan(page, info);
  await page.screenshot({ path: info.outputPath('replay.png'), fullPage: true });
  expect(controller.requests().filter((r) => r.method !== 'GET')).toEqual([]);
});

test('running, failed, missing and empty findings are shown with text', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'findings/fnd-3');
  const main = page.getByRole('main');
  await expect(main.getByText('No experiments yet.')).toBeVisible();
  await expect(main.getByRole('region', { name: 'Verdict' })).toContainText('No verdict yet');

  await controller.open(page, 'findings/fnd-4');
  await expect(main.getByRole('alert')).toContainText('worker exited');
  await scan(page, info);

  await controller.open(page, 'findings/missing');
  await expect(main.getByRole('alert')).toContainText('API endpoint not found');

  controller.respond('GET', '/api/findings', { json: [] });
  await controller.open(page, 'findings');
  await expect(main.getByText('No findings yet.', { exact: false })).toBeVisible();
  expect(controller.requests().filter((r) => r.method !== 'GET')).toEqual([]);
});
