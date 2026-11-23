import AxeBuilder from '@axe-core/playwright';
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures.mjs';

const wcagAA = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];
const SHA = 'b'.repeat(64);
const SHA_P = 'a'.repeat(64);
const XC = 'c0ffee'.padEnd(64, '0');
const PARENT = 'model-parent';
const CHILD = '00000000-0000-4000-8000-000000000021';
const SHOTS = process.env.X6_SHOTS ?? '/arboretum/tmp/x6-shots';

const scratchRev = {
  id: CHILD, kind: 'scratch', state: 'scratch', sha256: SHA, qualification: 'not release-qualified', library_model_id: null,
  provenance: {
    schema: 'brewery.scratch-provenance.v1', parent_id: PARENT, parent_sha256: SHA_P, revision_id: CHILD, revision_sha256: SHA, worker: 'frame', method: 'direct',
    operations: [{ op: 'add', tensor: 'blk.3.ffn_down.weight', origin: [0, 0], shape: [1, 2], values: [0.01, -0.02] }],
    tensors: [{ name: 'blk.3.ffn_down.weight', shape: [896, 4864], dtype: 'BF16' }], tool: 'brewery-weight-edit', tool_version: '0.4.1', runtime_version: 'lanternfish fixture', created_at_ms: 1_790_000_000_000,
  },
  delta: { schema: 'brewery.weight-delta.v1', parent_sha256: SHA_P, revision_sha256: SHA, changed_tensors: 1, changed_bytes: 4, delta_bytes: 128, delta_sha256: SHA },
};

// Deterministic synthetic features: every class, null values and clamped ratios.
const classes = ['shared', 'shared', 'shared', 'changed', 'parent_unique', 'child_unique', 'shared', 'dead'];
const features = Array.from({ length: 96 }, (_, feature) => {
  const classification = classes[feature % classes.length];
  const wave = ((feature * 37) % 23) / 23;
  const [parent_norm, child_norm, decoder_cosine] = {
    shared: [1 + wave, 1 + wave * 1.05, 0.9 + wave * 0.09],
    changed: [1, 0.6 + wave, 0.1 + wave * 0.4],
    parent_unique: [0.8 + wave, feature === 4 ? 0 : 0.02 * wave, feature === 4 ? null : 0.2 * wave],
    child_unique: [feature === 5 ? 0 : 0.01, 0.9 + wave, feature === 5 ? null : -0.1 + 0.3 * wave],
    dead: [0, 0, null],
  }[classification];
  return { feature, classification, parent_norm, child_norm, decoder_cosine, child_norm_ratio: parent_norm + child_norm === 0 ? null : child_norm / (parent_norm + child_norm) };
});
const report = {
  id: XC, schema_version: 1, version: 1, qualification: 'not_release_qualified',
  parent: { id: PARENT, sha256: SHA_P }, child: { id: CHILD, sha256: SHA }, layer: 6, role: 'l_out',
  width: 896, k: 8, method: 'batchtopk',
  pairs: [0, 1, 2, 3].map((i) => ({ parent_job_id: `trace-p${i}`, child_job_id: `trace-c${i}`, split: i < 3 ? 'train' : 'heldout' })), weights_sha256: 'd'.repeat(64), weights_bytes: 3_538_944,
  metrics: {
    parent: { mse: 0.0123, explained_variance: 0.912, l0: 7.9, dead_features: 12 },
    child: { mse: 0.0141, explained_variance: 0.897, l0: 8.1, dead_features: 12 },
  },
  features,
};

function mockReport(controller, body = report, status = 200) {
  controller.respond('GET', '/api/scratch', { json: [scratchRev] });
  controller.respond('GET', `/api/crosscoders/${XC}`, { status, json: body });
}

async function scan(page, info) {
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  const { violations } = await new AxeBuilder({ page }).withTags(wcagAA).analyze();
  if (violations.length) await info.attach('axe-violations', { body: JSON.stringify(violations, null, 2), contentType: 'application/json' });
  expect(violations.flatMap(({ id, nodes }) => nodes.map(({ target }) => ({ id, target })))).toEqual([]);
}

test('Edits opens a crosscoder report for the revision and the report reads only with GET', async ({ page, controller }, info) => {
  mockReport(controller);
  await controller.open(page, `edits?rev=${CHILD}`);
  const section = page.getByRole('region', { name: 'Crosscoder diff' });
  const input = section.getByRole('textbox', { name: 'Crosscoder report ID' });
  await input.fill('not-an-id');
  await section.getByRole('button', { name: 'Open report' }).click();
  await expect(input).toHaveAttribute('aria-invalid', 'true');
  await expect(section.getByText('Enter the 64-character hexadecimal report ID.')).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`#/edits\\?rev=${CHILD}$`));

  await input.fill(` ${XC.toUpperCase()} `);
  await section.getByRole('button', { name: 'Open report' }).click();
  await expect(page).toHaveURL(new RegExp(`#/edits/crosscoder\\?id=${XC}&rev=${CHILD}$`));
  const main = page.getByRole('main');
  await expect(main.getByRole('heading', { level: 1, name: 'Crosscoder diff' })).toBeFocused();
  await expect(page.getByRole('navigation', { name: 'Sections' }).getByRole('link', { name: 'Edits' })).toHaveAttribute('aria-current', 'page');
  await expect(main.getByText('Not release-qualified').first()).toBeVisible();

  const metrics = main.getByRole('table', { name: 'Reconstruction metrics' });
  await expect(metrics.getByRole('row', { name: /Mean squared error/ })).toContainText('0.0123');
  await expect(metrics.getByRole('row', { name: /Mean squared error/ })).toContainText('0.0141');
  await expect(metrics.getByRole('row', { name: /Explained variance/ })).toContainText('91.2%');

  const counts = main.getByRole('list', { name: 'Features per class' });
  await expect(counts.getByRole('listitem').filter({ hasText: 'Unique to parent' })).toContainText('12');
  await expect(counts.getByRole('listitem').filter({ hasText: 'Shared' })).toContainText('48');

  const plot = main.getByRole('img', { name: /Scatter of 82 features/ });
  await expect(plot).toBeVisible();
  await expect(main.getByText('14 features have no child share or cosine and are not plotted.', { exact: false })).toBeVisible();
  await expect(main.getByText('batchtopk, 96 features, k = 8')).toBeVisible();
  await expect(main.getByText('4 (3 train, 1 held out)')).toBeVisible();
  await expect(main.getByRole('definition').filter({ hasText: /^896$/ })).toBeVisible();

  const changed = main.getByRole('region', { name: /^Changed/ });
  await expect(changed.getByRole('table')).toBeVisible();
  await expect(changed.getByRole('row')).toHaveCount(13);
  await changed.getByRole('searchbox', { name: 'Feature number' }).fill('11');
  await expect(changed.getByRole('row')).toHaveCount(2);
  await expect(changed.getByRole('rowheader', { name: '11', exact: true })).toBeVisible();
  await changed.getByRole('searchbox', { name: 'Feature number' }).fill('999');
  await expect(changed.getByText('No feature matches the filter.')).toBeVisible();

  const parentUnique = main.getByRole('region', { name: /^Unique to parent/ });
  await expect(parentUnique.getByRole('rowheader').first()).toBeVisible();
  await parentUnique.getByRole('combobox', { name: 'Sort' }).selectOption('feature');
  await expect(parentUnique.getByRole('rowheader').first()).toHaveText('4');

  const download = main.getByRole('link', { name: /Download crosscoder weights/ });
  await expect(download).toHaveAttribute('href', `/api/crosscoders/${XC}/safetensors`);
  await expect(main.getByRole('link', { name: /Back to the revision/ })).toHaveAttribute('href', `#/edits?rev=${CHILD}`);

  expect(controller.requests().filter((r) => r.method !== 'GET')).toEqual([]);
  await scan(page, info);
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: `${SHOTS}/report-${info.project.name}.png`, fullPage: true });
});

test('crosscoder report shows a missing report, a bad schema and a child that is not the revision', async ({ page, controller }, info) => {
  mockReport(controller, { error: 'crosscoder not found' }, 404);
  await controller.open(page, `edits/crosscoder?id=${XC}`);
  const main = page.getByRole('main');
  await expect(main.getByRole('alert')).toContainText('crosscoder not found');

  controller.respond('GET', `/api/crosscoders/${XC}`, { json: { ...report, schema_version: 2 } });
  await page.reload();
  await expect(main.getByRole('alert')).toContainText('schema version 1. The report has version 2.');

  controller.respond('GET', `/api/crosscoders/${XC}`, { json: report });
  await controller.open(page, `edits/crosscoder?id=${XC}&rev=other-rev`);
  await page.reload();
  await expect(main.getByText('This report is not for revision other-rev.', { exact: false })).toBeVisible();

  await controller.open(page, 'edits/crosscoder?id=xyz');
  await page.reload();
  await expect(main.getByText('The report ID is not valid.', { exact: false })).toBeVisible();
  expect(controller.requests('GET', '/api/crosscoders/xyz')).toEqual([]);
  await scan(page, info);
});
