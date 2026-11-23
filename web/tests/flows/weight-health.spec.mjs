import AxeBuilder from '@axe-core/playwright';
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures.mjs';

const SHA = 'a'.repeat(64);
const SHA_P = 'b'.repeat(64);
const REV = '11111111-1111-4111-8111-111111111111';
const wcagAA = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];

const rev = {
  id: REV, kind: 'scratch', state: 'scratch', sha256: SHA, qualification: 'not release-qualified', library_model_id: null,
  provenance: {
    parent_id: 'source-a', parent_sha256: SHA_P, revision_id: REV, revision_sha256: SHA, worker: 'worker-a', method: 'direct',
    operations: [{ op: 'add', tensor: 'model.layers.2.mlp.down_proj.weight', origin: [0, 0], shape: [1, 2], values: [0.01, -0.02] }],
    tensors: [], tool: 'brewery-weights', tool_version: '1', runtime_version: 'fixture', created_at_ms: 1_700_000_000_000,
  },
  delta: { changed_tensors: 1, changed_bytes: 4, delta_bytes: 128, delta_sha256: SHA },
};

const sv = (top, n = 48) => Array.from({ length: n }, (_, i) => top / (1 + i) ** 1.1);
const side = (er, sr, alpha, top = 20) => ({ dtype: 'BF16', shape: [896, 4864], method: 'randomized', singular_values: sv(top), effective_rank: er, stable_rank: sr, alpha });
const tensors = Array.from({ length: 6 }, (_, layer) => {
  const edited = layer === 2;
  return ['mlp.down_proj.weight', 'self_attn.q_proj.weight'].map((part, j) => {
    const er = 600 - layer * 10 - j * 40, srk = 30 - layer - j * 3, a = 3 + layer * 0.1 + j * 0.2;
    const d = edited && j === 0 ? { er: -45.5, sr: -4.25, a: 0.62 } : { er: 0, sr: 0, a: 0 };
    return {
      name: `model.layers.${layer}.${part}`, layer,
      source: side(er + d.er, srk + d.sr, a + d.a, edited && j === 0 ? 26 : 20),
      reference: side(er, srk, a),
      delta: { effective_rank: d.er, stable_rank: d.sr, alpha: d.a, sigma_max: edited && j === 0 ? 6 : 0 },
    };
  });
}).flat();
tensors.push({ name: 'model.layers.0.input_layernorm.weight', layer: 0, source: { dtype: 'BF16', shape: [896], method: 'none', singular_values: null, effective_rank: null, stable_rank: null, alpha: null }, status: 'unsupported_1d' });

const result = {
  kind: 'weight_spectra', qualification: 'not release-qualified',
  source: { id: REV, sha256: SHA, format: 'safetensors' },
  reference: { id: 'source-a', sha256: SHA_P, format: 'safetensors' },
  tensors,
};
const jobAt = (state, extra = {}) => ({
  id: 'spectra-1', kind: 'weight_spectra', host: 'worker-a', input_ids: [REV, 'source-a'], options: {}, state,
  created_at: 1_700_000_000_000, updated_at: 1_700_000_000_000, has_result: state === 'succeeded', events: [], ...extra,
});

async function setup(controller) {
  controller.respond('GET', '/api/scratch', { json: [rev] });
  controller.respond('POST', '/api/weight-spectra', { status: 202, json: { id: 'spectra-1' } });
  let reads = 0;
  controller.respond('GET', '/api/jobs/spectra-1', () => ({ json: ++reads < 2 ? jobAt('running') : jobAt('succeeded', { result }) }));
}

test('Edits weight health: queue spectra against the parent and read the charts and table', async ({ page, controller }, info) => {
  await setup(controller);
  await controller.open(page, `edits?rev=${REV}`);
  const panel = page.getByRole('region', { name: 'Weight health' });
  await expect(panel).toBeVisible();
  await expect(panel.getByText('Reference: the parent', { exact: false })).toBeVisible();

  const run = panel.getByRole('button', { name: 'Measure spectra' });
  await run.focus();
  await page.keyboard.press('Enter');
  await expect(panel.getByText('Not release-qualified').first()).toBeVisible();

  const posts = controller.requests('POST', '/api/weight-spectra');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toEqual({ id: REV });
  expect(posts[0].headers['idempotency-key']).toBeTruthy();

  // Per-layer charts with the delta highlighted.
  for (const name of ['Effective rank by layer', 'Stable rank by layer', 'Alpha by layer']) {
    await expect(panel.getByRole('img', { name })).toBeVisible();
  }
  await expect(panel.getByText('Largest change: layer 2', { exact: false }).first()).toBeVisible();

  // The values table has labelled values.
  const table = panel.getByRole('table', { name: /Per-layer values/ });
  const row2 = table.getByRole('row', { name: /^(Layer )?2 / });
  await expect(row2.getByRole('cell').first()).toBeVisible();
  await expect(row2).toContainText('Changed');

  // Filter by tensor type.
  await panel.getByLabel('Tensor type').selectOption('mlp.down_proj.weight');
  await expect(table.getByRole('row', { name: /^(Layer )?2 / })).toContainText('-45.5');

  // Spectrum of a selected tensor.
  await panel.getByLabel('Spectrum tensor').selectOption('model.layers.2.mlp.down_proj.weight');
  await expect(panel.getByRole('img', { name: /Singular values of model.layers.2.mlp.down_proj.weight/ })).toBeVisible();
  await expect(panel.getByText('σmax 26', { exact: false })).toBeVisible();

  // The per-tensor table reports unsupported input.
  await panel.getByText('All tensor values').click();
  await expect(panel.getByText('unsupported_1d')).toBeVisible();

  await expect(page.locator('html')).toHaveAttribute('data-theme', info.project.use.colorScheme);
  const { violations } = await new AxeBuilder({ page }).include('#weight-health').withTags(wcagAA).analyze();
  expect(violations.flatMap(({ id, nodes }) => nodes.map(({ target }) => ({ id, target })))).toEqual([]);

  if (process.env.X2_SHOTS) {
    mkdirSync(process.env.X2_SHOTS, { recursive: true });
    await panel.screenshot({ path: `${process.env.X2_SHOTS}/weight-health-${info.project.name}.png`, animations: 'disabled' });
    await page.screenshot({ path: `${process.env.X2_SHOTS}/edits-${info.project.name}.png`, fullPage: true });
  }
  // The page must not scroll sideways.
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('Edits weight health: a controller rejection stays beside the action', async ({ page, controller }) => {
  await setup(controller);
  controller.respond('POST', '/api/weight-spectra', { status: 400, json: { error: 'compare_id is not a model' } });
  await controller.open(page, `edits?rev=${REV}`);
  const panel = page.getByRole('region', { name: 'Weight health' });
  await panel.getByText('Options').click();
  await panel.getByLabel('Layer').fill('x');
  await expect(panel.getByRole('button', { name: 'Measure spectra' })).toBeDisabled();
  await expect(panel.getByText('The layer must be a whole number.')).toBeVisible();
  await panel.getByLabel('Layer').fill('2');
  await panel.getByRole('button', { name: 'Measure spectra' }).click();
  await expect(panel.getByRole('alert')).toContainText('compare_id is not a model');
  expect(controller.requests('POST', '/api/weight-spectra')[0].body).toEqual({ id: REV, layer: 2 });
});
